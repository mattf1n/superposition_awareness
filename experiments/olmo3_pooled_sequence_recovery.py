import argparse
import json
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SEQUENCE_CANDIDATES = [
    "the red apple",
    "a quiet summer rain",
    "music drifted through the hall",
    "the window faced the river",
    "paper boats crossed the pond",
    "a brown dog chased leaves",
    "the candle lit the room",
    "small waves touched the shore",
    "the school bell rang early",
    "green vines covered the wall",
    "a camera rested on the table",
    "the storm rolled over town",
    "soft light filled the kitchen",
    "the old bridge crossed the water",
    "bright stars covered the sky",
    "a child opened the window",
    "the radio played all night",
    "fresh bread cooled on the rack",
    "the garden gate stood open",
    "a yellow train left at dawn",
    "the artist sketched the harbor",
    "morning fog covered the field",
    "the silver key unlocked the door",
    "warm coffee waited on the desk",
]

PLACEHOLDER_CANDIDATES = [
    "travel",
    "window",
    "garden",
    "music",
    "paper",
    "river",
    "signal",
    "camera",
    "ticket",
]


@dataclass
class TrialResult:
    mode: str
    target_text: str
    target_word_count: int
    target_token_count: int
    generated_text: str
    candidate_answers: list[str]
    contains_target_substring: bool
    exact_normalized_match: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether an OLMo 3 model can verbalize natural-language sequences from pooled embeddings."
    )
    parser.add_argument(
        "--model",
        default="allenai/Olmo-3-7B-Instruct",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=12,
        help="How many natural-language sequences to test.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=24,
        help="Maximum decoded answer length.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed used to sample sequences.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory where JSON output will be written.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    normalized = text.lower()
    normalized = normalized.replace("`", " ")
    normalized = normalized.replace('"', " ")
    normalized = normalized.replace("'", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"^[^a-z0-9]+", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+$", "", normalized)
    return normalized


def extract_candidate_answers(text: str) -> list[str]:
    candidates: list[str] = []
    raw_candidates: list[str] = [text]

    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            raw_candidates.append(stripped)

    for quote_match in re.findall(r'"([^"]+)"', text):
        raw_candidates.append(quote_match)
    for quote_match in re.findall(r"'([^']+)'", text):
        raw_candidates.append(quote_match)

    seen: set[str] = set()
    normalized_candidates: list[str] = []
    for candidate in raw_candidates:
        normalized = normalize_text(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        normalized_candidates.append(normalized)

    return normalized_candidates


def build_prompt(
    tokenizer,
    placeholder_token_id: int,
    placeholder_text: str,
    mode: str,
) -> tuple[torch.Tensor, int]:
    pool_name = "mean-pooling" if mode == "mean" else "elementwise max-pooling"
    user_content = (
        "You are given one synthetic embedding token after the marker TOKEN.\n"
        f"It was created by {pool_name} the input embeddings of every token in one short lowercase natural-language sequence from this model.\n"
        "The original sequence has between 3 and 6 words.\n"
        "Recover the original sequence exactly.\n"
        "Reply with only the original sequence and no explanation.\n\nTOKEN:\n"
        f"{placeholder_text}"
    )

    if getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        occurrences = [idx for idx, token_id in enumerate(input_ids) if token_id == placeholder_token_id]
        if len(occurrences) != 1:
            raise RuntimeError(
                f"Expected exactly one placeholder token occurrence in chat prompt, found {len(occurrences)}."
            )
        return torch.tensor([input_ids]), occurrences[0]

    instruction_prefix = user_content.rsplit(placeholder_text, maxsplit=1)[0]
    suffix = "\n\nAnswer:\n"
    input_ids: list[int] = []
    if tokenizer.bos_token_id is not None:
        input_ids.append(tokenizer.bos_token_id)
    input_ids.extend(tokenizer.encode(instruction_prefix, add_special_tokens=False))
    placeholder_index = len(input_ids)
    input_ids.append(placeholder_token_id)
    input_ids.extend(tokenizer.encode(suffix, add_special_tokens=False))
    return torch.tensor([input_ids]), placeholder_index


def greedy_decode(
    model,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> list[int]:
    attention_mask = torch.ones(input_ids.shape, device=inputs_embeds.device, dtype=torch.long)
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated: list[int] = []

    for _ in range(max_new_tokens):
        token_id = int(next_token.item())
        if eos_token_id is not None and token_id == eos_token_id:
            break
        generated.append(token_id)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype)],
            dim=1,
        )
        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    return generated


def select_sequences(tokenizer, count: int, seed: int) -> list[tuple[str, list[int]]]:
    usable: list[tuple[str, list[int]]] = []
    for sequence in SEQUENCE_CANDIDATES:
        token_ids = tokenizer.encode(sequence, add_special_tokens=False)
        if len(token_ids) < 2:
            continue
        usable.append((sequence, token_ids))

    if len(usable) < count:
        raise RuntimeError(f"Requested {count} sequences, but only {len(usable)} are usable.")

    rng = random.Random(seed)
    rng.shuffle(usable)
    return usable[:count]


def select_placeholder_token_id(tokenizer) -> tuple[int, str]:
    for word in PLACEHOLDER_CANDIDATES:
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(token_ids) == 1 and tokenizer.decode(token_ids, clean_up_tokenization_spaces=False) == word:
            return token_ids[0], word

    for token_id in range(tokenizer.vocab_size):
        if token_id in set(tokenizer.all_special_ids):
            continue
        decoded = tokenizer.decode([token_id], clean_up_tokenization_spaces=False).strip()
        if decoded:
            return token_id, decoded

    raise RuntimeError("Could not find a placeholder token.")


def evaluate_trial(
    tokenizer,
    model,
    embedding_layer,
    placeholder_token_id: int,
    placeholder_text: str,
    target_text: str,
    target_token_ids: list[int],
    mode: str,
    max_new_tokens: int,
    device: torch.device,
) -> TrialResult:
    input_ids, placeholder_index = build_prompt(tokenizer, placeholder_token_id, placeholder_text, mode)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        inputs_embeds = embedding_layer(input_ids)
        source_embeddings = embedding_layer(torch.tensor([target_token_ids], device=device))
        if mode == "mean":
            pooled = source_embeddings.mean(dim=1)[0]
        elif mode == "max":
            pooled = source_embeddings.max(dim=1).values[0]
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        inputs_embeds[0, placeholder_index] = pooled

        generated_ids = greedy_decode(
            model=model,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    candidate_answers = extract_candidate_answers(generated_text)
    normalized_target = normalize_text(target_text)
    exact_normalized_match = normalized_target in candidate_answers
    contains_target_substring = normalized_target in normalize_text(generated_text)

    return TrialResult(
        mode=mode,
        target_text=target_text,
        target_word_count=len(target_text.split()),
        target_token_count=len(target_token_ids),
        generated_text=generated_text,
        candidate_answers=candidate_answers,
        contains_target_substring=contains_target_substring,
        exact_normalized_match=exact_normalized_match,
    )


def summarize(results: Iterable[TrialResult]) -> dict[str, dict[str, float | int]]:
    by_mode: dict[str, list[TrialResult]] = {}
    for result in results:
        by_mode.setdefault(result.mode, []).append(result)

    summary: dict[str, dict[str, float | int]] = {}
    for mode, mode_results in by_mode.items():
        count = len(mode_results)
        exact = sum(result.exact_normalized_match for result in mode_results)
        contains = sum(result.contains_target_substring for result in mode_results)
        summary[mode] = {
            "trials": count,
            "contains_target_substring": contains,
            "contains_target_substring_rate": contains / count if count else 0.0,
            "exact_normalized_match": exact,
            "exact_normalized_match_rate": exact / count if count else 0.0,
        }
    return summary


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"[info] loading tokenizer: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[info] loading model on {device}: {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    embedding_layer = model.get_input_embeddings()

    sequences = select_sequences(tokenizer, args.num_sequences, args.seed)
    placeholder_token_id, placeholder_text = select_placeholder_token_id(tokenizer)
    print(
        f"[info] selected {len(sequences)} sequences; placeholder token={placeholder_text!r}",
        flush=True,
    )

    results: list[TrialResult] = []
    for mode in ("mean", "max"):
        for target_text, target_token_ids in sequences:
            result = evaluate_trial(
                tokenizer=tokenizer,
                model=model,
                embedding_layer=embedding_layer,
                placeholder_token_id=placeholder_token_id,
                placeholder_text=placeholder_text,
                target_text=target_text,
                target_token_ids=target_token_ids,
                mode=mode,
                max_new_tokens=args.max_new_tokens,
                device=device,
            )
            print(
                f"[trial] mode={mode} words={result.target_word_count} tokens={result.target_token_count} "
                f"target={target_text!r} generated={result.generated_text!r}",
                flush=True,
            )
            results.append(result)

    summary = summarize(results)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"olmo3_pooled_sequence_recovery_{timestamp}.json"
    payload = {
        "timestamp_utc": timestamp,
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "seed": args.seed,
        "num_sequences": args.num_sequences,
        "max_new_tokens": args.max_new_tokens,
        "placeholder_token": placeholder_text,
        "summary": summary,
        "results": [asdict(result) for result in results],
    }
    output_path.write_text(json.dumps(payload, indent=2))

    print("[summary]", json.dumps(summary, indent=2), flush=True)
    print(f"[info] wrote results to {output_path}", flush=True)


if __name__ == "__main__":
    main()
