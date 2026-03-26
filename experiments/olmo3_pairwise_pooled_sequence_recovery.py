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


PAIR_CANDIDATES = [
    ("soft light filled the kitchen", "bright stars covered the sky"),
    ("the radio played all night", "music drifted through the hall"),
    ("the candle lit the room", "a child opened the window"),
    ("the storm rolled over town", "the market opened at dawn"),
    ("the school bell rang early", "green vines covered the wall"),
    ("the window faced the river", "paper boats crossed the pond"),
    ("small waves touched the shore", "the garden gate stood open"),
    ("the old bridge crossed the water", "fresh bread cooled on the rack"),
    ("the silver key unlocked the door", "warm coffee waited on the desk"),
    ("a camera rested on the table", "a yellow train left at dawn"),
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
    target_a: str
    target_b: str
    token_count: int
    generated_text: str
    candidate_pairs: list[list[str]]
    contains_both_targets: bool
    exact_ordered_match: bool
    exact_unordered_match: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether an OLMo 3 model can verbalize two aligned sequences from pairwise pooled embeddings."
    )
    parser.add_argument(
        "--model",
        default="allenai/Olmo-3-7B-Instruct",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=10,
        help="How many aligned sequence pairs to test.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum decoded answer length.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed used to sample sequence pairs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory where JSON output will be written.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    normalized = text.lower().strip()
    normalized = re.sub(r"^[\-\*\d\.\)\(:\s]+", "", normalized)
    normalized = normalized.replace("`", " ")
    normalized = normalized.replace('"', " ")
    normalized = normalized.replace("'", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"^[^a-z0-9]+", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+$", "", normalized)
    return normalized


def extract_candidate_pairs(text: str) -> list[list[str]]:
    candidates: list[list[str]] = []
    seen: set[tuple[str, str]] = set()

    def add_pair(raw_a: str, raw_b: str) -> None:
        pair = (normalize_text(raw_a), normalize_text(raw_b))
        if not pair[0] or not pair[1]:
            return
        if pair in seen:
            return
        seen.add(pair)
        candidates.append([pair[0], pair[1]])

    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_lines = [re.sub(r"^(assistant|answer)\s*[:\-]?\s*", "", line, flags=re.IGNORECASE) for line in raw_lines]

    if len(clean_lines) >= 2:
        add_pair(clean_lines[0], clean_lines[1])

    separators = ["|", ";", "/", ",", " || "]
    for line in clean_lines:
        for separator in separators:
            if line.count(separator) == 1:
                left, right = line.split(separator, maxsplit=1)
                add_pair(left, right)

    numbered = []
    for line in raw_lines:
        match = re.match(r"^\s*(?:1|2)[\.\)]\s*(.+)$", line)
        if match:
            numbered.append(match.group(1))
    if len(numbered) >= 2:
        add_pair(numbered[0], numbered[1])

    return candidates


def build_prompt(
    tokenizer,
    placeholder_text: str,
    pooled_length: int,
    mode: str,
) -> tuple[torch.Tensor, list[int]]:
    pool_name = "mean-pooling" if mode == "mean" else "elementwise max-pooling"
    token_stream = " ".join([placeholder_text] * pooled_length)
    placeholder_stream_ids = tokenizer.encode(token_stream, add_special_tokens=False)
    if len(placeholder_stream_ids) != pooled_length:
        raise RuntimeError(
            f"Expected placeholder stream to encode to {pooled_length} tokens, found {len(placeholder_stream_ids)}."
        )
    user_content = (
        "You are given one synthetic token sequence after the marker TOKEN.\n"
        f"Each position was created by {pool_name} the corresponding token embeddings from two original lowercase natural-language sequences in this model.\n"
        f"Both original sequences had exactly {pooled_length} tokens.\n"
        "Recover both original sequences.\n"
        "Reply with exactly two lines and no explanation:\n"
        "line 1: first sequence\n"
        "line 2: second sequence\n\n"
        "TOKEN:\n"
        f"{token_stream}"
    )

    if getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        start_index = -1
        for idx in range(len(input_ids) - len(placeholder_stream_ids) + 1):
            if input_ids[idx : idx + len(placeholder_stream_ids)] == placeholder_stream_ids:
                start_index = idx
                break
        if start_index < 0:
            raise RuntimeError(
                "Could not locate the placeholder token stream inside the chat-formatted prompt."
            )
        placeholder_indices = list(range(start_index, start_index + len(placeholder_stream_ids)))
        return torch.tensor([input_ids]), placeholder_indices

    instruction_prefix = user_content.rsplit(token_stream, maxsplit=1)[0]
    suffix = "\n\nAnswer:\n"
    input_ids: list[int] = []
    if tokenizer.bos_token_id is not None:
        input_ids.append(tokenizer.bos_token_id)
    input_ids.extend(tokenizer.encode(instruction_prefix, add_special_tokens=False))
    placeholder_indices = list(range(len(input_ids), len(input_ids) + len(placeholder_stream_ids)))
    input_ids.extend(placeholder_stream_ids)
    input_ids.extend(tokenizer.encode(suffix, add_special_tokens=False))
    return torch.tensor([input_ids]), placeholder_indices


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


def select_sequence_pairs(tokenizer, count: int, seed: int) -> list[tuple[str, str, list[int], list[int]]]:
    usable: list[tuple[str, str, list[int], list[int]]] = []
    for sequence_a, sequence_b in PAIR_CANDIDATES:
        token_ids_a = tokenizer.encode(sequence_a, add_special_tokens=False)
        token_ids_b = tokenizer.encode(sequence_b, add_special_tokens=False)
        if len(token_ids_a) < 2 or len(token_ids_a) != len(token_ids_b):
            continue
        usable.append((sequence_a, sequence_b, token_ids_a, token_ids_b))

    if len(usable) < count:
        raise RuntimeError(f"Requested {count} pairs, but only {len(usable)} are usable.")

    rng = random.Random(seed)
    rng.shuffle(usable)
    return usable[:count]


def select_placeholder_token_id(tokenizer) -> tuple[int, str]:
    for word in PLACEHOLDER_CANDIDATES:
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(token_ids) == 1 and tokenizer.decode(token_ids, clean_up_tokenization_spaces=False) == word:
            return token_ids[0], word

    special_ids = set(tokenizer.all_special_ids)
    for token_id in range(tokenizer.vocab_size):
        if token_id in special_ids:
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
    target_a: str,
    target_b: str,
    token_ids_a: list[int],
    token_ids_b: list[int],
    mode: str,
    max_new_tokens: int,
    device: torch.device,
) -> TrialResult:
    input_ids, placeholder_indices = build_prompt(
        tokenizer=tokenizer,
        placeholder_text=placeholder_text,
        pooled_length=len(token_ids_a),
        mode=mode,
    )
    input_ids = input_ids.to(device)

    with torch.no_grad():
        inputs_embeds = embedding_layer(input_ids)
        source_a = embedding_layer(torch.tensor([token_ids_a], device=device))[0]
        source_b = embedding_layer(torch.tensor([token_ids_b], device=device))[0]
        if mode == "mean":
            pooled = (source_a + source_b) / 2
        elif mode == "max":
            pooled = torch.maximum(source_a, source_b)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        for pool_index, placeholder_index in enumerate(placeholder_indices):
            inputs_embeds[0, placeholder_index] = pooled[pool_index]

        generated_ids = greedy_decode(
            model=model,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    candidate_pairs = extract_candidate_pairs(generated_text)
    normalized_a = normalize_text(target_a)
    normalized_b = normalize_text(target_b)
    ordered = [normalized_a, normalized_b]
    swapped = [normalized_b, normalized_a]
    exact_ordered_match = ordered in candidate_pairs
    exact_unordered_match = exact_ordered_match or swapped in candidate_pairs
    normalized_generation = normalize_text(generated_text)
    contains_both_targets = normalized_a in normalized_generation and normalized_b in normalized_generation

    return TrialResult(
        mode=mode,
        target_a=target_a,
        target_b=target_b,
        token_count=len(token_ids_a),
        generated_text=generated_text,
        candidate_pairs=candidate_pairs,
        contains_both_targets=contains_both_targets,
        exact_ordered_match=exact_ordered_match,
        exact_unordered_match=exact_unordered_match,
    )


def summarize(results: Iterable[TrialResult]) -> dict[str, dict[str, float | int]]:
    by_mode: dict[str, list[TrialResult]] = {}
    for result in results:
        by_mode.setdefault(result.mode, []).append(result)

    summary: dict[str, dict[str, float | int]] = {}
    for mode, mode_results in by_mode.items():
        count = len(mode_results)
        contains = sum(result.contains_both_targets for result in mode_results)
        ordered = sum(result.exact_ordered_match for result in mode_results)
        unordered = sum(result.exact_unordered_match for result in mode_results)
        summary[mode] = {
            "trials": count,
            "contains_both_targets": contains,
            "contains_both_targets_rate": contains / count if count else 0.0,
            "exact_ordered_match": ordered,
            "exact_ordered_match_rate": ordered / count if count else 0.0,
            "exact_unordered_match": unordered,
            "exact_unordered_match_rate": unordered / count if count else 0.0,
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

    sequence_pairs = select_sequence_pairs(tokenizer, args.num_pairs, args.seed)
    placeholder_token_id, placeholder_text = select_placeholder_token_id(tokenizer)
    print(
        f"[info] selected {len(sequence_pairs)} aligned sequence pairs; placeholder token={placeholder_text!r}",
        flush=True,
    )

    results: list[TrialResult] = []
    for mode in ("mean", "max"):
        for target_a, target_b, token_ids_a, token_ids_b in sequence_pairs:
            result = evaluate_trial(
                tokenizer=tokenizer,
                model=model,
                embedding_layer=embedding_layer,
                placeholder_token_id=placeholder_token_id,
                placeholder_text=placeholder_text,
                target_a=target_a,
                target_b=target_b,
                token_ids_a=token_ids_a,
                token_ids_b=token_ids_b,
                mode=mode,
                max_new_tokens=args.max_new_tokens,
                device=device,
            )
            print(
                f"[trial] mode={mode} tokens={result.token_count} "
                f"target_a={target_a!r} target_b={target_b!r} generated={result.generated_text!r}",
                flush=True,
            )
            results.append(result)

    summary = summarize(results)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"olmo3_pairwise_pooled_sequence_recovery_{timestamp}.json"
    payload = {
        "timestamp_utc": timestamp,
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "seed": args.seed,
        "num_pairs": args.num_pairs,
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
