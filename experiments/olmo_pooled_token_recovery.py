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


WORD_RE = re.compile(r"[a-z]{3,8}")
COMMON_WORD_CANDIDATES = [
    "apple", "baker", "beach", "black", "blue", "bread", "brick", "brown",
    "cabin", "candy", "chair", "cloud", "coast", "coffee", "dance", "delta",
    "dream", "earth", "field", "flame", "flower", "forest", "garden", "glass",
    "grape", "green", "happy", "honey", "house", "light", "magic", "metal",
    "money", "moon", "music", "ocean", "olive", "orange", "paper", "pearl",
    "pepper", "piano", "pilot", "planet", "plaza", "poetry", "queen", "quiet",
    "radio", "rain", "river", "robot", "rocket", "round", "sable", "salad",
    "scale", "shadow", "silver", "smile", "snow", "solar", "sound", "spark",
    "spice", "spirit", "spring", "stone", "storm", "sugar", "summer", "sunset",
    "table", "thunder", "tiger", "toast", "travel", "velvet", "violet", "water",
    "whisper", "window", "winter", "yellow", "zebra", "almond", "artist", "autumn",
    "basket", "button", "camera", "castle", "copper", "cotton", "desert", "dragon",
    "engine", "fabric", "feather", "galaxy", "gentle", "harbor", "island", "jungle",
    "kitten", "ladder", "lantern", "marble", "meadow", "mirror", "moment", "monkey",
    "mountain", "novel", "orchard", "pencil", "peppermint", "planet", "pocket", "prairie",
    "puppet", "purple", "raven", "ribbon", "sailor", "saturn", "school", "signal",
    "spider", "spoon", "star", "studio", "ticket", "turbo", "valley", "violin",
    "walnut", "willow", "yogurt",
]


@dataclass
class TrialResult:
    mode: str
    target_tokens: list[str]
    prompt_suffix: str
    generated_text: str
    generated_words: list[str]
    contains_all_targets: bool
    exact_match_first_n: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether an OLMo model can verbalize tokens from pooled embeddings."
    )
    parser.add_argument(
        "--model",
        default="allenai/OLMo-7B-Instruct-hf",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=24,
        help="How many two-token pairs to evaluate for mean/max pooling.",
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=256,
        help="How many clean single-token words to sample candidate pairs from.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=12,
        help="Maximum answer length to decode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed used for candidate selection and pairing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory where JSON output will be written.",
    )
    return parser.parse_args()


def roundtrip_single_token(tokenizer, token_id: int) -> str | None:
    if token_id in set(tokenizer.all_special_ids):
        return None
    decoded = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    if not WORD_RE.fullmatch(decoded):
        return None
    if tokenizer.encode(decoded, add_special_tokens=False) != [token_id]:
        return None
    return decoded


def collect_candidate_words(tokenizer, target_count: int) -> list[tuple[int, str]]:
    candidates: list[tuple[int, str]] = []
    seen_ids: set[int] = set()

    for word in COMMON_WORD_CANDIDATES:
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(token_ids) != 1:
            continue
        token_id = token_ids[0]
        decoded = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        if decoded != word:
            continue
        if token_id in seen_ids:
            continue
        seen_ids.add(token_id)
        candidates.append((token_id, decoded))
        if len(candidates) >= target_count:
            return candidates

    special_ids = set(tokenizer.all_special_ids)
    for token_id in range(tokenizer.vocab_size):
        if token_id in special_ids:
            continue
        decoded = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        if not WORD_RE.fullmatch(decoded):
            continue
        if tokenizer.encode(decoded, add_special_tokens=False) != [token_id]:
            continue
        if token_id in seen_ids:
            continue
        seen_ids.add(token_id)
        candidates.append((token_id, decoded))
        if len(candidates) >= target_count:
            break
    return candidates


def build_prompt(tokenizer, placeholder_token_id: int, mode: str) -> tuple[torch.Tensor, int, str]:
    if mode == "single":
        instruction = (
            "You are given one synthetic embedding token after the marker TOKEN.\n"
            "It is exactly the input embedding of one lowercase English word token from this model.\n"
            "Recover that word.\n"
            "Reply with only the word.\n\nTOKEN:\n"
        )
        suffix = "\n\nAnswer:\n"
    else:
        pool_name = "mean-pooling" if mode == "mean" else "elementwise max-pooling"
        instruction = (
            "You are given one synthetic embedding token after the marker TOKEN.\n"
            f"It was created by {pool_name} the input embeddings of exactly two lowercase English word tokens from this model.\n"
            "Recover the original two words.\n"
            "Reply with only the two words, separated by a comma.\n\nTOKEN:\n"
        )
        suffix = "\n\nAnswer:\n"

    input_ids: list[int] = []
    if tokenizer.bos_token_id is not None:
        input_ids.append(tokenizer.bos_token_id)
    input_ids.extend(tokenizer.encode(instruction, add_special_tokens=False))
    placeholder_index = len(input_ids)
    input_ids.append(placeholder_token_id)
    input_ids.extend(tokenizer.encode(suffix, add_special_tokens=False))
    return torch.tensor([input_ids]), placeholder_index, suffix.strip()


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


def extract_words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def evaluate_trial(
    tokenizer,
    model,
    embedding_layer,
    placeholder_token_id: int,
    target_token_ids: list[int],
    target_words: list[str],
    mode: str,
    max_new_tokens: int,
    device: torch.device,
) -> TrialResult:
    input_ids, placeholder_index, prompt_suffix = build_prompt(tokenizer, placeholder_token_id, mode)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        inputs_embeds = embedding_layer(input_ids)
        source_embeddings = embedding_layer(torch.tensor([target_token_ids], device=device))
        if mode == "single":
            pooled = source_embeddings[0, 0]
        elif mode == "mean":
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
    generated_words = extract_words(generated_text)
    target_set = set(target_words)
    contains_all_targets = target_set.issubset(set(generated_words))
    exact_match_first_n = generated_words[: len(target_words)] == target_words or generated_words[: len(target_words)] == list(
        reversed(target_words)
    )

    return TrialResult(
        mode=mode,
        target_tokens=target_words,
        prompt_suffix=prompt_suffix,
        generated_text=generated_text,
        generated_words=generated_words,
        contains_all_targets=contains_all_targets,
        exact_match_first_n=exact_match_first_n,
    )


def summarize(results: Iterable[TrialResult]) -> dict[str, dict[str, float | int]]:
    by_mode: dict[str, list[TrialResult]] = {}
    for result in results:
        by_mode.setdefault(result.mode, []).append(result)

    summary: dict[str, dict[str, float | int]] = {}
    for mode, mode_results in by_mode.items():
        count = len(mode_results)
        contains = sum(result.contains_all_targets for result in mode_results)
        exact = sum(result.exact_match_first_n for result in mode_results)
        summary[mode] = {
            "trials": count,
            "contains_all_targets": contains,
            "contains_all_targets_rate": contains / count if count else 0.0,
            "exact_match_first_n": exact,
            "exact_match_first_n_rate": exact / count if count else 0.0,
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

    candidate_target_count = max(args.candidate_pool_size, (2 * args.num_pairs) + 16)
    candidates = collect_candidate_words(tokenizer, candidate_target_count)
    if len(candidates) < (2 * args.num_pairs) + 2:
        raise RuntimeError(
            f"Not enough clean single-token lowercase words in the tokenizer vocabulary. Found {len(candidates)}."
        )

    placeholder_token_id, placeholder_word = candidates[-1]
    usable_candidates = candidates[:-1]
    random.shuffle(usable_candidates)

    single_targets = usable_candidates[: args.num_pairs]
    pair_tokens = usable_candidates[args.num_pairs : args.num_pairs + (2 * args.num_pairs)]
    pair_trials = [
        (pair_tokens[2 * idx], pair_tokens[(2 * idx) + 1])
        for idx in range(args.num_pairs)
    ]

    print(
        f"[info] selected {len(single_targets)} single-token controls and {len(pair_trials)} pooled pairs; "
        f"placeholder token='{placeholder_word}'",
        flush=True,
    )

    results: list[TrialResult] = []

    for token_id, token_word in single_targets:
        result = evaluate_trial(
            tokenizer=tokenizer,
            model=model,
            embedding_layer=embedding_layer,
            placeholder_token_id=placeholder_token_id,
            target_token_ids=[token_id],
            target_words=[token_word],
            mode="single",
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        print(
            f"[trial] mode=single target={token_word} generated={result.generated_text!r}",
            flush=True,
        )
        results.append(result)

    for mode in ("mean", "max"):
        for (token_id_a, word_a), (token_id_b, word_b) in pair_trials:
            result = evaluate_trial(
                tokenizer=tokenizer,
                model=model,
                embedding_layer=embedding_layer,
                placeholder_token_id=placeholder_token_id,
                target_token_ids=[token_id_a, token_id_b],
                target_words=[word_a, word_b],
                mode=mode,
                max_new_tokens=args.max_new_tokens,
                device=device,
            )
            print(
                f"[trial] mode={mode} target={[word_a, word_b]} generated={result.generated_text!r}",
                flush=True,
            )
            results.append(result)

    summary = summarize(results)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"olmo_pooled_token_recovery_{timestamp}.json"
    payload = {
        "timestamp_utc": timestamp,
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "seed": args.seed,
        "num_pairs": args.num_pairs,
        "candidate_pool_size": args.candidate_pool_size,
        "max_new_tokens": args.max_new_tokens,
        "placeholder_token": placeholder_word,
        "summary": summary,
        "results": [asdict(result) for result in results],
    }
    output_path.write_text(json.dumps(payload, indent=2))

    print("[summary]", json.dumps(summary, indent=2), flush=True)
    print(f"[info] wrote results to {output_path}", flush=True)


if __name__ == "__main__":
    main()
