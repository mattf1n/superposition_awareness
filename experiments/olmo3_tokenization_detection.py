import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TEXT_CANDIDATES = [
    "the red apple",
    "soft light filled the kitchen",
    "the window faced the river",
    "music drifted through the hall",
    "the candle lit the room",
    "the school bell rang early",
    "green vines covered the wall",
    "paper boats crossed the pond",
    "small waves touched the shore",
    "the silver key unlocked the door",
    "warm coffee waited on the desk",
    "a camera rested on the table",
    "a yellow train left at dawn",
    "the market opened at dawn",
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

LABELS = ("canonical", "character", "random")
ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyz ")


@dataclass
class TrialResult:
    text: str
    true_label: str
    predicted_label: str
    token_count: int
    canonical_token_count: int
    character_token_count: int
    random_token_count: int
    generated_text: str
    token_pieces: list[str]
    exact_match: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether OLMo 3 can detect non-canonical tokenization from embedded token streams."
    )
    parser.add_argument(
        "--model",
        default="allenai/Olmo-3-7B-Instruct",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--num-texts",
        type=int,
        default=10,
        help="How many texts to include; each text yields one trial per tokenization scheme.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        help="Maximum decoded answer length.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed used for text selection and random tokenization.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory where JSON output will be written.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def decode_piece(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)


def build_piece_index(tokenizer) -> tuple[dict[str, list[tuple[str, int]]], dict[str, int]]:
    pieces_by_first: dict[str, list[tuple[str, int]]] = defaultdict(list)
    single_char_tokens: dict[str, int] = {}
    special_ids = set(tokenizer.all_special_ids)

    for token_id in range(tokenizer.vocab_size):
        if token_id in special_ids:
            continue
        piece = decode_piece(tokenizer, token_id)
        if not piece:
            continue
        if any(ch not in ALLOWED_CHARS for ch in piece):
            continue
        pieces_by_first[piece[0]].append((piece, token_id))
        if len(piece) == 1 and piece not in single_char_tokens:
            single_char_tokens[piece] = token_id

    for key in pieces_by_first:
        pieces_by_first[key].sort(key=lambda item: (len(item[0]), item[0], item[1]), reverse=True)
    return pieces_by_first, single_char_tokens


def build_segmenter(text: str, pieces_by_first: dict[str, list[tuple[str, int]]]):
    @lru_cache(maxsize=None)
    def valid_choices(position: int) -> tuple[tuple[str, int], ...]:
        if position == len(text):
            return ()
        first_char = text[position]
        candidates: list[tuple[str, int]] = []
        for piece, token_id in pieces_by_first.get(first_char, ()):
            if text.startswith(piece, position) and can_finish(position + len(piece)):
                candidates.append((piece, token_id))
        return tuple(candidates)

    @lru_cache(maxsize=None)
    def can_finish(position: int) -> bool:
        if position == len(text):
            return True
        return any(can_finish(position + len(piece)) for piece, _ in valid_choices(position))

    return valid_choices, can_finish


def build_character_tokenization(text: str, single_char_tokens: dict[str, int]) -> list[int] | None:
    token_ids: list[int] = []
    for ch in text:
        token_id = single_char_tokens.get(ch)
        if token_id is None:
            return None
        token_ids.append(token_id)
    return token_ids


def sample_random_tokenization(
    text: str,
    valid_choices,
    canonical_ids: list[int],
    character_ids: list[int],
    seed: int,
) -> list[int] | None:
    for attempt in range(256):
        rng = random.Random(seed + attempt)
        position = 0
        sampled_ids: list[int] = []
        while position < len(text):
            choices = list(valid_choices(position))
            if not choices:
                break
            piece, token_id = rng.choice(choices)
            sampled_ids.append(token_id)
            position += len(piece)
        if position != len(text):
            continue
        if sampled_ids != canonical_ids and sampled_ids != character_ids:
            return sampled_ids
    return None


def select_texts(
    tokenizer,
    num_texts: int,
    seed: int,
) -> list[dict[str, object]]:
    pieces_by_first, single_char_tokens = build_piece_index(tokenizer)
    usable: list[dict[str, object]] = []

    for idx, text in enumerate(TEXT_CANDIDATES):
        canonical_ids = tokenizer.encode(text, add_special_tokens=False)
        if normalize_text(tokenizer.decode(canonical_ids, clean_up_tokenization_spaces=False)) != text:
            continue

        character_ids = build_character_tokenization(text, single_char_tokens)
        if character_ids is None:
            continue

        valid_choices, can_finish = build_segmenter(text, pieces_by_first)
        if not can_finish(0):
            continue

        random_ids = sample_random_tokenization(
            text=text,
            valid_choices=valid_choices,
            canonical_ids=canonical_ids,
            character_ids=character_ids,
            seed=seed + (idx * 1000),
        )
        if random_ids is None:
            continue

        usable.append(
            {
                "text": text,
                "canonical_ids": canonical_ids,
                "character_ids": character_ids,
                "random_ids": random_ids,
            }
        )

    if len(usable) < num_texts:
        raise RuntimeError(f"Requested {num_texts} texts, but only {len(usable)} support all three tokenizations.")

    rng = random.Random(seed)
    rng.shuffle(usable)
    return usable[:num_texts]


def select_placeholder_token_id(tokenizer) -> tuple[int, str]:
    for word in PLACEHOLDER_CANDIDATES:
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(token_ids) == 1 and decode_piece(tokenizer, token_ids[0]) == word:
            return token_ids[0], word

    special_ids = set(tokenizer.all_special_ids)
    for token_id in range(tokenizer.vocab_size):
        if token_id in special_ids:
            continue
        piece = decode_piece(tokenizer, token_id)
        if piece and all(ch in ALLOWED_CHARS for ch in piece):
            return token_id, piece
    raise RuntimeError("Could not find a placeholder token.")


def build_prompt(
    tokenizer,
    placeholder_text: str,
    token_count: int,
) -> tuple[torch.Tensor, list[int]]:
    token_stream = " ".join([placeholder_text] * token_count)
    placeholder_stream_ids = tokenizer.encode(token_stream, add_special_tokens=False)
    if len(placeholder_stream_ids) != token_count:
        raise RuntimeError(
            f"Expected placeholder stream to encode to {token_count} tokens, found {len(placeholder_stream_ids)}."
        )

    user_content = (
        "You are given a hidden text as a sequence of synthetic embedding tokens after the marker TOKEN.\n"
        "The underlying text is the same lowercase text in every case, but the text was tokenized in exactly one of three ways before embedding:\n"
        "canonical\n"
        "character\n"
        "random\n\n"
        "Definitions:\n"
        "- canonical: the model's normal tokenizer output.\n"
        "- character: one token per character, including spaces.\n"
        "- random: random valid tokens were chosen so that their decoded pieces still exactly cover the full text.\n\n"
        "Reply with exactly one label: canonical, character, or random.\n\n"
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
            raise RuntimeError("Could not locate the placeholder token stream inside the prompt.")
        placeholder_indices = list(range(start_index, start_index + len(placeholder_stream_ids)))
        return torch.tensor([input_ids]), placeholder_indices

    prefix = user_content.rsplit(token_stream, maxsplit=1)[0]
    suffix = "\n\nAnswer:\n"
    input_ids = tokenizer.encode(prefix, add_special_tokens=False)
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


def parse_label(generated_text: str) -> str:
    normalized = normalize_text(generated_text)
    for label in LABELS:
        if label in normalized:
            return label
    return "unparsed"


def evaluate_trial(
    tokenizer,
    model,
    embedding_layer,
    placeholder_text: str,
    text: str,
    label: str,
    token_ids: list[int],
    canonical_ids: list[int],
    character_ids: list[int],
    random_ids: list[int],
    max_new_tokens: int,
    device: torch.device,
) -> TrialResult:
    input_ids, placeholder_indices = build_prompt(tokenizer, placeholder_text, len(token_ids))
    input_ids = input_ids.to(device)

    with torch.no_grad():
        inputs_embeds = embedding_layer(input_ids)
        source_embeddings = embedding_layer(torch.tensor([token_ids], device=device))[0]
        for source_index, placeholder_index in enumerate(placeholder_indices):
            inputs_embeds[0, placeholder_index] = source_embeddings[source_index]

        generated_ids = greedy_decode(
            model=model,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    predicted_label = parse_label(generated_text)
    token_pieces = [decode_piece(tokenizer, token_id) for token_id in token_ids]

    return TrialResult(
        text=text,
        true_label=label,
        predicted_label=predicted_label,
        token_count=len(token_ids),
        canonical_token_count=len(canonical_ids),
        character_token_count=len(character_ids),
        random_token_count=len(random_ids),
        generated_text=generated_text,
        token_pieces=token_pieces,
        exact_match=(predicted_label == label),
    )


def summarize(results: list[TrialResult]) -> dict[str, object]:
    overall_accuracy = sum(result.exact_match for result in results) / len(results) if results else 0.0
    by_label: dict[str, dict[str, object]] = {}
    confusion: dict[str, dict[str, int]] = {label: {} for label in LABELS}
    confusion["unparsed"] = {}

    grouped: dict[str, list[TrialResult]] = defaultdict(list)
    for result in results:
        grouped[result.true_label].append(result)
        confusion.setdefault(result.true_label, {})
        confusion[result.true_label][result.predicted_label] = confusion[result.true_label].get(result.predicted_label, 0) + 1

    for label in LABELS:
        label_results = grouped[label]
        correct = sum(result.exact_match for result in label_results)
        by_label[label] = {
            "trials": len(label_results),
            "correct": correct,
            "accuracy": correct / len(label_results) if label_results else 0.0,
        }

    return {
        "overall_accuracy": overall_accuracy,
        "total_trials": len(results),
        "by_label": by_label,
        "confusion": confusion,
    }


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

    texts = select_texts(tokenizer, args.num_texts, args.seed)
    placeholder_token_id, placeholder_text = select_placeholder_token_id(tokenizer)
    print(
        f"[info] selected {len(texts)} texts; placeholder token={placeholder_text!r} id={placeholder_token_id}",
        flush=True,
    )

    results: list[TrialResult] = []
    for text_spec in texts:
        text = text_spec["text"]
        variants = {
            "canonical": text_spec["canonical_ids"],
            "character": text_spec["character_ids"],
            "random": text_spec["random_ids"],
        }
        for label in LABELS:
            result = evaluate_trial(
                tokenizer=tokenizer,
                model=model,
                embedding_layer=embedding_layer,
                placeholder_text=placeholder_text,
                text=text,
                label=label,
                token_ids=variants[label],
                canonical_ids=text_spec["canonical_ids"],
                character_ids=text_spec["character_ids"],
                random_ids=text_spec["random_ids"],
                max_new_tokens=args.max_new_tokens,
                device=device,
            )
            print(
                f"[trial] text={text!r} true={label} predicted={result.predicted_label} "
                f"tokens={result.token_count} generated={result.generated_text!r}",
                flush=True,
            )
            results.append(result)

    summary = summarize(results)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"olmo3_tokenization_detection_{timestamp}.json"
    payload = {
        "timestamp_utc": timestamp,
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "seed": args.seed,
        "num_texts": args.num_texts,
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
