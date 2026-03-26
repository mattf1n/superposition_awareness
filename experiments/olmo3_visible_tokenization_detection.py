import argparse
import json
import random
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent))

from olmo3_tokenization_detection import (
    LABELS,
    TEXT_CANDIDATES,
    decode_piece,
    normalize_text,
    select_texts,
)


@dataclass
class TrialResult:
    text: str
    true_label: str
    predicted_label: str
    token_count: int
    generated_text: str
    token_pieces: list[str]
    exact_match: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether OLMo 3 can classify visible token-piece sequences by tokenization scheme."
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


def render_token_pieces(token_pieces: list[str]) -> str:
    return " ".join(json.dumps(piece) for piece in token_pieces)


def build_prompt(tokenizer, token_pieces: list[str]) -> torch.Tensor:
    user_content = (
        "You are given the token pieces of one lowercase text.\n\n"
        "The tokenization scheme is exactly one of:\n"
        "canonical\n"
        "character\n"
        "random\n\n"
        "Definitions:\n"
        "- canonical: the model's normal tokenizer output.\n"
        "- character: one token per character, including spaces.\n"
        "- random: random valid token pieces were chosen, but together they still decode to exactly the same full text.\n\n"
        "Reply with exactly one label:\n"
        "canonical\n"
        "character\n"
        "random\n\n"
        "TOKENS:\n"
        f"{render_token_pieces(token_pieces)}"
    )

    if getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return torch.tensor([tokenizer.encode(prompt_text, add_special_tokens=False)])

    prompt_text = f"{user_content}\n\nAnswer:\n"
    return torch.tensor([tokenizer.encode(prompt_text, add_special_tokens=False)])


def greedy_decode(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> list[int]:
    attention_mask = torch.ones(input_ids.shape, device=input_ids.device, dtype=torch.long)
    outputs = model(
        input_ids=input_ids,
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
    text: str,
    label: str,
    token_ids: list[int],
    max_new_tokens: int,
    device: torch.device,
) -> TrialResult:
    token_pieces = [decode_piece(tokenizer, token_id) for token_id in token_ids]
    input_ids = build_prompt(tokenizer, token_pieces).to(device)

    with torch.no_grad():
        generated_ids = greedy_decode(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    predicted_label = parse_label(generated_text)
    return TrialResult(
        text=text,
        true_label=label,
        predicted_label=predicted_label,
        token_count=len(token_ids),
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

    texts = select_texts(tokenizer, args.num_texts, args.seed)
    print(f"[info] selected {len(texts)} texts from {len(TEXT_CANDIDATES)} candidates", flush=True)

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
                text=text,
                label=label,
                token_ids=variants[label],
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
    output_path = args.output_dir / f"olmo3_visible_tokenization_detection_{timestamp}.json"
    payload = {
        "timestamp_utc": timestamp,
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "seed": args.seed,
        "num_texts": args.num_texts,
        "max_new_tokens": args.max_new_tokens,
        "summary": summary,
        "results": [asdict(result) for result in results],
    }
    output_path.write_text(json.dumps(payload, indent=2))

    print("[summary]", json.dumps(summary, indent=2), flush=True)
    print(f"[info] wrote results to {output_path}", flush=True)


if __name__ == "__main__":
    main()
