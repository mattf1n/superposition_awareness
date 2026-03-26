from __future__ import annotations

import random
import re
from collections import defaultdict
from functools import lru_cache


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


def select_texts(tokenizer, num_texts: int, seed: int) -> list[dict[str, object]]:
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
