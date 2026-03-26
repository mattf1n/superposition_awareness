# Superposition Awareness

Small OLMo / OLMo 3 experiments for testing whether models can detect or recover information from unusual token and embedding constructions.

All code was written by AI. It is highly recommended to use AI to continue development, since reading slop is often a waste of time.

## Layout

- `experiments/`: Python experiment entrypoints.
- `slurm/`: SLURM batch scripts.
- `results/`: JSON outputs from completed runs.
- `logs/`: stdout/stderr from SLURM jobs.
- `pyproject.toml`: `uv` project dependencies.

## Environment

This repo is set up to run with `uv` and Hugging Face `transformers`.

Local run pattern:

```bash
uv run python experiments/<experiment>.py ...
```

SLURM run pattern:

```bash
sbatch slurm/<job>.sbatch
```

`sbatch` reads `SBATCH_*` environment variables automatically.
Set any cluster-specific `SBATCH_*` environment variables you need before submitting jobs.
- `SBATCH_ACCOUNT`
- `SBATCH_PARTITION`

All batch scripts write logs to `logs/%x-%j.out` and `logs/%x-%j.err`.

## Experiment Catalog

### `experiments/olmo_pooled_token_recovery.py`

Tests whether `allenai/OLMo-7B-Instruct-hf` can verbalize one or two lowercase word tokens from a single synthetic embedding token.

- `single`: inject one token embedding and ask for the word back.
- `mean`: mean-pool two token embeddings into one synthetic token and ask for both words.
- `max`: elementwise max-pool two token embeddings into one synthetic token and ask for both words.

This is the original OLMo pooled-token recovery path, preserved alongside the OLMo 3 version.

Batch script:
- `slurm/run_olmo_pooled_token_recovery.sbatch`

Historical result summary:
- Here, `broad` means a wider tokenizer-clean pool: any lowercase token that round-tripped cleanly as a single token was eligible.
- `2026-03-13`, broad tokenizer-clean run, job `2865689`
- Single-token control: `21/24` exact
- Mean-pooled two-token recovery: `1/24` exact
- Max-pooled two-token recovery: `1/24` exact
- The only exact recovered pooled pair in that run was `ell, ack` in both mean and max modes.
- Result file: `results/olmo_pooled_token_recovery_20260313T180055Z.json`

- Here, `strict` means a curated common-word-only pool: only the hand-picked common lowercase words in the script were eligible.
- `2026-03-13`, stricter common-word rerun, job `2865710`
- Single-token control: `12/12` exact
- Mean-pooled two-token recovery: `0/12` exact
- Max-pooled two-token recovery: `4/12` exact
- The exact recovered max-pooled pairs were `music, metal`, `glass, brown`, `spring, apple`, and `paper, cloud`.
- Result file: `results/olmo_pooled_token_recovery_20260313T180216Z.json`

### `experiments/olmo3_pooled_token_recovery.py`

Tests whether `allenai/Olmo-3-7B-Instruct` can verbalize one or two lowercase word tokens from a single synthetic embedding token.

- `single`: inject one token embedding and ask for the word back.
- `mean`: mean-pool two token embeddings into one synthetic token and ask for both words.
- `max`: elementwise max-pool two token embeddings into one synthetic token and ask for both words.

This is the token-level pooled-embedding probe adapted to OLMo 3.

Batch script:
- `slurm/run_olmo3_pooled_token_recovery.sbatch`

Latest result summary:
- `2026-03-26` rerun, job `3045250`
- Single-token control: `23/24` exact
- Mean-pooled two-token recovery: `0/24` exact
- Max-pooled two-token recovery: `2/24` exact
- The only exact recovered max-pooled pairs were `monkey, stone` and `money, queen`.

So in the clean OLMo 3 word-only setup, the model reliably verbalizes a directly injected single token, does not recover mean-pooled token pairs, and only rarely recovers max-pooled pairs.

### `experiments/olmo3_pairwise_pooled_sequence_recovery.py`

Tests whether `allenai/Olmo-3-7B-Instruct` can recover two aligned sequences when each position is pooled across the two originals.

- Uses sequence pairs with matching token length.
- At position `i`, replaces the prompt token embedding with the mean or max pool of token `i` from sequence A and sequence B.
- Asks the model to output both original sequences on separate lines.
- Scores ordered and unordered exact recovery.

This preserves sequence length and only mixes information across paired sequences positionwise.

Batch script:
- `slurm/run_olmo3_pairwise_pooled_sequence_recovery.sbatch`

Latest result summary:
- `2026-03-26` rerun, job `3045237`
- Mean pooling: `0/10` exact
- Max pooling: `0/10` exact
- Neither mode produced both full target sequences in any trial

Qualitatively, the recovered text is closer to blended paraphrase than reconstruction.
- Mean pooling often produced generic or weakly mixed outputs with low overlap to either source sentence.
- Max pooling leaked more local structure and content words, but still produced hybrids rather than the originals.
- A crude word-overlap check on the rerun gave average best-match Jaccard overlap of about `0.16` for mean pooling and `0.15` for max pooling.

Representative examples:
- Mean: `the window faced the river` + `paper boats crossed the pond` -> `the river crossed paper / cared windows crossed boats`
- Max: `the candle lit the room` + `a child opened the window` -> `the candle opened the door / a child lit the window`
- Max: `the old bridge crossed the water` + `fresh bread cooled on the rack` -> `fresh old bridge crossed the lake / new cold bridge crossed the sand`

## Outputs

Each experiment writes:

- one JSON file in `results/`
- one stdout log in `logs/`
- one stderr log in `logs/`

The JSON files include run metadata, summary metrics, and per-trial outputs.
