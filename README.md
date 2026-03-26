# Superposition Awareness

Small OLMo / OLMo 3 experiments for testing whether models can detect or recover information from unusual token and embedding constructions.

## Layout

- `experiments/`: Python experiment entrypoints.
- `src/superposition_awareness/`: shared Python package code used across experiments.
- `slurm/`: SLURM batch scripts.
- `results/`: JSON outputs from completed runs.
- `logs/`: stdout/stderr from SLURM jobs.
- `pyproject.toml`: `uv` project dependencies.

## Environment

This repo is set up to run with `uv` and Hugging Face `transformers`.

Shared helpers live in the `superposition_awareness` package under `src/`. `uv run` installs the local project so experiment scripts can import that package cleanly.

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

This is the original token-level pooled-embedding probe.

Batch script:
- `slurm/run_olmo_pooled_token_recovery.sbatch`

### `experiments/olmo3_pooled_sequence_recovery.py`

Tests whether `allenai/Olmo-3-7B-Instruct` can recover a short natural-language sequence from a single pooled embedding token.

- Selects short lowercase sequences.
- Pools all token embeddings in one sequence down to one token.
- Evaluates `mean` and `max` pooling.
- Scores substring and exact normalized sequence recovery.

This is the sequence-level analogue of the original token probe.

Batch script:
- `slurm/run_olmo3_pooled_sequence_recovery.sbatch`

### `experiments/olmo3_pairwise_pooled_sequence_recovery.py`

Tests whether `allenai/Olmo-3-7B-Instruct` can recover two aligned sequences when each position is pooled across the two originals.

- Uses sequence pairs with matching token length.
- At position `i`, replaces the prompt token embedding with the mean or max pool of token `i` from sequence A and sequence B.
- Asks the model to output both original sequences on separate lines.
- Scores ordered and unordered exact recovery.

This preserves sequence length and only mixes information across paired sequences positionwise.

Batch script:
- `slurm/run_olmo3_pairwise_pooled_sequence_recovery.sbatch`

### `experiments/olmo3_tokenization_detection.py`

Embedding-based tokenization-classification probe for `allenai/Olmo-3-7B-Instruct`.

- Builds three exact tokenizations of the same lowercase text:
  - `canonical`
  - `character`
  - `random`
- Feeds the model a synthetic token sequence whose embeddings are replaced with the chosen tokenization.
- Asks the model to classify which tokenization scheme was used.

This measures whether tokenization scheme is detectable from hidden token embeddings alone.

Batch script:
- `slurm/run_olmo3_tokenization_detection.sbatch`

### `experiments/olmo3_visible_tokenization_detection.py`

Visible-token classification probe for `allenai/Olmo-3-7B-Instruct`.

- Shows the token pieces directly in the prompt as quoted strings.
- Uses the same three tokenization schemes: `canonical`, `character`, `random`.
- Asks the model to classify the scheme from the visible token-piece list.

This makes token boundaries explicit to the model.

Batch script:
- `slurm/run_olmo3_visible_tokenization_detection.sbatch`

### `experiments/olmo3_same_text_tokenization_detection.py`

Same-visible-text tokenization-classification probe for `allenai/Olmo-3-7B-Instruct`.

- Uses the same three tokenization schemes: `canonical`, `character`, `random`.
- Ensures all variants decode to exactly the same visible text.
- Splices the chosen tokenization directly into the prompt so only token boundaries differ.
- Asks the model to classify the internal tokenization scheme.

This is the strictest tokenization-detection setup in the repo because the rendered text is identical across conditions.

Batch script:
- `slurm/run_olmo3_same_text_tokenization_detection.sbatch`

## Outputs

Each experiment writes:

- one JSON file in `results/`
- one stdout log in `logs/`
- one stderr log in `logs/`

The JSON files include run metadata, summary metrics, and per-trial outputs.
