#!/bin/bash
#SBATCH --job-name=olmo3-token-detect
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=swabhas_1625
#SBATCH --partition=nlp_hiprio
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

cd /home1/mfinlays/superposition_awareness
mkdir -p logs results

echo "[job] host=$(hostname)"
echo "[job] started=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "[job] python=$(command -v python || true)"
echo "[job] uv=$(command -v uv || true)"
echo "[job] nvidia-smi"
nvidia-smi || true

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run python experiments/olmo3_tokenization_detection.py "$@"

echo "[job] finished=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
