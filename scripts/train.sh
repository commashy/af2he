#!/usr/bin/env bash
# Masked-reconstruction pretraining. Stops if the launch itself fails.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi
conda activate stain_translation

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

CONFIG="${CONFIG:-configs/pretrain.yaml}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-29531}"

exec torchrun \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  train.py \
  --config "${CONFIG}" \
  "$@"
