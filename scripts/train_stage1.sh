#!/usr/bin/env bash
# Stage-1 training. Does not resume unless --resume is passed.
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

CONFIG="${CONFIG:-configs/stage1_pseudo.yaml}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-29532}"

exec torchrun \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  train_stage1.py \
  --config "${CONFIG}" \
  "$@"
