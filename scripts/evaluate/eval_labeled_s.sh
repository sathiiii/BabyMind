#!/usr/bin/env bash
# Evaluate a checkpoint on Labeled-S / SAYCam forced-choice trials.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

CKPT="${CKPT:?Set CKPT to a checkpoint path, e.g. checkpoints/BabyMind_full_DDP_seed0/last.ckpt}"
EVAL_METADATA="${EVAL_METADATA:-eval_test.json}"
STAGE="${STAGE:-test}"

python "$REPO_ROOT/eval.py" \
  --checkpoint "$CKPT" \
  --eval_type image \
  --eval_dataset saycam \
  --stage "$STAGE" \
  --eval_metadata_filename "$EVAL_METADATA" \
  --use_kitty_label \
  --save_predictions
