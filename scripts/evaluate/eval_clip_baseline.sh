#!/usr/bin/env bash
# Evaluate OpenAI CLIP on the same forced-choice interface.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

EVAL_METADATA="${EVAL_METADATA:-eval_test.json}"
STAGE="${STAGE:-test}"
EVAL_DATASET="${EVAL_DATASET:-saycam}"

python "$REPO_ROOT/eval.py" \
  --clip_eval \
  --eval_type image \
  --eval_dataset "$EVAL_DATASET" \
  --stage "$STAGE" \
  --eval_metadata_filename "$EVAL_METADATA" \
  --save_predictions
