#!/usr/bin/env bash
# Convert generated mask npz files into compact per-frame .pt files for training.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

SAM_OUTPUT_DIR="${SAM_OUTPUT_DIR:-$BABYMIND_DATA_DIR/train_sam_masks}"
PREPACKED_DIR="${SAM_PREPACKED_DIR:-$SAM_OUTPUT_DIR/sam_prepacked}"

python "$REPO_ROOT/prepack_sam_masks.py" \
  --output-dir "$SAM_OUTPUT_DIR" \
  --prepacked-dir "$PREPACKED_DIR" \
  --image-height "${IMAGE_HEIGHT:-224}" \
  --image-width "${IMAGE_WIDTH:-224}"
