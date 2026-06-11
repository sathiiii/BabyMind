#!/usr/bin/env bash
# Generate GroundingDINO + SAM + CLIP masks for SAYCam training frames.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

FRAMES_ROOT="${FRAMES_ROOT:-$BABYMIND_DATA_DIR/train_5fps}"
OUT_DIR="${SAM_OUTPUT_DIR:-$BABYMIND_DATA_DIR/train_sam_masks}"
SAM_CKPT="${SAM_CKPT:?Set SAM_CKPT to your SAM checkpoint, e.g. /path/to/sam_vit_h_4b8939.pth}"
PRECISION="${PRECISION:-fp16}"

python "$REPO_ROOT/generate_sam_silhouettes.py" \
  --frames-root "$FRAMES_ROOT" \
  --output-dir "$OUT_DIR" \
  --sam-checkpoint "$SAM_CKPT" \
  --sam-model-type "${SAM_MODEL_TYPE:-vit_h}" \
  --precision "$PRECISION" \
  --box-threshold "${BOX_THRESHOLD:-0.30}" \
  --text-threshold "${TEXT_THRESHOLD:-0.75}" \
  --sam-score-threshold "${SAM_SCORE_THRESHOLD:-0.87}" \
  --clip-sim-threshold "${CLIP_SIM_THRESHOLD:-0.30}"
