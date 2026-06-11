#!/usr/bin/env bash
# Reproduce the CVCL baseline used for controlled comparison.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

EXP_NAME="${EXP_NAME:-CVCL_baseline_DDP_seed${SEED:-0}}"
SEED="${SEED:-0}"
DEVICES="${DEVICES:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
CNN_MODEL="${CNN_MODEL:-$BABYMIND_DATA_DIR/dino_sfp_resnext50.pth}"
MAX_EPOCHS="${MAX_EPOCHS:-400}"

python "$REPO_ROOT/train.py" \
  --dataset saycam \
  --exp_name "$EXP_NAME" \
  --accelerator gpu --devices "$DEVICES" --strategy ddp \
  --pretrained_cnn \
  --max_epochs "$MAX_EPOCHS" \
  --multiple_frames --augment_frames \
  --cnn_model "$CNN_MODEL" \
  --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS" \
  --weight_decay 0.1 \
  --fix_temperature \
  --lr 4e-4 --lr_scheduler \
  --seed "$SEED" \
  --normalize_features --embedding_type flat \
  --drop_last \
  --no_mil --no_proto --no_track_coh --no_go --no_img_adapter \
  --lambda_lm 0.0 --lambda_ar 0.0
