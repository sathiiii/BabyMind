#!/usr/bin/env bash
# Reproduce the BabyMind full model: CVCL + object-file MIL + prototype memory
# + track coherence + global-object agreement.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

EXP_NAME="${EXP_NAME:-BabyMind_full_DDP_seed${SEED:-0}}"
SEED="${SEED:-0}"
DEVICES="${DEVICES:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
CNN_MODEL="${CNN_MODEL:-$BABYMIND_DATA_DIR/dino_sfp_resnext50.pth}"
MAX_EPOCHS="${MAX_EPOCHS:-400}"
SAM_MASKS_DIR="${SAM_MASKS_DIR:-$BABYMIND_DATA_DIR/train_sam_masks}"
SAM_PREPACKED_DIR="${SAM_PREPACKED_DIR:-$SAM_MASKS_DIR/sam_prepacked}"
SAM_FREQ_JSON="${SAM_FREQ_JSON:-$SAM_PREPACKED_DIR/concept_frequency.json}"

EXTRA_ARGS=()
USER_ARGS=("$@")
# Optional extra flags can be supplied either as command-line args to this script
# or as a whitespace-separated EXTRA_TRAIN_ARGS environment variable.
EXTRA_TRAIN_ARGS_ARRAY=()
if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_TRAIN_ARGS_ARRAY=(${EXTRA_TRAIN_ARGS})
fi
if [[ "${ENABLE_SAM_CONCEPT_ALIGN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--sam_concept_align_enable --sam_concept_align_lambda "${SAM_CONCEPT_ALIGN_LAMBDA:-0.05}" --sam_concept_align_tau "${SAM_CONCEPT_ALIGN_TAU:-0.07}")
fi

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
  --use_sam_masks --sam_masks_dir "$SAM_MASKS_DIR" --sam_prepacked_dir "$SAM_PREPACKED_DIR" \
  --sam_concept_frequency_json "$SAM_FREQ_JSON" \
  --bag_num_frames 5 --bag_contiguous \
  --mil_enable --mil_run_val \
  --mil_lambda 0.10 --mil_tau 0.05 --mil_min_mask_area 0.01 \
  --mil_track --mil_track_sim_thresh 0.55 --mil_track_max_tracks 16 \
  --mil_obj_ring_weight 0.05 --mil_obj_ring_px_fmap 1 \
  --proto_enable \
  --proto_num 64 --proto_tau 0.07 \
  --proto_use_sinkhorn --proto_sinkhorn_iters 3 --proto_sinkhorn_epsilon 0.05 --proto_sinkhorn_min_samples 32 \
  --proto_ema_decay 0.99 --proto_ema_eps 1e-3 --proto_ema_ddp_sync \
  --proto_warm_start \
  --proto_warm_min_local 128 --proto_warm_sim_thresh 0.25 --proto_warm_max_total 4096 --proto_warm_kmeans_iters 10 \
  --w_align_sim0 0.10 --w_align_simscale 0.05 --w_align_warmup_steps 500 --w_align_min 0.05 \
  --track_coh_enable --track_coh_lambda 0.05 --track_coh_match_thresh 0.30 --track_coh_min_frames 2 \
  --go_enable --go_lambda 0.05 \
  --img_adapter_enable \
  --lambda_lm 0.0 --lambda_ar 0.0 \
  --debug_save_tracks --debug_tracks_topk 6 --debug_tracks_every_n_epochs 50 --debug_tracks_overlay_alpha 0.45 \
  --sam_use_concept_weights --sam_registry_verbose \
  --mil_text_mode noun_multi --mil_noun_max 5 \
  "${EXTRA_ARGS[@]}" \
  "${EXTRA_TRAIN_ARGS_ARRAY[@]}" \
  "${USER_ARGS[@]}"
