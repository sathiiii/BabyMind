#!/usr/bin/env bash
# Run a BabyMind ablation. Usage: VARIANT=no_go bash scripts/train/train_ablation.sh
# Variants: mil_only, no_tracking, no_track_coh, no_go, no_warm_start
set -euo pipefail
source "$(dirname "$0")/../env.sh"

VARIANT="${VARIANT:-mil_only}"
SEED="${SEED:-0}"
BASE_EXP="Abl_${VARIANT}_seed${SEED}"
export EXP_NAME="${EXP_NAME:-$BASE_EXP}"

COMMON_EXTRA=""
case "$VARIANT" in
  mil_only)
    COMMON_EXTRA="--no_proto --no_track_coh --no_go"
    ;;
  no_tracking)
    COMMON_EXTRA="--mil_no_track --no_track_coh --go_enable"
    ;;
  no_track_coh)
    COMMON_EXTRA="--no_track_coh --go_enable"
    ;;
  no_go)
    COMMON_EXTRA="--track_coh_enable --no_go"
    ;;
  no_warm_start)
    COMMON_EXTRA="--track_coh_enable --go_enable --proto_no_warm_start"
    ;;
  *)
    echo "Unknown VARIANT=$VARIANT" >&2
    exit 2
    ;;
esac

# Build from the full command but append ablation overrides.
# The last occurrence of duplicated argparse flags wins for bool stores.
# shellcheck disable=SC2206
EXTRA_OVERRIDE_ARGS=($COMMON_EXTRA)
bash "$REPO_ROOT/scripts/train/train_babymind_full.sh" "${EXTRA_OVERRIDE_ARGS[@]}"
