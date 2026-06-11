#!/usr/bin/env bash
# Common environment defaults for BabyMind scripts.
# Set BABYMIND_DATA_DIR before running if your SAYCam-derived files live elsewhere.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export BABYMIND_DATA_DIR="${BABYMIND_DATA_DIR:-$REPO_ROOT/expt_saycam}"
export SAYCAM_VOCAB="${SAYCAM_VOCAB:-$BABYMIND_DATA_DIR/vocab.json}"
