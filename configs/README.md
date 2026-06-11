# Configs

The public release uses shell wrappers in `scripts/` instead of large YAML files so that the exact paper commands are visible and easy to edit.

Primary commands:

- `scripts/train/train_cvcl_baseline.sh`
- `scripts/train/train_babymind_full.sh`
- `scripts/train/train_ablation.sh`
- `scripts/evaluate/eval_labeled_s.sh`
- `scripts/evaluate/eval_konkle.sh`

Set environment variables such as `BABYMIND_DATA_DIR`, `SEED`, `DEVICES`, `BATCH_SIZE`, `MAX_EPOCHS`, and `CKPT` to customize runs.
