# Reproducibility checklist

1. Install the environment and spaCy model.
2. Set `BABYMIND_DATA_DIR`.
3. Verify the SAYCam metadata and frame layout.
4. Generate/prepack masks or place an existing `sam_prepacked` cache in the expected location.
5. Train the CVCL baseline.
6. Train BabyMind full.
7. Run ablations as needed.
8. Evaluate with `eval_labeled_s.sh` and optional object-category evaluation.

Recommended smoke checks before a long cluster run:

```bash
pytest -q
MAX_EPOCHS=1 DEVICES=1 BATCH_SIZE=2 NUM_WORKERS=0 bash scripts/train/train_cvcl_baseline.sh --fast_dev_run
```

For BabyMind, use a very small temporary subset of the training metadata if you want a fast end-to-end test of mask loading and MIL losses.
