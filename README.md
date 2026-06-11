# BabyMind

**Objects Before Words: Object-First Inductive Biases for Grounding Language in Child-View Video**

This repository contains the code used for BabyMind, an object-first extension of Child-View Contrastive Learning (CVCL) for learning grounded word meaning from child-view video paired with caregiver speech.

BabyMind keeps the CVCL global image-text contrastive objective and adds an object-file pathway: offline mask candidates, short-window object files, prototype-space multiple-instance contrastive alignment, track-coherence regularization, and global-object agreement.

## Repository status

This release is intended for the CogSci paper. It contains source code, training/evaluation scripts, and configuration-style shell wrappers. It does **not** include SAYCam videos/frames, extracted masks, checkpoints, or logs.

SAYCam-derived data cannot be redistributed here. Researchers should obtain the original SAYCam/SAYCam-S data through the appropriate data-use process and then prepare the local directory layout described below.

## Layout

```text
BabyMind/
  train.py                         # main training entry point
  eval.py                          # forced-choice evaluation entry point
  generate_sam_silhouettes.py      # GroundingDINO + SAM + CLIP mask mining
  prepack_sam_masks.py             # convert mined masks to per-frame .pt cache
  make_eval_object_categories.py    # helper for object-category evaluation metadata
  multimodal/                      # model, data modules, MIL, prototype memory
  neuro_symbolic/                  # concept lists / optional symbolic resources
  scripts/
    preprocess/                    # mask generation and prepacking wrappers
    train/                         # CVCL, BabyMind, and ablation wrappers
    evaluate/                      # Labeled-S / object-category evaluation wrappers
  analysis/                        # optional figure / diagnostic utilities
  docs/                            # data and reproducibility notes
  tests/                           # lightweight smoke tests
```

## Installation

The code was developed with Python 3.8-style dependencies and PyTorch Lightning 1.x.

```bash
git clone https://github.com/sathiiii/BabyMind.git
cd BabyMind

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[masking,dev]"
python -m spacy download en_core_web_sm
```

If you use `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[masking,dev]"
uv run python -m spacy download en_core_web_sm
```

If your cluster already provides PyTorch/ROCm/CUDA, install PyTorch according to your cluster instructions first, then install this package without letting pip replace it.

## Expected data layout

By default the code looks under `./expt_saycam`. You can override this with:

```bash
export BABYMIND_DATA_DIR=/path/to/expt_saycam
```

Expected structure after preprocessing:

```text
$BABYMIND_DATA_DIR/
  train.json
  val.json
  test.json
  vocab.json
  dino_sfp_resnext50.pth
  train_5fps/
    <frame files used by train.json>
  eval/
    dev/
    test/
  eval_dev.json
  eval_test.json
  train_sam_masks/
    gdino_sam_clip_multi_index_rank*.jsonl
    sam_prepacked/
      concept_vocab.json
      sam_prepacked_index.json
      concept_frequency.json
      *.pt
```

The code also supports Konkle/Object Categories evaluation metadata under `$BABYMIND_DATA_DIR` and image files under `eval_datasets/17-objects` by default. See `docs/DATA.md` for more detail.

## 1. Generate and prepack object masks

Skip this section if you already have `train_sam_masks/sam_prepacked`.

```bash
export BABYMIND_DATA_DIR=/path/to/expt_saycam
export SAM_CKPT=/path/to/sam_vit_h_4b8939.pth

bash scripts/preprocess/generate_masks.sh
bash scripts/preprocess/prepack_masks.sh
```

The first command mines candidate masks with GroundingDINO, SAM, and CLIP. The second command converts the per-mask files into a fast per-frame cache used by training.

## 2. Train the CVCL baseline

```bash
export BABYMIND_DATA_DIR=/path/to/expt_saycam
bash scripts/train/train_cvcl_baseline.sh
```

Useful overrides:

```bash
SEED=1 DEVICES=1 BATCH_SIZE=8 MAX_EPOCHS=400 bash scripts/train/train_cvcl_baseline.sh
```

## 3. Train BabyMind

```bash
export BABYMIND_DATA_DIR=/path/to/expt_saycam
bash scripts/train/train_babymind_full.sh
```

This enables the core BabyMind components:

- `--use_sam_masks --bag_num_frames 5 --bag_contiguous`
- `--mil_enable --mil_track`
- `--proto_enable --proto_warm_start`
- `--track_coh_enable`
- `--go_enable`

Useful overrides:

```bash
SEED=1 DEVICES=1 BATCH_SIZE=8 MAX_EPOCHS=400 bash scripts/train/train_babymind_full.sh
```

To append additional training flags:

```bash
bash scripts/train/train_babymind_full.sh --debug_tracks_every_n_epochs 25
```

The implementation also contains an optional SAM concept-name alignment auxiliary loss. It is disabled by default because it is not part of the minimal BabyMind method. To enable it:

```bash
ENABLE_SAM_CONCEPT_ALIGN=1 bash scripts/train/train_babymind_full.sh
```

## 4. Run ablations

```bash
VARIANT=mil_only bash scripts/train/train_ablation.sh
VARIANT=no_tracking bash scripts/train/train_ablation.sh
VARIANT=no_track_coh bash scripts/train/train_ablation.sh
VARIANT=no_go bash scripts/train/train_ablation.sh
VARIANT=no_warm_start bash scripts/train/train_ablation.sh
```

The ablation script reuses the full BabyMind command and appends the appropriate disabling flags at the end.

## 5. Evaluate checkpoints

Labeled-S / SAYCam forced-choice:

```bash
export BABYMIND_DATA_DIR=/path/to/expt_saycam
CKPT=checkpoints/BabyMind_full_DDP_seed0/last.ckpt \
  bash scripts/evaluate/eval_labeled_s.sh
```

Object Categories / Konkle-style forced-choice:

```bash
CKPT=checkpoints/BabyMind_full_DDP_seed0/last.ckpt \
EVAL_METADATA=eval_konkle_object_categories.json \
  bash scripts/evaluate/eval_konkle.sh
```

CLIP baseline on the same interface:

```bash
EVAL_DATASET=saycam EVAL_METADATA=eval_test.json \
  bash scripts/evaluate/eval_clip_baseline.sh
```

Evaluation writes JSON predictions and per-class metrics under `results/`.

## 6. Diagnostics and paper figures

Optional utilities live in `analysis/`. Common examples:

```bash
python analysis/dump_vm_embeddings.py --help
python analysis/make_prototype_diagnostics.py --help
python analysis/plot_vm_tsne.py --help
```

Some analysis scripts assume local result files or checkpoint paths; they are provided as utilities rather than required reproduction steps.

## Development checks

The lightweight tests cover shape and packing behavior for the public release code:

```bash
pytest -q
```

A syntax check over the repository can be run with:

```bash
python -m compileall -q .
```

## Notes on checkpoints and data

Large artifacts are intentionally excluded from the repository:

- model checkpoints (`*.ckpt`, `*.pth`, `*.pt`),
- extracted frames and videos,
- SAM mask caches,
- logs and W&B runs,
- generated result files.

Use `.gitignore` as the source of truth for what should remain out of version control.

## Citation

If you use this code, please cite:

<!-- ```bibtex
@inproceedings{silva2026babymind,
  title     = {Objects Before Words: Object-First Inductive Biases for Grounding Language in Child-View Video},
  author    = {Silva, Sathira and Gebreselasie, Abrham Kahsay and Sheikh, Muhammad Umer and Kuckreja, Kartik and Harari, Daniel and Khan, Muhammad Haris},
  booktitle = {Proceedings of the Annual Meeting of the Cognitive Science Society},
  year      = {2026}
}
``` -->

