# Data setup

BabyMind uses child-view frames, utterance metadata, forced-choice evaluation metadata, and precomputed object-mask candidates. The repository does not redistribute SAYCam data or derived frames.

## Environment variable

Set:

```bash
export BABYMIND_DATA_DIR=/path/to/expt_saycam
```

If unset, the code uses `./expt_saycam`.

## Core SAYCam files

The training code expects:

```text
$BABYMIND_DATA_DIR/
  train.json
  val.json
  test.json
  vocab.json
  train_5fps/
  eval_dev.json
  eval_test.json
  eval/
```

The `train.json`, `val.json`, and `test.json` files should follow the existing CVCL/SAYCam metadata format used by `multimodal/multimodal_saycam_data_module.py`.

## Pretrained visual backbone

By default the scripts expect:

```text
$BABYMIND_DATA_DIR/dino_sfp_resnext50.pth
```

Override with:

```bash
CNN_MODEL=/path/to/backbone.pth bash scripts/train/train_babymind_full.sh
```

## SAM / object-mask cache

BabyMind full training expects the prepacked mask cache:

```text
$BABYMIND_DATA_DIR/train_sam_masks/sam_prepacked/
  concept_vocab.json
  sam_prepacked_index.json
  concept_frequency.json
  *.pt
```

Generate it with:

```bash
export SAM_CKPT=/path/to/sam_vit_h_4b8939.pth
bash scripts/preprocess/generate_masks.sh
bash scripts/preprocess/prepack_masks.sh
```

## Object-category evaluation

The default object-category data path is:

```text
eval_datasets/17-objects/object_categories/
```

You can override relevant paths with the arguments exposed by `multimodal/object_categories_data_module.py`, for example `--konkle_data_dir`, `--konkle_categories_dir`, and `--eval_metadata_filename`.
