# Cleanup report

This public-release bundle was created from the larger research workspace by keeping the BabyMind reproduction path and removing local artifacts / exploratory clutter.

Kept:

- core training and evaluation entry points (`train.py`, `eval.py`),
- BabyMind implementation modules (`multimodal/object_mil.py`, `multimodal/visual_memory.py`, `multimodal/multimodal_lit.py`),
- SAYCam data and mask-loading modules,
- mask generation and prepacking scripts,
- object-category evaluation helper,
- concept-list resources,
- lightweight diagnostics and tests.

Removed from the release bundle:

- checkpoints, logs, results, frames, mask tensors, images, videos,
- old SLURM scripts containing machine-specific paths,
- old UI/demo notebooks and inherited CVCL analysis helpers,
- personal/local absolute paths.

Main public-release changes:

- Added a complete README with install, data layout, preprocessing, training, ablation, and evaluation commands.
- Added `BABYMIND_DATA_DIR` support so data need not live under `./expt_saycam`.
- Added clean shell wrappers under `scripts/`.
- Fixed package metadata in `pyproject.toml` / `setup.py`.
- Added `.gitignore`, `CITATION.cff`, and lightweight tests.
