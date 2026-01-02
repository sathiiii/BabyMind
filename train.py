#!/usr/bin/env python3
import argparse
from pathlib import Path

import os
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchinfo import summary

from multimodal.multimodal_data_module import MultiModalDataModule
from multimodal.multimodal_saycam_data_module import MultiModalSAYCamDataModule
from multimodal.coco_captions_data_module import COCOCaptionsDataModule
from multimodal.multimodal import VisionEncoder, TextEncoder, MultiModalModel, LanguageModel
from multimodal.multimodal_lit import MultiModalLitModel
from grad_norm_logger import GradNormLogger

# To make sure the results are reproducible for the VM
env = os.environ.copy()
env["PYTHONHASHSEED"] = "0"


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser()

    # ---- Lightning Trainer args (PL 1.9.5) ----
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # ---- Data/model args ----
    data_group = parser.add_argument_group("Data Args")
    MultiModalDataModule.add_to_argparse(data_group)
    MultiModalSAYCamDataModule.add_additional_to_argparse(data_group)
    COCOCaptionsDataModule.add_additional_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    VisionEncoder.add_to_argparse(model_group)
    TextEncoder.add_to_argparse(model_group)
    MultiModalModel.add_to_argparse(model_group)
    LanguageModel.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    MultiModalLitModel.add_to_argparse(lit_model_group)

    # ---- Script args ----
    parser.add_argument("--exp_name", type=str, default="multimodal_test",
                        help="experiment name for logging")
    parser.add_argument("--dataset", type=str, choices=["saycam", "coco"],
                        default="saycam", help="which dataset to use")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for everything")
    parser.add_argument("--save_top_k", type=int, default=1,
                        help="saves best k models; 0 saves none; -1 saves all")
    parser.add_argument("--resume_ckpt", type=Path, default=None,
                        help="path to the checkpoint to resume from; if it's "
                             "\"last\", resume from the last checkpoint.")

    return parser


def _inspect_ckpt_compat(model: torch.nn.Module, ckpt_path: Path):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(sd.keys())

    unexpected = sorted(list(ckpt_keys - model_keys))
    missing = sorted(list(model_keys - ckpt_keys))
    return unexpected, missing


def _load_ckpt_nonstrict_drop_prefixes(
    model: torch.nn.Module,
    ckpt_path: Path,
    drop_prefixes=("obj_encoder.",),
):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)

    def _keep(k: str) -> bool:
        return not any(k.startswith(p) for p in drop_prefixes)

    sd = {k: v for k, v in sd.items() if _keep(k)}

    incompatible = model.load_state_dict(sd, strict=False)
    print("[ckpt] Loaded with strict=False")
    print("[ckpt] Missing keys (first 50):", incompatible.missing_keys[:50])
    print("[ckpt] Unexpected keys (first 50):", incompatible.unexpected_keys[:50])


def main():
    # Parse args
    parser = _setup_parser()
    args = parser.parse_args()

    # Paths
    ckpt_dir = Path("checkpoints") / args.exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if str(args.resume_ckpt) == "last":
        args.resume_ckpt = ckpt_dir / "last.ckpt"

    # Seed
    pl.seed_everything(args.seed)

    # Data + models
    DataModuleClass = {
        "saycam": MultiModalSAYCamDataModule,
        "coco": COCOCaptionsDataModule,
    }[args.dataset]
    data = DataModuleClass(args)
    vocab = data.read_vocab()
    vision_encoder = VisionEncoder(args=args)
    text_encoder = TextEncoder(
        vocab, image_feature_map_dim=vision_encoder.last_cnn_out_dim, args=args)
    lit_model = MultiModalLitModel(vision_encoder, text_encoder, args)

    # Checkpointing (monitor matches names logged by your LightningModule, e.g. "val/loss")
    checkpoint_cb = ModelCheckpoint(
        monitor="val/loss",
        save_last=True,
        save_top_k=args.save_top_k,
        dirpath=ckpt_dir,
        filename="{epoch}",
    )

    # Trainer
    if args.logger:
        wandb_logger = WandbLogger(project="multimodal-saycam", name=args.exp_name, log_model=True)
        trainer = pl.Trainer.from_argparse_args(
            args,
            enable_checkpointing=True,
            callbacks=[checkpoint_cb, GradNormLogger(log_per_layer=False)],
            logger=wandb_logger,
        )
    else:
        trainer = pl.Trainer.from_argparse_args(
            args,
            enable_checkpointing=True,
            callbacks=[checkpoint_cb],
        )

    print(args)

    # ---- Resume logic ----
    ckpt_path = args.resume_ckpt
    if ckpt_path is not None:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--resume_ckpt not found: {ckpt_path}")

        unexpected, missing = _inspect_ckpt_compat(lit_model, ckpt_path)

        # If incompatible, do warm-start instead of Lightning resume
        if unexpected or missing:
            print("[ckpt] Checkpoint is NOT compatible for strict resume.")
            print(f"[ckpt] unexpected keys: {len(unexpected)} (showing up to 10)")
            print(unexpected[:10])
            print(f"[ckpt] missing keys: {len(missing)} (showing up to 10)")
            print(missing[:10])

            print("[ckpt] Warm-starting weights (dropping obj_encoder.*) and starting fresh training state.")
            _load_ckpt_nonstrict_drop_prefixes(
                lit_model,
                ckpt_path,
                drop_prefixes=("obj_encoder.",),
            )
            ckpt_path = None  # IMPORTANT: do NOT pass ckpt_path to trainer.fit

    # fit model
    trainer.fit(lit_model, data, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
