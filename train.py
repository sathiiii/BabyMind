#!/usr/bin/env python3
import argparse
from pathlib import Path

import os
import sys
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torchinfo import summary

from multimodal.multimodal_data_module import MultiModalDataModule
from multimodal.multimodal_saycam_data_module import MultiModalSAYCamDataModule
from multimodal.coco_captions_data_module import COCOCaptionsDataModule
from multimodal.multimodal import VisionEncoder, TextEncoder, MultiModalModel, LanguageModel
from multimodal.multimodal_lit import MultiModalLitModel
from grad_norm_logger import GradNormLogger

# To make sure the results are reproducible for the VM
os.environ["PYTHONHASHSEED"] = "0"

def redirect_stdout_to_file(log_dir: Path, filename_prefix: str = "stdout") -> None:
    log_dir.mkdir(parents=True, exist_ok=True)

    # DDP-safe: one file per process
    rank = int(os.environ.get("RANK", "0"))
    log_path = log_dir / f"{filename_prefix}_rank{rank}.log"

    f = open(log_path, "a", buffering=1)  # line-buffered
    sys.stdout = f
    sys.stderr = f

    # Optional: make prints flush immediately
    os.environ["PYTHONUNBUFFERED"] = "1"
    print(f"[logging] Redirected stdout/stderr to: {log_path}", flush=True)

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

    IGNORE = {"mask_concept_count", "mask_concept_weight"}  # add others if needed

    model_keys = set(model.state_dict().keys()) - IGNORE
    ckpt_keys = set(sd.keys()) - IGNORE

    unexpected = sorted(list(ckpt_keys - model_keys))
    missing = sorted(list(model_keys - ckpt_keys))
    return unexpected, missing


def main():
    # Parse args
    parser = _setup_parser()
    args = parser.parse_args()

    # Paths
    ckpt_dir = Path("checkpoints") / args.exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if str(args.resume_ckpt) == "last":
        args.resume_ckpt = ckpt_dir / "last.ckpt"

    # redirect_stdout_to_file(ckpt_dir, filename_prefix="train")

    # Seed
    pl.seed_everything(args.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

    # fit model
    trainer.fit(lit_model, data, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
