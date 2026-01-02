import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from multimodal.multimodal_data_module import (
    EVAL_DATA_DIR,
    SOS_TOKEN_ID,
    EOS_TOKEN_ID,
    load_data,
)
from multimodal.multimodal_saycam_data_module import MultiModalSAYCamDataModule, DATA_DIR
from multimodal.object_categories_data_module import (
    ObjectCategoriesDataModule,
    _get_object_categories,
)
from multimodal.multimodal_lit import MultiModalLitModel
from train import _setup_parser

import clip

EVAL_FRAMES_DIRNAME = EVAL_DATA_DIR / "eval"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int = 0) -> None:
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_ckpt_path(ckpt_arg: str) -> Path:
    """
    User passes a full path to last.ckpt (or any .ckpt). We just validate it.
    """
    ckpt_path = Path(ckpt_arg).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if ckpt_path.suffix != ".ckpt":
        raise ValueError(f"--checkpoint must point to a .ckpt file, got: {ckpt_path}")
    return ckpt_path


def _infer_run_tag(ckpt_path: Path) -> str:
    # Try to infer run_tag as checkpoints/<exp_name>/last.ckpt -> exp_name
    parts = list(ckpt_path.parts)
    if "checkpoints" in parts:
        i = parts.index("checkpoints")
        if i + 1 < len(parts):
            return parts[i + 1]
    return ckpt_path.parent.name


def _load_lit_disable_vm(ckpt_path: Path, map_location) -> MultiModalLitModel:
    """
    Load a Lightning checkpoint while forcibly disabling VM at construction time.

    This avoids failing inside MultiModalLitModel.__init__ when vm_enable=True
    but vm_fmap_dim (or a usable feature map) is unavailable at eval time.

    We override the saved hyper_parameters['args'] before Lightning instantiates the model.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    hp = ckpt.get("hyper_parameters", {})

    saved_args = hp.get("args", None)
    if saved_args is None:
        raise RuntimeError(
            "[eval] Checkpoint missing hyper_parameters['args']; cannot override vm_enable safely."
        )

    # saved_args might be a dict or Namespace-like
    if isinstance(saved_args, dict):
        saved_args = argparse.Namespace(**saved_args)

    # Force-disable VM and VM-only logging knobs
    setattr(saved_args, "vm_enable", False)
    setattr(saved_args, "vm_lambda", 0.0)
    setattr(saved_args, "vm_log_gradcam", False)
    setattr(saved_args, "vm_debug_verbose", False)
    setattr(saved_args, "vm_debug_log_images_every", 0)

    lit = MultiModalLitModel.load_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        map_location=map_location,
        strict=False,
        args=saved_args,
    )
    lit.eval()
    return lit


def _ckpt_key_sanity_report(ckpt_path: Path, lit: MultiModalLitModel) -> None:
    """
    strict=False is fine only if we ensure mismatches are expected.
    We report and (optionally) fail on surprising mismatches.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if "state_dict" not in ckpt:
        print("[eval] WARNING: checkpoint missing 'state_dict' key; cannot sanity-check keys.")
        return

    ckpt_keys = set(ckpt["state_dict"].keys())
    model_keys = set(lit.state_dict().keys())

    unexpected = sorted(list(ckpt_keys - model_keys))
    missing = sorted(list(model_keys - ckpt_keys))

    print(f"[eval] checkpoint keys: {len(ckpt_keys)} | model keys: {len(model_keys)}")
    print(f"[eval] unexpected keys in checkpoint: {len(unexpected)}")
    print(f"[eval] missing keys from checkpoint: {len(missing)}")

    if unexpected:
        print("[eval] unexpected examples:", unexpected[:20])
    if missing:
        print("[eval] missing examples:", missing[:20])

    # Allow known benign unexpected prefixes (common when we disable VM at eval time)
    allowed_unexpected_prefixes = (
        "obj_encoder.",  # legacy / VM variants
        "vm.",           # VM module disabled at eval
    )
    bad_unexpected = [k for k in unexpected if not k.startswith(allowed_unexpected_prefixes)]
    if bad_unexpected:
        raise RuntimeError(
            "[eval] Refusing to evaluate: checkpoint contains unexpected keys that are not "
            f"whitelisted. Examples: {bad_unexpected[:10]}"
        )


def main(args: argparse.Namespace) -> None:
    _seed_everything(0)

    # ---------------- model and config loading ----------------
    if args.clip_eval:
        print("[eval] Loading CLIP ViT-L/14")
        checkpoint_name = "clip_vitl_14"
        run_tag = checkpoint_name
        model, _ = clip.load("ViT-L/14", device=device)
        model.eval()
        print("[eval] CLIP loaded")

        parser = _setup_parser()
        data_args = parser.parse_args([])

        config = {
            "model": "clip",
            "seed": None,
            "shuffle_utterances": None,
            "cnn": "clip",
            "augment_frames": None,
            "multiple_frames": None,
        }
    else:
        ckpt_path = _resolve_ckpt_path(args.checkpoint)
        checkpoint_name = ckpt_path.name
        run_tag = _infer_run_tag(ckpt_path)

        print(f"[eval] Using checkpoint: {ckpt_path}")
        print(f"[eval] run_tag: {run_tag}")
        print("[eval] Forcing vm_enable=False during checkpoint load")

        # IMPORTANT FIX: disable VM during construction
        lit = _load_lit_disable_vm(ckpt_path, map_location=device)
        model = lit

        # Key sanity check: allow vm.* and obj_encoder.* as expected unexpected keys
        _ckpt_key_sanity_report(ckpt_path, lit)

        # Hyperparameters from checkpoint (as stored by the model)
        hp = getattr(lit, "args", {}) or getattr(lit, "hparams", {}) or {}

        pretrained = bool(hp.get("pretrained_cnn", False))
        finetune = bool(hp.get("finetune_cnn", False))
        if pretrained and finetune:
            cnn_str = "finetune_pretrained"
        elif (not pretrained) and finetune:
            cnn_str = "finetune_random_init"
        elif (not pretrained) and (not finetune):
            cnn_str = "frozen_random_init"
        else:
            cnn_str = "frozen_pretrained"

        config = {
            "model": hp.get("text_encoder", "embedding"),
            "seed": hp.get("seed", None),
            "shuffle_utterances": bool(hp.get("shuffle_utterances", False)),
            "augment_frames": bool(hp.get("augment_frames", True)),
            "multiple_frames": bool(hp.get("multiple_frames", True)),
            "cnn": cnn_str,
        }

        parser = _setup_parser()
        data_args = parser.parse_args([])

        # Apply checkpoint hparams onto data_args when compatible
        for k, v in hp.items():
            if hasattr(data_args, k):
                setattr(data_args, k, v)

    # ---------------- data module and eval split ----------------
    data_args.augment_frames = False
    data_args.eval_include_sos_eos = args.eval_include_sos_eos
    data_args.eval_type = args.eval_type
    data_args.eval_metadata_filename = args.eval_metadata_filename

    if args.clip_eval:
        data_args.clip_eval = True

    if args.eval_dataset == "saycam":
        data = MultiModalSAYCamDataModule(data_args)

        # Do not call prepare_data() during evaluation
        data.setup()

        vocab = data.read_vocab()
        eval_data = load_data(EVAL_DATA_DIR / data_args.eval_metadata_filename)

        dl_fn = {"dev": data.val_dataloader, "test": data.test_dataloader}[args.stage]
        dataloader = dl_fn()[1]

        if getattr(dataloader, "batch_size", 1) != 1:
            raise RuntimeError(
                f"[eval] This eval loop assumes batch_size=1, got {dataloader.batch_size}."
            )

        if data_args.eval_metadata_filename == "eval_manual_filtered_test.json":
            classes = sorted(os.listdir(DATA_DIR / "eval_manual_filtered" / "test"))
        else:
            classes = sorted(os.listdir(EVAL_FRAMES_DIRNAME / "dev"))

    elif args.eval_dataset == "object_categories":
        object_categories_dm = ObjectCategoriesDataModule(data_args)
        object_categories_dm.setup()

        dataloader = object_categories_dm.test_dataloader()
        if getattr(dataloader, "batch_size", 1) != 1:
            raise RuntimeError(
                f"[eval] This eval loop assumes batch_size=1, got {dataloader.batch_size}."
            )

        vocab = object_categories_dm.read_vocab()
        eval_data = load_data(EVAL_DATA_DIR / "eval_object_categories.json")
        classes = _get_object_categories(vocab)
    else:
        raise ValueError(f"Unknown eval_dataset: {args.eval_dataset}")

    if len(eval_data) != len(dataloader):
        print(
            f"[eval] WARNING: eval_data length ({len(eval_data)}) != dataloader length ({len(dataloader)}). "
            "Result logging that uses eval_data[i] may be misaligned."
        )

    # replace cat with kitty if requested
    if args.use_kitty_label and not args.clip_eval:
        if "cat" in classes:
            classes.remove("cat")
        if "kitty" not in classes:
            classes.append("kitty")

    # ---------------- metrics containers ----------------
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    results = []

    # ---------------- main evaluation loop ----------------
    for i, batch in enumerate(dataloader):
        img, label, label_len, raw_label = batch
        class_label = raw_label[0][0]

        # optional kitty relabel
        if args.use_kitty_label and class_label == "cat" and not args.clip_eval:
            class_label = "kitty"
            if args.eval_type == "image":
                new_label = [vocab[class_label]]
                if args.eval_include_sos_eos:
                    new_label = [SOS_TOKEN_ID] + new_label + [EOS_TOKEN_ID]
                label = torch.LongTensor([new_label])
            elif args.eval_type == "text":
                label[0][0] = vocab[class_label]

        # ------------ forward pass to get logits ------------
        if args.eval_type == "image":
            img = img.squeeze(0).to(device)
            label = label.to(device)
            label_len = label_len.to(device)

            with torch.no_grad():
                if args.clip_eval:
                    label_ = label.squeeze(0)
                    _, logits_per_text = model(img, label_)
                else:
                    _, logits_per_text = model(img, label, label_len)

                logits = logits_per_text[0]
                logits_soft = torch.softmax(logits, dim=-1)
                logits_list = logits_soft.detach().cpu().numpy().tolist()

                pred = int(torch.argmax(logits))
                ground_truth = 0

        elif args.eval_type == "text":
            img = img.squeeze(0).to(device)
            label = label.squeeze(0).to(device)
            label_len = label_len.squeeze(0).to(device)

            with torch.no_grad():
                if args.clip_eval:
                    logits_per_image, _ = model(img, label)
                else:
                    logits_per_image, _ = model(img, label, label_len)

                logits = logits_per_image[0]
                logits_soft = torch.softmax(logits, dim=-1)
                logits_list = logits_soft.detach().cpu().numpy().tolist()

                pred = int(torch.argmax(logits))
                ground_truth = 0
        else:
            raise ValueError(f"Unknown eval_type: {args.eval_type}")

        # ------------ update metrics ------------
        correct = pred == ground_truth
        correct_pred[class_label] += int(correct)
        total_pred[class_label] += 1

        # Logging categories (best-effort)
        curr_eval_categories = None
        if i < len(eval_data) and isinstance(eval_data[i], dict):
            curr_trial = eval_data[i]
            curr_target_category = curr_trial.get("target_category", None)
            curr_foil_categories = curr_trial.get("foil_categories", None)
            if curr_target_category is not None and curr_foil_categories is not None:
                curr_eval_categories = [curr_target_category] + list(curr_foil_categories)

        curr_results = {
            "checkpoint": checkpoint_name,
            "model": config["model"],
            "seed": config["seed"],
            "shuffle_utterances": config["shuffle_utterances"],
            "augment_frames": config["augment_frames"],
            "multiple_frames": config["multiple_frames"],
            "cnn": config["cnn"],
            "eval_type": args.eval_type,
            "eval_dataset": args.eval_dataset,
            "stage": args.stage,
            "trial_idx": i,
            "categories": curr_eval_categories,
            "logits": logits_list,
            "pred": pred,
            "correct": bool(correct),
        }
        results.append(curr_results)

    # ---------------- aggregate metrics ----------------
    print("\nPer class accuracies:")
    for classname, correct_count in correct_pred.items():
        t = total_pred.get(classname, 0)
        if t > 0:
            accuracy = float(correct_count) / float(t)
            print(f"Accuracy for class {classname:8s} is: {accuracy:.1%}")
        else:
            print(f"There are no evaluation samples for {classname:8s}")

    total_correct = sum(correct_pred.values())
    total = sum(total_pred.values())
    overall_accuracy = float(total_correct) / float(total) if total > 0 else 0.0
    print(f"\nTotal accuracy: {overall_accuracy:%}")

    per_class_metrics = {}
    for classname in classes:
        t = int(total_pred.get(classname, 0))
        c = int(correct_pred.get(classname, 0))
        acc = float(c) / t if t > 0 else None
        per_class_metrics[classname] = {"correct": c, "total": t, "accuracy": acc}

    valid_accs = [m["accuracy"] for m in per_class_metrics.values() if m["accuracy"] is not None]
    macro_accuracy = float(np.mean(valid_accs)) if len(valid_accs) > 0 else None

    metrics = {
        "overall_accuracy": overall_accuracy,
        "macro_accuracy": macro_accuracy,
        "per_class": per_class_metrics,
        "n_trials": total,
    }

    if args.save_predictions:
        results_dict = {"data": results, "metrics": metrics}

        base_dir = Path("results") / args.eval_dataset / run_tag
        base_dir.mkdir(parents=True, exist_ok=True)

        if args.clip_eval:
            leaf = f"clip_{args.eval_type}_{args.eval_dataset}_{args.stage}_eval_predictions.json"
        elif args.eval_metadata_filename == "eval_filtered_test.json":
            leaf = (
                f"{config['model']}_{config['cnn']}_seed_{config['seed']}_"
                f"{args.eval_type}_{args.eval_dataset}_{args.stage}_eval_filtered_predictions.json"
            )
        elif args.eval_metadata_filename == "eval_manual_filtered_test.json":
            leaf = (
                f"{config['model']}_{config['cnn']}_seed_{config['seed']}_"
                f"{args.eval_type}_{args.eval_dataset}_{args.stage}_eval_manual_filtered_predictions.json"
            )
        elif config["shuffle_utterances"]:
            leaf = (
                f"shuffle_{config['model']}_{config['cnn']}_seed_{config['seed']}_"
                f"{args.eval_type}_{args.eval_dataset}_{args.stage}_eval_predictions.json"
            )
        elif not config["augment_frames"]:
            leaf = (
                f"{config['model']}_{config['cnn']}_augment_frames_{config['augment_frames']}_"
                f"seed_{config['seed']}_{args.eval_type}_{args.eval_dataset}_{args.stage}_eval_predictions.json"
            )
        elif not config["multiple_frames"]:
            leaf = (
                f"{config['model']}_{config['cnn']}_multiple_frames_{config['multiple_frames']}_"
                f"seed_{config['seed']}_{args.eval_type}_{args.eval_dataset}_{args.stage}_eval_predictions.json"
            )
        else:
            leaf = (
                f"{config['model']}_{config['cnn']}_seed_{config['seed']}_"
                f"{args.eval_type}_{args.eval_dataset}_{args.stage}_eval_predictions.json"
            )

        results_filename = base_dir / leaf
        print(f"\nSaving predictions to {results_filename}")
        with open(results_filename, "w") as f:
            json.dump(results_dict, f)

        # per class metrics CSV
        try:
            per_class_rows = [{"class": k, **v} for k, v in per_class_metrics.items()]
            metrics_csv = os.path.splitext(str(results_filename))[0] + "_class_metrics.csv"
            pd.DataFrame(per_class_rows).to_csv(metrics_csv, index=False)
            print(f"Saved per-class metrics CSV to {metrics_csv}")
        except Exception as e:
            print(f"Could not write per-class metrics CSV: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        default=None,
        help="Path to checkpoint (.ckpt). Pass last.ckpt here for stable evaluation.",
    )
    parser.add_argument(
        "--clip_eval",
        action="store_true",
        help="Use CLIP model for evaluation",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="test",
        choices=["dev", "test"],
        help="which evaluation stage to use",
    )
    parser.add_argument(
        "--eval_include_sos_eos",
        action="store_true",
        help="include SOS/EOS tokens for eval labels",
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        default="image",
        choices=["image", "text"],
        help="Run evaluation using multiple images or multiple labels",
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="saycam",
        choices=["saycam", "object_categories"],
        help="Which evaluation dataset to use",
    )
    parser.add_argument(
        "--eval_metadata_filename",
        type=str,
        default="eval_test.json",
        help="JSON file with metadata evaluation split to use",
    )
    parser.add_argument(
        "--use_kitty_label",
        action="store_true",
        help="replaces cat label with kitty",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="save model predictions to JSON",
    )
    args = parser.parse_args()

    if not args.clip_eval and not args.checkpoint:
        raise ValueError("Provide --checkpoint /path/to/last.ckpt (or use --clip_eval).")

    main(args)
