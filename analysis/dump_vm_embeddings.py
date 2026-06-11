import argparse
import os
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from tqdm import tqdm

from multimodal.multimodal_lit import MultiModalLitModel
from multimodal.multimodal_saycam_data_module import MultiModalSAYCamDataModule


# ------------------------------------------------------------
# Distributed + seeding helpers
# ------------------------------------------------------------

def setup_distributed():
    """
    Initialize torch.distributed if launched with torchrun.

    Returns:
        use_ddp (bool), rank (int), world_size (int), device (torch.device)
    """
    if not torch.cuda.is_available():
        # CPU fallback, no DDP
        return False, 0, 1, torch.device("cpu")

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        return True, rank, world_size, device
    else:
        # Single GPU run
        return False, 0, 1, torch.device("cuda:0")


def seed_all(base_seed: int, rank: int = 0):
    """
    Seed Python, NumPy and Torch. For DDP we offset by rank so each process
    gets a different but reproducible stream.
    """
    seed = int(base_seed) + 1000 * int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Argparse and datamodule
# ------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--max_batches",
        type=int,
        default=250,
        help="Total number of shuffled batches to scan (across all GPUs).",
    )
    p.add_argument(
        "--max_instances_per_concept",
        type=int,
        default=200,
        help="Cap on stored objects per concept for t-SNE",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Base random seed for shuffling and sampling",
    )
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def build_saycam_datamodule(batch_size: int, num_workers: int, use_sam_masks: bool = True):
    """
    Build a MultiModalSAYCamDataModule with the same defaults as training,
    but with batch size / workers overridden and SAM masks forced on.
    """
    dm_parser = argparse.ArgumentParser(add_help=False)
    dm_parser = MultiModalSAYCamDataModule.add_to_argparse(dm_parser)
    dm_args_ns = dm_parser.parse_args([])  # Namespace with defaults

    # override a few key things on the Namespace
    dm_args_ns.batch_size = batch_size
    dm_args_ns.num_workers = num_workers
    dm_args_ns.use_sam_masks = bool(use_sam_masks)
    # for analysis we do not need shuffled utterances or multiple frames;
    # shuffling will be handled by a Sampler at the DataLoader level.
    dm_args_ns.multiple_frames = False
    dm_args_ns.shuffle_utterances = False

    debug_keys = [
        "batch_size",
        "num_workers",
        "use_sam_masks",
        "multiple_frames",
        "shuffle_utterances",
        "sam_masks_dir",
    ]
    debug_view = {k: getattr(dm_args_ns, k, None) for k in debug_keys}
    print("[datamodule] args:", debug_view, flush=True)

    dm = MultiModalSAYCamDataModule(dm_args_ns)
    dm.setup(stage="fit")
    return dm


def build_dataloader(dm, split, batch_size, num_workers, seed, use_ddp, rank, world_size):
    """
    Wrap the DM dataloader with a (Distributed)Sampler to get
    shuffled batches and multi-GPU support.
    """
    if split == "train":
        base_dl = dm.train_dataloader()
    elif split == "val":
        vdl = dm.val_dataloader()
        base_dl = vdl[0] if isinstance(vdl, (list, tuple)) else vdl
    else:
        tdl = dm.test_dataloader()
        base_dl = tdl[0] if isinstance(tdl, (list, tuple)) else tdl

    dataset = base_dl.dataset

    if use_ddp:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
        )
    else:
        sampler = RandomSampler(dataset)

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=getattr(base_dl, "collate_fn", None),
        pin_memory=getattr(base_dl, "pin_memory", False),
        drop_last=getattr(base_dl, "drop_last", False),
    )

    return dl, sampler

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


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    args = parse_args()
    use_ddp, rank, world_size, device = setup_distributed()
    seed_all(args.seed, rank)

    if rank == 0:
        print(f"[dist] use_ddp={use_ddp}, world_size={world_size}, device={device}", flush=True)
        print("Loading model from", args.checkpoint, flush=True)

    # load model
    model = _load_lit_disable_vm(args.checkpoint, map_location=device)

    vm = getattr(model, "visual_memory", None) or getattr(model, "vm", None)
    assert vm is not None, "Could not find VisualMemory on the model"
    num_concepts = getattr(vm, "num_concepts", 0)

    # build data module with SAM masks enabled
    dm = build_saycam_datamodule(args.batch_size, args.num_workers, use_sam_masks=True)
    dl, sampler = build_dataloader(
        dm,
        args.split,
        args.batch_size,
        args.num_workers,
        args.seed,
        use_ddp,
        rank,
        world_size,
    )

    # choose how many batches PER RANK so total ≈ args.max_batches
    if use_ddp:
        max_batches_per_rank = (args.max_batches + world_size - 1) // world_size
    else:
        max_batches_per_rank = args.max_batches

    if use_ddp and hasattr(sampler, "set_epoch"):
        # single epoch, but this makes sampling deterministic for a given seed
        sampler.set_epoch(0)

    # per-rank storage
    per_concept_kept = defaultdict(int)
    per_concept_total = np.zeros(num_concepts, dtype=np.int64) if num_concepts > 0 else None

    all_embeds = []
    all_concepts = []
    all_proto_idx = []
    all_clip = []
    all_frame = []
    all_paths = []

    batches_seen = 0
    batches_with_vm = 0
    total_objs_before_cap = 0

    with torch.no_grad():
        for b_idx, batch in enumerate(tqdm(dl, desc=f"Rank {rank} scanning")):
            if b_idx >= max_batches_per_rank:
                break
            batches_seen += 1

            vm_inputs = model.build_vm_inputs_from_batch(batch, device=device)
            if vm_inputs is None:
                continue

            batches_with_vm += 1

            embeds = vm_inputs["embeds"]        # (N_obj, D)
            cids = vm_inputs["concept_ids"]     # (N_obj,)
            clip_id = vm_inputs["clip_id"]      # (N_obj,)
            frame_idx = vm_inputs["frame_idx"]  # (N_obj,)
            relpaths = vm_inputs["frame_relpath"]  # list[str] or (N_obj,)

            if embeds.numel() == 0:
                continue

            total_objs_before_cap += embeds.size(0)

            # per-concept totals: only count concept ids in [0, num_concepts)
            if num_concepts > 0 and per_concept_total is not None:
                cids_np = cids.cpu().numpy()
                valid_mask = (cids_np >= 0) & (cids_np < num_concepts)
                cids_valid = cids_np[valid_mask]
                if cids_valid.size > 0:
                    per_concept_total_np = np.bincount(
                        cids_valid,
                        minlength=num_concepts,
                    ).astype(np.int64)
                    per_concept_total += per_concept_total_np

            # assignment to prototypes (no temporal info needed here)
            assign = vm.assign(embeds, cids, batch_meta=None)
            proto_idx = assign["proto_indices"]

            for i in range(embeds.size(0)):
                c = int(cids[i].item())
                if c < 0:
                    continue
                if per_concept_kept[c] >= args.max_instances_per_concept:
                    continue

                all_embeds.append(embeds[i].cpu().numpy())
                all_concepts.append(c)
                all_proto_idx.append(int(proto_idx[i].item()))
                all_clip.append(int(clip_id[i].item()))
                all_frame.append(int(frame_idx[i].item()))
                all_paths.append(relpaths[i])

                per_concept_kept[c] += 1

    # local summary struct for gathering
    local_summary = {
        "embeds": all_embeds,
        "concepts": all_concepts,
        "proto_idx": all_proto_idx,
        "clip": all_clip,
        "frame": all_frame,
        "paths": all_paths,
        "per_concept_total": per_concept_total,
        "batches_seen": batches_seen,
        "batches_with_vm": batches_with_vm,
        "total_objs_before_cap": total_objs_before_cap,
        "usage_counts": vm.usage_counts.detach().cpu().numpy(),
    }

    # gather summaries from all ranks
    if use_ddp:
        summaries = [None for _ in range(world_size)]
        dist.all_gather_object(summaries, local_summary)
    else:
        summaries = [local_summary]

    # rank 0 merges and saves
    if (not use_ddp) or rank == 0:
        merged_embeds = []
        merged_concepts = []
        merged_proto_idx = []
        merged_clip = []
        merged_frame = []
        merged_paths = []

        global_per_concept_total = None
        global_usage_counts = None

        global_batches_seen = 0
        global_batches_with_vm = 0
        global_total_objs_before_cap = 0

        for s in summaries:
            merged_embeds.extend(s["embeds"])
            merged_concepts.extend(s["concepts"])
            merged_proto_idx.extend(s["proto_idx"])
            merged_clip.extend(s["clip"])
            merged_frame.extend(s["frame"])
            merged_paths.extend(s["paths"])

            if s["per_concept_total"] is not None:
                if global_per_concept_total is None:
                    global_per_concept_total = np.zeros_like(s["per_concept_total"])
                global_per_concept_total += s["per_concept_total"]

            # usage_counts is saved in the checkpoint and identical on all ranks;
            # just take it from the first summary.
            if global_usage_counts is None:
                global_usage_counts = s["usage_counts"].astype(np.float32)

            global_batches_seen += s["batches_seen"]
            global_batches_with_vm += s["batches_with_vm"]
            global_total_objs_before_cap += s["total_objs_before_cap"]

        # final numpy arrays
        D = vm.prototypes.shape[1]
        if merged_embeds:
            obj_embeds = np.stack(merged_embeds, axis=0).astype(np.float32)
        else:
            obj_embeds = np.zeros((0, D), dtype=np.float32)

        obj_concepts = (
            np.array(merged_concepts, dtype=np.int64)
            if merged_concepts
            else np.zeros((0,), dtype=np.int64)
        )
        proto_indices = (
            np.array(merged_proto_idx, dtype=np.int64)
            if merged_proto_idx
            else np.zeros((0,), dtype=np.int64)
        )
        clip_id_arr = (
            np.array(merged_clip, dtype=np.int64)
            if merged_clip
            else np.zeros((0,), dtype=np.int64)
        )
        frame_idx_arr = (
            np.array(merged_frame, dtype=np.int64)
            if merged_frame
            else np.zeros((0,), dtype=np.int64)
        )

        proto_embeds = vm.prototypes.detach().cpu().numpy()         # (K, D)
        proto_concepts = vm.proto_concept_ids.detach().cpu().numpy()
        usage_counts = (
            global_usage_counts
            if global_usage_counts is not None
            else vm.usage_counts.detach().cpu().numpy()
        )

        concept_names = getattr(model, "concepts", None)
        if concept_names is None:
            concept_names = [f"c{idx}" for idx in range(vm.num_concepts)]

        if global_per_concept_total is None:
            global_per_concept_total = np.zeros(vm.num_concepts, dtype=np.int64)

        output = {
            "obj_embeds": obj_embeds,
            "obj_concepts": obj_concepts,
            "proto_indices": proto_indices,
            "clip_id": clip_id_arr,
            "frame_idx": frame_idx_arr,
            "frame_relpaths": merged_paths,
            "per_concept_total": global_per_concept_total,
            "proto_embeds": proto_embeds,
            "proto_concepts": proto_concepts,
            "proto_usage_counts": usage_counts,
            "concept_names": concept_names,
        }

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output, out_path)

        print(f"[summary] world_size:           {world_size}", flush=True)
        print(f"[summary] batches_seen (tot):   {global_batches_seen}", flush=True)
        print(f"[summary] batches_with_vm (tot):{global_batches_with_vm}", flush=True)
        print(f"[summary] total objs (raw):     {global_total_objs_before_cap}", flush=True)
        print(f"[summary] total objs kept:      {obj_embeds.shape[0]}", flush=True)
        print("Saved visual memory dump to", out_path, flush=True)
        print("Final obj_embeds shape:", obj_embeds.shape, flush=True)

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
