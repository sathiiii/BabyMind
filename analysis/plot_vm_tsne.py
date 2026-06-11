import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_pt",
        type=str,
        required=True,
        help="Output of dump_vm_embeddings.py",
    )
    p.add_argument(
        "--max_points",
        type=int,
        default=4000,
        help="Max total object points for global t-SNE",
    )
    p.add_argument(
        "--per_concept_points",
        type=int,
        default=200,
        help="Points per concept in per-concept plots",
    )
    p.add_argument("--outdir", type=str, required=True)
    return p.parse_args()


def l2_normalize_np(x, eps: float = 1e-8):
    """Row-wise L2 normalization for numpy arrays, safe on empty inputs."""
    x = np.asarray(x)
    if x.size == 0:
        return x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def subsample_indices_per_concept(concepts, max_points_per_concept):
    rng = np.random.RandomState(0)
    unique = np.unique(concepts)
    idxs = []
    for c in unique:
        if c < 0:
            continue
        c_idx = np.where(concepts == c)[0]
        if len(c_idx) > max_points_per_concept:
            c_idx = rng.choice(c_idx, size=max_points_per_concept, replace=False)
        idxs.append(c_idx)
    if not idxs:
        return np.array([], dtype=np.int64)
    return np.concatenate(idxs, axis=0)


def compute_assignment_stats(obj_embeds, proto_embeds, obj_concepts, proto_indices):
    """
    Print some high-D stats: how far are objects from the prototypes
    they were assigned to (in dump_vm_embeddings)?
    """
    if proto_indices is None or obj_embeds.size == 0:
        print("[assign] No proto_indices or empty obj_embeds, skipping stats.")
        return

    # global distances
    diffs = obj_embeds - proto_embeds[proto_indices]
    dists = np.linalg.norm(diffs, axis=1)
    print(f"[assign] mean L2 dist obj->proto (global): {dists.mean():.4f}")
    print(f"[assign] median L2 dist obj->proto:        {np.median(dists):.4f}")
    print(f"[assign] min / max L2 dist:                 {dists.min():.4f} / {dists.max():.4f}")

    # per concept, just to see if some concepts behave differently
    concepts = np.unique(obj_concepts)
    concepts = [int(c) for c in concepts if c >= 0]
    for c in concepts:
        mask = (obj_concepts == c)
        if not mask.any():
            continue
        d_c = dists[mask]
        print(
            f"[assign] concept {c:2d}: n={d_c.size:4d}, "
            f"mean={d_c.mean():.4f}, median={np.median(d_c):.4f}"
        )


def plot_global_tsne(
    obj_embeds,
    obj_concepts,
    proto_embeds,
    proto_concepts,
    proto_indices,
    concept_names,
    outdir,
    max_links_per_proto: int = 20,
):
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # normalize both objects and prototypes for fair geometry in t-SNE
    obj_embeds_n = l2_normalize_np(obj_embeds)
    proto_embeds_n = l2_normalize_np(proto_embeds)

    # basic sanity prints
    print(f"[global] #prototypes: {proto_embeds_n.shape[0]}")
    print(f"[global] #objects:    {obj_embeds_n.shape[0]}")
    if obj_embeds_n.shape[0] == 0:
        print("[global] Warning: no object embeddings found, will plot prototypes only.")

    # t-SNE on prototypes + objects
    X = np.concatenate([proto_embeds_n, obj_embeds_n], axis=0)
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=min(30, max(5, X.shape[0] // 5)),
        random_state=0,
    )
    X_2d = tsne.fit_transform(X)

    K = proto_embeds_n.shape[0]
    proto_2d = X_2d[:K]
    obj_2d = X_2d[K:]

    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(10, 10))

    # iterate over all concept ids that actually appear
    all_ids = np.unique(
        np.concatenate([proto_concepts, obj_concepts]) if obj_embeds_n.size > 0 else proto_concepts
    )
    all_ids = [int(c) for c in all_ids if c >= 0]

    for c in all_ids:
        color = cmap(c % 20)

        # objects of this concept (dots)
        o_mask = obj_concepts == c
        if obj_embeds_n.size > 0 and o_mask.any():
            ax.scatter(
                obj_2d[o_mask, 0],
                obj_2d[o_mask, 1],
                marker="o",
                s=8,
                alpha=0.4,
                linewidths=0,
                color=color,
            )

        # prototypes of this concept (big stars, appear in legend)
        p_mask = proto_concepts == c
        if p_mask.any():
            if c < len(concept_names):
                label = f"{concept_names[c]} protos"
            else:
                label = f"c{c} protos"
            ax.scatter(
                proto_2d[p_mask, 0],
                proto_2d[p_mask, 1],
                marker="*",
                s=140,
                edgecolors="black",
                linewidths=0.7,
                color=color,
                label=label,
                zorder=3,
            )

    # draw light links from each prototype to up to max_links_per_proto
    # of its assigned objects (after subsampling), to visualize how far
    # they are in the 2D embedding.
    if proto_indices is not None and obj_embeds_n.size > 0:
        N_obj = obj_embeds_n.shape[0]
        if proto_indices.shape[0] != N_obj:
            print(
                f"[global] Warning: proto_indices length {proto_indices.shape[0]} "
                f"does not match obj_embeds {N_obj}, skipping links."
            )
        else:
            for k in range(K):
                mask_k = (proto_indices == k)
                idx_k = np.where(mask_k)[0]
                if idx_k.size == 0:
                    continue
                if idx_k.size > max_links_per_proto:
                    idx_k = idx_k[:max_links_per_proto]
                pts = obj_2d[idx_k]
                p = proto_2d[k]
                for q in pts:
                    ax.plot(
                        [p[0], q[0]],
                        [p[1], q[1]],
                        color="gray",
                        alpha=0.15,
                        linewidth=0.5,
                        zorder=1,
                    )

    ax.set_title("Visual memory: global t-SNE of prototypes (stars) and objects (dots)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=8, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(Path(outdir) / "vm_tsne_global.png", dpi=300)
    plt.close(fig)


def plot_per_concept_tsne(
    obj_embeds,
    obj_concepts,
    proto_embeds,
    proto_concepts,
    concept_names,
    per_concept_points,
    outdir,
):
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # normalized copies for consistent geometry
    obj_embeds_n = l2_normalize_np(obj_embeds)
    proto_embeds_n = l2_normalize_np(proto_embeds)

    all_ids = np.unique(
        np.concatenate([proto_concepts, obj_concepts]) if obj_embeds_n.size > 0 else proto_concepts
    )
    all_ids = [int(c) for c in all_ids if c >= 0]

    rng = np.random.RandomState(0)

    for c in all_ids:
        o_mask = obj_concepts == c
        p_mask = proto_concepts == c
        if not p_mask.any() or not o_mask.any():
            continue

        obj_idx = np.where(o_mask)[0]
        if len(obj_idx) > per_concept_points:
            obj_idx = rng.choice(obj_idx, size=per_concept_points, replace=False)

        X_proto = proto_embeds_n[p_mask]
        X_obj = obj_embeds_n[obj_idx]

        X = np.concatenate([X_proto, X_obj], axis=0)
        tsne = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=min(30, max(5, len(X) // 3)),
            random_state=0,
        )
        X_2d = tsne.fit_transform(X)
        Kc = X_proto.shape[0]
        proto_2d = X_2d[:Kc]
        obj_2d = X_2d[Kc:]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(
            obj_2d[:, 0],
            obj_2d[:, 1],
            marker="o",
            s=10,
            alpha=0.6,
            label="objects",
        )
        ax.scatter(
            proto_2d[:, 0],
            proto_2d[:, 1],
            marker="*",
            s=180,
            edgecolors="black",
            linewidths=0.7,
            label="prototypes",
            zorder=3,
        )

        name = concept_names[c] if c < len(concept_names) else f"c{c}"
        ax.set_title(f"{name}: prototypes vs objects")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend()
        fig.tight_layout()
        fig.savefig(Path(outdir) / f"vm_tsne_{name}.png", dpi=300)
        plt.close(fig)


def plot_stats(
    per_concept_total,
    proto_concepts,
    usage_counts,
    concept_names,
    outdir,
):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    num_concepts = len(concept_names)
    xs = np.arange(num_concepts)

    # instances per concept
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(xs, per_concept_total)
    ax.set_xticks(xs)
    ax.set_xticklabels(concept_names, rotation=45, ha="right")
    ax.set_ylabel("# SAM instances in scanned data")
    ax.set_title("Number of SAM instances per concept")
    fig.tight_layout()
    fig.savefig(Path(outdir) / "vm_instances_per_concept.png", dpi=300)
    plt.close(fig)

    # prototype usage per concept
    proto_usage_per_concept = np.zeros(num_concepts, dtype=np.float32)
    for c in range(num_concepts):
        mask = proto_concepts == c
        if mask.any():
            proto_usage_per_concept[c] = usage_counts[mask].sum()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(xs, proto_usage_per_concept)
    ax.set_xticks(xs)
    ax.set_xticklabels(concept_names, rotation=45, ha="right")
    ax.set_ylabel("Total prototype assignments (usage_counts)")
    ax.set_title("Visual memory prototype usage per concept")
    fig.tight_layout()
    fig.savefig(Path(outdir) / "vm_proto_usage_per_concept.png", dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = torch.load(args.data_pt, map_location="cpu")
    obj_embeds = data["obj_embeds"]
    obj_concepts = data["obj_concepts"]
    proto_embeds = data["proto_embeds"]
    proto_concepts = data["proto_concepts"]
    usage_counts = data["proto_usage_counts"]
    per_concept_total = data["per_concept_total"]
    concept_names = data["concept_names"]
    proto_indices = data.get("proto_indices", None)

    # ensure numpy
    obj_embeds = np.asarray(obj_embeds)
    obj_concepts = np.asarray(obj_concepts)
    proto_embeds = np.asarray(proto_embeds)
    proto_concepts = np.asarray(proto_concepts)
    usage_counts = np.asarray(usage_counts)
    per_concept_total = np.asarray(per_concept_total)
    if proto_indices is not None:
        proto_indices = np.asarray(proto_indices)

    num_concepts = len(concept_names)

    print(f"[load] obj_embeds shape: {obj_embeds.shape}")
    print(f"[load] proto_embeds shape: {proto_embeds.shape}")
    print(f"[load] unique obj_concepts: {np.unique(obj_concepts)}")
    print(f"[load] unique proto_concepts: {np.unique(proto_concepts)}")

    # subsample objects for global t-SNE if needed
    if obj_embeds.shape[0] > args.max_points and num_concepts > 0:
        idx = subsample_indices_per_concept(
            obj_concepts,
            max_points_per_concept=args.max_points // max(num_concepts, 1),
        )
        obj_embeds = obj_embeds[idx]
        obj_concepts = obj_concepts[idx]
        if proto_indices is not None and proto_indices.shape[0] == data["obj_embeds"].shape[0]:
            proto_indices = proto_indices[idx]
        print(f"[load] subsampled objects to {obj_embeds.shape[0]} for global t-SNE")

    # high-D diagnostics before any 2D embedding
    compute_assignment_stats(obj_embeds, proto_embeds, obj_concepts, proto_indices)

    plot_global_tsne(
        obj_embeds,
        obj_concepts,
        proto_embeds,
        proto_concepts,
        proto_indices,
        concept_names,
        outdir,
    )

    plot_per_concept_tsne(
        obj_embeds,
        obj_concepts,
        proto_embeds,
        proto_concepts,
        concept_names,
        args.per_concept_points,
        outdir,
    )

    plot_stats(
        per_concept_total,
        proto_concepts,
        usage_counts,
        concept_names,
        outdir,
    )


if __name__ == "__main__":
    main()
