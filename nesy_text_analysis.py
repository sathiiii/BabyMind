# tools/nesy_text_analysis.py
# -*- coding: utf-8 -*-
"""
SAYCam neuro-symbolic text diagnostics (CVCL-style).

Pulls utterances via MultiModalSAYCamDataModule (same pipeline as train.py),
uses your ALIAS/canonicalizer + build_targets/build_default_rules, and
produces plots + CSVs to inspect supervision signal and rule (A->B) quality.

New in this version:
- Alias mining: suggest token->concept mappings from SAYCam text.
- Rule mining: extract A->B implication edges from text with support+thresholds,
  optional transitive pruning, and CSV/JSON export.

Usage examples:
    python tools/nesy_text_analysis.py --split train --outdir nesy_out \
        --max_batches 200 --debug_show_tokens 3 \
        --ckpt checkpoints/CVCL_nesy_loss/last.ckpt --device cpu \
        --alias_suggest --alias_min_support 50 --alias_p_thresh 0.7 \
        --edges_mine --edge_min_support 80 --edge_p_thresh 0.8
"""

import os
import re
import json
import argparse
import collections
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# --- repo imports ---
from multimodal.nesy_constraints import (
    _canon_tokens, build_targets, build_default_rules
)
from multimodal.multimodal_saycam_data_module import MultiModalSAYCamDataModule
from train import _setup_parser as build_repo_parser  # same parser CVCL uses

# Optional: for model-level rule gap with concept head (if --ckpt is given)
try:
    from multimodal.multimodal_lit import MultiModalLitModel
except Exception:
    MultiModalLitModel = None


DEFAULT_22 = [
    "ball","basket","car","cat","chair","computer","crib","door","floor","foot",
    "ground","hand","kitchen","paper","puzzle","road","room","sand","stairs",
    "table","toy","window"
]


# ---------- filename helpers ----------

def _with_split(outdir: Path, split: str, leaf: str) -> Path:
    """Prepend split (train/val/test) to the filename leaf."""
    return outdir / f"{split}_{leaf}"


# ---------- vocab-aware decoding utilities ----------

def _build_id_maps_and_ignores(dm):
    """Build id->token map and a set of token IDs to ignore (specials, punct)."""
    vocab = dm.read_vocab()
    if isinstance(vocab, dict):
        id2tok = {int(i): tok for tok, i in vocab.items()}
    elif hasattr(vocab, "itos"):  # torchtext-like
        itos = list(vocab.itos)
        id2tok = {i: tok for i, tok in enumerate(itos)}
    else:
        raise ValueError("Unsupported vocab type returned by DataModule.read_vocab()")

    ignore_tokens = {"<pad>", "<unk>", "<sos>", "<eos>", ".", ",", "?", "!", "...", "..", "...."}
    ignore_ids = {vocab[t] for t in ignore_tokens if (isinstance(vocab, dict) and t in vocab)}
    return vocab, id2tok, ignore_ids


def _decode_any(obj, id2tok, ignore_ids):
    """
    Robustly decode raw_y (which may be strings, ints, tensors, lists of lists, etc.)
    into a *flat list of tokens* with specials removed.
    """
    import torch

    def _flatten(x):
        if torch.is_tensor(x):
            for v in x.reshape(-1).tolist():
                yield v
        elif isinstance(x, (list, tuple)):
            for y in x:
                yield from _flatten(y)
        else:
            yield x

    toks: List[str] = []
    for item in _flatten(obj):
        if isinstance(item, int):
            if item in ignore_ids:
                continue
            toks.append(id2tok.get(int(item), "<unk>"))
        elif isinstance(item, str):
            toks.extend([w for w in re.split(r"[^a-zA-Z]+", item.lower()) if w])
        else:
            s = str(item).lower()
            toks.extend([w for w in re.split(r"[^a-zA-Z]+", s) if w])
    return toks


def _gather_raw_utterances(loader, id2tok, ignore_ids, max_batches: Optional[int] = None) -> List[List[str]]:
    """
    Return a list of per-utterance token lists (already vocab-decoded & de-puncted).
    This fixes the 'coverage==0' issue that arises when raw_y is a tensor of IDs.
    """
    raws: List[List[str]] = []
    for bi, batch in enumerate(loader):
        # expected batch structure: (x, y, y_len, raw_y)
        raw_y = batch[-1]
        if isinstance(raw_y, (list, tuple)):
            for u in raw_y:
                toks = _decode_any(u, id2tok, ignore_ids)
                raws.append(toks)
        else:
            raws.append(_decode_any(raw_y, id2tok, ignore_ids))
        if max_batches is not None and (bi + 1) >= max_batches:
            break
    return raws


def _canon_per_utt(raws: List[List[str]]) -> List[set]:
    """
    Canonicalize (alias + simple plural handling) and return per-utterance *sets*
    for co-occurrence/adherence computations downstream.
    """
    return [set(_canon_tokens(u)) for u in raws]


# ---------- alias mining ----------

def suggest_aliases(
    raws: List[List[str]],
    concepts: List[str],
    min_support: int = 50,
    p_thresh: float = 0.7
) -> Tuple[Dict[str, str], pd.DataFrame]:
    """
    Suggest token -> concept aliases using conditional P(c|t).
    For every token t not in concept list with count >= min_support:
        if max_c P(c|t) >= p_thresh, propose t -> argmax_c P(c|t).

    Returns:
        mapping dict {token: concept}, and a DataFrame with details per suggestion.
    """
    names = [c.lower() for c in concepts]
    name_set = set(names)

    # Utterance-level concept mask
    mask = build_targets(raws, names)  # (B,C)

    # Canonicalize tokens per utterance once
    tok_sets = _canon_per_utt(raws)

    # Token frequencies (utterance-level)
    vocab = collections.Counter(t for ts in tok_sets for t in ts)

    suggestions: Dict[str, str] = {}
    rows = []

    # Index list per token for fast selection
    tok_to_uttidxs: Dict[str, List[int]] = collections.defaultdict(list)
    for i, ts in enumerate(tok_sets):
        for t in ts:
            tok_to_uttidxs[t].append(i)

    for t, n in vocab.items():
        if t in name_set or n < min_support:
            continue
        idxs = tok_to_uttidxs.get(t, [])
        if not idxs:
            continue
        submask = mask[idxs]  # (#mentions, C)
        counts = submask.sum(0).cpu().numpy()
        total = counts.sum()
        if total <= 0:
            continue
        probs = counts / total
        c_star_idx = int(probs.argmax())
        pmax = float(probs[c_star_idx])
        if pmax >= p_thresh:
            c_star = names[c_star_idx]
            suggestions[t] = c_star
            rows.append({
                "token": t,
                "suggested_concept": c_star,
                "support_n_utts": int(n),
                "pmax": pmax
            })

    df = pd.DataFrame(rows).sort_values(["pmax", "support_n_utts"], ascending=[False, False])
    return suggestions, df


# ---------- rule (A->B) mining ----------

def _path_exists_without_direct_edge(u: int, v: int, adj: Dict[int, set]) -> bool:
    """
    Return True if there exists a path u -> ... -> v of length >= 2, i.e.,
    excluding the direct edge (u, v).
    """
    from collections import deque
    q = deque([u])
    visited = set([u])
    while q:
        x = q.popleft()
        for y in adj.get(x, []):
            if x == u and y == v:
                # skip the direct edge to ensure length >= 2
                continue
            if y == v:
                return True
            if y not in visited:
                visited.add(y)
                q.append(y)
    return False


def prune_transitive_edges(edges: List[Tuple[int, int]], C: int) -> List[Tuple[int, int]]:
    """
    Remove (u,v) if there exists another path u -> ... -> v (length >=2).
    Keeps a minimal equivalent edge set when the graph is a DAG.
    """
    # Build adjacency
    adj: Dict[int, set] = {i: set() for i in range(C)}
    for a, b in edges:
        adj[a].add(b)

    keep = []
    for (a, b) in edges:
        if not _path_exists_without_direct_edge(a, b, adj):
            keep.append((a, b))
    return keep


def mine_edges_from_text(
    mask: torch.Tensor,
    names: List[str],
    min_support: int = 80,
    p_thresh: float = 0.8,
    reduce_transitive: bool = True
) -> Tuple[List[Tuple[int, int]], pd.DataFrame]:
    """
    Mine A->B edges where:
      support n(A) >= min_support AND P(B|A) >= p_thresh.

    Returns:
      edges list [(a_idx, b_idx), ...] and a DataFrame with stats.
    """
    B, C = int(mask.shape[0]), int(mask.shape[1])
    rows = []
    edges = []

    for a in range(C):
        A = mask[:, a].bool()
        nA = int(A.sum())
        if nA < min_support:
            continue
        for b in range(C):
            if a == b:
                continue
            Bv = mask[:, b].bool()
            nAB = int((A & Bv).sum())
            p = float(nAB / nA) if nA > 0 else float("nan")
            if p >= p_thresh:
                edges.append((a, b))
            rows.append({
                "A": names[a],
                "B": names[b],
                "n(A)": nA,
                "n(B)": int(Bv.sum()),
                "n(A∧B)": nAB,
                "P(B|A)": p,
                "violation_rate": float((nA - nAB) / nA) if nA > 0 else float("nan"),
            })

    if reduce_transitive and edges:
        edges = prune_transitive_edges(edges, C)

    # DataFrame for inspection (sorted by P(B|A) desc then n(A) desc)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["P(B|A)", "n(A)"], ascending=[False, False])

    return edges, df


# ---------- analysis ----------

def analyze_mentions(raws: List[List[str]], concepts: List[str]) -> Dict[str, Any]:
    """
    raws: list of token lists (already vocab-decoded).
    concepts: concept name list (e.g., the 22 Labeled-S).
    """
    names = [c.lower() for c in concepts]
    # build_targets accepts strings OR token-lists; pass token-lists, it will canonicalize
    mask = build_targets(raws, names)  # (B,C)
    B, C = int(mask.shape[0]), int(mask.shape[1])

    mentions_per = mask.sum(dim=1).cpu().numpy()
    coverage = float((mentions_per > 0).mean()) if B else 0.0
    avg_mentions = float(mentions_per.mean()) if B else 0.0

    per_class_counts = mask.sum(dim=0).cpu().numpy().astype(int)
    per_class_freq = per_class_counts / max(1, B)

    # size histogram: how many concepts per utterance
    counter = collections.Counter(mentions_per.tolist())
    size_hist = dict(sorted(counter.items(), key=lambda kv: int(kv[0])))

    # co-occurrence (utterance-level)
    cooccur = (mask.T @ mask).cpu().numpy().astype(int)  # (C,C)

    # rule stats for *default* rules
    edges = build_default_rules(names)
    rows = []
    for a_idx, b_idx in edges:
        A, Bn = names[a_idx], names[b_idx]
        a = mask[:, a_idx].bool()
        b = mask[:, b_idx].bool()
        nA = int(a.sum())
        nB = int(b.sum())
        nAB = int((a & b).sum())
        nA_notB = int((a & (~b)).sum())
        p_B_given_A = (nAB / nA) if nA > 0 else float("nan")
        rows.append({
            "antecedent": A,
            "consequent": Bn,
            "n(A)": nA,
            "n(B)": nB,
            "n(A∧B)": nAB,
            "n(A∧¬B)": nA_notB,
            "P(B|A)": p_B_given_A,
            "violation_rate": (nA_notB / nA) if nA > 0 else float("nan"),
        })

    # utterance-level adherence: all triggered rules satisfied?
    # For each utterance, collect all As present; check Bs present (after canonicalization).
    edges_set = [(names[a], names[b]) for a, b in edges]
    tok_sets = _canon_per_utt(raws)
    ok = 0
    trig_any = 0
    for toks in tok_sets:
        triggered = [(a, b) for (a, b) in edges_set if a in toks]
        if not triggered:
            continue
        trig_any += 1
        if all((b in toks) for (_, b) in triggered):
            ok += 1
    adherence = (ok / trig_any) if trig_any > 0 else float("nan")

    return dict(
        B=B,
        concepts=names,
        coverage=coverage,
        avg_mentions=avg_mentions,
        per_class_counts=per_class_counts,
        per_class_freq=per_class_freq,
        cooccur_counts=cooccur,
        size_hist=size_hist,
        rule_stats=rows,
        rule_adherence=adherence,
        mask=mask,               # for miners
        tok_sets=tok_sets,       # for alias miner
    )


@torch.no_grad()
def model_rule_gap(ckpt: str, loader, concepts: List[str], device: str = "cuda") -> pd.DataFrame:
    """Avg ReLU(pA - pB) per rule using the concept head."""
    if MultiModalLitModel is None:
        print("Could not import MultiModalLitModel; skipping model-level rule gap.")
        return pd.DataFrame()

    names = [c.lower() for c in concepts]
    edges = build_default_rules(names)
    if not edges:
        return pd.DataFrame()

    dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    lit = MultiModalLitModel.load_from_checkpoint(ckpt, map_location=dev)
    lit.eval().to(dev)

    sums = np.zeros(len(edges), dtype=np.float64)
    n = 0

    for batch in loader:
        x = batch[0].to(dev)
        # encode image -> concept logits
        img_feats, fmap = lit.model.encode_image(x)
        logits = lit.vision_encoder.concept_logits(img_feats, image_feature_map=fmap)  # (B,C) or (B,T,C)
        probs = torch.sigmoid(logits).amax(dim=1) if logits.dim() == 3 else torch.sigmoid(logits)  # (B,C)
        p = probs.cpu().numpy()
        for ei, (a, b) in enumerate(edges):
            gap = np.maximum(0.0, p[:, a] - p[:, b])  # ReLU
            sums[ei] += gap.sum()
        n += p.shape[0]

    avg_gap = sums / max(1, n)
    rows = [{"A": names[a], "B": names[b], "avg_ReLU(pA-pB)": float(g)} for (a, b), g in zip(edges, avg_gap)]
    df = pd.DataFrame(rows)
    df.loc["__mean__"] = {"A": "", "B": "MEAN", "avg_ReLU(pA-pB)": float(np.mean(avg_gap))}
    return df


# ---------- plotting (plain matplotlib, no style/colors set) ----------

def plot_per_class_counts(names: List[str], counts: np.ndarray, out: Optional[Path]):
    fig, ax = plt.subplots(figsize=(10, 4))
    idx = np.arange(len(names))
    ax.bar(idx, counts)
    ax.set_xticks(idx); ax.set_xticklabels(names, rotation=60, ha="right")
    ax.set_ylabel("Utterance count")
    ax.set_title("Per-class mention counts (after ALIAS + canonicalization)")
    plt.tight_layout()
    if out: fig.savefig(out, dpi=160)
    plt.show()

def plot_size_hist(size_hist: Dict[int, int], out: Optional[Path]):
    xs = sorted(size_hist.keys()); ys = [size_hist[k] for k in xs]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(int(x)) for x in xs], ys)
    ax.set_xlabel("# concepts mentioned in an utterance")
    ax.set_ylabel("Count")
    ax.set_title("Mention set-size distribution")
    plt.tight_layout()
    if out: fig.savefig(out, dpi=160)
    plt.show()

def plot_cooccur(names: List[str], M: np.ndarray, out: Optional[Path]):
    M = M.astype(float); np.fill_diagonal(M, 0.0)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(M, interpolation="nearest", aspect="auto")
    ax.set_xticks(np.arange(len(names))); ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=90); ax.set_yticklabels(names)
    ax.set_title("Co-occurrence heatmap (utterance-level)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if out: fig.savefig(out, dpi=160)
    plt.show()

def plot_rule_viols(rule_rows: List[dict], out: Optional[Path]) -> pd.DataFrame:
    if not rule_rows:
        print("No rules found for these concepts.")
        return pd.DataFrame()
    df = pd.DataFrame(rule_rows)
    labels = [f"{r['antecedent']}→{r['consequent']}" for r in rule_rows]
    vals = df["violation_rate"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(labels)), np.nan_to_num(vals))
    ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("P(A ∧ ¬B)")
    ax.set_title("Text-level rule violation rate")
    plt.tight_layout()
    if out: fig.savefig(out, dpi=160)
    plt.show()
    return df


# ---------- main ----------

def main():
    p = argparse.ArgumentParser("SAYCam NeSy diagnostics (CVCL-style)")
    p.add_argument("--concepts", type=str, default=None,
                   help="Path to JSON list of concept names (default: Labeled-S 22).")
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"],
                   help="Which split to analyze.")
    p.add_argument("--max_batches", type=int, default=None,
                   help="Limit #batches (for quick runs).")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Optional .ckpt for model-level rule gap via concept head.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--outdir", type=str, default="nesy_analysis_out")
    p.add_argument("--debug_show_tokens", type=int, default=0,
                   help="If >0, print the first N decoded utterances' tokens.")

    # NEW: alias mining flags
    p.add_argument("--alias_suggest", action="store_true",
                   help="Run alias miner and export suggestions.")
    p.add_argument("--alias_min_support", type=int, default=50,
                   help="Minimum #utterances a token must appear in to be considered for aliasing.")
    p.add_argument("--alias_p_thresh", type=float, default=0.7,
                   help="Threshold on max_c P(c|t) to propose t->c alias.")

    # NEW: edge mining flags
    p.add_argument("--edges_mine", action="store_true",
                   help="Mine A->B edges from text and export.")
    p.add_argument("--edge_min_support", type=int, default=80,
                   help="Minimum n(A) to consider A->B.")
    p.add_argument("--edge_p_thresh", type=float, default=0.8,
                   help="Threshold on P(B|A) to keep A->B.")
    p.add_argument("--no_transitive_prune", action="store_true",
                   help="Disable transitive edge pruning (keeps redundant edges).")

    # Parse our flags, then pass the rest to the repo parser so data args behave like train.py
    known, unknown = p.parse_known_args()
    args = known

    # Build the same parser your repo uses and parse remaining CLI args into data_args
    repo_parser = build_repo_parser()
    data_args = repo_parser.parse_args(unknown if unknown is not None else [])

    # Force SAYCam (you said you only care about SAYCam here)
    data_args.dataset = "saycam"

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Concepts
    if args.concepts:
        with open(args.concepts) as f:
            concepts = json.load(f)
        assert isinstance(concepts, list)
    else:
        concepts = DEFAULT_22

    # ---- build datamodule & loader exactly like training ----
    dm = MultiModalSAYCamDataModule(data_args)
    dm.prepare_data(); dm.setup()

    if args.split == "train":
        loader = dm.train_dataloader()
    elif args.split == "val":
        loader = dm.val_dataloader()[0]  # first val dataloader is the paired data
    else:  # "test"
        loader = dm.test_dataloader()[0]

    # ---- vocab maps & ignore IDs (mirrors training) ----
    vocab, id2tok, ignore_ids = _build_id_maps_and_ignores(dm)

    # ---- collect utterances and analyze text signal ----
    raws = _gather_raw_utterances(loader, id2tok=id2tok, ignore_ids=ignore_ids, max_batches=args.max_batches)

    if args.debug_show_tokens > 0:
        for i in range(min(args.debug_show_tokens, len(raws))):
            print(f"[DEBUG] utt #{i} tokens:", raws[i][:50])

    report = analyze_mentions(raws, concepts)

    # Save tables (prefix with split)
    per_class_csv = _with_split(outdir, args.split, "per_class_counts.csv")
    pd.DataFrame({
        "concept": report["concepts"],
        "count": report["per_class_counts"],
        "freq": report["per_class_freq"],
    }).sort_values("count", ascending=False).to_csv(per_class_csv, index=False)

    if report["rule_stats"]:
        rule_stats_csv = _with_split(outdir, args.split, "text_rule_stats.csv")
        pd.DataFrame(report["rule_stats"]).to_csv(rule_stats_csv, index=False)

    # Print quick summary
    print(f"Utterances analyzed: {report['B']}")
    print(f"Coverage (≥1 mention): {report['coverage']:.3f}")
    print(f"Avg mentions / utterance: {report['avg_mentions']:.3f}")
    print(f"Rule adherence (utterance-level): {report['rule_adherence']:.3f} "
          f"(fraction of utterances whose triggered A→B are all satisfied)")
    print(f"[Saved] {per_class_csv}")
    if report["rule_stats"]:
        print(f"[Saved] {rule_stats_csv}")

    # Plots (prefix with split)
    plot_per_class_counts(
        report["concepts"],
        report["per_class_counts"],
        _with_split(outdir, args.split, "per_class_counts.png"),
    )
    plot_size_hist(
        report["size_hist"],
        _with_split(outdir, args.split, "setsize_hist.png"),
    )
    plot_cooccur(
        report["concepts"],
        report["cooccur_counts"],
        _with_split(outdir, args.split, "cooccur_heatmap.png"),
    )
    _ = plot_rule_viols(
        report["rule_stats"],
        _with_split(outdir, args.split, "rule_violation_rates.png"),
    )

    # ----------------- NEW: Alias miner -----------------
    if args.alias_suggest:
        print("[Alias] Mining alias suggestions...")
        suggestions, alias_df = suggest_aliases(
            raws,
            report["concepts"],
            min_support=args.alias_min_support,
            p_thresh=args.alias_p_thresh,
        )
        alias_json = _with_split(outdir, args.split, "alias_suggestions.json")
        alias_csv  = _with_split(outdir, args.split, "alias_suggestions.csv")
        with open(alias_json, "w") as f:
            json.dump(suggestions, f, indent=2)
        alias_df.to_csv(alias_csv, index=False)
        print(f"[Alias] {len(suggestions)} suggestions.")
        print(f"[Saved] {alias_json}")
        print(f"[Saved] {alias_csv}")

        # Also save a token frequency table to help manual review
        tok_sets = report["tok_sets"]
        vocab_counts = collections.Counter(t for ts in tok_sets for t in ts)
        vocab_df = pd.DataFrame(
            [{"token": t, "n_utts": c} for t, c in vocab_counts.items()]
        ).sort_values("n_utts", ascending=False)
        vocab_csv = _with_split(outdir, args.split, "token_utterance_counts.csv")
        vocab_df.to_csv(vocab_csv, index=False)
        print(f"[Saved] {vocab_csv}")

    # ----------------- NEW: Edge miner ------------------
    if args.edges_mine:
        print("[Rules] Mining A→B edges from text...")
        mined_edges, edges_df = mine_edges_from_text(
            report["mask"],
            report["concepts"],
            min_support=args.edge_min_support,
            p_thresh=args.edge_p_thresh,
            reduce_transitive=not args.no_transitive_prune
        )
        edges_json = _with_split(outdir, args.split, "mined_edges.json")
        edges_csv  = _with_split(outdir, args.split, "mined_edges_table.csv")
        with open(edges_json, "w") as f:
            json.dump(mined_edges, f, indent=2)  # list of [a_idx, b_idx]
        edges_df.to_csv(edges_csv, index=False)
        print(f"[Rules] {len(mined_edges)} edges kept "
              f"(transitive_prune={'OFF' if args.no_transitive_prune else 'ON'}).")
        print(f"[Saved] {edges_json}")
        print(f"[Saved] {edges_csv}")

    # Optional: model-level rule gap using concept head (prefix with split)
    if args.ckpt:
        print(f"Computing model-level rule gap from {args.ckpt} …")
        df_gap = model_rule_gap(args.ckpt, loader, concepts, device=args.device)
        if len(df_gap):
            gap_csv = _with_split(outdir, args.split, "model_rule_gap.csv")
            df_gap.to_csv(gap_csv, index=False)
            print(df_gap)
            print(f"[Saved] {gap_csv}")


if __name__ == "__main__":
    main()
