#!/usr/bin/env python3
import json, argparse
from pathlib import Path
import numpy as np

def load_vocab(path: Path):
    v = json.loads(path.read_text())
    # common formats: {"word": id, ...} OR {"w2i": {...}} OR {"stoi": {...}}
    if isinstance(v, dict):
        for k in ["w2i", "stoi", "token_to_id"]:
            if k in v and isinstance(v[k], dict):
                return v[k]
        return v
    raise ValueError(f"Unrecognized vocab format in {path}")

def list_imgs(cat_dir: Path):
    exts = {".jpg", ".jpeg", ".png"}
    # recursive is safest across different zip layouts
    return sorted([p for p in cat_dir.rglob("*") if p.suffix.lower() in exts])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s_multimodal", type=Path, required=True,
                    help="Root: /misc/.../S_multimodal")
    ap.add_argument("--n_foils", type=int, default=3)
    ap.add_argument("--n_repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    vocab_path = args.s_multimodal / "../vocab.json"
    obj_root = args.s_multimodal
    vocab = load_vocab(vocab_path)

    # categories present on disk
    categories = sorted([p.name for p in obj_root.iterdir() if p.is_dir()])
    # match CVCL convention: only categories that exist in training vocab :contentReference[oaicite:2]{index=2}
    categories = [c for c in categories if c in vocab]

    rng = np.random.default_rng(args.seed)

    # pre-index images per category
    imgs_by_cat = {}
    for c in categories:
        imgs = list_imgs(obj_root / c)
        if len(imgs) == 0:
            continue
        imgs_by_cat[c] = imgs

    categories = [c for c in categories if c in imgs_by_cat]
    if len(categories) < args.n_foils + 1:
        raise RuntimeError("Not enough categories with images to sample foils.")

    trials = []
    trial_num = 0
    for target_cat in categories:
        target_imgs = imgs_by_cat[target_cat]
        for target_img in target_imgs:
            for rep in range(args.n_repeats):
                foil_cats = [c for c in categories if c != target_cat]
                foil_cats = rng.choice(foil_cats, size=args.n_foils, replace=False).tolist()

                foil_imgs = []
                for fc in foil_cats:
                    foil_imgs.append(str(rng.choice(imgs_by_cat[fc])))

                trials.append({
                    "trial_num": trial_num,
                    "target_category": target_cat,
                    "target_img_filename": str(target_img),
                    "foil_categories": foil_cats,
                    "foil_img_filenames": foil_imgs,
                })
                trial_num += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"data": trials}))
    print(f"Wrote {len(trials)} trials to {args.out}")

if __name__ == "__main__":
    main()
