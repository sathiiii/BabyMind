import json

import torch

from multimodal.sam_concept_registry import build_sam_concept_registry


def test_sam_registry_maps_and_weights(tmp_path):
    root = tmp_path / "sam_prepacked"
    root.mkdir()
    (root / "concept_vocab.json").write_text(json.dumps({"concepts": ["ball", "cat", "rare"]}))
    (root / "sam_prepacked_index.json").write_text(json.dumps({"frame_0001.jpg": "frame_0001.pt"}))
    (root / "concept_frequency.json").write_text(
        json.dumps({"counts": {"ball": 100, "cat": 10, "rare": 1}})
    )

    reg = build_sam_concept_registry(
        sam_prepacked_dir=root,
        concept_frequency_json=None,
        min_masks_per_concept=5,
        concept_list_file=None,
        alpha=0.5,
        clip_min=0.5,
        clip_max=2.0,
        verbose=False,
    )

    assert reg.local_to_global.tolist()[:2] == [0, 1]
    assert reg.local_to_global.tolist()[2] == -1
    assert torch.isfinite(reg.weights).all()
    assert "rare" in reg.dropped_local
