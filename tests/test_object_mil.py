import torch

from multimodal.object_mil import (
    TrackConfig,
    build_object_tracks_greedy,
    masked_pool_k,
    mil_logsumexp_logits,
    pack_candidates_with_null,
    per_sample_candidate_weights,
)


def test_masked_pool_k_shapes():
    fmap = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).view(2, 3, 4, 4)
    masks = torch.ones(2, 5, 1, 8, 8)
    pooled = masked_pool_k(fmap, masks)
    assert pooled.shape == (2, 5, 3)
    assert torch.isfinite(pooled).all()


def test_tracking_and_packing_with_null():
    z = torch.randn(3, 4, 8)
    z = torch.nn.functional.normalize(z, dim=-1)
    valid = torch.tensor(
        [[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1]],
        dtype=torch.bool,
    )
    tracks = build_object_tracks_greedy(z, valid, TrackConfig(sim_thresh=-1.0, max_tracks=5))
    assert tracks.ndim == 2
    assert tracks.shape[1] == 8
    assert tracks.shape[0] <= 5

    null = torch.zeros(8)
    cand, mask = pack_candidates_with_null([tracks, tracks[:1]], null)
    assert cand.shape[0] == 2
    assert cand.shape[2] == 8
    assert mask.shape[:2] == cand.shape[:2]
    assert mask[:, -1].any()


def test_mil_logits_and_candidate_weights():
    text = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    cand = torch.nn.functional.normalize(torch.randn(4, 3, 8), dim=-1)
    mask = torch.ones(4, 3, dtype=torch.bool)
    logits = mil_logsumexp_logits(text, cand, mask, tau=0.05)
    assert logits.shape == (4, 4)
    assert torch.isfinite(logits).all()

    weights = per_sample_candidate_weights(text, cand, mask, tau=0.05)
    assert weights.shape == (4, 3)
    assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5)
