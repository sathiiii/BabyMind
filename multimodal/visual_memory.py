from __future__ import annotations

"""Prototype-based visual memory.

The memory is an EMA-updated codebook of prototype vectors. It is used as a
"visual vocabulary" that supports:

  * Soft assignment of embeddings to prototypes (with or without gradient).
  * Recall / reconstruction by mixing prototypes.
  * A SwAV-style clustering regularizer (optional) with DDP-safe updates.
  * A one-shot warm-start initializer that seeds the prototype vectors from
    early confident embeddings.

The design goals are:
  * DDP-safety: collective calls are shape-stable and invoked consistently.
  * Minimal dependencies: only PyTorch.
  * Stability: prototypes are stored as buffers and updated with EMA.

Note: Prototypes are not trainable parameters; gradients flow to the inputs.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch.distributed as dist
except Exception:
    dist = None


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize a tensor along dimension ``dim``."""
    return x / (x.norm(dim=dim, keepdim=True) + eps)


@torch.no_grad()
def sinkhorn_knopp(
    logits: torch.Tensor,
    n_iters: int = 3,
    epsilon: float = 0.05,
) -> torch.Tensor:
    """SwAV-style balanced assignment.

    Args:
        logits: (N, K) similarity scores.
        n_iters: number of normalization iterations.
        epsilon: temperature-like smoothing.

    Returns:
        Q: (N, K) matrix with rows summing to 1.
    """
    if logits.ndim != 2:
        raise ValueError(f"sinkhorn_knopp expects (N,K), got {tuple(logits.shape)}")

    eps = max(float(epsilon), 1e-6)

    # Numerical stability: subtract per-row max before exp.
    logits = logits - logits.max(dim=1, keepdim=True).values

    Q = torch.exp(logits / eps).t().contiguous()  # (K,N)
    Q /= Q.sum().clamp(min=1e-12)

    K, N = Q.shape
    r = torch.ones(K, device=Q.device, dtype=Q.dtype) / float(K)
    c = torch.ones(N, device=Q.device, dtype=Q.dtype) / float(N)

    for _ in range(int(n_iters)):
        u = Q.sum(dim=1).clamp(min=1e-12)
        Q *= (r / u).unsqueeze(1)

        v = Q.sum(dim=0).clamp(min=1e-12)
        Q *= (c / v).unsqueeze(0)

    Q = Q / Q.sum(dim=0, keepdim=True).clamp(min=1e-12)
    return Q.t().contiguous()  # (N,K)


@dataclass
class ProtoConfig:
    embedding_dim: int
    num_prototypes: int = 256
    tau: float = 0.5

    # Balanced assignments.
    use_sinkhorn: bool = True
    sinkhorn_iters: int = 3
    sinkhorn_epsilon: float = 0.05
    sinkhorn_min_samples: int = 2

    # EMA updates.
    ema_decay: float = 0.99
    ema_eps: float = 1e-3
    ema_ddp_sync: bool = True

    # Warm-start.
    warm_start_max_samples: int = 4096


class PrototypeMemory(nn.Module):
    """EMA prototype memory.

    Prototypes are stored as buffers and updated with an exponential moving
    average driven by pseudo-label assignments.

    The memory exposes both no-grad (teacher-like) and with-grad assignment and
    recall helpers.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_prototypes: int = 256,
        tau: float = 0.5,
        use_sinkhorn: bool = True,
        sinkhorn_iters: int = 3,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_min_samples: int = 2,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-3,
        ema_ddp_sync: bool = True,
        warm_start_max_samples: int = 4096,
    ) -> None:
        super().__init__()

        self.cfg = ProtoConfig(
            embedding_dim=int(embedding_dim),
            num_prototypes=int(num_prototypes),
            tau=float(tau),
            use_sinkhorn=bool(use_sinkhorn),
            sinkhorn_iters=int(sinkhorn_iters),
            sinkhorn_epsilon=float(sinkhorn_epsilon),
            sinkhorn_min_samples=int(sinkhorn_min_samples),
            ema_decay=float(ema_decay),
            ema_eps=float(ema_eps),
            ema_ddp_sync=bool(ema_ddp_sync),
            warm_start_max_samples=int(warm_start_max_samples),
        )

        K = int(self.cfg.num_prototypes)
        D = int(self.cfg.embedding_dim)

        protos = l2norm(torch.randn(K, D, dtype=torch.float32))
        self.register_buffer("prototypes", protos)

        self.register_buffer("ema_cluster_size", torch.ones(K, dtype=torch.float32))
        self.register_buffer("ema_prototype_sum", self.prototypes.clone())

        # Warm-start state (buffer-level, not saved to checkpoints).
        self.register_buffer("_warm_started", torch.tensor(False), persistent=False)

    # ---------------------------------------------------------------------
    # DDP helpers
    # ---------------------------------------------------------------------
    def _ddp_is_active(self) -> bool:
        if dist is None:
            return False
        try:
            return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        except Exception:
            return False

    # ---------------------------------------------------------------------
    # Assignments (teacher / student)
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def soft_assign(self, z: torch.Tensor, tau: Optional[float] = None) -> torch.Tensor:
        """Teacher-style soft assignment (no grad).

        Args:
            z: (N, D) embeddings (need not be normalized).
            tau: optional override for temperature.

        Returns:
            q: (N, K) assignment probabilities.
        """
        if z.ndim != 2:
            raise ValueError(f"soft_assign expects (N,D), got {tuple(z.shape)}")

        z_n = l2norm(z.float())
        p_n = l2norm(self.prototypes.float()).to(device=z_n.device, dtype=z_n.dtype)

        tau_eff = float(self.cfg.tau if tau is None else tau)
        tau_eff = max(tau_eff, 1e-6)

        logits = (z_n @ p_n.t()) / tau_eff
        return torch.softmax(logits, dim=1)

    def assign_with_grad(self, z: torch.Tensor, tau: Optional[float] = None) -> torch.Tensor:
        """Student-style soft assignment with gradients w.r.t. ``z``.

        Prototypes remain buffers, so gradients do not flow into the codebook.
        """
        if z.ndim != 2:
            raise ValueError(f"assign_with_grad expects (N,D), got {tuple(z.shape)}")

        z_n = l2norm(z.float())
        p_n = l2norm(self.prototypes.float()).to(device=z_n.device, dtype=z_n.dtype)

        tau_eff = float(self.cfg.tau if tau is None else tau)
        tau_eff = max(tau_eff, 1e-6)

        logits = (z_n @ p_n.t()) / tau_eff
        return torch.softmax(logits, dim=1)

    def recall_with_grad(
        self,
        z: torch.Tensor,
        tau: Optional[float] = None,
        *,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recall in prototype space.

        Args:
            z: (N, D) query embeddings.
            tau: optional temperature override.
            normalize: if True, L2-normalize recalled vectors.

        Returns:
            q: (N, K) assignment probabilities (with grad to z).
            z_tilde: (N, D) reconstructed embeddings via prototype mixture.
        """
        q = self.assign_with_grad(z, tau=tau)  # (N,K)
        p = l2norm(self.prototypes.detach().to(device=q.device, dtype=q.dtype))  # (K,D)
        z_tilde = q @ p
        if normalize:
            z_tilde = l2norm(z_tilde)
        return q, z_tilde

    # ---------------------------------------------------------------------
    # EMA update
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def _ema_update(self, cluster_batch: torch.Tensor, sum_batch: torch.Tensor) -> None:
        decay = float(self.cfg.ema_decay)
        eps = float(self.cfg.ema_eps)

        cluster_batch = cluster_batch.to(device=self.ema_cluster_size.device, dtype=torch.float32)
        sum_batch = sum_batch.to(device=self.ema_prototype_sum.device, dtype=torch.float32)

        if bool(self.cfg.ema_ddp_sync) and self._ddp_is_active():
            assert dist is not None
            dist.all_reduce(cluster_batch, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_batch, op=dist.ReduceOp.SUM)

        if float(cluster_batch.sum().item()) <= 0.0:
            return

        self.ema_cluster_size.mul_(decay).add_(cluster_batch, alpha=(1.0 - decay))
        self.ema_prototype_sum.mul_(decay).add_(sum_batch, alpha=(1.0 - decay))

        n = self.ema_cluster_size.clamp(min=0.0) + eps
        p = self.ema_prototype_sum / n.unsqueeze(1)
        self.prototypes.copy_(l2norm(p))

    # ---------------------------------------------------------------------
    # Clustering regularizer (SwAV-like)
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def usage_eff_k(self) -> torch.Tensor:
        """Effective number of used prototypes.

        Computed from the EMA cluster-size distribution:
          eff_k = exp( H(p) ),  p_k ∝ ema_cluster_size[k]
        """

        p = self.ema_cluster_size.float().clamp(min=0.0)
        p = p / p.sum().clamp(min=1e-12)
        ent = -(p * p.clamp(min=1e-12).log()).sum()
        return torch.exp(ent)

    def proto_loss_ddp_gather(
        self,
        z_local: torch.Tensor,
        *,
        update_ema: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """DDP-safe clustering loss.

        The method gathers embeddings (no grad) across ranks to compute global
        balanced assignments, then computes the local loss with gradients.

        Args:
            z_local: (N_local, D) embeddings.
            update_ema: if True, update the prototype EMA using the gathered
                assignments. Set False for val/test monitoring.

        Returns:
            loss_local: scalar tensor for this rank.
            stats: dict with keys:
              - q_local: (N_local, K) assignment targets (no grad)
              - usage_eff_k: scalar effective-K statistic
        """
        if z_local.ndim != 2:
            raise ValueError(f"proto_loss_ddp_gather expects (N,D), got {tuple(z_local.shape)}")

        # If not in DDP, fall back.
        if not self._ddp_is_active():
            return self.proto_loss(z_local, update_ema=bool(update_ema))

        assert dist is not None
        rank = dist.get_rank()
        world = dist.get_world_size()

        device = z_local.device
        z_local = l2norm(z_local.float())

        Kp = int(self.cfg.num_prototypes)
        D = int(self.cfg.embedding_dim)
        tau = max(float(self.cfg.tau), 1e-6)

        p = l2norm(self.prototypes.float()).to(device=device, dtype=z_local.dtype)

        # 1) Gather sizes.
        n_local = torch.tensor([int(z_local.size(0))], device=device, dtype=torch.long)
        n_list = [torch.zeros_like(n_local) for _ in range(world)]
        dist.all_gather(n_list, n_local)
        ns = [int(t.item()) for t in n_list]
        max_n = max(ns)

        if max_n == 0:
            loss0 = z_local.new_tensor(0.0)
            q0 = torch.zeros((0, Kp), device=device, dtype=z_local.dtype)
            stats0: Dict[str, torch.Tensor] = {
                "q_local": q0,
                "usage_eff_k": self.usage_eff_k().to(device=device),
            }
            return loss0, stats0

        # 2) Gather z (detached) with padding.
        z_pad = torch.zeros((max_n, D), device=device, dtype=z_local.dtype)
        if ns[rank] > 0:
            z_pad[: ns[rank]] = z_local.detach()

        z_gather = [torch.zeros_like(z_pad) for _ in range(world)]
        dist.all_gather(z_gather, z_pad)

        z_all = torch.cat([zg[:n] for zg, n in zip(z_gather, ns)], dim=0)  # (N_all,D)
        N_all = int(z_all.size(0))

        # 3) Global targets (no grad).
        with torch.no_grad():
            sim_all = z_all @ p.t()  # (N_all,K)

            if (
                bool(self.cfg.use_sinkhorn)
                and N_all >= max(int(self.cfg.sinkhorn_min_samples), Kp)
                and Kp > 1
            ):
                q_all = sinkhorn_knopp(
                    sim_all.detach(),
                    n_iters=int(self.cfg.sinkhorn_iters),
                    epsilon=float(self.cfg.sinkhorn_epsilon),
                ).to(dtype=z_local.dtype)
            else:
                hard = sim_all.detach().argmax(dim=1)
                q_all = F.one_hot(hard, num_classes=Kp).float().to(dtype=z_local.dtype)

        # Slice local assignments.
        start = sum(ns[:rank])
        end = start + ns[rank]
        q_local = q_all[start:end]  # (N_local,K)

        # 4) Local loss with gradients.
        if int(z_local.size(0)) == 0:
            loss_local = z_local.new_tensor(0.0)
        else:
            sim_local = z_local @ p.t()
            logp_local = F.log_softmax(sim_local / tau, dim=1)
            loss_local = -(q_local.to(logp_local.dtype) * logp_local).sum(dim=1).mean()

        # 5) EMA update (optional).
        if bool(update_ema):
            with torch.no_grad():
                cluster_batch = q_all.sum(dim=0).float()  # (K,)
                sum_batch = (q_all.t().float() @ z_all.float())  # (K,D)

                # Avoid double-counting when _ema_update performs all_reduce.
                if bool(self.cfg.ema_ddp_sync) and self._ddp_is_active():
                    cluster_batch = cluster_batch / float(world)
                    sum_batch = sum_batch / float(world)

                self._ema_update(cluster_batch, sum_batch)

        stats: Dict[str, torch.Tensor] = {
            "q_local": q_local,
            "usage_eff_k": self.usage_eff_k().to(device=device),
        }
        return loss_local, stats

    def proto_loss(self, z: torch.Tensor, *, update_ema: bool = True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Single-process clustering loss + optional EMA update."""
        if z.ndim != 2:
            raise ValueError(f"proto_loss expects (N,D), got {tuple(z.shape)}")

        device = z.device
        z = l2norm(z.float())
        p = l2norm(self.prototypes.float()).to(device=device, dtype=z.dtype)

        sim = z @ p.t()  # (N,K)
        logp = F.log_softmax(sim / max(float(self.cfg.tau), 1e-6), dim=1)

        N = int(z.size(0))
        Kp = int(self.cfg.num_prototypes)
        D = int(self.cfg.embedding_dim)

        if (
            bool(self.cfg.use_sinkhorn)
            and N >= max(int(self.cfg.sinkhorn_min_samples), Kp)
            and Kp > 1
        ):
            q_tgt = sinkhorn_knopp(
                sim.detach(),
                n_iters=int(self.cfg.sinkhorn_iters),
                epsilon=float(self.cfg.sinkhorn_epsilon),
            )
        else:
            if N == 0:
                q_tgt = torch.zeros((0, Kp), device=device, dtype=z.dtype)
            else:
                hard = sim.detach().argmax(dim=1)
                q_tgt = F.one_hot(hard, num_classes=Kp).float()

        if N == 0:
            loss = z.new_tensor(0.0)
            cluster_batch = torch.zeros(Kp, device=device, dtype=torch.float32)
            sum_batch = torch.zeros(Kp, D, device=device, dtype=torch.float32)
        else:
            loss = -(q_tgt.to(logp.dtype) * logp).sum(dim=1).mean()
            cluster_batch = q_tgt.detach().sum(dim=0).float()
            sum_batch = (q_tgt.detach().t().float() @ z.detach().float())

        if bool(update_ema):
            self._ema_update(cluster_batch, sum_batch)

        stats: Dict[str, torch.Tensor] = {
            "q_local": q_tgt,
            "usage_eff_k": self.usage_eff_k().to(device=device),
        }
        return loss, stats

    # ---------------------------------------------------------------------
    # One-shot warm start
    # ---------------------------------------------------------------------
    @property
    def warm_started(self) -> bool:
        return bool(self._warm_started.item())

    @torch.no_grad()
    def warm_start_ddp(
        self,
        z_local: torch.Tensor,
        *,
        seed: int = 0,
        n_iters: int = 10,
        max_samples: Optional[int] = None,
        verbose: bool = False,
        # Backwards-compatible aliases (used by older callers)
        kmeans_iters: Optional[int] = None,
        max_total: Optional[int] = None,
    ) -> bool:
        """One-shot warm-start of prototypes from embeddings.

        This routine gathers (detached) embeddings across ranks and runs a small
        kmeans-style initializer on rank 0, then broadcasts the resulting
        prototypes.

        Args:
            z_local: (N_local,D) embeddings. Can be empty.
            seed: RNG seed for deterministic init.
            n_iters: number of k-means refinement iterations after init.
            max_samples: cap the total gathered samples (subsample if needed).
            verbose: print a short message on rank 0.

        Returns:
            True if warm-start succeeded and prototypes were updated.
        """
        if self.warm_started:
            return False

        # Backwards-compatible kwarg aliases.
        if kmeans_iters is not None:
            n_iters = int(kmeans_iters)
        if max_total is not None:
            max_samples = int(max_total)

        if z_local.ndim != 2:
            raise ValueError(f"warm_start_ddp expects (N,D), got {tuple(z_local.shape)}")

        if max_samples is None:
            max_samples = int(self.cfg.warm_start_max_samples)
        max_samples = int(max_samples)

        # Normalize and detach.
        z_local = l2norm(z_local.float()).detach()

        # Gather across ranks.
        if self._ddp_is_active():
            assert dist is not None
            rank = dist.get_rank()
            world = dist.get_world_size()

            device = z_local.device
            D = int(z_local.size(1)) if z_local.numel() > 0 else int(self.cfg.embedding_dim)

            n_local = torch.tensor([int(z_local.size(0))], device=device, dtype=torch.long)
            n_list = [torch.zeros_like(n_local) for _ in range(world)]
            dist.all_gather(n_list, n_local)
            ns = [int(t.item()) for t in n_list]
            max_n = max(ns)

            # Pad then all_gather.
            z_pad = torch.zeros((max_n, D), device=device, dtype=z_local.dtype)
            if ns[rank] > 0:
                z_pad[: ns[rank]] = z_local

            z_gather = [torch.zeros_like(z_pad) for _ in range(world)]
            dist.all_gather(z_gather, z_pad)

            z_all = torch.cat([zg[:n] for zg, n in zip(z_gather, ns)], dim=0)
        else:
            z_all = z_local
            rank = 0

        # Possibly subsample.
        N_all = int(z_all.size(0))
        K = int(self.cfg.num_prototypes)

        if N_all < K:
            # Not enough samples to initialize.
            if self._ddp_is_active():
                assert dist is not None
                dist.barrier()
            return False

        if max_samples > 0 and N_all > max_samples:
            g = torch.Generator(device=z_all.device)
            g.manual_seed(int(seed))
            idx = torch.randperm(N_all, generator=g, device=z_all.device)[:max_samples]
            z_use = z_all[idx]
        else:
            z_use = z_all

        # Compute prototypes on rank 0.
        if rank == 0:
            protos = _kmeans_cosine(z_use, K=K, n_iters=int(n_iters), seed=int(seed))
            protos = l2norm(protos)
            if verbose:
                print(f"[PrototypeMemory] warm_start_ddp: initialized K={K} from N={int(z_use.size(0))} samples")
        else:
            protos = torch.empty((K, int(self.cfg.embedding_dim)), device=z_all.device, dtype=torch.float32)

        # Broadcast prototypes.
        if self._ddp_is_active():
            assert dist is not None
            dist.broadcast(protos, src=0)

        # Commit to buffers.
        self.prototypes.copy_(protos.to(device=self.prototypes.device, dtype=self.prototypes.dtype))
        self.ema_cluster_size.fill_(1.0)
        self.ema_prototype_sum.copy_(self.prototypes)

        self._warm_started.fill_(True)

        if self._ddp_is_active():
            assert dist is not None
            dist.barrier()

        return True


# -----------------------------------------------------------------------------
# Small k-means helper (cosine / spherical)
# -----------------------------------------------------------------------------

@torch.no_grad()
def _kmeans_cosine(z: torch.Tensor, K: int, n_iters: int = 10, seed: int = 0) -> torch.Tensor:
    """Spherical k-means with a farthest-point init.

    Args:
        z: (N,D) L2-normalized embeddings.
        K: number of clusters.
        n_iters: refinement iterations.
        seed: random seed for choosing the first center.

    Returns:
        centers: (K,D) L2-normalized.
    """
    if z.ndim != 2:
        raise ValueError(f"_kmeans_cosine expects (N,D), got {tuple(z.shape)}")

    z = l2norm(z.float())
    N, D = z.shape

    K = int(K)
    if K <= 0:
        raise ValueError("K must be > 0")
    if N < K:
        raise ValueError(f"Need N>=K for kmeans init, got N={N}, K={K}")

    # Farthest-point / kmeans++-like init.
    g = torch.Generator(device=z.device)
    g.manual_seed(int(seed))

    first = int(torch.randint(low=0, high=N, size=(1,), generator=g, device=z.device).item())
    centers = [z[first]]

    # Track best similarity to any chosen center for each point.
    best_sim = (z @ centers[0].view(D, 1)).squeeze(1)  # (N,)

    for _ in range(1, K):
        # Choose the point with minimum similarity (farthest in cosine distance).
        idx = int(torch.argmin(best_sim).item())
        centers.append(z[idx])
        sim_new = (z @ centers[-1].view(D, 1)).squeeze(1)
        best_sim = torch.maximum(best_sim, sim_new)

    C = torch.stack(centers, dim=0)  # (K,D)
    C = l2norm(C)

    # Lloyd iterations.
    for _ in range(int(n_iters)):
        sim = z @ C.t()  # (N,K)
        assign = sim.argmax(dim=1)  # (N,)

        C_new = torch.zeros_like(C)
        for k in range(K):
            mask = assign == k
            if bool(mask.any().item()):
                C_new[k] = z[mask].mean(dim=0)
            else:
                # Re-seed empty cluster with a far point.
                idx = int(torch.argmin(best_sim).item())
                C_new[k] = z[idx]

        C = l2norm(C_new)
        best_sim = (z @ C.t()).amax(dim=1)

    return C
