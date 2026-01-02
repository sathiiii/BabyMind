# grad_norm_logger.py
import torch
import pytorch_lightning as pl

class GradNormLogger(pl.Callback):
    def __init__(self, log_per_layer: bool = False):
        self.log_per_layer = log_per_layer

    def on_before_zero_grad(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer, *args, **kwargs):
        grads = []
        per_layer = {}
        for name, p in pl_module.named_parameters():
            if p.grad is not None:
                g = p.grad.detach().float()
                grads.append(g)
                if self.log_per_layer:
                    per_layer[f"train/grad_norms/{name}"] = g.norm(2)

        if grads:
            total = torch.norm(torch.stack([g.norm(2) for g in grads]), 2)
            pl_module.log("train/grad_norm", total, on_step=True, on_epoch=False, sync_dist=True)

        # optional per-layer logging (can be a lot of series)
        if self.log_per_layer and per_layer:
            pl_module.log_dict(per_layer, on_step=True, on_epoch=False, sync_dist=True)

        # log LR
        try:
            lr = optimizer.param_groups[0]["lr"]
            pl_module.log("train/lr", lr, on_step=True, on_epoch=False, sync_dist=True)
        except Exception:
            pass
