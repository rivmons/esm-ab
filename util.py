# ------------------------------- util.py ---------------------------------
import torch


def freeze_module(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)


@torch.inference_mode()
def pearson_corrcoef(x: torch.Tensor, y: torch.Tensor) -> float:
    """x,y: (N,)"""
    vx = x - x.mean()
    vy = y - y.mean()
    corr = (vx * vy).sum() / (torch.sqrt((vx**2).sum()) * torch.sqrt((vy**2).sum()) + 1e-8)
    return corr.item()

@torch.inference_mode()
def spearman_corrcoef(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute Spearman correlation between two 1D tensors"""
    rx = x.argsort().argsort().float()
    ry = y.argsort().argsort().float()
    return pearson_corrcoef(rx, ry)
# -------------------------------------------------------------------------
