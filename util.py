# ------------------------------- util.py ---------------------------------
import torch
import subprocess

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

def gpu_lowestvram():
    try:
        smi_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
        )
        memory_usages = [int(x) for x in smi_output.decode("utf-8").strip().split('\n')]
        min_mem = min(memory_usages)
        best_gpu = memory_usages.index(min_mem)
        return best_gpu
    except Exception as e:
        print(f"Error checking GPU usage: {e}")
        return -1
# -------------------------------------------------------------------------
