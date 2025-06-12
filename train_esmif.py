# ----------------------------- train_esminv.py ---------------------------

from __future__ import annotations

import argparse
import math
from pathlib import Path
import os
import time
import sys
from inspect import signature

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from peft import get_peft_model, get_peft_config, LoraConfig, TaskType
from transformers.optimization import get_scheduler

from esm import pretrained

from data_utils import get_dataloaders
from util import freeze_module, spearman_corrcoef
from esm.pretrained import esm_if1_gvp4_t16_142M_UR50

# ----------------------------- Dist --------------------------------------

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

# ----------------------------- Model -------------------------------------
class ESMIFRegressor(nn.Module):
    """
    - Freeze the ESM-IF
    - MLP regression head
    - Input: coords (B, L, 3, 3), mask (B, L), seqs (list[str])
    """

    def __init__(self, model_name: str = "esm_if1_gvp4_t16_142M"):
        super().__init__()
        # self.backbone, alphabet = pretrained.load_model_and_alphabet(model_name)
        self.backbone, self.alphabet = esm_if1_gvp4_t16_142M_UR50()

        # print(self.backbone)
        # freeze_module(self.backbone)

        hidden_dim = self.backbone.decoder.embed_dim
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, coords, mask, seqs):
        device = next(self.parameters()).device
        coords = coords.to(device)
        mask   = mask.to(device)

        seqs = [("", s) for s in seqs]  # Add an empty label to each sequence.
        batch_tokens = self.alphabet.get_batch_converter()(seqs)[2].to(device)
        
        confidence = torch.ones_like(~mask, dtype=coords.dtype)
        reps, _ = self.backbone(
            coords=coords,
            padding_mask=~mask,                  # pad==1
            prev_output_tokens=batch_tokens,
            confidence=confidence,
            features_only=True,
        )

        # pooled = reps[:, :, 0]                                   # [CLS], change to mean pooling
        # pred = self.reg_head(pooled).squeeze(-1)                 # (B,)

        # <<< 2. masked mean pooling Obtain sentence-level representations. >>>
        reps = reps[:, :, 1:] 
        lengths = mask.sum(-1, keepdim=True).clamp(min=1e-8)   # (B,1)
        pooled  = (reps * mask.unsqueeze(1).float()).sum(-1) / lengths  # (B,C)

        pred = self.reg_head(pooled).squeeze(-1)                # (B,)
        return pred

def get_pair_indices(n):
    return torch.combinations(torch.tensor(list(range(n))), 2, with_replacement=False)

# ------------------------- Train / Eval Loop -----------------------------
def train_epoch(model, loader, optimizer, criterion, device, val_loader=None, best_val_corr=-math.inf, save_path=None, scheduler=None):
    model.train()
    tot_loss = tot_corr = 0.0
    eval_every = len(loader) // 3
    step = 0
    pair_ind = get_pair_indices(loader.batch_size)
    batch_size = loader.batch_size
    scaler = GradScaler()

    for batch in loader:
        # with torch.profiler.profile(
        #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        #     record_shapes=True,
        #     with_stack=True,
        # ) as prof:
        with autocast("cuda"):
            stime = time.time()
            labels = batch["labels"].to(device)
            preds  = model(batch["coords"], batch["mask"], batch["seqs"])

            bs = preds.size(0)
            if bs == batch_size: p_idx = pair_ind
            else: p_idx = get_pair_indices(bs)

            p_i, p_j = p_idx[:, 0], p_idx[:, 1]
            targets = torch.where(labels[p_i] > labels[p_j], torch.tensor(1), torch.tensor(-1))
            loss   = criterion(preds[p_i], preds[p_j], targets)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        # print(torch.cuda.memory_summary())

        tot_loss += loss.item()
        step += 1
        
        print(f'train loss: {loss.item()}, time={(time.time() - stime):.3f}, lr={optimizer.param_groups[0]["lr"]:.8f}')
        # Perform validation every N batches and save the best model based on the highest average correlation coefficient on the validation set.
        if val_loader is not None and step % eval_every == 0:
            val_loss, val_corr, corr_dict = eval_epoch(model, val_loader, criterion, device)
            print(f"[Step {step}] Val Loss {val_loss:.4f} ρ {val_corr:.3f}")
            print(corr_dict)
            print('Current training loss:', loss.item())

            if val_corr > best_val_corr and save_path is not None:
                print(f"New best model found (ρ={val_corr:.3f}), saving to {save_path}")
                torch.save(model.state_dict(), save_path)
                best_val_corr = val_corr

    n = len(loader)
    return tot_loss / n, tot_corr / n, best_val_corr



@torch.inference_mode()
def eval_epoch(model, loader, criterion, device):
    
    model.eval()
    all_preds, all_labels, all_antigens = [], [], []

    total_loss = 0.0
    for batch in loader:
        labels = batch["labels"].to(device)
        preds  = model.forward(batch["coords"], batch["mask"], batch["seqs"])

        # Accumulate loss (weighted by the number of samples).
        pairs = torch.combinations(preds, 2, False)
        pairs_t = torch.combinations(labels, 2, False)
        targets = torch.where(pairs_t[:, 0] > pairs_t[:, 1], torch.tensor(1), torch.tensor(-1))
        loss   = criterion(pairs[:, 0], pairs[:, 1], targets)
        total_loss += loss.item() * labels.size(0)

        # Collect results for grouping later.
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_antigens.extend(batch["antigen"])

    preds   = torch.cat(all_preds)
    labels  = torch.cat(all_labels)

    # ---- Calculate Pearson correlation grouped by antigen. ----
    from collections import defaultdict
    group = defaultdict(list)
    for p, y, ag in zip(preds, labels, all_antigens):
        group[ag].append((p, y))

    # corr_list = []
    corr_dict = {}
    for ag, pairs in group.items():
        if len(pairs) < 2:        # When there is only one mutant, the correlation coefficient is meaningless and should be skipped.
            continue
        p_stack = torch.stack([p for p, _ in pairs])
        y_stack = torch.stack([y for _, y in pairs])
        corr = spearman_corrcoef(p_stack, y_stack)
        # corr_list.append(corr)
        corr_dict[ag] = corr     

    # mean_corr = sum(corr_dict) / len(corr_dict) if corr_dict else 0.0
    mean_corr = sum(corr_dict.values()) / len(corr_dict) if corr_dict else 0.0
    mean_loss = total_loss / len(preds)

    return mean_loss, mean_corr, corr_dict

# ------------------------------ Main -------------------------------------

class PEFTWrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model

    def forward(self, 
                input_ids, 
                attention_mask, 
                inputs_embeds, 
                output_attentions=None, 
                output_hidden_states=None, 
                return_dict=None, 
                task_ids=None, 
                **kwargs):
        return self.original_model(input_ids, attention_mask, inputs_embeds)
    
def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(3)
    print(device)

    train_loader, val_loader, test_loader = get_dataloaders(
        args.meta_json, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=777
    )

    model = ESMIFRegressor(args.model).to(device)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=['q_proj', 'v_proj', 'k_proj', 'out_proj', 'fc1', 'fc2', 'output_projection'],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )
    model = PEFTWrapper(model)
    model = get_peft_model(model, lora_config)
    for name, param in model.named_parameters():
        if "reg_head" in name: param.requires_grad = True
    criterion = nn.MarginRankingLoss(margin=0.01)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    train_steps = args.epochs * len(train_loader)
    scheduler = get_scheduler(
        name="cosine",  
        optimizer=optimizer,
        num_warmup_steps=int(0.05 * train_steps),
        num_training_steps=train_steps
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "runs"))

    best_val_corr = -math.inf
    for epoch in range(1, args.epochs + 1):
        # tr_loss, tr_corr = train_epoch(model, train_loader, optimizer, criterion, device, val_loader)
        save_path = out_dir / "best_model.pt"
        tr_loss, tr_corr, best_val_corr = train_epoch(
            model, train_loader, optimizer, criterion, device,
            val_loader=val_loader,
            best_val_corr=best_val_corr,
            save_path=save_path,
            scheduler=scheduler
        )
        val_loss, val_corr, val_corr_dict = eval_epoch(model, val_loader, criterion, device)


        # writer.add_scalars("Loss", {"train": tr_loss, "val": val_loss}, epoch)
        # writer.add_scalars("Pearson", {"train": tr_corr, "val": va_corr}, epoch)

        print(
            f"[Epoch {epoch:03d}] "
            f"Train Loss {tr_loss:.4f} ρ {tr_corr:.3f} | "
            f"Val Loss {val_loss:.4f} ρ {val_corr:.3f}"
        )
        
        for ag, r in sorted(val_corr_dict.items()):
            print(f"    {ag:<12s}:  ρ = {r: .3f}")

        if val_corr > best_val_corr:
            best_val_corr = val_corr
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    # ------------------------- Final Test -------------------------
    model.load_state_dict(torch.load(out_dir / "best_model.pt"))
    te_loss, te_corr, te_corr_dict = eval_epoch(model, test_loader, criterion, device)
    print(f"\n[Test]  Loss {te_loss:.4f}  |  Pearson ρ {te_corr:.3f}\n {te_corr_dict}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_json", type=str, default="./data/metadata.json", help="metadata.json (contains pdb paths & affinity csvs)")
    parser.add_argument("--pdb_dir", type=str, default="./data/complex_structure", help="PDB folder")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--model",   type=str, default="esm_if1_gvp4_t16_142M")
    parser.add_argument("--epochs",  type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr",      type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    main(args)
# -------------------------------------------------------------------------
