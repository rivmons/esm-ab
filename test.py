import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import itertools
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import os
from torch.nn.parallel import DistributedDataParallel as DDP


def test():
    data, y, y_hat = torch.randn((64, 10)), torch.randn((64,)), torch.randn((64,))
    pairs = torch.combinations(y_hat, 2, False) # predicted
    pairs_t = torch.combinations(y, 2, False)
    targets = (pairs_t[:, 0] > pairs_t[:, 1]).int()
    print(torch.where(pairs_t[:, 0] > pairs_t[:, 1], torch.tensor(1), torch.tensor(-1)))
    print(targets)

    l = nn.MarginRankingLoss(1)
    loss = l(pairs[:, 0], pairs[:, 1], targets)
    return loss

# x = []
# for i in range(1):
#     x.append(test())
# x = np.array(x)
# print(np.mean(x), np.std(x))

# def example(rank, world_size):
#     # create default process group
#     dist.init_process_group("gloo", rank=rank, world_size=world_size)
#     # create local model
#     model = nn.Linear(10, 10).to(rank)
#     # construct DDP model
#     ddp_model = DDP(model, device_ids=[rank])
#     # define loss function and optimizer
#     loss_fn = nn.MSELoss()
#     optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

#     # forward pass
#     outputs = ddp_model(torch.randn(20, 10).to(rank))
#     labels = torch.randn(20, 10).to(rank)
#     # backward pass
#     loss_fn(outputs, labels).backward()
#     # update parameters
#     optimizer.step()

# def main():
#     world_size = 2
#     mp.spawn(example,
#         args=(world_size,),
#         nprocs=world_size,
#         join=True)

# if __name__=="__main__":
#     # Environment variables which need to be
#     # set when using c10d's default "env"
#     # initialization mode.
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "29500"
#     main()

import torch
import time
from esm.pretrained import esm_if1_gvp4_t16_142M_UR50

# Set GPU index explicitly
GPU_INDEX = 5
device = torch.device(f"cuda:{GPU_INDEX}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

# Load model and move to the correct GPU
model, alphabet = esm_if1_gvp4_t16_142M_UR50()
model.eval()
model.to(device)

batch_converter = alphabet.get_batch_converter()

# Example dummy batch (batch of 32 sequences of length ~150)
sequences = [("seq{}".format(i), "M" * 150) for i in range(32)]
_, _, tokens = batch_converter(sequences)
tokens = tokens.to(device)

# Dummy 3D coordinates and mask
B, L = 32, 300
coords = torch.rand(B, L, 3, 3).to(device)
mask = torch.ones(B, L, dtype=torch.bool).to(device)
confidence = torch.ones_like(~mask, dtype=coords.dtype).to(device)

# Warm-up
with torch.inference_mode():
    for _ in range(3):
        model(coords, padding_mask=~mask, prev_output_tokens=tokens, confidence=confidence, features_only=True)

# Timing
start = time.time()
with torch.inference_mode():
    for _ in range(10):
        torch.cuda.synchronize()
        model(coords, padding_mask=~mask, prev_output_tokens=tokens, confidence=confidence, features_only=True)
        torch.cuda.synchronize()
end = time.time()

print(f"Average inference time per batch: {(end - start) / 10:.4f} s")
