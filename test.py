import numpy as np
import torch
import torch.nn as nn
import itertools

def test():
    data, y, y_hat = torch.randn((64, 10)), torch.randn((64,)), torch.randn((64,))
    pairs = torch.combinations(y_hat, 2, False) # predicted
    pairs_t = torch.combinations(y, 2, False)
    targets = (pairs_t[:, 0] > pairs_t[:, 1]).int()

    l = nn.MarginRankingLoss(1)
    loss = l(pairs[:, 0], pairs[:, 1], targets)
    return loss

x = []
for i in range(10000):
    x.append(test())
x = np.array(x)
print(np.mean(x), np.std(x))