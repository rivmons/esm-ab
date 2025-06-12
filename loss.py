import torch
from random import shuffle

def get_pair_indices(n):
    return torch.combinations(torch.tensor(list(range(n))), 2, with_replacement=False)

def pairwiseRankingLoss_random(preds, labels, p_idx=None):
    if p_idx is None:
        p_idx = get_pair_indices(preds.size(0))
    p_i, p_j = p_idx[:, 0], p_idx[:, 1]
    targets = torch.where(labels[p_i] >= labels[p_j], torch.tensor(1), torch.tensor(-1))
    return preds[p_i], preds[p_j], targets

def pairwiseRankingLoss(antigens, preds, labels):
    ag2idx = {}
    for idx, ag in enumerate(antigens):
        ag2idx.setdefault(ag, []).append(idx)
    p_i, p_j, targets = [], [], []

    for idxs in ag2idx.values():
        if len(idxs) < 2: continue

        pair_ind = torch.combinations(torch.tensor(idxs), 2, with_replacement=False)
        idxs_t, partner_t = pair_ind[:, 0], pair_ind[:, 1]
        score_i, score_j = labels[idxs_t], labels[partner_t]
        targets.append(torch.where(score_i >= score_j, torch.tensor(1), torch.tensor(-1)))

        p_i.append(idxs_t)
        p_j.append(partner_t)

    return preds[torch.cat(p_i)], preds[torch.cat(p_j)], torch.cat(targets)

def dpoLoss(antigens, preds, labels, beta):
    ag2idx = {}
    for idx, ag in enumerate(antigens):
        ag2idx.setdefault(ag, []).append(idx)
    chosen_s, rejected_s = [], []

    for idxs in ag2idx.values():
        if len(idxs) < 2: continue

        idxs = idxs.copy()
        shuffle(idxs)
        partner = idxs[1:] + idxs[:1]

        idxs_t = torch.tensor(idx, device=preds.device)
        partner_t = torch.tensor(partner, device=preds.device)
        score_i, score_j = labels[idxs_t], labels[partner_t]
        targets = score_i >= score_j

        chosen_s.append(torch.where(targets, preds[idxs_t], preds[partner_t]))
        rejected_s.append(torch.where(targets, preds[partner_t], preds[idxs_t]))

    chosen_scores, rejected_scores = torch.cat(chosen_s), torch.cat(rejected_s)
    return chosen_scores, rejected_scores, beta
