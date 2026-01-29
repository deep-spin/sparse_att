import os
import sys
from types import SimpleNamespace

# ensure package import works (add Library to path)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from torch.nn import functional as F

from memory_mosaics.models.memory_mosaics_eft import StackAssoMem
from memory_mosaics.data.dataset_synthetic import SPECIAL

torch.manual_seed(0)

def make_model(task, skip_tokens=0):
    config = SimpleNamespace(skip_tokens=skip_tokens)
    # small model for MWE
    model = StackAssoMem(method='softmax', vocab_size=60, n_head=2, n_embd=8, n_layer=1, v_shift=1, block_size=16, task=task, config=config)
    model.eval()
    return model


def compute_manual_loss(logits, targets, idx, task, config):
    b, t = idx.size()
    device = idx.device
    sep_id = SPECIAL[task]["<sep>"]
    pad_id = SPECIAL[task]["<pad>"]

    sep_positions = (idx == sep_id)
    sep_exists = sep_positions.any(dim=1)
    sep_idx = (t - 1) - sep_positions.int().flip(dims=[1]).argmax(dim=1)
    sep_idx = torch.where(sep_exists, sep_idx, torch.full_like(sep_idx, -1))

    positions = torch.arange(t, device=device).unsqueeze(0).expand(b, t)
    loss_mask = (positions > sep_idx.unsqueeze(1)) & (idx != pad_id) & sep_exists.unsqueeze(1)

    skip = getattr(config, 'skip_tokens', 0)
    aligned_mask = loss_mask[:, skip:]

    per_token = F.cross_entropy(logits.view(-1, logits.size(-1)), targets[:, skip:].contiguous().view(-1), ignore_index=-1, reduction='none')
    denom = aligned_mask.contiguous().view(-1).sum().item()
    if denom == 0:
        return None, aligned_mask
    manual_loss = (per_token * aligned_mask.contiguous().view(-1)).sum() / denom
    return manual_loss, aligned_mask


def run_case(task, skip_tokens=0):
    model = make_model(task, skip_tokens=skip_tokens)
    cfg = model.config

    # Build batch of 4 sequences length 6
    b = 4
    t = 6
    sep = SPECIAL[task]["<sep>"]
    pad = SPECIAL[task]["<pad>"]

    idx = torch.tensor([
        [1, 2, sep, 10, 11, 12],      # SEP at 2 -> mask positions 3,4,5
        [sep, 20, 21, 22, pad, pad],  # SEP at 0 -> mask 1,2,3 (pads excluded)
        [1, sep, sep, 30, 31, 32],    # multiple SEP -> last at 2 -> mask 3,4,5
        [1, 2, 3, 4, 5, 6],          # no SEP -> excluded
    ], dtype=torch.long)

    # Create simple targets within vocab
    targets = torch.randint(0, 59, (b, t), dtype=torch.long)

    # get logits and loss from model
    logits, loss = model(idx, targets=targets)

    manual_loss, aligned_mask = compute_manual_loss(logits, targets, idx, task, cfg)

    print('Task:', task, 'skip_tokens=', skip_tokens)
    print('idx:\n', idx)
    print('aligned_mask:\n', aligned_mask)
    print('model loss:', loss)
    print('manual loss:', manual_loss)

    if manual_loss is not None:
        assert torch.isclose(loss, manual_loss, atol=1e-5), f"Loss mismatch: model {loss} vs manual {manual_loss}"
        print('OK: losses match')
    else:
        print('No valid positions to compute manual loss (denominator==0)')


if __name__ == '__main__':
    # pick a task from SPECIAL
    task = list(SPECIAL.keys())[0]
    print('Using task:', task)
    # run with skip_tokens = 0
    run_case(task, skip_tokens=0)
    # run with skip_tokens = 1 to test alignment
    run_case(task, skip_tokens=1)
