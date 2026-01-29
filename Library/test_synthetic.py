import torch
import os
import time
from contextlib import nullcontext
import copy
from collections import defaultdict
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
import torch.distributed as dist

from utils import create_logger, TimeEstimater
from utils import init_ddp
from utils import str2bool, get_step_lr, get_cosine_lr

import tiktoken
import argparse
import torch._dynamo
import glob
from torch.utils.data import ConcatDataset
torch._dynamo.config.suppress_errors = True
# always import numpy after import torch.
import numpy as np

from memory_mosaics.data.dataset_synthetic import IndexedPairDataset, InfiniteDataLoader, SPECIAL, make_indexedpair_collate
from memory_mosaics.evaluation.common_metrics import estimate_loss

#####################
# two version of memory mosaics. Pick one as you wish!
from memory_mosaics.models.memory_mosaics_eft import StackAssoMem
#####################
#from memory_mosaics.models.memory_mosaics import StackAssoMem

parser = argparse.ArgumentParser()

# general 
parser.add_argument('--seed', type=int, default=1337, help='seed ')
parser.add_argument('--out_dir',type=str,default='results/copy', help='output directory')
parser.add_argument('--backend',type=str, default='nccl', help='distributed training backend. [nccl, gloo, etc]. nccl is faster in general.')
parser.add_argument('--device', type=str, default='cuda', help="examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks")
parser.add_argument('--dtype', type=str, default='bfloat16', help="'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler")
parser.add_argument('--compile', type=str2bool, default=True, help=' use PyTorch 2.0 to compile the model to be faster')


# evaluation / logging process
parser.add_argument('--eval_interval', type=int, default=10000000, help='validation set evaluation interval')
parser.add_argument('--save_checkpoint_interval', type=int, default=1000, help='save checkpoint (on disk) interval')
parser.add_argument('--log_interval', type=int, default=100, help='log performance interval')
parser.add_argument('--eval_iters',type=int, default=1000, help='validation evaluation iterations (per gpu)')

# dataset 
parser.add_argument('--task', type=str, default="mqmtar", help='task: mqmtar')
parser.add_argument('--datapath', type=str, default="mqmtar", help='your data directory')

# training process 
parser.add_argument('--batch_size',type=int, default=8, help='batch size per gpu')


# model
parser.add_argument('--block_size',type=int, default=512, help='block size, aka in-context length')
parser.add_argument('--n_embd', type=int, default=768, help='embedding dim')
parser.add_argument('--v_shift',type=int, default=1, help='value right shift')
parser.add_argument('--att_shift',type=int, default=0, help='additional attn shift')


parser.add_argument('--pmem_size', type=int, default=2688, help='memory size')
parser.add_argument('--pmem_count', type=int, default=1, help='memory count')

parser.add_argument('--ic_dropout', type=float, default=0.05, help='in-context attention score dropout rate')
parser.add_argument('--hd_dropout', type=float, default=0.05, help='hidden representation vector dropout rate')

parser.add_argument('--bias', type=str2bool, default=False, help = "do we use bias inside LayerNorm and Linear layers?")
parser.add_argument('--weight_tying', type=str2bool, default=True, help = "True: last linear layer and first embedding share weights. False: do not share.")


parser.add_argument('--pre_ln', type=str2bool, default=True, help="pre-layernorm or post-layernorm. Try post-layernorm if hm_dropout > 0. ")

parser.add_argument('--k_kernel_size',type=int, default=1, help='key kernel size')
parser.add_argument('--v_kernel_size',type=int, default=2, help='value kernel size')
parser.add_argument('--k_fe_type', type=str, default='linearconv', help="key feature extractor type")
parser.add_argument('--v_fe_type', type=str, default='lowrlinearconv', help="value feature extractor type")

parser.add_argument('--skip_tokens', type=int, default=0, help="select last tokens in loss function. ")
parser.add_argument('--ckpt_path', type=str, default="ckpt.pt")

args = parser.parse_args()

import os

def load_args_from_ckpt_path(ckpt_path):
    """
    Example ckpt_path:
    /mnt/.../results/synthetic_test/softmax/2_4/ckpt.pt

    Returns:
        method = "softmax"
        n_layer = 2
        n_head = 4
    """
    ckpt_dir = os.path.dirname(ckpt_path)

    # The folder structure ends like .../<method>/<layer>_<head>
    # Example: .../softmax/2_4
    method = os.path.basename(os.path.dirname(ckpt_dir))       # softmax
    lh = os.path.basename(ckpt_dir)                            # "2_4"
    if "_" not in lh:
        raise ValueError(f"Expected folder like '2_4', got: {lh}")

    n_layer, n_head = map(int, lh.split("_"))
    return method, n_layer, n_head




args.method, args.n_layer, args.n_head = load_args_from_ckpt_path(args.ckpt_path)
print(args.method)
master_process, seed_offset, ddp_world_size, ddp_rank, device, ddp, rank, ddp_local_rank = init_ddp(
    args.device, args.backend
)
torch.manual_seed(args.seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
args.out_dir = args.out_dir + "/" + args.method + "/" + str(args.n_layer) + "_" + str(args.n_head)
logger = create_logger(os.path.join(args.out_dir, "test.log"), rank=rank)

if master_process:
    logger.info("### STARTING SHARD-BY-SHARD TEST ###")
    logger.info(f"Using checkpoint = {args.ckpt_path}")

# ---------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------
checkpoint = torch.load(args.ckpt_path, map_location="cuda")
conf = argparse.Namespace(**checkpoint["model_args"])
conf.vocab_size = SPECIAL[args.task]["<pad>"] + 1
# Build model exactly as in training
model = StackAssoMem(
    method=conf.method, task=conf.task, vocab_size=conf.vocab_size,
    n_head=conf.n_head, n_embd=conf.n_embd, n_layer=conf.n_layer,
    v_shift=conf.v_shift, block_size=conf.block_size,
    k_fe_type=conf.k_fe_type, v_fe_type=conf.v_fe_type,
    k_kernel_size=conf.k_kernel_size, v_kernel_size=conf.v_kernel_size,
    ic_dropout=0.0, hd_dropout=0.0,
    bias=conf.bias, pmem_count=conf.pmem_count, pmem_size=conf.pmem_size,
    pre_ln=conf.pre_ln, weight_tying=conf.weight_tying,
    skip_tokens=conf.skip_tokens, attn_only=conf.attn_only,
    config=conf
)
# Load weights
state_dict = checkpoint["model"]
unwanted = "_orig_mod."
for k in list(state_dict.keys()):
    if k.startswith(unwanted):
        state_dict[k[len(unwanted):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.to(device)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model
raw_model.eval()

# ---------------------------------------------------------------
# Utility evaluation functions
# ---------------------------------------------------------------
def compute_assoc_and_exact_accuracy(pred_seq, tgt_seq, eos_token=261):
    """
    Compute associative recall and exact match accuracy for last 13 tokens.
    - Associative recall: per key-value pair
    - Exact match: all key-value pairs at once
    """
    # Remove EOS if present
    if pred_seq[-1] == eos_token:
        pred_seq = pred_seq[:-1]
    if tgt_seq[-1] == eos_token:
        tgt_seq = tgt_seq[:-1]
    # Key-value pair indices
    kv_indices = torch.tensor([1,2, 4,5, 7,8, 10,11], device=pred_seq.device)
    # --- Associative recall ---
    ncorrect = 0
    npairs = 0
    for i in range(0, len(kv_indices)-1, 2):
        idx1, idx2 = kv_indices[i], kv_indices[i+1]
        if idx1 < len(pred_seq) and idx2 < len(pred_seq):
            npairs += 1
            if pred_seq[idx1] == tgt_seq[idx1] and pred_seq[idx2] == tgt_seq[idx2]:
                ncorrect += 1

    # --- Exact match across all key-value pairs ---
    valid_idx = kv_indices[kv_indices < len(pred_seq)]
    if valid_idx.numel() == 0:
        exact_acc = False
    else:
        exact_acc = bool(torch.all(pred_seq[valid_idx] == tgt_seq[valid_idx]).item())

    return npairs, ncorrect, exact_acc

def compute_reverse_exact_accuracy(pred_seq, tgt_seq, sep_token):
    """
    Returns:
      exact_match (0/1),
      num_token_matches,
      num_tokens
    """

    # remove EOS
    tgt_seq = tgt_seq[:-1]
    pred_seq = pred_seq[:-1]

    # find SEP index (assume single SEP)
    sep_idx = (tgt_seq == sep_token).nonzero(as_tuple=True)[0]
    if len(sep_idx) != 1:
        return 0.0, 0, 0  # defensive

    sep_idx = sep_idx.item()

    # tail after SEP
    tgt_tail = tgt_seq[sep_idx + 1:]
    pred_tail = pred_seq[sep_idx + 1:]


    if tgt_tail.numel() == 0 or tgt_tail.shape != pred_tail.shape:
        return 0.0, 0, 0

    # counts
    token_matches = (pred_tail == tgt_tail).sum().item()
    num_tokens = tgt_tail.numel()

    exact_match = float(token_matches == num_tokens)

    return exact_match, token_matches, num_tokens


# ---------------------------------------------------------------------
# Evaluation for a single shard
# ---------------------------------------------------------------------
def evaluate_shard(src_file, trg_file, shard_id, task):
    """Returns: avg_loss, assoc_accuracy, exact_accuracy, predictions"""
    dataset = IndexedPairDataset(src_file, trg_file, task)
    collate_fn = make_indexedpair_collate(SPECIAL[task]["<pad>"])
    loader = InfiniteDataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=collate_fn,
        task=args.task
    )

    losses = []
    predictions = []

    total_pairs = 0
    total_correct = 0
    total_exact = 0
    total_samples = 0
    total_token_matches = 0
    total_tokens = 0
    pad_token = SPECIAL[args.task]["<pad>"]
    eos_token = SPECIAL[args.task]["<eos>"]

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for it, batch in enumerate(loader):
            if it >= args.eval_iters:
                break

            X, Y, context_lengths = batch
            context_lengths = context_lengths.to(device)
            X = X.to(device)
            Y = Y.to(device)
            logits, loss = raw_model(X, Y, ctx=context_lengths)
            losses.append(loss.item())
            preds = torch.argmax(logits, dim=-1)

            # --- Evaluate according to task ---
            for b in range(preds.size(0)):
                # We will compare tails of preds and tails of Y (consistent logic for both tasks)
                # Slice to last tokens equal to block size used during training if needed, but we use actual sequence tail.
                # For MQMTAR we expect last 13 tokens to contain the KV pairs layout.
                if task == "mqmtar":
                    # Keep original behavior: compare last 13 tokens for KV associative recall + exact match
                    pred_seq = preds[b, -13:]
                    tgt_seq  = Y[b,   -13:]
                    npairs, ncorrect, exact_acc = compute_assoc_and_exact_accuracy(pred_seq, tgt_seq, eos_token=eos_token)
                    total_pairs   += npairs
                    total_correct += ncorrect
                    total_exact   += int(exact_acc)
                    total_samples += 1

                elif task == "reverse" or task == "copy" or task == "sort":
                    sep_tok = SPECIAL[args.task]["<sep>"]

                    exact_acc, token_matches, num_tokens = compute_reverse_exact_accuracy(
                        preds[b], Y[b], sep_tok
                    )

                    total_exact += exact_acc
                    total_token_matches += token_matches
                    total_tokens += num_tokens
                    total_samples += 1

                else:
                    # Unknown task: fallback to token-level exact on last min-length segment
                    # This should rarely happen; we keep it defensive.
                    tgt_mask = (Y[b] != pad_token)
                    if not tgt_mask.any():
                        continue
                    tgt_len = int(tgt_mask.sum().item())
                    pred_tail = preds[b, -tgt_len:]
                    tgt_tail = Y[b, -tgt_len:]
                    exact_acc = float(torch.all(pred_tail == tgt_tail).item())
                    total_exact += exact_acc
                    total_samples += 1

            predictions.append(preds.cpu())

    # --- Loss reduction ---
    if ddp:
        tensor = torch.tensor([sum(losses), len(losses)], device=device)
        torch.distributed.all_reduce(tensor)
        avg_loss = tensor[0].item() / tensor[1].item()
    else:
        avg_loss = float(np.mean(losses)) if losses else float("nan")

    # --- Accuracy ---
    if task == "mqmtar":
        assoc_acc = total_correct / total_pairs if total_pairs > 0 else 0.0
        exact_acc = total_exact / total_samples if total_samples > 0 else 0.0
    elif task == "reverse" or task == "copy":
        token_acc = total_token_matches / total_tokens if total_tokens > 0 else 0.0
        exact_acc = total_exact / total_samples if total_samples > 0 else 0.0
        assoc_acc = 0.0  # not used
    if master_process:
        N = len(dataset)
        if task == "mqmtar":
            logger.info(
                f"[shard {shard_id}] size={N} loss={avg_loss:.4f}, assoc_acc={assoc_acc*100:}%, exact_acc={exact_acc*100:}%"
            )
        else:
            logger.info(
                f"[shard {shard_id}] size={N} loss={avg_loss:.4f}, reverse_exact_acc={exact_acc*100:}%, token_acc={token_acc*100:}%"
            )

    preds_tensor = torch.cat(predictions, dim=0) if predictions else None
    return avg_loss, assoc_acc, exact_acc, preds_tensor

# ---------------------------------------------------------------------
# Run per-shard evaluation
# ---------------------------------------------------------------------
eval_folders = [
    "copy",
    #"reverse/30M_tr-32-64_val_64-256_t-64-512",
]


'''
eval_folders = [
    "copy",
    #"50M_abc-256_vocab-10K_kv-len-2_num_kv-80_num_q-4",
    #"eval-8K-16K-abc-256_vocab-10K_kv-len-2_num_kv-80",
    #"eval-32K-abc-256_vocab-10K_kv-len-2_num_kv-80",
    #"eval-65K-abc-256_vocab-10K_kv-len-2_num_kv-80",
]'''
data_dir = args.datapath

src_shards = []
trg_shards = []

for folder in eval_folders:
    full = os.path.join(data_dir, folder)
    src_shards.extend(sorted(glob.glob(os.path.join(full, "test_*.src"))))
    trg_shards.extend(sorted(glob.glob(os.path.join(full, "test_*.trg"))))

all_results = {}

for shard_id, (src_file, trg_file) in enumerate(zip(src_shards, trg_shards)):
    if master_process:
        logger.info(f"Evaluating shard {shard_id}: {os.path.basename(src_file)}")

    avg_loss, assoc_acc, exact_acc, preds = evaluate_shard(src_file, trg_file, shard_id, args.task)

    if master_process:
        pred_path = os.path.join(args.out_dir, f"preds_shard_{shard_id}.pt")

        all_results[f"shard_{shard_id}"] = {
            "src": os.path.basename(src_file),
            "trg": os.path.basename(trg_file),
            "loss": avg_loss,
            "assoc_accuracy": assoc_acc,
            "exact_accuracy": exact_acc,
            "preds_path": pred_path,
        }

        if preds is not None:
            torch.save(preds, pred_path)

# ---------------------------------------------------------------------
# Save summary file
# ---------------------------------------------------------------------
if master_process:
    summary_path = os.path.join(args.out_dir, "test_summary.pt")
    torch.save(all_results, summary_path)
    logger.info(f"Saved shard-by-shard summary → {summary_path}")
    print(f"Saved summary → {summary_path}")