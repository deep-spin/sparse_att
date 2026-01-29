import os
import struct
import torch
from torch.utils.data import Dataset
SPECIAL = {"mqmtar" : {
        "<bos>": 259,
        "<sep>": 260,
        "<eos>": 261,
        "<pad>": 262,
    },
      "reverse" : {
        "<bos>": 32,
        "<sep>": 33,
        "<eos>": 34,
        "<pad>": 35,
    },
    "copy" : {
        "<bos>": 32,
        "<sep>": 33,
        "<eos>": 34,
        "<pad>": 35,
    },
    "sort" : {
        "<bos>": 32,
        "<sep>": 33,
        "<eos>": 34,
        "<pad>": 35,
    }}

class IndexedPairDataset(Dataset):
    def __init__(self, src_path, trg_path, task):
        self.SPECIAL = SPECIAL[task]
        self.src_path = src_path
        self.trg_path = trg_path
        self.src_idx_path = f"{src_path}.idx"
        self.trg_idx_path = f"{trg_path}.idx"

        self.src_lines = os.path.getsize(self.src_idx_path) // 8
        self.trg_lines = os.path.getsize(self.trg_idx_path) // 8
        self.line_count = min(self.src_lines, self.trg_lines)

    def __len__(self):
        return self.line_count

    def _get_line(self, file_path, line_idx):
        idx_path = f"{file_path}.idx"
        with open(idx_path, "rb") as f_idx, open(file_path, "rb") as f_data:
            f_idx.seek(line_idx * 8)
            offset = struct.unpack("Q", f_idx.read(8))[0]
            f_data.seek(offset)
            line_bytes = f_data.readline()
        return line_bytes.decode("utf-8", errors="replace").strip()

    def _parse_tokens(self, line):
        return [int(tok) for tok in line.split()] if line else []

    def __getitem__(self, idx):
        src_line = self._get_line(self.src_path, idx)
        trg_line = self._get_line(self.trg_path, idx)

        src = self._parse_tokens(src_line)
        trg = self._parse_tokens(trg_line)

        BOS = self.SPECIAL["<bos>"]
        SEP = self.SPECIAL["<sep>"]
        EOS = self.SPECIAL["<eos>"]

        # Build raw sequence:    [BOS] src [SEP] trg [EOS]
        seq = [BOS] + src + [SEP] + trg + [EOS]
        # Compute prefix length = [BOS] + src + [SEP]
        context_length = len(seq)

        # DO NOT TRUNCATE — raw sequence returned
        # DO NOT PAD — padding done in collate_fn

        xs = torch.tensor(seq[:-1], dtype=torch.long)
        ys = torch.tensor(seq[1:], dtype=torch.long)

        return xs, ys, context_length


class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

    def __len__(self):
        raise ValueError("InfiniteSampler has no length")


class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, num_workers=1, collate_fn=None, task=None):
        self.batch_size = batch_size

        # Random sampler with replacement
        sampler = torch.utils.data.RandomSampler(dataset, replacement=True)

        # Batch sampler
        batch_sampler = torch.utils.data.BatchSampler(
            sampler, batch_size=batch_size, drop_last=True
        )

        self._infinite_iterator = iter(
            torch.utils.data.DataLoader(
                dataset,
                num_workers=num_workers,
                batch_sampler=_InfiniteSampler(batch_sampler),
                collate_fn=collate_fn,
            )
        )

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError("InfiniteDataLoader has no length")


def make_indexedpair_collate(PAD):
    def indexedpair_collate(batch):
        xs, ys, ctx = zip(*batch)
        max_len = max(ctx) - 1

        batch_x = []
        batch_y = []

        for x, y in zip(xs, ys):
            pad_x = torch.full((max_len - len(x),), PAD, dtype=torch.long)
            pad_y = torch.full((max_len - len(y),), PAD, dtype=torch.long)
            batch_x.append(torch.cat([x, pad_x], dim=0))
            batch_y.append(torch.cat([y, pad_y], dim=0))

        batch_x = torch.stack(batch_x, dim=0)
        batch_y = torch.stack(batch_y, dim=0)
        ctx = torch.tensor(ctx, dtype=torch.long) - 1
        

        return batch_x, batch_y, ctx

    return indexedpair_collate