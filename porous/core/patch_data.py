import hashlib
import math

import numpy as np


def volume_hash(volume, chunk=1 << 26):
    h = hashlib.blake2b(digest_size=16)
    flat = volume.reshape(-1)
    for i in range(0, flat.size, chunk):
        h.update(flat[i : i + chunk].tobytes())
    return h.hexdigest()


class DistributedSampler:

    def __init__(self, dataset, num_replicas, rank, shuffle=True, seed=0, drop_last=False):
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        if self.shuffle:
            g = np.random.RandomState(self.seed + self.epoch)
            indices = g.permutation(len(self.dataset)).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)


def _collate(items):
    if isinstance(items[0], tuple):
        return tuple(np.stack([it[j] for it in items]) for j in range(len(items[0])))
    return np.stack(items)


def _stack_ranks(per_rank):
    if isinstance(per_rank[0], tuple):
        return tuple(np.stack([pr[j] for pr in per_rank]) for j in range(len(per_rank[0])))
    return np.stack(per_rank)
