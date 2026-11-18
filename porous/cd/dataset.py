import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from porous.core.patch_data import _collate, _stack_ranks


class PorousPatchDataset:

    def __init__(self, volume, origins, porosity, patch_size, flip=True):
        self.volume = volume
        self.origins = np.asarray(origins, np.int64).reshape(-1, 3)
        self.porosity = np.asarray(porosity, np.float32).reshape(-1)
        assert len(self.origins) == len(self.porosity), (len(self.origins), len(self.porosity))
        self.patch = tuple(int(p) for p in patch_size)
        self.flip = bool(flip)

    def __len__(self):
        return len(self.origins)

    def raw_patch(self, idx):
        z, y, x = (int(v) for v in self.origins[idx])
        pd, ph, pw = self.patch
        return self.volume[z : z + pd, y : y + ph, x : x + pw]

    def __getitem__(self, idx, rng=None):
        rng = np.random if rng is None else rng
        coins = rng.rand(3)
        v = self.raw_patch(idx)
        for ax in range(3):
            if self.flip and coins[ax] < 0.5:
                v = np.flip(v, axis=ax)
        img = (2.0 * v.astype(np.float32) - 1.0)[None]
        return img, self.porosity[idx : idx + 1].copy()


class ShardedLoader:

    def __init__(self, dataset, samplers, ranks, batch_size, num_workers, base_seed, steps_per_epoch=None):
        self.dataset, self.samplers, self.ranks = dataset, samplers, ranks
        self.batch_size, self.num_workers, self.base_seed = batch_size, num_workers, base_seed
        self.steps_per_epoch = int(steps_per_epoch) if steps_per_epoch else None
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        for s in self.samplers:
            s.set_epoch(epoch)

    def __len__(self):
        n = len(self.samplers[0]) // self.batch_size
        return min(n, self.steps_per_epoch) if self.steps_per_epoch else n

    def _build(self, epoch, b, idx_lists):
        per_rank = []
        for r_i, rank in enumerate(self.ranks):
            seed = np.random.SeedSequence([int(self.base_seed), int(rank), int(epoch), int(b)]).generate_state(1)[0]
            rng = np.random.RandomState(seed)
            ids = idx_lists[r_i][b * self.batch_size : (b + 1) * self.batch_size]
            if hasattr(self.dataset, 'get_batch'):
                per_rank.append(self.dataset.get_batch(ids, rng))
            else:
                per_rank.append(_collate([self.dataset.__getitem__(i, rng=rng) for i in ids]))
        return _stack_ranks(per_rank)

    def __iter__(self):
        epoch, n_batches = self.epoch, len(self)
        idx_lists = [list(s) for s in self.samplers]
        if self.num_workers <= 0:
            for b in range(n_batches):
                yield self._build(epoch, b, idx_lists)
            return
        depth = 2 * self.num_workers
        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            futures, nxt = deque(), 0
            while nxt < min(depth, n_batches):
                futures.append(ex.submit(self._build, epoch, nxt, idx_lists))
                nxt += 1
            for b in range(n_batches):
                out = futures.popleft().result()
                if nxt < n_batches:
                    futures.append(ex.submit(self._build, epoch, nxt, idx_lists))
                    nxt += 1
                yield out
