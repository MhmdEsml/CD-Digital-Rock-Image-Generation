import math
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from porous.edm.config import get_stride, resolve_sigma_data
from porous.core.patch_data import _collate, _stack_ranks, volume_hash


def grid_starts(dim, patch, stride):
    return np.arange(0, int(dim) - int(patch) + 1, int(stride), dtype=np.int64)


def make_origins(starts):
    zz, yy, xx = np.meshgrid(*starts, indexing='ij')
    return np.stack([zz, yy, xx], axis=-1).reshape(-1, 3).astype(np.int32)


def assign_split(origins, patch, dims, axis, holdout_size):
    split = np.zeros(len(origins), np.uint8)
    if holdout_size and holdout_size > 0:
        b = int(dims[axis]) - int(holdout_size)
        if not (0 < b):
            raise ValueError(f'holdout_size={holdout_size} >= volume extent {dims[axis]} along axis {axis}')
        lo = origins[:, axis].astype(np.int64)
        hi = lo + int(patch)
        split = np.where(hi <= b, 0, np.where(lo >= b, 1, 2)).astype(np.uint8)
    return split


def window_void_counts(vol, starts, patch, chunk=32):
    d, h, w = vol.shape
    sz, sy, sx = [np.asarray(s, np.int64) for s in starts]
    p = int(patch)
    a = np.empty((d, h, len(sx)), np.int32)
    for z0 in range(0, d, chunk):
        slab = vol[z0 : z0 + chunk] == 0
        c = np.zeros(slab.shape[:2] + (w + 1,), np.int32)
        c[..., 1:] = np.cumsum(slab, axis=2, dtype=np.int32)
        a[z0 : z0 + chunk] = c[..., sx + p] - c[..., sx]
    c = np.zeros((d, h + 1, len(sx)), np.int64)
    c[:, 1:] = np.cumsum(a, axis=1, dtype=np.int64)
    b = c[:, sy + p] - c[:, sy]
    del a, c
    c = np.zeros((d + 1,) + b.shape[1:], np.int64)
    c[1:] = np.cumsum(b, axis=0, dtype=np.int64)
    return c[sz + p] - c[sz]


def compute_patch_stats(volume, origins, patch, n_samples, seed=0):
    rng = np.random.RandomState(seed)
    n = min(int(n_samples), len(origins))
    sel = rng.choice(len(origins), size=n, replace=False)
    cnt, s, s2 = 0, 0.0, 0.0
    for i in sel:
        z, y, x = [int(v) for v in origins[i]]
        xx = 2.0 * volume[z : z + patch, y : y + patch, x : x + patch].astype(np.float64) - 1.0
        cnt += xx.size
        s += float(xx.sum())
        s2 += float((xx * xx).sum())
    mean = s / cnt
    return {
        'data_mean': mean,
        'data_std': math.sqrt(max(s2 / cnt - mean * mean, 0.0)),
        'data_rms': math.sqrt(s2 / cnt),
        'n_patches': int(n),
    }


def build_patch_cache(volume, data_cfg, n_check=16, verbose=True):
    p = int(data_cfg['patch_size'])
    strides = get_stride(data_cfg)
    dims = tuple(int(v) for v in volume.shape)
    assert volume.dtype == np.uint8 and volume.ndim == 3, (volume.dtype, volume.shape)
    if int(volume.max()) > 1:
        raise ValueError(f'the volume must be binary {{0 = void, 1 = grain}}; found max value {int(volume.max())}')
    starts = [grid_starts(dims[a], p, strides[a]) for a in range(3)]
    if any(len(s) == 0 for s in starts):
        raise ValueError(f'patch_size {p} does not fit into the volume {dims}')
    origins = make_origins(starts)
    counts = window_void_counts(volume, starts, p).reshape(-1)
    assert counts.shape[0] == origins.shape[0]
    porosity = (counts / float(p**3)).astype(np.float32)

    rng = np.random.RandomState(0)
    for i in rng.choice(len(origins), size=min(n_check, len(origins)), replace=False):
        z, y, x = [int(v) for v in origins[i]]
        bf = int((volume[z : z + p, y : y + p, x : x + p] == 0).sum())
        assert bf == int(counts[i]), f'porosity table mismatch at origin {(z, y, x)}: box sum {int(counts[i])} vs brute force {bf}'
    if verbose:
        print(f'porosity table verified against brute force on {min(n_check, len(origins))} random patches')

    ax, hs = int(data_cfg['holdout_axis']), int(data_cfg['holdout_size'])
    split = assign_split(origins, p, dims, ax, hs)
    tr = np.flatnonzero(split == 0)
    if len(tr) == 0:
        raise ValueError('no training patch left after the hold-out split: reduce holdout_size')
    if hs > 0 and not (split == 1).any():
        raise ValueError(
            f'holdout_size={hs} leaves no hold-out patch (patch {p}, stride {strides}); use holdout_size >= patch_size + stride'
        )
    st = compute_patch_stats(volume, origins[tr], p, data_cfg.get('stats_n_samples', 256), data_cfg.get('stats_seed', 0))
    por_tr = porosity[tr].astype(np.float64)
    cache = {
        'origins': origins,
        'void_count': counts.astype(np.int64),
        'porosity': porosity,
        'split': split,
        'dataset': np.array(str(data_cfg['dataset'])),
        'source': np.array(str(data_cfg['source'])),
        'volume_dims': np.array(dims, np.int64),
        'patch_size': np.array(p, np.int64),
        'stride': np.array(strides, np.int64),
        'holdout_axis': np.array(ax, np.int64),
        'holdout_size': np.array(hs, np.int64),
        'volume_hash': np.array(volume_hash(volume)),
        'data_mean': np.array(st['data_mean']),
        'data_std': np.array(st['data_std']),
        'data_rms': np.array(st['data_rms']),
        'stats_n_patches': np.array(st['n_patches'], np.int64),
        'stats_seed': np.array(int(data_cfg.get('stats_seed', 0)), np.int64),
        'porosity_train_min': np.array(por_tr.min()),
        'porosity_train_max': np.array(por_tr.max()),
        'porosity_train_mean': np.array(por_tr.mean()),
        'porosity_train_std': np.array(por_tr.std()),
    }
    return cache


def save_patch_cache(path, cache):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[: -len('.npz')] + '.tmp.npz'
    np.savez(tmp, **cache)
    os.replace(tmp, path)


def _cache_problems(cache, data_cfg, volume=None):
    p = int(data_cfg['patch_size'])
    probs = []
    chk = [
        ('dataset', str(cache['dataset'].item()), str(data_cfg['dataset'])),
        ('source', str(cache['source'].item()), str(data_cfg['source'])),
        ('patch_size', int(cache['patch_size']), p),
        ('stride', tuple(int(v) for v in cache['stride']), get_stride(data_cfg)),
        ('holdout_axis', int(cache['holdout_axis']), int(data_cfg['holdout_axis'])),
        ('holdout_size', int(cache['holdout_size']), int(data_cfg['holdout_size'])),
        (
            'stats_n_samples',
            int(cache['stats_n_patches']),
            min(int(data_cfg.get('stats_n_samples', 256)), int((cache['split'] == 0).sum())),
        ),
        ('stats_seed', int(cache['stats_seed']), int(data_cfg.get('stats_seed', 0))),
    ]
    for k, a, b in chk:
        if a != b:
            probs.append(f'{k}: cache={a!r} vs config={b!r}')
    if volume is not None:
        if tuple(int(v) for v in cache['volume_dims']) != tuple(volume.shape):
            probs.append(f"volume_dims: cache={tuple(cache['volume_dims'])} vs volume={tuple(volume.shape)}")
        elif str(cache['volume_hash'].item()) != volume_hash(volume):
            probs.append('volume_hash differs (the volume file changed)')
    return probs


def load_patch_cache(path, data_cfg, volume=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f'patch cache not found: {path}. Build it with prepare_data() before training.')
    z = np.load(path, allow_pickle=False)
    cache = {k: z[k] for k in z.files}
    probs = _cache_problems(cache, data_cfg, volume)
    if probs:
        raise RuntimeError(
            f'Stale patch cache {path}: ' + '; '.join(probs) + '. Delete the cache file so it is rebuilt together with the config.'
        )
    return cache


def load_volume(path, dims=None):
    path = os.path.expanduser(str(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f'volume file not found: {path}')
    print(f"Loading volume '{path}' ...")
    if path.lower().endswith('.npy'):
        vol = np.load(path, mmap_mode='r')
        if vol.ndim != 3:
            raise ValueError(f'{path}: expected a 3-D array, got shape {vol.shape}')
        if dims is not None and tuple(int(v) for v in dims) != tuple(vol.shape):
            raise ValueError(f'{path} has shape {tuple(vol.shape)} but volume_dims={list(dims)}')
        vol = np.ascontiguousarray(vol).astype(np.uint8, copy=False)
    else:
        if dims is None:
            raise ValueError('a raw volume needs data.volume_dims (--volume-dims D H W)')
        d, h, w = [int(v) for v in dims]
        size = os.path.getsize(path)
        if size != d * h * w:
            raise ValueError(f'{path} has {size} bytes, expected {d * h * w} for volume_dims={[d, h, w]} (uint8)')
        vol = np.fromfile(path, dtype=np.uint8).reshape((d, h, w))
    print('Volume loaded with shape', vol.shape, f'({vol.nbytes / 1e6:.0f} MB, uint8)')
    return vol


_VOLUME_MEMO = {}


def get_volume(cfg):
    d = cfg['data']
    key = (d['source'], str(d.get('volume_path')), tuple(d['volume_dims']))
    if key not in _VOLUME_MEMO:
        vol = load_volume(d['volume_path'], None if str(d['volume_path']).lower().endswith('.npy') else d['volume_dims'])
        if tuple(vol.shape) != tuple(int(v) for v in d['volume_dims']):
            d['volume_dims'] = [int(v) for v in vol.shape]
        _VOLUME_MEMO.clear()
        _VOLUME_MEMO[key] = vol
    return _VOLUME_MEMO[key]


def prepare_data(CFG):
    PATCH = int(CFG['data']['patch_size'])
    volume = get_volume(CFG)
    PATCH_CACHE_PATH = CFG['patch_cache_path']
    try:
        patch_cache = load_patch_cache(PATCH_CACHE_PATH, CFG['data'], volume)
        print(f'reusing {PATCH_CACHE_PATH} (matches config and volume)')
    except (FileNotFoundError, RuntimeError) as e:
        print('building the patch cache:', str(e)[:160])
        t0 = time.time()
        patch_cache = build_patch_cache(volume, CFG['data'])
        save_patch_cache(PATCH_CACHE_PATH, patch_cache)
        print(f'patch cache built in {time.time() - t0:.1f} s -> {PATCH_CACHE_PATH} ({os.path.getsize(PATCH_CACHE_PATH) / 1e6:.2f} MB)')

    _sp = patch_cache['split']
    print(
        f"patch grid {tuple(int(len(grid_starts(volume.shape[a], PATCH, get_stride(CFG['data'])[a]))) for a in range(3))} = {len(_sp)} origins | "
        f"training {int((_sp == 0).sum())} | hold-out {int((_sp == 1).sum())} | excluded (straddling) {int((_sp == 2).sum())}"
    )
    if CFG['data']['holdout_size'] == 0:
        print('NOTE: data.holdout_size = 0 -> every patch of the volume is used for training (no hold-out region).')
    print(
        f"porosity of the training patches: min {float(patch_cache['porosity_train_min']):.4f} max {float(patch_cache['porosity_train_max']):.4f} "
        f"mean {float(patch_cache['porosity_train_mean']):.4f} std {float(patch_cache['porosity_train_std']):.4f}"
    )
    if (_sp == 1).any():
        _ph = patch_cache['porosity'][_sp == 1]
        print(f'porosity of the hold-out patches:  min {_ph.min():.4f} max {_ph.max():.4f} mean {_ph.mean():.4f} std {_ph.std():.4f}')

    data_mean, data_std, data_rms = float(patch_cache['data_mean']), float(patch_cache['data_std']), float(patch_cache['data_rms'])
    _an_mean = 1.0 - 2.0 * float(patch_cache['porosity_train_mean'])
    print(
        f"data_mean = {data_mean:.5f} (analytic 1 - 2*mean porosity over all training origins: {_an_mean:.5f}) | data_std = {data_std:.5f} "
        f"(= sqrt(1 - mean^2) = {math.sqrt(max(1 - data_mean ** 2, 0)):.5f}) | data_rms = {data_rms:.5f}"
    )
    resolve_sigma_data(CFG['diffusion'], data_rms)
    print(f"diffusion.sigma_data = {CFG['diffusion']['sigma_data']}")
    return volume, patch_cache


class PorousPatchDataset:

    def __init__(self, volume, origins, porosity, patch_size, flip_prob=0.5, with_porosity=True):
        assert volume.dtype == np.uint8 and volume.ndim == 3
        assert len(origins) == len(porosity)
        self.volume, self.origins, self.porosity = volume, np.asarray(origins), np.asarray(porosity, np.float32)
        self.p = int(patch_size)
        self.flip_prob = float(flip_prob)
        self.with_porosity = bool(with_porosity)

    def __len__(self):
        return len(self.origins)

    def raw_patch(self, idx):
        z, y, x = [int(v) for v in self.origins[idx]]
        p = self.p
        return self.volume[z : z + p, y : y + p, x : x + p]

    def __getitem__(self, idx, rng=None):
        rng = np.random if rng is None else rng
        patch = self.raw_patch(idx)
        if self.flip_prob > 0:
            flips = rng.rand(3) < self.flip_prob
            patch = patch[tuple(slice(None, None, -1) if f else slice(None) for f in flips)]
        img = (2.0 * patch.astype(np.float32) - 1.0)[None]
        if self.with_porosity:
            return img, self.porosity[idx : idx + 1].copy()
        return img


class ShardedLoader:

    def __init__(self, dataset, samplers, ranks, batch_size, num_workers, base_seed):
        self.dataset, self.samplers, self.ranks = dataset, samplers, ranks
        self.batch_size, self.num_workers, self.base_seed = batch_size, num_workers, base_seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        for s in self.samplers:
            s.set_epoch(epoch)

    def __len__(self):
        return len(self.samplers[0]) // self.batch_size

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


def build_split_dataset(volume, cache, data_cfg, which, with_porosity, max_patches=None):
    idx = np.flatnonzero(cache['split'] == which)
    if max_patches and len(idx) > max_patches:
        idx = idx[np.linspace(0, len(idx) - 1, int(max_patches)).astype(np.int64)]
    ds = PorousPatchDataset(
        volume,
        cache['origins'][idx],
        cache['porosity'][idx],
        int(cache['patch_size']),
        flip_prob=float(data_cfg.get('flip_prob', 0.5)),
        with_porosity=with_porosity,
    )
    return ds, idx
