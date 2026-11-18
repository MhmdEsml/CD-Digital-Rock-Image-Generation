import os
import zipfile

import numpy as np

from porous.cd.dataset import PorousPatchDataset
from porous.core.patch_data import volume_hash


def _norm3(v):
    return (int(v),) * 3 if np.ndim(v) == 0 else tuple(int(x) for x in np.asarray(v).reshape(-1))


def resolve_data_spec(CDCFG, TEACHER, bundle_config):
    info = TEACHER['info']
    need = ('dataset', 'volume_dims', 'patch_size', 'stride', 'holdout_axis', 'holdout_size')
    miss = [k for k in need if info.get(k) is None]
    if miss:
        raise KeyError(
            f'teacher_info.json has no {miss}: it was not written by the EDM teacher training code (fields: dataset, volume_dims, patch_size, stride, holdout_axis, holdout_size, ...)'
        )
    blocks = (
        ('edm_config.json[data]', (bundle_config or {}).get('data') or {}),
        ('checkpoint config[data]', (TEACHER.get('ck_config') or {}).get('data') or {}),
    )
    norm = {'dataset': str, 'volume_dims': _norm3, 'patch_size': _norm3, 'stride': _norm3, 'holdout_axis': int, 'holdout_size': int}
    for k in need:
        for bname, blk in blocks:
            if blk.get(k) is not None and norm[k](blk[k]) != norm[k](info[k]):
                raise RuntimeError(
                    f'{bname}.{k} = {blk[k]!r} disagrees with teacher_info.json.{k} = {info[k]!r}: the bundle / checkpoint / info do not belong to the same teacher run'
                )
    spec = {k: info[k] for k in need}
    spec['seed'] = int(info.get('seed', 0))
    ov = CDCFG['data'].get('override') or {}
    if ov:
        print(
            f'[WARN] data.override = {ov} replaces values read from the teacher: the student will NOT be trained on the teacher\'s data definition'
        )
        spec.update({k: v for k, v in ov.items() if k in spec})
    vd, ps, st = _norm3(spec['volume_dims']), _norm3(spec['patch_size']), _norm3(spec['stride'])
    ax, hs = int(spec['holdout_axis']), int(spec['holdout_size'])
    assert len(vd) == 3 and ax in (0, 1, 2) and hs >= 0, (vd, ax, hs)
    out = dict(
        dataset=str(spec['dataset']),
        volume_dims=vd,
        patch_size=ps,
        stride=st,
        holdout_axis=ax,
        holdout_size=hs,
        holdout_declared=hs > 0,
        holdout=({'axis': ax, 'size': hs, 'side': 'end'} if hs > 0 else None),
        seed=int(spec['seed']),
        volume_hash=info.get('volume_hash'),
    )
    print(
        f"data spec (teacher_info.json): dataset={out['dataset']} volume_dims={vd} patch_size={ps} stride={st} hold-out: axis {ax}, last {hs} voxels"
        f"{'' if hs > 0 else ' (NONE: no hold-out)'} seed={out['seed']}"
    )
    return out


def local_volume_path(path):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f'volume file not found: {path!r} (give the local .raw file with --volume)')
    return os.path.abspath(path)


def load_volume(path, dims):
    n = int(np.prod(dims))
    size = os.path.getsize(path)
    if size != n:
        raise ValueError(f'{path} has {size} bytes but volume_dims {tuple(dims)} need {n}')
    return np.fromfile(path, dtype=np.uint8).reshape(tuple(dims))


def volume_stats(vol, chunk=8):
    counts = np.zeros(256, np.int64)
    for i in range(0, vol.shape[0], chunk):
        counts += np.bincount(vol[i : i + chunk].ravel(), minlength=256)
    total = float(counts.sum())
    vals = np.arange(256, dtype=np.float64)
    other = int(counts[2:].sum())
    return dict(
        void_fraction=float(counts[0] / total), rms=float(np.sqrt((counts * (2 * vals - 1) ** 2).sum() / total)), n_non_binary=other
    )


def patch_porosity(volume, origins, patch_size):
    pd, ph, pw = patch_size
    return np.array([float(np.mean(volume[z : z + pd, y : y + ph, x : x + pw] == 0)) for z, y, x in np.asarray(origins)], np.float32)


def load_patch_index(npz_paths, spec, volume):
    paths = [p for p in npz_paths if os.path.basename(p) == 'porosity_cache.npz']
    if len(paths) != 1:
        raise FileNotFoundError(
            f"eval_bundle.zip must contain exactly one porosity_cache.npz (found {[os.path.basename(p) for p in npz_paths]})"
        )
    with np.load(paths[0], allow_pickle=False) as z:
        keys = set(z.files)
        required = ('origins', 'porosity', 'split', 'volume_dims', 'patch_size', 'stride', 'holdout_axis', 'holdout_size', 'volume_hash')
        miss = [k for k in required if k not in keys]
        if miss:
            raise KeyError(f'porosity_cache.npz has no {miss} (has {sorted(keys)}): not written by the EDM teacher training code')
        origins, por, split = (
            np.asarray(z['origins']).astype(np.int64),
            np.asarray(z['porosity'], np.float32),
            np.asarray(z['split']).astype(np.int64),
        )
        stored = dict(
            volume_dims=_norm3(z['volume_dims']),
            patch_size=_norm3(z['patch_size']),
            stride=_norm3(z['stride']),
            holdout_axis=int(z['holdout_axis']),
            holdout_size=int(z['holdout_size']),
            volume_hash=str(z['volume_hash'].item()),
        )
    print('porosity_cache.npz arrays: ' + ', '.join(sorted(keys)))
    if not (origins.ndim == 2 and origins.shape[1] == 3 and len(origins) == len(por) == len(split)):
        raise ValueError(f'inconsistent patch index: origins {origins.shape}, porosity {por.shape}, split {split.shape}')
    if not set(np.unique(split).tolist()) <= {0, 1, 2}:
        raise ValueError(f'split must contain only 0 (train) / 1 (hold-out) / 2 (excluded), found {np.unique(split).tolist()}')
    for k in ('volume_dims', 'patch_size', 'stride', 'holdout_axis', 'holdout_size'):
        want = spec[k] if k != 'volume_dims' else tuple(spec['volume_dims'])
        if stored[k] != want:
            raise RuntimeError(
                f'porosity_cache.npz {k} = {stored[k]} differs from the data definition used here ({want}): the student would not be trained on the teacher\'s patch grid'
            )
    tr, ho, ex = split == 0, split == 1, split == 2
    if spec['holdout_declared'] and not ho.any():
        raise RuntimeError(
            f"the teacher declares a hold-out (last {spec['holdout_size']} voxels of axis {spec['holdout_axis']}) but its split has NO hold-out patch"
        )
    if not spec['holdout_declared'] and (ho.any() or ex.any()):
        raise RuntimeError('holdout_size = 0 but the split contains hold-out / excluded patches: inconsistent porosity_cache.npz')
    how = f"split array of porosity_cache.npz: {int(tr.sum())} train (0), {int(ho.sum())} hold-out (1), {int(ex.sum())} excluded (2, never used)"
    print(f"patch-index selection: {how}")
    return dict(
        origins_train=origins[tr],
        por_train=por[tr],
        origins_hold=origins[ho],
        por_hold=por[ho],
        how=how,
        rebuilt_holdout=False,
        volume_hash_stored=stored['volume_hash'],
    )


def build_data(CDCFG, TEACHER, bundle, volume_path, spec=None):
    spec = spec or resolve_data_spec(CDCFG, TEACHER, bundle['config_dict'])
    pd_ = spec['patch_size']
    vol = load_volume(volume_path, spec['volume_dims'])
    assert all(p <= v for p, v in zip(pd_, vol.shape)), (pd_, vol.shape)
    vs = volume_stats(vol)
    print(f"volume {vol.shape} uint8 ({vol.nbytes / 1e9:.2f} GB) | void fraction {vs['void_fraction']:.4f} | RMS(2v-1) = {vs['rms']:.4f}")
    if vs['n_non_binary']:
        raise ValueError(f"the volume has {vs['n_non_binary']} voxels with values > 1: not a binary uint8 volume")
    t_rms = next(
        (
            float(s[k])
            for s in (
                TEACHER['info'],
                bundle['config_dict'],
                TEACHER.get('ck_config') or {},
                (TEACHER.get('ck_config') or {}).get('data') or {},
            )
            for k in ('data_rms',)
            if isinstance(s, dict) and s.get(k) is not None
        ),
        None,
    )
    tol = float(CDCFG['data']['rms_tolerance'])
    if t_rms is None:
        print('[warn] the teacher stores no data_rms: skipped the RMS comparison')
    else:
        print(f"teacher data_rms = {t_rms:.4f} (relative difference {abs(vs['rms'] - t_rms) / t_rms:.2%})")
        if abs(vs['rms'] - t_rms) > tol * t_rms:
            print(
                f'[WARN] the measured volume RMS differs from the teacher data_rms by more than {tol:.0%}: a different dataset / preprocessing?'
            )
    sd = TEACHER['diffusion'].sigma_data
    if abs(vs['rms'] - sd) > 0.02 * sd:
        print(f'[warn] the volume RMS ({vs["rms"]:.4f}) differs from the teacher sigma_data ({sd:.4f}) by > 2%')

    idx = load_patch_index(bundle['npz'], spec, vol)
    vh = volume_hash(vol)
    for src_name, want in (('porosity_cache.npz', idx['volume_hash_stored']), ('teacher_info.json', spec.get('volume_hash'))):
        if want is not None and not CDCFG['data'].get('override') and vh != want:
            raise RuntimeError(
                f'the volume file (hash {vh}) is NOT the volume the teacher was trained on ({src_name}: {want}): wrong or modified file'
            )
    print(f'volume hash {vh} matches the teacher\'s volume_hash')
    o_tr, p_tr, o_ho, p_ho = idx['origins_train'], idx['por_train'], idx['origins_hold'], idx['por_hold']
    print(f"patch index: {len(o_tr)} training / {len(o_ho)} hold-out patches (hold-out: {idx['how']})")
    rs = np.random.RandomState(spec['seed'] + 7)
    for name, o, p in (('training', o_tr, p_tr), ('hold-out', o_ho, p_ho)):
        if len(o):
            pick = rs.choice(len(o), size=min(64, len(o)), replace=False)
            err = float(np.abs(patch_porosity(vol, o[pick], pd_) - p[pick]).max())
            if err > 1e-5:
                raise RuntimeError(
                    f'the recomputed porosity of {name} patches differs from the teacher\'s stored values (max |diff| = {err:.2e}): this is not the volume / '
                    'patch grid the teacher was trained on'
                )
    print('stored patch porosities match the recomputed ones (up to 64 random patches per set)')
    if len(o_tr) == 0:
        raise ValueError('no training patches')
    mx = CDCFG['training'].get('max_train_patches')
    if mx and mx < len(o_tr):
        sel = np.sort(np.random.RandomState(spec['seed']).choice(len(o_tr), size=int(mx), replace=False))
        o_tr, p_tr = o_tr[sel], p_tr[sel]
    holdout_ok = len(o_ho) > 0
    if spec['holdout_declared'] and not holdout_ok:
        raise RuntimeError(
            f"the teacher declares a hold-out ({spec['holdout']}) but the hold-out patch set is EMPTY (patch index / slab geometry do not produce hold-out patches): "
            'refusing to continue, because the evaluation would then not be out-of-sample'
        )
    if not holdout_ok:
        print(
            'the teacher defines no hold-out: training patches are used wherever a reference set is needed (no evaluation is run by this code)'
        )
        o_ho, p_ho = o_tr, p_tr
    dataset = PorousPatchDataset(vol, o_tr, p_tr, pd_, flip=True)
    return dict(
        volume=vol,
        dataset=dataset,
        spec=spec,
        patch_size=pd_,
        image_channels=TEACHER['image_channels'],
        data_rms=vs['rms'],
        void_fraction=vs['void_fraction'],
        origins_train=o_tr,
        por_train=p_tr,
        origins_hold=o_ho,
        por_hold=p_ho,
        holdout_is_real=holdout_ok,
    )


def unpack_batch(batch, has_por):
    x_1, por = batch
    return {'x_1': x_1, 'por': por} if has_por else {'x_1': x_1}


def extract_bundle(bundle_zip, out_dir):
    paths = {'config': None, 'npz': []}
    with zipfile.ZipFile(bundle_zip) as zf:
        for m in zf.namelist():
            base = os.path.basename(m)
            if base == 'edm_config.json' or m.lower().endswith('.npz'):
                zf.extract(m, out_dir)
                (
                    paths.__setitem__('config', os.path.join(out_dir, m))
                    if base == 'edm_config.json'
                    else paths['npz'].append(os.path.join(out_dir, m))
                )
    assert paths['config'], 'eval_bundle.zip has no edm_config.json'
    return paths
