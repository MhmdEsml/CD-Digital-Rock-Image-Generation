import json
import os

import numpy as np

from porous.edm.sampler import edm_sample_many


def binarise_pore(x):
    x = np.asarray(x)
    if x.ndim >= 3 and x.shape[1] == 1:
        x = x[:, 0]
    return x < 0.0


def porosity_of(pore):
    pore = np.asarray(pore)
    return pore.reshape(pore.shape[0], -1).mean(axis=1)


def run_preview(unet, diffusion, ema_params, epoch, global_step, results_dir, cfg, patch_porosity, key, rng, use_porosity=True):
    sc = cfg['sampling']
    n_p, n_g = int(sc.get('preview_n', 8)), (int(sc.get('preview_grid_n', 8)) if use_porosity else 0)
    pp = np.asarray(patch_porosity, np.float64)
    tgt_real = pp[rng.choice(len(pp), size=n_p, replace=n_p > len(pp))]
    lo, hi = np.percentile(pp, [5, 95])
    grid = np.linspace(lo, hi, n_g) if n_g else np.zeros(0)
    por_all = np.concatenate([tgt_real, grid]) if use_porosity else None
    x = edm_sample_many(
        unet, diffusion, ema_params, cfg['image_channels'], int(cfg['data']['patch_size']), sc, n_p + n_g, porosity=por_all, key=key
    )
    pore = binarise_pore(x)
    ach = porosity_of(pore)
    rec = {'epoch': int(epoch), 'global_step': int(global_step)}
    if use_porosity:
        rec['porosity_err'] = float(np.abs(ach[:n_p] - tgt_real).mean())
        rec['porosity_bias'] = float((ach[:n_p] - tgt_real).mean())
        if n_g:
            rec['grid_porosity_err'] = float(np.abs(ach[n_p:] - grid).mean())
            rec['grid_porosity_bias'] = float((ach[n_p:] - grid).mean())
        rec['target_porosity'] = [float(v) for v in por_all]
    else:
        rec['porosity_mean_gen'] = float(ach.mean())
    rec['achieved_porosity'] = [float(v) for v in ach]
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, f'samples_epoch_{epoch:04d}.npy'), pore.astype(np.uint8))
    path = os.path.join(results_dir, 'preview_log.json')
    records = []
    if os.path.exists(path):
        try:
            with open(path) as f:
                records = json.load(f)
        except Exception as e:
            print('[preview] could not read the existing log, starting a new one:', repr(e))
    records = [r for r in records if r.get('epoch') != rec['epoch']] + [rec]
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(records, f, indent=1)
    os.replace(tmp, path)
    print('[preview] ' + ' | '.join(f'{k}={v:.4g}' for k, v in rec.items() if isinstance(v, float)))
    return rec
