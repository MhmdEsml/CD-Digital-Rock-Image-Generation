import math

import numpy as np
import jax
import jax.numpy as jnp

from porous.core.nn import Ctx
from porous.cd.config import amp_dtype
from porous.cd.teacher import EMAOutputWrapper
from porous.core.consistency_math import unet_F
from porous.core.consistency_math import cm_denoise, snap_taus_to_grid


def pair_probs(grid, kind, P_mean=-1.1, P_std=2.0, k=1):
    lo, hi = grid[:-k], grid[k:]
    if kind == 'uniform':
        p = np.ones_like(lo)
    elif kind == 'lognormal':
        erf = np.vectorize(math.erf)
        f = lambda s: erf((np.log(s) - P_mean) / (math.sqrt(2.0) * P_std))
        p = f(hi) - f(lo)
    elif kind == 'uniform_t':
        p = hi / (1.0 + hi) - lo / (1.0 + lo)
    else:
        raise ValueError(kind)
    p = np.maximum(p, 1e-12)
    return p / p.sum()


def pair_weights(grid, probs, kind, sigma_data, k=1):
    lo, hi = grid[:-k], grid[k:]
    if kind == 'none':
        return np.ones_like(lo)
    if kind == 'inv_delta':
        w = 1.0 / (hi - lo)
    elif kind == 'edm':
        w = (lo**2 + sigma_data**2) / (lo * sigma_data) ** 2
    else:
        raise ValueError(kind)
    return w / float(np.sum(probs * w))


def cd_distance(a, b, kind, huber_c):
    ax = tuple(range(1, a.ndim))
    if kind == 'l1':
        return jnp.mean(jnp.abs(a - b), axis=ax)
    d2 = jnp.mean((a - b) ** 2, axis=ax)
    if kind == 'l2':
        return d2
    return jnp.sqrt(d2 + huber_c**2) - huber_c


def make_extra(d):
    e = {}
    if d.get('por') is not None:
        e['fraction'] = d['por']
    return e


def make_teacher_denoiser(unet, params, diffusion, amp=None):
    def denoiser(x, sigma, extra=None):
        ctx = Ctx(training=False, amp_dtype=amp)
        cast = (lambda a: a) if amp is None else (lambda a: a.astype(amp))
        return diffusion.denoise(lambda xi, cn: unet_F(unet, params, cast(xi), cn, extra, ctx), x, sigma, clip=False)

    return denoiser


def make_teacher_wrapper(TEACHER, CDCFG, t_params, por, amp=None):
    por = por if TEACHER['has_por'] else None
    return EMAOutputWrapper(
        make_teacher_denoiser(TEACHER['unet'], t_params, TEACHER['diffusion'], amp),
        TEACHER['image_channels'],
        porosity=por,
        cfg_scale=float(CDCFG['teacher']['porosity_cfg_scale']),
    )


def cast_tree(tree, dtype_name):
    if dtype_name == 'float32':
        return tree
    return jax.tree_util.tree_map(lambda a: np.asarray(jnp.asarray(a, jnp.bfloat16)), tree)


_SAMPLER_CACHE = {}


def student_ctx(CDCFG, training=False, key=None):
    return Ctx(training=training, amp_dtype=amp_dtype(CDCFG['student']['compute_dtype']), key=key)


def get_cm_sampler(TEACHER, CDCFG, taus, gamma=1.0):
    cd = CDCFG['consistency']
    smin, smax = float(cd['sigma_min']), float(cd['sigma_max'])
    sd = TEACHER['diffusion'].sigma_data
    taus = snap_taus_to_grid(taus, cd)
    gamma = float(gamma)
    ck = ('cm', id(TEACHER['student_unet']), taus, gamma, smin, smax, sd, CDCFG['student']['compute_dtype'])
    if ck not in _SAMPLER_CACHE:
        unet = TEACHER['student_unet']

        def fn(params, z, key, por):
            extra = make_extra({'por': por if TEACHER['has_por'] else None})
            ctx = student_ctx(CDCFG)
            n = z.shape[0]
            x0 = cm_denoise(unet, params, z.astype(jnp.float32) * smax, jnp.full((n,), smax, jnp.float32), extra, ctx, sd, smin)
            x, s_cur = z.astype(jnp.float32) * smax, smax
            for t in taus:
                key, k = jax.random.split(key)
                s = max((1.0 - gamma) * t, smin)
                x_s = x0 + (s / s_cur) * (x - x0)
                x = x_s + math.sqrt(max(t * t - s * s, 0.0)) * jax.random.normal(k, x0.shape, jnp.float32)
                s_cur = t
                x0 = cm_denoise(unet, params, x, jnp.full((n,), t, jnp.float32), extra, ctx, sd, smin)
            return x0

        _SAMPLER_CACHE[ck] = jax.jit(fn)
    return _SAMPLER_CACHE[ck]


def sample_in_chunks(fn, params, z, por, key, batch_size):
    n, bs, out = z.shape[0], int(batch_size), []
    params = jax.device_put(params)
    for i in range(0, n, bs):
        zc = z[i : i + bs]
        pc = None if por is None else por[i : i + bs]
        m = zc.shape[0]
        if m < bs:
            zc = np.concatenate([zc, np.repeat(zc[-1:], bs - m, 0)], 0)
            pc = None if pc is None else np.concatenate([pc, np.repeat(pc[-1:], bs - m, 0)], 0)
        out.append(np.asarray(fn(params, zc, jax.random.fold_in(key, i // bs), pc))[:m])
    return np.concatenate(out, 0)
