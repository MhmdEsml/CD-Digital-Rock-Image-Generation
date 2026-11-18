import math

import numpy as np
import jax.numpy as jnp

from porous.core.nn import Ctx


def karras_grid(n, smin, smax, rho):
    ramp = np.linspace(0.0, 1.0, int(n), dtype=np.float64)
    return (smin ** (1 / rho) + ramp * (smax ** (1 / rho) - smin ** (1 / rho))) ** rho


def parse_schedule(spec):
    if isinstance(spec, dict):
        return tuple(float(t) for t in spec.get('taus', [])), float(spec.get('gamma', 1.0))
    return tuple(float(t) for t in spec), 1.0


def snap_taus_to_grid(taus, cd):
    grid = karras_grid(int(cd['num_scales']), float(cd['sigma_min']), float(cd['sigma_max']), float(cd['rho']))
    out = []
    for t in taus:
        g = float(grid[int(np.argmin(np.abs(np.log(grid) - math.log(float(t)))))])
        if g not in out:
            out.append(g)
    out = tuple(out)
    key = (tuple(float(t) for t in taus), out)
    if out != tuple(float(t) for t in taus) and key not in _SNAP_ANNOUNCED:
        _SNAP_ANNOUNCED.add(key)
        print(f'[sampler] schedule sigmas {list(taus)} snapped to the training grid -> {[round(g, 4) for g in out]}')
    return out


def unet_F(unet, params, x, c_noise, extra=None, ctx=None):
    if ctx is None:
        ctx = Ctx(training=False)
    if c_noise.ndim == 0:
        c_noise = jnp.broadcast_to(c_noise, (x.shape[0],))
    return unet.apply(params['net'], x, c_noise, extra if extra is not None else {}, ctx)


def cm_coeffs(sigma, sigma_data, smin):
    s2 = sigma * sigma
    c_skip = sigma_data**2 / ((sigma - smin) ** 2 + sigma_data**2)
    c_out = sigma_data * (sigma - smin) / jnp.sqrt(s2 + sigma_data**2)
    c_in = 1.0 / jnp.sqrt(s2 + sigma_data**2)
    c_noise = jnp.log(sigma) / 4.0
    return c_skip, c_out, c_in, c_noise


def _bc(v, ndim):
    return v.reshape(v.shape + (1,) * (ndim - v.ndim))


def cm_denoise(unet, params, x, sigma, extra, ctx, sigma_data, smin):
    x = x.astype(jnp.float32)
    c_skip, c_out, c_in, c_noise = cm_coeffs(sigma, sigma_data, smin)
    x_in = _bc(c_in, x.ndim) * x
    if ctx.amp_dtype is not None:
        x_in = x_in.astype(ctx.amp_dtype)
    F = unet_F(unet, params, x_in, c_noise, extra, ctx).astype(jnp.float32)
    return _bc(c_skip, x.ndim) * x + _bc(c_out, x.ndim) * F


_SNAP_ANNOUNCED = set()
