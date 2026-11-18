import math

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(it, **kw):
        return it


from porous.core.nn import Ctx
from porous.core.consistency_math import snap_taus_to_grid
from porous.core.consistency_math import _bc, cm_denoise, unet_F


def build_edm_sample_fn(unet, info, spec):
    d = info["diffusion"]
    sd, smin, smax, rho = info["sigma_data"], float(d.get("sigma_min", 0.002)), float(d.get("sigma_max", 80.0)), float(d.get("rho", 7.0))
    clip, has_por = bool(d.get("clip_denoised", False)), info["has_por"]
    N, sampler, fs = int(spec["steps"]), spec["sampler"], float(spec["cfg_scale"])
    S_churn, S_tmin, S_tmax, S_noise = spec["S_churn"], spec["S_tmin"], spec["S_tmax"], spec["S_noise"]
    ramp = np.linspace(0.0, 1.0, N, dtype=np.float64)
    sig = np.concatenate([(smax ** (1 / rho) + ramp * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho, [0.0]])
    gam = np.where((sig[:-1] >= S_tmin) & (sig[:-1] <= S_tmax), min(S_churn / N, math.sqrt(2.0) - 1.0), 0.0)
    f32 = lambda a: jnp.asarray(a, jnp.float32)

    def fn(params, z, key, por):
        ctx = Ctx(training=False)
        n = z.shape[0]

        def D(x, s):
            sigma = jnp.full((n,), s, jnp.float32)
            s2 = sigma * sigma
            c_skip, c_out = sd**2 / (s2 + sd**2), sigma * sd / jnp.sqrt(s2 + sd**2)
            c_in, c_noise = 1.0 / jnp.sqrt(s2 + sd**2), jnp.log(sigma) / 4.0

            def run(use_por):
                extra = {"fraction": por} if (use_por and has_por) else {}
                F = unet_F(unet, params, _bc(c_in, x.ndim) * x, c_noise, extra, ctx).astype(jnp.float32)
                return _bc(c_skip, x.ndim) * x + _bc(c_out, x.ndim) * F

            if not has_por or fs == 1.0:
                out = run(True)
            elif fs == 0.0:
                out = run(False)
            else:
                o_c, o_u = run(True), run(False)
                out = o_u + fs * (o_c - o_u)
            return jnp.clip(out, -1.0, 1.0) if clip else out

        def churn(x, t_cur, g, k):
            t_hat = t_cur * (1.0 + g)
            zz = S_noise * jax.random.normal(k, x.shape, dtype=x.dtype)
            return x + jnp.sqrt(jnp.maximum(t_hat * t_hat - t_cur * t_cur, 0.0)) * zz, t_hat

        keys = jax.random.split(key, N)
        x = z.astype(jnp.float32) * jnp.float32(sig[0])

        def body(x, inp):
            t_cur, t_next, g, k = inp
            x_hat, t_hat = churn(x, t_cur, g, k)
            dd = (x_hat - D(x_hat, t_hat)) / t_hat
            x_new = x_hat + (t_next - t_hat) * dd
            if sampler == "heun":
                d2 = (x_new - D(x_new, t_next)) / t_next
                x_new = x_hat + (t_next - t_hat) * (0.5 * dd + 0.5 * d2)
            return x_new, None

        if N > 1:
            x, _ = lax.scan(body, x, (f32(sig[: N - 1]), f32(sig[1:N]), f32(gam[: N - 1]), keys[: N - 1]))
        x_hat, t_hat = churn(x, jnp.float32(sig[N - 1]), jnp.float32(gam[N - 1]), keys[N - 1])
        return D(x_hat, t_hat)

    meta = {
        "steps": N,
        "sampler": sampler,
        "cfg_scale": fs,
        "S_churn": float(S_churn),
        "network_calls_per_volume": (2 * N - 1 if sampler == "heun" else N) * (1 if (not has_por or fs in (0.0, 1.0)) else 2),
    }
    return fn, meta


def replicate_for_pmap(tree, n):
    stacked = jax.tree_util.tree_map(lambda x: np.stack([np.asarray(x)] * n), tree)
    return jax.pmap(lambda s: s, axis_name="batch")(stacked)


def to_host(x):
    try:
        shards = sorted(x.addressable_shards, key=lambda s: (s.index[0].start or 0))
        return np.concatenate([np.asarray(s.data) for s in shards], axis=0)
    except AttributeError:
        return np.asarray(x)


def build_sample_fn(unet, info, taus, gamma):
    cd = info["cd"]
    smin, smax = float(cd["sigma_min"]), float(cd["sigma_max"])
    sd, has_por = float(info["sigma_data"]), info["has_por"]
    taus = snap_taus_to_grid(taus, cd)
    gamma = float(gamma)
    amp = None if info["compute_dtype"] == "float32" else jnp.bfloat16

    def fn(params, z, key, por):
        extra = {"fraction": por} if has_por else {}
        ctx = Ctx(training=False, amp_dtype=amp)
        n = z.shape[0]
        x = z.astype(jnp.float32) * smax
        x0 = cm_denoise(unet, params, x, jnp.full((n,), smax, jnp.float32), extra, ctx, sd, smin)
        s_cur = smax
        for t in taus:
            key, k = jax.random.split(key)
            s = max((1.0 - gamma) * t, smin)
            x_s = x0 + (s / s_cur) * (x - x0)
            x = x_s + math.sqrt(max(t * t - s * s, 0.0)) * jax.random.normal(k, x0.shape, jnp.float32)
            s_cur = t
            x0 = cm_denoise(unet, params, x, jnp.full((n,), t, jnp.float32), extra, ctx, sd, smin)
        return x0

    return fn, taus


def sample_all_devices(fn_pmap, params_rep, z, por, key, n_dev, per_dev_bs, desc="sampling"):
    n = z.shape[0]
    chunk = n_dev * per_dev_bs
    out = []
    for ci, i in enumerate(tqdm(range(0, n, chunk), desc=desc, leave=False)):
        zc, pc = z[i : i + chunk], por[i : i + chunk]
        m = zc.shape[0]
        if m < chunk:
            zc = np.concatenate([zc, np.repeat(zc[-1:], chunk - m, 0)], 0)
            pc = np.concatenate([pc, np.repeat(pc[-1:], chunk - m, 0)], 0)
        keys = jax.random.split(jax.random.fold_in(key, ci), n_dev)
        y = fn_pmap(params_rep, zc.reshape((n_dev, per_dev_bs) + zc.shape[1:]), keys, pc.reshape((n_dev, per_dev_bs, 1)))
        y = to_host(y)
        out.append(y.reshape((chunk,) + y.shape[2:])[:m])
    return np.concatenate(out, 0).astype(np.float32)
