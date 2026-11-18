import numpy as np
import jax
import jax.numpy as jnp

from porous.core.edm import ModelWrapper
from porous.core.nn import Ctx
from porous.edm.model import ref_velocity_net


class EMAOutputWrapper(ModelWrapper):

    def __init__(self, model, mean_channels, fraction=None, porosity_cfg_scale=1.0):
        super().__init__(model)
        self.mean_channels = mean_channels
        self.fraction = fraction
        self.porosity_cfg_scale = float(porosity_cfg_scale)

    def _extra(self, use_porosity):
        return {"fraction": self.fraction} if (use_porosity and self.fraction is not None) else {}

    def forward(self, x, t, **extras):
        if t.ndim == 0:
            t = jnp.broadcast_to(t, (x.shape[0],))
        s = self.porosity_cfg_scale if self.fraction is not None else 1.0
        run = lambda use_porosity: self.model(x, t, extra=self._extra(use_porosity))
        if s == 1.0:
            return run(True)
        if s == 0.0:
            return run(False)
        o_c, o_u = run(True), run(False)
        return o_u + s * (o_c - o_u)

    def __call__(self, x, t, **extras):
        return self.forward(x, t, **extras)


_SAMPLER_CACHE = {}


def _get_sampler(
    unet, diffusion, image_channels, num_steps, sampler, rho, churn, clip, fs, sigma_min=None, sigma_max=None, sigma_start=None
):
    ck = (id(unet), id(diffusion), image_channels, num_steps, sampler, rho, churn, clip, fs, sigma_min, sigma_max, sigma_start)
    if ck not in _SAMPLER_CACHE:

        def sample_fn(params, x_init, key, fraction, x_clean=None):
            ctx = Ctx(training=False)

            def denoiser(x, sigma, extra=None):
                return diffusion.denoise(lambda xi, cn: ref_velocity_net(unet, params, xi, cn, extra, ctx), x, sigma, clip=False)

            wrapped = EMAOutputWrapper(denoiser, image_channels, fraction=fraction, porosity_cfg_scale=fs)
            S_churn, S_tmin, S_tmax, S_noise = churn
            return diffusion.sample(
                wrapped,
                x_init,
                key,
                num_steps=num_steps,
                sampler=sampler,
                rho=rho,
                S_churn=S_churn,
                S_tmin=S_tmin,
                S_tmax=S_tmax,
                S_noise=S_noise,
                clip_denoised=clip,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                x_clean=x_clean,
                sigma_start=sigma_start,
            )

        _SAMPLER_CACHE[ck] = jax.jit(sample_fn)
    return _SAMPLER_CACHE[ck]


def _spatial(patch, dims=3):
    return (int(patch),) * dims if np.isscalar(patch) else tuple(int(v) for v in patch)


def edm_sample(
    unet,
    diffusion,
    params,
    image_channels,
    patch,
    sampling_cfg,
    num_samples=8,
    num_steps=None,
    sampler=None,
    rho=None,
    S_churn=None,
    S_tmin=None,
    S_tmax=None,
    S_noise=None,
    clip_denoised=None,
    porosity=None,
    porosity_cfg_scale=None,
    key=None,
    sigma_min=None,
    sigma_max=None,
    x_clean=None,
    sigma_start=None,
):
    g = lambda v, name, default: v if v is not None else sampling_cfg.get(name, default)
    spatial = _spatial(patch, unet.dims)
    num_steps = int(g(num_steps, 'num_steps', 32))
    sampler = g(sampler, 'sampler', 'heun')
    rho = float(g(rho, 'rho', diffusion.rho))
    churn = (
        float(g(S_churn, 'S_churn', 0.0)),
        float(g(S_tmin, 'S_tmin', 0.0)),
        float(g(S_tmax, 'S_tmax', float('inf'))),
        float(g(S_noise, 'S_noise', 1.0)),
    )
    clip = bool(g(clip_denoised, 'clip_denoised', diffusion.clip_denoised))
    sigma_min = float(g(sigma_min, 'sigma_min', diffusion.sigma_min))
    sigma_max = float(g(sigma_max, 'sigma_max', diffusion.sigma_max))
    sigma_start = None if x_clean is None else (float(sigma_start) if sigma_start is not None else None)
    if x_clean is not None:
        if sigma_start is None:
            raise ValueError('x_clean needs sigma_start')
        x_clean = jnp.asarray(x_clean, dtype=jnp.float32)
        assert x_clean.shape == (num_samples, image_channels) + spatial, x_clean.shape
    fs = float(porosity_cfg_scale if porosity_cfg_scale is not None else sampling_cfg.get('porosity_cfg_scale', 1.0))
    if porosity is not None:
        porosity = jnp.asarray(porosity, dtype=jnp.float32).reshape(num_samples, 1)
    else:
        fs = 1.0
    if key is None:
        key = jax.random.PRNGKey(int(np.random.randint(0, 2**31 - 1)))
    k_init, k_sample = jax.random.split(key)
    x_init = jax.random.normal(k_init, (num_samples, image_channels) + spatial, dtype=jnp.float32)
    fn = _get_sampler(unet, diffusion, image_channels, num_steps, sampler, rho, churn, clip, fs, sigma_min, sigma_max, sigma_start)
    return fn(params, x_init, k_sample, porosity, x_clean)


def edm_sample_many(
    unet, diffusion, params, image_channels, patch, sampling_cfg, num_samples, porosity=None, x_clean=None, key=None, chunk=None, **kw
):
    chunk = int(chunk or sampling_cfg.get('max_batch', 8))
    key = jax.random.PRNGKey(int(np.random.randint(0, 2**31 - 1))) if key is None else key
    por = None if porosity is None else np.asarray(porosity, np.float32).reshape(num_samples, 1)
    xc = None if x_clean is None else np.asarray(x_clean, np.float32)
    outs = []
    for c, i in enumerate(range(0, num_samples, chunk)):
        j = min(i + chunk, num_samples)
        idx = np.minimum(np.arange(i, i + chunk), num_samples - 1)
        o = edm_sample(
            unet,
            diffusion,
            params,
            image_channels,
            patch,
            sampling_cfg,
            num_samples=chunk,
            porosity=None if por is None else por[idx],
            x_clean=None if xc is None else xc[idx],
            key=jax.random.fold_in(key, c),
            **kw
        )
        outs.append(np.asarray(o)[: j - i])
    return np.concatenate(outs, axis=0)
