import math
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax


class ModelWrapper:

    def __init__(self, model):
        self.model = model

    def forward(self, x, t, **extras):
        return self.model(x=x, t=t, **extras)

    def __call__(self, x, t, **extras):
        return self.forward(x, t, **extras)


LOSS_TYPES = ('mse', 'l1', 'huber')
LOSS_WEIGHTINGS = ('edm', 'uniform')
SAMPLERS = ('heun', 'euler')
SIGMA_DISTS = ('lognormal', 'uniform_t', 'uniform_log_sigma')

DEFAULT_DIFFUSION_CFG = dict(
    sigma_data=0.5,
    P_mean=-1.2,
    P_std=1.2,
    sigma_min=0.002,
    sigma_max=80.0,
    rho=7.0,
    loss_type='mse',
    loss_weighting='edm',
    huber_delta=1.0,
    clip_denoised=False,
    sigma_dist='lognormal',
    sigma_train_min=0.002,
    sigma_train_max=80.0,
)

EDM_STRUCTURAL_KEYS = ('sigma_data',)


def sigma_bin_index(sigma, lo, hi, n_bins):
    ls = jnp.log(sigma)
    l0, l1 = math.log(lo), math.log(hi)
    idx = jnp.clip(jnp.floor((ls - l0) / (l1 - l0) * n_bins).astype(jnp.int32) + 1, 1, n_bins)
    eps = 1e-4
    return jnp.where(ls < l0 - eps, 0, jnp.where(ls > l1 + eps, n_bins + 1, idx)).astype(jnp.int32)


def sigma_bin_edges(lo, hi, n_bins):
    return np.exp(np.linspace(math.log(lo), math.log(hi), int(n_bins) + 1))


def _mean_flat(x):
    return jnp.mean(x, axis=tuple(range(1, x.ndim)))


class EDM:

    model_out_mult = 1

    def __init__(
        self,
        sigma_data=0.5,
        P_mean=-1.2,
        P_std=1.2,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
        loss_type='mse',
        loss_weighting='edm',
        huber_delta=1.0,
        clip_denoised=False,
        sigma_dist='lognormal',
        sigma_train_min=0.002,
        sigma_train_max=80.0,
        **_ignored,
    ):
        if isinstance(sigma_data, str):
            raise ValueError(
                f"sigma_data={sigma_data!r} is not resolved: 'auto' must be turned into a number (resolve_sigma_data(CFG['diffusion'], data_rms), "
                "done by the data-cache cell and by run_training) BEFORE the diffusion is built"
            )
        assert loss_type in LOSS_TYPES, loss_type
        assert sigma_dist in SIGMA_DISTS, f'sigma_dist must be one of {SIGMA_DISTS}, got {sigma_dist!r}'
        assert 0 < sigma_train_min < sigma_train_max, (sigma_train_min, sigma_train_max)
        assert loss_weighting in LOSS_WEIGHTINGS, loss_weighting
        assert sigma_data > 0 and P_std > 0 and 0 < sigma_min < sigma_max and rho > 0
        self.sigma_data, self.P_mean, self.P_std = float(sigma_data), float(P_mean), float(P_std)
        self.sigma_min, self.sigma_max, self.rho = float(sigma_min), float(sigma_max), float(rho)
        self.loss_type, self.loss_weighting, self.huber_delta = loss_type, loss_weighting, float(huber_delta)
        self.clip_denoised = bool(clip_denoised)
        self.sigma_dist = sigma_dist
        self.sigma_train_min, self.sigma_train_max = float(sigma_train_min), float(sigma_train_max)
        self.t_lo = self.sigma_train_min / (1.0 + self.sigma_train_min)
        self.t_hi = self.sigma_train_max / (1.0 + self.sigma_train_max)

    @staticmethod
    def _bc(v, ndim):
        return v.reshape(v.shape + (1,) * (ndim - v.ndim))

    def _sig(self, sigma, n):
        sigma = jnp.asarray(sigma, jnp.float32)
        return jnp.broadcast_to(sigma, (n,)) if sigma.ndim == 0 else sigma

    def coeffs(self, sigma):
        sd = self.sigma_data
        s2 = sigma * sigma
        c_skip = sd**2 / (s2 + sd**2)
        c_out = sigma * sd / jnp.sqrt(s2 + sd**2)
        c_in = 1.0 / jnp.sqrt(s2 + sd**2)
        c_noise = jnp.log(sigma) / 4.0
        return c_skip, c_out, c_in, c_noise

    def loss_weight(self, sigma):
        if self.loss_weighting == 'uniform':
            return jnp.ones_like(sigma)
        sd = self.sigma_data
        return (sigma * sigma + sd**2) / (sigma * sd) ** 2

    def denoise(self, net_fn, x, sigma, clip=None):
        sigma = self._sig(sigma, x.shape[0])
        c_skip, c_out, c_in, c_noise = self.coeffs(sigma)
        x = x.astype(jnp.float32)
        F = net_fn((self._bc(c_in, x.ndim) * x), c_noise).astype(jnp.float32)
        D = self._bc(c_skip, x.ndim) * x + self._bc(c_out, x.ndim) * F
        if self.clip_denoised if clip is None else clip:
            D = jnp.clip(D, -1.0, 1.0)
        return D

    def sample_sigma(self, key, n):
        if self.sigma_dist == 'lognormal':
            return jnp.exp(self.P_mean + self.P_std * jax.random.normal(key, (n,), dtype=jnp.float32))
        u = jax.random.uniform(key, (n,), dtype=jnp.float32)
        if self.sigma_dist == 'uniform_t':
            t = jnp.float32(self.t_lo) + jnp.float32(self.t_hi - self.t_lo) * u
            sigma = t / (1.0 - t)
        else:
            lo, hi = math.log(self.sigma_train_min), math.log(self.sigma_train_max)
            sigma = jnp.exp(jnp.float32(lo) + jnp.float32(hi - lo) * u)
        return jnp.clip(sigma, self.sigma_train_min, self.sigma_train_max)

    def sigma_cdf(self, s):
        s = np.asarray(s, np.float64)
        ls = np.log(s)
        if self.sigma_dist == 'lognormal':
            return 0.5 * (1.0 + np.vectorize(math.erf)((ls - self.P_mean) / (self.P_std * math.sqrt(2.0))))
        if self.sigma_dist == 'uniform_t':
            return np.clip((s / (1.0 + s) - self.t_lo) / (self.t_hi - self.t_lo), 0.0, 1.0)
        lo, hi = math.log(self.sigma_train_min), math.log(self.sigma_train_max)
        return np.clip((ls - lo) / (hi - lo), 0.0, 1.0)

    def sigma_pdf(self, s):
        s = np.asarray(s, np.float64)
        inside = (s >= self.sigma_train_min) & (s <= self.sigma_train_max)
        if self.sigma_dist == 'lognormal':
            return np.exp(-0.5 * ((np.log(s) - self.P_mean) / self.P_std) ** 2) / (s * self.P_std * math.sqrt(2.0 * math.pi))
        if self.sigma_dist == 'uniform_t':
            return np.where(inside, 1.0 / ((1.0 + s) ** 2 * (self.t_hi - self.t_lo)), 0.0)
        return np.where(inside, 1.0 / (s * (math.log(self.sigma_train_max) - math.log(self.sigma_train_min))), 0.0)

    def training_losses(self, apply_fn, x_start, sigma, noise):
        x_start = x_start.astype(jnp.float32)
        sig = self._bc(self._sig(sigma, x_start.shape[0]), x_start.ndim)
        x_noisy = x_start + sig * noise.astype(jnp.float32)
        D = self.denoise(lambda xi, cn: apply_fn(xi, cn), x_noisy, sigma, clip=False)
        e = jnp.sqrt(self._bc(self.loss_weight(self._sig(sigma, x_start.shape[0])), x_start.ndim)) * (D - x_start)
        if self.loss_type == 'mse':
            per = e * e
        elif self.loss_type == 'l1':
            per = jnp.abs(e)
        else:
            d = self.huber_delta
            a = jnp.abs(e)
            per = jnp.where(a <= d, 0.5 * e * e, d * (a - 0.5 * d)) * 2.0
        return _mean_flat(per), D

    def sigma_schedule(self, num_steps, rho=None, sigma_min=None, sigma_max=None):
        rho = self.rho if rho is None else float(rho)
        smin = self.sigma_min if sigma_min is None else float(sigma_min)
        smax = self.sigma_max if sigma_max is None else float(sigma_max)
        ramp = np.linspace(0.0, 1.0, int(num_steps), dtype=np.float64)
        sig = (smax ** (1 / rho) + ramp * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho
        return np.concatenate([sig, [0.0]])

    def sample(
        self,
        model_fn,
        x_init,
        key,
        num_steps=32,
        sampler='heun',
        rho=None,
        sigma_min=None,
        sigma_max=None,
        S_churn=0.0,
        S_tmin=0.0,
        S_tmax=float('inf'),
        S_noise=1.0,
        clip_denoised=None,
        x_clean=None,
        sigma_start=None,
    ):
        assert sampler in SAMPLERS, sampler
        N = int(num_steps)
        n = x_init.shape[0]
        if x_clean is not None:
            if sigma_start is None:
                raise ValueError('x_clean needs sigma_start (the noise level the clean image is re-noised to)')
            sigma_max = float(sigma_start)
            if not sigma_max > (self.sigma_min if sigma_min is None else float(sigma_min)):
                raise ValueError(f'sigma_start={sigma_max} must be larger than sigma_min')
        sig = self.sigma_schedule(N, rho, sigma_min, sigma_max)
        gam = np.where((sig[:-1] >= S_tmin) & (sig[:-1] <= S_tmax), min(S_churn / N, math.sqrt(2.0) - 1.0), 0.0)
        clip = self.clip_denoised if clip_denoised is None else bool(clip_denoised)
        D = lambda x, s: model_fn(x, jnp.full((n,), s, jnp.float32))
        if clip:
            D_raw = D
            D = lambda x, s: jnp.clip(D_raw(x, s), -1.0, 1.0)

        def churn(x, t_cur, g, k):
            t_hat = t_cur * (1.0 + g)
            z = S_noise * jax.random.normal(k, x.shape, dtype=x.dtype)
            return x + jnp.sqrt(jnp.maximum(t_hat * t_hat - t_cur * t_cur, 0.0)) * z, t_hat

        keys = jax.random.split(key, N)
        if x_clean is None:
            x = x_init.astype(jnp.float32) * jnp.float32(sig[0])
        else:
            x = x_clean.astype(jnp.float32) + jnp.float32(sig[0]) * x_init.astype(jnp.float32)

        def body(x, inp):
            t_cur, t_next, g, k = inp
            x_hat, t_hat = churn(x, t_cur, g, k)
            d = (x_hat - D(x_hat, t_hat)) / t_hat
            x_new = x_hat + (t_next - t_hat) * d
            if sampler == 'heun':
                d2 = (x_new - D(x_new, t_next)) / t_next
                x_new = x_hat + (t_next - t_hat) * (0.5 * d + 0.5 * d2)
            return x_new, None

        if N > 1:
            f32 = lambda a: jnp.asarray(a, jnp.float32)
            x, _ = lax.scan(body, x, (f32(sig[: N - 1]), f32(sig[1:N]), f32(gam[: N - 1]), keys[: N - 1]))
        x_hat, t_hat = churn(x, jnp.float32(sig[N - 1]), jnp.float32(gam[N - 1]), keys[N - 1])
        return D(x_hat, t_hat)

    def describe(self):
        return (
            f"EDM(sigma_data={self.sigma_data}, ln sigma ~ N({self.P_mean}, {self.P_std}^2), sigma in [{self.sigma_min}, {self.sigma_max}], "
            f"rho={self.rho}, loss={self.loss_type}/{self.loss_weighting}, clip_denoised={self.clip_denoised}, sigma_dist={self.sigma_dist}"
            + (f", train range [{self.sigma_train_min}, {self.sigma_train_max}])" if self.sigma_dist != 'lognormal' else ")")
        )


def build_diffusion(dcfg):
    cfg = {**DEFAULT_DIFFUSION_CFG, **{k: v for k, v in dict(dcfg or {}).items() if k in DEFAULT_DIFFUSION_CFG}}
    if isinstance(cfg['sigma_data'], str):
        raise ValueError(
            f"diffusion.sigma_data={cfg['sigma_data']!r} is still unresolved: resolve 'auto' (resolve_sigma_data) before building the diffusion"
        )
    return EDM(**cfg)


device = jax.default_backend()
n_gpus = jax.local_device_count() if device != 'cpu' else 0
