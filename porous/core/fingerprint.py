import numpy as np
import jax
import jax.numpy as jnp

from porous.core.nn import Ctx

FP_POROSITIES = (0.05, 0.15, 0.30)


FP_SPATIAL = (16, 16, 16)


FP_SEED = 0


FP_C_NOISE = (-1.0, 0.0, 1.0)


FP_N_VALUES = 64


FP_RTOL_NORM = 1e-3


FP_ATOL_NORM = 1e-6


FP_ATOL_ELEM, FP_RTOL_ELEM = 1e-3, 1e-3


def porosity_fingerprint(unet, net_params, porosities=FP_POROSITIES):
    if not getattr(unet, 'fraction_dim', 0):
        return None
    ctx = Ctx(training=False)
    por = jnp.asarray(list(porosities), jnp.float32)[:, None]
    with jax.default_matmul_precision('highest'):
        emb = np.asarray(unet.porosity_term(net_params, por, len(porosities), jnp.float32, ctx), np.float64)
        null = np.asarray(unet.porosity_term(net_params, None, 1, jnp.float32, ctx), np.float64)[0]
    return {'porosities': [float(p) for p in porosities], 'embedding': emb.tolist(), 'null': null.tolist()}


def network_fingerprint(unet, net_params, shape=None, c_noise=FP_C_NOISE, porosities=FP_POROSITIES, seed=FP_SEED, n_values=FP_N_VALUES):
    shape = tuple(shape) if shape is not None else (1, int(unet.in_channels)) + FP_SPATIAL
    ctx = Ctx(training=False)
    x = jnp.asarray(np.random.RandomState(int(seed)).standard_normal(shape), jnp.float32)
    plist = list(porosities) if getattr(unet, 'fraction_dim', 0) else []
    rows = []
    with jax.default_matmul_precision('highest'):
        for cn in c_noise:
            cnv = jnp.full((shape[0],), float(cn), jnp.float32)
            for p in plist + [None]:
                extra = {} if p is None else {'fraction': jnp.full((shape[0], 1), float(p), jnp.float32)}
                f = np.asarray(unet.apply(net_params, x, cnv, extra, ctx), np.float64).reshape(-1)
                rows.append(f[:: max(f.size // int(n_values), 1)][: int(n_values)].tolist())
    return {
        'shape': list(shape),
        'seed': int(seed),
        'c_noise': [float(c) for c in c_noise],
        'porosities': [float(p) for p in plist],
        'n_values': int(n_values),
        'forward': rows,
    }


def fingerprints_for(unet, net_params):
    out = {'porosity_fingerprint': porosity_fingerprint(unet, net_params), 'network_fingerprint': network_fingerprint(unet, net_params)}
    if float(np.linalg.norm(np.asarray(out['network_fingerprint']['forward'], np.float64))) < 1e-8:
        raise ValueError(
            'the network output on the fingerprint input is identically zero (an untrained / zero-output network): its fingerprint would be meaningless. Use trained weights.'
        )
    return out


def fp_compare(ref, got):
    a, b = np.asarray(ref, np.float64), np.asarray(got, np.float64)
    if a.shape != b.shape:
        return dict(norm_rel=float('inf'), max_abs=float('inf'), ok=False, why=f'shape {a.shape} vs {b.shape}')
    d = np.abs(a - b)
    na = float(np.linalg.norm(a))
    ok_norm = float(np.linalg.norm(a - b)) <= FP_RTOL_NORM * na + FP_ATOL_NORM
    ok_elem = bool(np.all(d <= FP_ATOL_ELEM + FP_RTOL_ELEM * np.abs(a)))
    return dict(
        norm_rel=float(np.linalg.norm(a - b)) / max(na, 1e-30), max_abs=float(d.max()) if d.size else 0.0, ok=bool(ok_norm and ok_elem)
    )
