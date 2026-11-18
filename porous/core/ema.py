import numpy as np
import jax
import jax.numpy as jnp


def ema_init(params):
    return {
        'num_updates': np.int32(0),
        'shadow_params': jax.tree_util.tree_map(lambda p: np.array(p, copy=True), params),
    }


def ema_update(shadow_params, params, num_updates, decay):
    num_updates = num_updates + 1
    decay = jnp.minimum(decay, (1.0 + num_updates) / (10.0 + num_updates))
    shadow_params = jax.tree_util.tree_map(lambda shadow, param: shadow - (1.0 - decay) * (shadow - param), shadow_params, params)
    return shadow_params, num_updates


def resolve_ema_decays(train_cfg):
    ds = train_cfg.get('ema_decays') or [train_cfg['ema_decay']]
    ds = [float(d) for d in ds]
    if len(set(ds)) != len(ds) or not all(0.0 < d < 1.0 for d in ds):
        raise ValueError(f'ema_decays must be distinct values in (0, 1), got {ds}')
    return ds


def resolve_ema_index(decays, which=None):
    if which is None:
        return 0
    for j, d in enumerate(decays):
        if abs(d - float(which)) <= 1e-12:
            return j
    raise ValueError(f'ema_for_sampling={which} is not one of ema_decays={decays}')


def ema_init_tracks(params, decays, sel=0):
    tracks = [jax.tree_util.tree_map(lambda p: np.array(p, copy=True), params) for _ in decays]
    return {'num_updates': np.int32(0), 'shadow_params': tracks[sel], 'tracks': tracks, 'decays': [float(d) for d in decays]}
