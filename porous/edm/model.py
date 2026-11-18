import jax
import jax.numpy as jnp

from porous.core.nn import Ctx, KeyGen, flatten_tree
from porous.core.unet import count_params, make_unet, unet_level_widths


def ref_velocity_net(unet, params, x, t, extra=None, ctx=None, **extras):
    if ctx is None:
        ctx = Ctx(training=False)
    if t.ndim == 0:
        t = jnp.broadcast_to(t, (x.shape[0],))
    return unet.apply(params['net'], x, t, extra if extra is not None else {}, ctx)


def estimate_activation_memory(arch, patch, bytes_per_value=4, use_checkpoint=None):
    use_checkpoint = arch.get('use_checkpoint', False) if use_checkpoint is None else use_checkpoint
    mc, mult, nrb = arch['model_channels'], arch['channel_mult'], arch['num_res_blocks']
    widths = [int(m * mc) for m in mult]
    blocks = []
    skips = [patch**3 * widths[0]]
    for lvl, w in enumerate(widths):
        vox = (patch // 2**lvl) ** 3
        cin = widths[max(lvl - 1, 0)] if lvl else widths[0]
        for k in range(nrb):
            blocks.append((vox, cin if k == 0 else w, w))
            skips.append(vox * w)
        if lvl < len(widths) - 1:
            skips.append(vox // 8 * w)
    vox = (patch // 2 ** (len(widths) - 1)) ** 3
    blocks += [(vox, widths[-1], widths[-1])] * 2
    for lvl in range(len(widths) - 1, -1, -1):
        vox = (patch // 2**lvl) ** 3
        for k in range(nrb + 1):
            cin = widths[lvl] + widths[lvl]
            blocks.append((vox, cin, widths[lvl]))
            skips.append(vox * widths[lvl])
    full = [v * (3 * ci + 6 * co) for v, ci, co in blocks]
    kept = sum(v * ci for v, ci, co in blocks) if use_checkpoint else sum(full)
    peak = max(full) if use_checkpoint else 0
    return (kept + peak + sum(skips)) * bytes_per_value


def build_model(CFG, verbose=True):
    ref_arch = CFG['architecture']
    PATCH = int(CFG['data']['patch_size'])
    ref_unet = make_unet(ref_arch)
    ref_params = {'net': ref_unet.init(KeyGen(jax.random.PRNGKey(CFG['training']['seed'])))}
    param_order = list(flatten_tree(ref_params))
    n_params = count_params(ref_params)
    if verbose:
        ema_decays = CFG['training'].get('ema_decays') or [CFG['training']['ema_decay']]
        print(
            f'UNetModel parameter count: {n_params:,}  ({n_params * 4 / 1e6:.1f} MB float32; AdamW state + EMA tracks ~ {n_params * 4 * (3 + len(ema_decays)) / 1e6:.0f} MB)'
        )
        _w = unet_level_widths(ref_arch['model_channels'], ref_arch['channel_mult'])
        print(
            'levels (resolution: width):',
            ', '.join(f"{PATCH // 2 ** i}^3: {w}" for i, w in enumerate(_w)),
            f"| porosity embedding: {ref_arch['porosity_embedding']} | porosity_dropout {ref_arch['porosity_dropout']}",
        )
        for _name, _ds, _res, _tok in ref_unet.attention_report((PATCH,) * 3):
            print(
                f"attention layer {_name}: downsample x{_ds}, {'x'.join(map(str, _res))} = {_tok} tokens (limit {ref_arch['max_attention_tokens']})"
            )
        for _ck in (False, True):
            print(
                f"activation memory estimate per sample (float32, use_checkpoint={_ck}): {estimate_activation_memory(ref_arch, PATCH, 4, _ck) / 1e9:.2f} GB"
                f"  -> per-device batch {CFG['training']['batch_size']}: {CFG['training']['batch_size'] * estimate_activation_memory(ref_arch, PATCH, 4, _ck) / 1e9:.2f} GB  (rough, +/- 2x)"
            )
    return ref_unet, ref_params, param_order, n_params
