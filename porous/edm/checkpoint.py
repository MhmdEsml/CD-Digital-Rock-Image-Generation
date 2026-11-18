import numpy as np
import jax

from porous.core.edm import DEFAULT_DIFFUSION_CFG, EDM_STRUCTURAL_KEYS
from porous.core.nn import flatten_tree
from porous.core.ema import resolve_ema_index
from porous.core.adamw import adamw_init
from porous.core.checkpoint_io import load_state_dict_into


def make_checkpoint_payload(
    epoch, global_step, params, ema, opt, opt_state, lr_schedule, loss_scaler, config, conditioning, facies_values_, param_order
):
    n = len(param_order)
    live = flatten_tree(params)
    tracks = ema.get('tracks')
    shadow = flatten_tree(tracks[0] if tracks else ema['shadow_params'])
    m1, m2 = flatten_tree(opt_state['exp_avg']), flatten_tree(opt_state['exp_avg_sq'])
    assert set(live) == set(param_order) == set(shadow) == set(m1) == set(m2), 'param_order does not match the parameter tree'
    net_sd = {k: live[k] for k in param_order}
    ema_sd = {'num_updates': np.asarray(int(ema['num_updates']), dtype=np.int64)}
    ema_sd.update({f'model.{k}': net_sd[k] for k in param_order})
    ema_sd.update({f'shadow_params.{i}': shadow[k] for i, k in enumerate(param_order)})
    step = np.float32(opt_state['step'])
    opt_sd = {
        'state': {i: {'step': np.array(step), 'exp_avg': m1[k], 'exp_avg_sq': m2[k]} for i, k in enumerate(param_order)},
        'param_groups': [dict(g, params=list(range(n))) for g in opt.param_groups],
    }
    payload = {
        'epoch': epoch,
        'global_step': global_step,
        'net_state_dict': net_sd,
        'ema_state_dict': ema_sd,
        'opt_state_dict': opt_sd,
        'lr_schedule_state_dict': lr_schedule.state_dict(),
        'scaler_state_dict': loss_scaler.state_dict(),
        'config': config,
        'mode': 'unconditional',
        'conditioning': conditioning,
        'facies_values': facies_values_,
        'enable_fraction_conditioning': conditioning == 'porosity',
    }
    if tracks:
        decays_ = [float(d) for d in ema['decays']]
        assert len(decays_) == len(tracks), (decays_, len(tracks))
        flat_tracks = []
        for j, t in enumerate(tracks):
            ft = shadow if j == 0 else flatten_tree(t)
            assert set(ft) == set(param_order), f'EMA track {j} does not match param_order'
            flat_tracks.append({k: ft[k] for k in param_order})
        payload['ema_tracks'] = {'decays': decays_, 'num_updates': int(ema['num_updates']), 'tracks': flat_tracks}
    return payload


def _build_ema_tracks(ckpt, shadow0, params_template, ema_decays, sel, n_updates):
    cp = lambda t: jax.tree_util.tree_map(lambda a: np.array(a, copy=True), t)
    saved = ckpt.get('ema_tracks')
    if saved is None:
        tracks = [cp(shadow0) for _ in ema_decays]
        if len(ema_decays) > 1:
            print(
                f"[EMA] the checkpoint holds a single EMA track -> the {len(ema_decays)} requested tracks (decays {ema_decays}) are all initialised from it; "
                "they start identical and diverge as training continues"
            )
    else:
        old = [float(d) for d in saved['decays']]
        old_trees = [load_state_dict_into(params_template, saved['tracks'][j], f'ema_tracks[{j}]') for j in range(len(old))]
        tracks, new = [], []
        for d in ema_decays:
            hit = [j for j, o in enumerate(old) if abs(o - d) <= 1e-12]
            if hit:
                tracks.append(old_trees[hit[0]])
            else:
                tracks.append(cp(old_trees[0]))
                new.append(d)
        if new:
            print(f"[EMA] the checkpoint has tracks {old}; the new decays {new} are initialised from its first track (decay {old[0]})")
        dropped = [o for o in old if not any(abs(o - d) <= 1e-12 for d in ema_decays)]
        if dropped:
            print(f"[EMA] checkpoint tracks {dropped} are not requested any more and are dropped")
    return {
        'num_updates': np.int32(int(n_updates)),
        'shadow_params': tracks[sel],
        'tracks': tracks,
        'decays': [float(d) for d in ema_decays],
    }


def restore_from_checkpoint(ckpt, params_template, opt, lr_schedule, loss_scaler, ema_decays=None, ema_for_sampling=None):
    order = list(flatten_tree(params_template))
    n = len(order)
    params = load_state_dict_into(params_template, ckpt['net_state_dict'], 'net_state_dict')

    es = ckpt['ema_state_dict']
    sh_keys = [k for k in es if k.startswith('shadow_params.')]
    if len(sh_keys) != n:
        raise KeyError(f'ema_state_dict has {len(sh_keys)} shadow_params entries, the model has {n} parameters')
    shadow = load_state_dict_into(params_template, {order[int(k.split(".")[1])]: es[k] for k in sh_keys}, 'ema shadow_params')
    ema = {'num_updates': np.int32(int(es['num_updates'])), 'shadow_params': shadow}
    if ema_decays is not None:
        ema = _build_ema_tracks(
            ckpt,
            shadow,
            params_template,
            [float(d) for d in ema_decays],
            resolve_ema_index([float(d) for d in ema_decays], ema_for_sampling),
            int(es['num_updates']),
        )

    os_ = ckpt['opt_state_dict']
    st = os_['state']
    if len(st) == 0:
        opt_state = adamw_init(params_template)
    else:
        if sorted(st) != list(range(n)):
            raise KeyError('opt_state_dict["state"] must be indexed 0..n_params-1')
        opt_state = {
            'step': np.int32(int(round(float(np.asarray(st[0]['step']))))),
            'exp_avg': load_state_dict_into(params_template, {order[i]: st[i]['exp_avg'] for i in range(n)}, 'opt exp_avg'),
            'exp_avg_sq': load_state_dict_into(params_template, {order[i]: st[i]['exp_avg_sq'] for i in range(n)}, 'opt exp_avg_sq'),
        }
    opt.param_groups = [{k: v for k, v in g.items() if k != 'params'} for g in os_['param_groups']]
    lr_schedule.load_state_dict(ckpt['lr_schedule_state_dict'])
    loss_scaler.load_state_dict(ckpt['scaler_state_dict'])
    return params, ema, opt_state, ckpt['epoch'] + 1, ckpt.get('global_step', 0)


def check_checkpoint_compat(ckpt, path_, conditioning, arch, facies_values_, diffusion_cfg=None, data_cfg=None):
    if ckpt.get('mode') == 'inpainting':
        raise ValueError(
            f"Checkpoint {path_} is an inpainting (well-conditioned) checkpoint; it cannot be resumed by this porous-media trainer. "
            "Change training.checkpoint_path or set training.resume=false."
        )
    ck_cond = ckpt.get('conditioning')
    if ck_cond is None and 'enable_fraction_conditioning' in ckpt:
        ck_cond = 'porosity' if ckpt['enable_fraction_conditioning'] else 'none'
    if ck_cond is not None and ck_cond != conditioning:
        raise ValueError(
            f"Checkpoint {path_} was trained with conditioning={ck_cond!r}, but the config says conditioning={conditioning!r}. "
            "Use a matching run_tag / checkpoint_path or set training.resume=false."
        )
    ckpt_arch = ckpt.get('config', {}).get('architecture', {})
    structural = (
        'in_channels',
        'out_channels',
        'model_channels',
        'num_res_blocks',
        'attention_resolutions',
        'channel_mult',
        'conv_resample',
        'dims',
        'num_classes',
        'fraction_dim',
        'num_heads',
        'num_head_channels',
        'num_heads_upsample',
        'use_scale_shift_norm',
        'resblock_updown',
        'use_new_attention_order',
        'with_fourier_features',
        'ignore_time',
        'input_projection',
        'porosity_embedding',
        'porosity_fourier_levels',
    )
    norm = lambda v: list(v) if isinstance(v, (list, tuple)) else v
    diffs = [
        f"{key}: checkpoint={ckpt_arch[key]!r} vs config={arch.get(key)!r}"
        for key in structural
        if key in ckpt_arch and norm(ckpt_arch[key]) != norm(arch.get(key))
    ]
    if diffs:
        raise ValueError(
            f"Checkpoint {path_} was trained with a different architecture -> " + "; ".join(diffs) + ". "
            "Cannot resume: delete the checkpoint, change training.checkpoint_path, or set training.resume=false."
        )
    if diffusion_cfg is not None and ckpt.get('config', {}).get('diffusion') is not None:
        old_d = ckpt['config']['diffusion']
        if any(k in old_d for k in ('num_timesteps', 'beta_schedule', 'prediction_type', 'variance_type')):
            raise ValueError(
                f"Checkpoint {path_} is a DDPM checkpoint (its diffusion block has timestep / beta-schedule / prediction-type keys). "
                "Its network was trained on integer timesteps and epsilon / x0 / v targets, not on the EDM denoiser parameterisation, so it "
                "cannot be resumed here: change training.checkpoint_path or set training.resume=false."
            )
        cur = {**DEFAULT_DIFFUSION_CFG, **diffusion_cfg}
        old = {**DEFAULT_DIFFUSION_CFG, **old_d}
        nrm = lambda v: np.asarray(v, dtype=np.float64).tolist() if isinstance(v, (list, tuple, np.ndarray)) else v
        ddiffs = [f"{k}: checkpoint={nrm(old[k])!r} vs config={nrm(cur[k])!r}" for k in EDM_STRUCTURAL_KEYS if nrm(old[k]) != nrm(cur[k])]
        if ddiffs:
            raise ValueError(
                f"Checkpoint {path_} was trained with a different diffusion setup -> " + "; ".join(ddiffs) + ". "
                "Cannot resume: delete the checkpoint, change training.checkpoint_path, or set training.resume=false."
            )
    if 'facies_values' in ckpt and [int(v) for v in ckpt['facies_values']] != list(facies_values_):
        raise ValueError(
            f"Checkpoint {path_} was trained with facies_values={list(ckpt['facies_values'])}, "
            f"but the current data has {list(facies_values_)}."
        )
    old_data = (ckpt.get('config') or {}).get('data')
    if data_cfg is not None and old_data is not None:
        for k in ('dataset', 'source', 'volume_dims', 'patch_size', 'stride', 'holdout_axis', 'holdout_size'):
            if k in old_data and k in data_cfg and norm(old_data[k]) != norm(data_cfg[k]):
                extra = (
                    '  -> a changed hold-out moves previously held-out patches into the training set: evaluation is no longer clean'
                    if k.startswith('holdout')
                    else ''
                )
                print(f"[resume] WARNING: data.{k} differs from the checkpoint ({old_data[k]!r} -> {data_cfg[k]!r}).{extra}")
