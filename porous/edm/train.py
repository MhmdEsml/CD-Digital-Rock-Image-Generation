import itertools
import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from tqdm import tqdm

from porous.core.edm import build_diffusion, sigma_bin_edges, sigma_bin_index
from porous.core.nn import Ctx, KeyGen, flatten_tree
from porous.core.unet import count_params, make_unet
from porous.edm.checkpoint import check_checkpoint_compat, make_checkpoint_payload, restore_from_checkpoint
from porous.core.checkpoint_io import _CheckpointWriter, _snapshot_payload, _write_checkpoint_files, load_checkpoint_file
from porous.edm.config import get_stride, resolve_sigma_data
from porous.edm.data import ShardedLoader, build_split_dataset, get_volume, load_patch_cache
from porous.core.patch_data import DistributedSampler
from porous.core.ema import ema_init_tracks, ema_update, resolve_ema_decays, resolve_ema_index
from porous.edm.model import ref_velocity_net
from porous.edm.optim import (
    AdamWHost,
    NativeScalerWithGradNormCount,
    build_lr_schedule,
    clip_grad_norm,
    grad_scaler_update,
    lr_at_step,
    resolve_precision,
    set_lr,
)
from porous.core.adamw import adamw_init, adamw_update
from porous.edm.preview import run_preview
from porous.core.parallel import _replica0, _replicate_for_pmap, _sum_local


def make_step_fns(unet, diffusion, *, use_amp, amp_dtype, accum_iter, grad_clip_norm, ema_decays, betas, eps, weight_decay, bin_cfg=None):

    def grad_part(state, batch, first):
        rng, sub = jax.random.split(state['rng'])
        k_noise, k_sigma, k_model = jax.random.split(sub, 3)

        extra = {}
        if 'frac' in batch:
            extra['fraction'] = batch['frac']

        x_start = batch['x_1']
        noise = jax.random.normal(k_noise, x_start.shape, x_start.dtype)
        sigma = diffusion.sample_sigma(k_sigma, x_start.shape[0])

        def loss_fn(params):
            ctx = Ctx(training=True, amp_dtype=amp_dtype, key=k_model)
            apply_fn = lambda x_in, c_noise: ref_velocity_net(unet, params, x_in, c_noise, extra, ctx)
            per_sample, _ = diffusion.training_losses(apply_fn, x_start, sigma, noise)
            loss = per_sample.mean()
            scaled = loss / accum_iter
            if use_amp:
                scaled = scaled * state['scaler']['scale']
            return scaled, (loss, per_sample)

        (_, (loss, per_sample)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state['params'])

        if accum_iter > 1:
            accum = grads if first else jax.tree_util.tree_map(jnp.add, state['accum'], grads)
        else:
            accum = grads

        new_state = dict(state)
        new_state['rng'] = rng
        if accum_iter > 1:
            new_state['accum'] = accum
        if bin_cfg is not None:
            b_lo, b_hi, b_n = bin_cfg
            idx = sigma_bin_index(sigma, b_lo, b_hi, b_n)
            ps = lax.stop_gradient(per_sample).astype(jnp.float32)
            new_state['bins'] = {
                'sum': state['bins']['sum'] + jnp.zeros((b_n + 2,), jnp.float32).at[idx].add(ps),
                'cnt': state['bins']['cnt'] + jnp.zeros((b_n + 2,), jnp.float32).at[idx].add(1.0),
            }
        return new_state, loss, accum

    def update_part(state, g, lr):
        found_inf = jnp.bool_(False)
        if use_amp:
            inv_scale = 1.0 / state['scaler']['scale']
            g = jax.tree_util.tree_map(lambda a: a * inv_scale, g)
            finite = jnp.stack([jnp.all(jnp.isfinite(a)) for a in jax.tree_util.tree_leaves(g)])
            found_inf = jnp.logical_not(jnp.all(finite))
        if grad_clip_norm and grad_clip_norm > 0:
            g = clip_grad_norm(g, grad_clip_norm)

        new_state = dict(state)
        new_params, new_opt = adamw_update(state['params'], g, state['opt'], lr, betas, eps, weight_decay)
        if use_amp:
            new_params = jax.tree_util.tree_map(lambda n, o: jnp.where(found_inf, o, n), new_params, state['params'])
            new_opt = jax.tree_util.tree_map(lambda n, o: jnp.where(found_inf, o, n), new_opt, state['opt'])
            new_scale, new_tracker = grad_scaler_update(state['scaler']['scale'], state['scaler']['growth_tracker'], found_inf)
            new_state['scaler'] = {'scale': new_scale, 'growth_tracker': new_tracker}

        new_ema = []
        for tree_, d_ in zip(state['ema'], ema_decays):
            t_new, new_ema_n = ema_update(tree_, new_params, state['ema_n'], d_)
            new_ema.append(t_new)

        new_state['params'], new_state['opt'] = new_params, new_opt
        new_state['ema'], new_state['ema_n'] = new_ema, new_ema_n
        return new_state

    def fused(state, batch, lr, first, update):
        new_state, loss, accum = grad_part(state, batch, first)
        if not update:
            return new_state, loss
        g = lax.pmean(accum, axis_name='batch')
        return update_part(new_state, g, lr), loss

    return grad_part, update_part, fused


class SigmaLossLog:
    LAYOUT = (
        'index 0: sigma < edges[0]; index i = 1..N: edges[i-1] <= sigma < edges[i] (the last bin is closed); '
        'index N+1: sigma > edges[N]; mean = sum / count (null when count == 0)'
    )

    def __init__(self, results_dir, edges, resume=False):
        os.makedirs(results_dir, exist_ok=True)
        self.results_dir = results_dir
        self.edges = [float(e) for e in edges]
        self.path = os.path.join(results_dir, 'loss_by_sigma.json')
        self.epochs = {}
        if resume and os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    old = json.load(f)
                if np.allclose(old['sigma_edges'], self.edges, rtol=1e-6):
                    self.epochs = old['epochs']
                    print(f'[sigma bins] continuing {self.path} ({len(self.epochs)} epochs already logged)')
            except Exception as e:
                print('[sigma bins] could not read the existing log, starting a new one:', repr(e))

    def record(self, epoch, s, c):
        mean = [float(a / b) if b > 0 else None for a, b in zip(s, c)]
        self.epochs[str(epoch)] = {'sum': [float(v) for v in s], 'count': [float(v) for v in c], 'mean': mean}
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump({'sigma_edges': self.edges, 'layout': self.LAYOUT, 'epochs': self.epochs}, f)
        os.replace(tmp, self.path)


def _normalise_resume(v):
    if v is True:
        return 'from_disk'
    if v is False or v is None:
        return 'no'
    t = str(v).strip().lower()
    if t in ('from_disk', 'disk', 'true'):
        return 'from_disk'
    if t in ('no', 'false', 'none', ''):
        return 'no'
    raise ValueError(f"training.resume must be 'no' or 'from_disk' (or True / False); got {v!r}")


def run_training(CFG, grad_sync='nccl', volume=None):
    if grad_sync not in ('nccl', 'host', 'split'):
        raise ValueError(f"grad_sync must be 'nccl', 'host' or 'split', got {grad_sync!r}")
    n_local = jax.local_device_count()
    world_size = jax.device_count()
    process_index = jax.process_index()
    ranks = [process_index * n_local + i for i in range(n_local)]
    is_main = process_index == 0
    train_cfg = CFG['training']
    ref_arch_ = CFG['architecture']
    data_cfg = CFG['data']
    precision = resolve_precision(train_cfg)
    use_amp, amp_dtype = precision['use_scaler'], precision['amp_dtype']

    COND = CFG.get('conditioning')
    if COND not in ('porosity', 'none'):
        raise ValueError(f"config conditioning must be 'porosity' or 'none', got {COND!r}")
    use_porosity = COND == 'porosity'
    image_channels = CFG['image_channels']
    patch = int(data_cfg['patch_size'])
    if ref_arch_.get('dims') != 3:
        raise ValueError('architecture.dims must be 3')
    CFG.setdefault('diffusion', {})

    if volume is None:
        volume = get_volume(CFG)
    cache = load_patch_cache(CFG['patch_cache_path'], data_cfg, volume)
    if isinstance(CFG['diffusion'].get('sigma_data', 0.5), str):
        resolve_sigma_data(CFG['diffusion'], float(cache['data_rms']))
    diffusion = build_diffusion(CFG.get('diffusion', {}))
    ema_decays = resolve_ema_decays(train_cfg)
    ema_sel = resolve_ema_index(ema_decays, train_cfg.get('ema_for_sampling'))
    log_bins = bool(train_cfg.get('log_sigma_bins', True))
    n_bins = int(train_cfg.get('sigma_bins', 16))
    bin_cfg = (diffusion.sigma_train_min, diffusion.sigma_train_max, n_bins) if log_bins else None
    expected_out = image_channels * diffusion.model_out_mult
    if ref_arch_['out_channels'] != expected_out:
        raise ValueError(
            f"architecture.out_channels={ref_arch_['out_channels']} but EDM needs {expected_out} (one denoiser output, "
            f"no learned variance). The config is stale -- re-derive the config (derive_config)."
        )
    if ref_arch_['in_channels'] != image_channels:
        raise ValueError(
            f"architecture.in_channels={ref_arch_['in_channels']} but this trainer has no concatenated conditioning: it must be {image_channels}."
        )
    fraction_dim = ref_arch_.get('fraction_dim', 0)
    if fraction_dim != (1 if use_porosity else 0):
        raise ValueError(
            f"architecture.fraction_dim={fraction_dim} but conditioning={COND!r} needs {1 if use_porosity else 0}. The config is stale -- re-derive the config (derive_config)."
        )
    if int(cache['patch_size']) != patch:
        raise RuntimeError('patch cache / config patch_size mismatch')

    dataset, _tr_idx = build_split_dataset(volume, cache, data_cfg, 0, use_porosity, max_patches=train_cfg.get('max_train_patches'))
    sample_every = train_cfg.get('sample_every', 0)

    samplers = [
        DistributedSampler(dataset, num_replicas=world_size, rank=r, shuffle=True, seed=train_cfg.get('seed', 0), drop_last=True)
        for r in ranks
    ]
    start_epoch, global_step = 0, 0
    ckpt_path = train_cfg['checkpoint_path']

    unet = make_unet(ref_arch_)
    params = {'net': unet.init(KeyGen(jax.random.PRNGKey(train_cfg['seed'])))}
    param_order = list(flatten_tree(params))
    if is_main:
        print(f'UNetModel parameter count: {count_params(params):,}')
        print(diffusion.describe())
        print(f"Precision: compute_dtype={precision['compute_dtype']} -> {precision['label']}")

    opt = AdamWHost(train_cfg['lr'], betas=tuple(train_cfg['optimizer_betas']))
    lr_schedule = build_lr_schedule(opt, train_cfg)
    loss_scaler = NativeScalerWithGradNormCount(enabled=use_amp)
    opt_state = adamw_init(params)
    ema = ema_init_tracks(params, ema_decays, ema_sel)

    _resume_mode = _normalise_resume(train_cfg['resume'])
    if _resume_mode != 'no' and os.path.exists(ckpt_path):
        ckpt = load_checkpoint_file(ckpt_path)
        check_checkpoint_compat(ckpt, ckpt_path, COND, ref_arch_, CFG['facies_values'], CFG.get('diffusion', {}), data_cfg)
        params, ema, opt_state, start_epoch, global_step = restore_from_checkpoint(
            ckpt, params, opt, lr_schedule, loss_scaler, ema_decays=ema_decays, ema_for_sampling=train_cfg.get('ema_for_sampling')
        )
        del ckpt
        if is_main:
            print(f'Resumed from {ckpt_path}: starting at epoch {start_epoch}, global_step {global_step}')
    elif is_main:
        if _resume_mode != 'no':
            print(f'resume={_resume_mode} but no checkpoint at {ckpt_path} -> starting FRESH at epoch 0')
        else:
            print(
                'resume=false -> starting FRESH at epoch 0'
                + (f' (existing {ckpt_path} will be overwritten)' if os.path.exists(ckpt_path) else '')
            )

    rank_seeds = [train_cfg['seed'] + r + 1000 * start_epoch for r in ranks]
    loader = ShardedLoader(
        dataset,
        samplers,
        ranks,
        batch_size=train_cfg['batch_size'],
        num_workers=train_cfg['num_workers'],
        base_seed=train_cfg['seed'] + 1000 * start_epoch,
    )
    if len(loader) == 0:
        raise ValueError(
            f"0 batches per rank: {len(dataset)} patches / {world_size} ranks < per-device batch_size={train_cfg['batch_size']}."
        )
    accum_iter = train_cfg['accum_iter']
    spe = train_cfg.get('steps_per_epoch')
    n_steps_epoch = min(int(spe), len(loader)) if spe else len(loader)
    n_steps_epoch -= n_steps_epoch % accum_iter
    if n_steps_epoch <= 0:
        raise ValueError(f'steps_per_epoch={spe} is smaller than accum_iter={accum_iter}')
    if is_main:
        print(
            f"Conditioning: {COND} | grad_sync: {grad_sync} | World size: {world_size} ({n_local} local devices, backend={jax.default_backend()}) | "
            f"training patches: {len(dataset)} (hold-out: {int((cache['split'] == 1).sum())}) | per-device batch: {train_cfg['batch_size']} | "
            f"global micro-batch = per-device * world_size = {train_cfg['batch_size'] * world_size} x accum_iter {train_cfg['accum_iter']} -> EFFECTIVE batch {train_cfg['batch_size'] * world_size * train_cfg['accum_iter']} | "
            f"batches per full pass/rank: {len(loader)} | micro-steps per epoch: {n_steps_epoch} = {n_steps_epoch // accum_iter} optimizer updates"
            + (' (steps_per_epoch)' if spe and n_steps_epoch < len(loader) else ' (full pass)')
        )
        print(
            f"fraction_dim (porosity)={fraction_dim} porosity_embedding={ref_arch_.get('porosity_embedding')} porosity_dropout={ref_arch_.get('porosity_dropout')} "
            f"| patch {patch}^3, stride {get_stride(data_cfg)}, flip_prob {data_cfg.get('flip_prob', 0.5)}"
        )

    sigma_log = None
    if is_main:
        print(f'EMA tracks: decays {ema_decays} (previews / sampling use decay {ema_decays[ema_sel]}) | sigma_dist={diffusion.sigma_dist}')
        if log_bins:
            sigma_log = SigmaLossLog(
                train_cfg['results_dir'],
                sigma_bin_edges(diffusion.sigma_train_min, diffusion.sigma_train_max, n_bins),
                resume=start_epoch > 0,
            )
            print(
                f'Loss-per-sigma logging: {n_bins} log-spaced bins on [{diffusion.sigma_train_min}, {diffusion.sigma_train_max}] + under/overflow -> {sigma_log.path}'
            )

    total_epochs = train_cfg['total_epochs']
    end_epoch = min(start_epoch + train_cfg['session_epochs'], total_epochs)
    warmup_steps = train_cfg.get('lr_warmup_steps', 0)
    grad_clip_norm = train_cfg.get('grad_clip_norm', 0.0)
    save_every = int(train_cfg.get('save_every', 0) or 0)
    checkpoint_every = int(train_cfg.get('checkpoint_every', 1) or 1)
    async_checkpoint = bool(train_cfg.get('async_checkpoint', True))

    host_state = {
        'params': params,
        'ema': list(ema['tracks']),
        'ema_n': np.int32(ema['num_updates']),
        'opt': opt_state,
        'scaler': {
            'scale': np.float32(loss_scaler._scale if use_amp else 1.0),
            'growth_tracker': np.int32(loss_scaler._growth_tracker if use_amp else 0),
        },
    }
    if accum_iter > 1:
        host_state['accum'] = jax.tree_util.tree_map(lambda p: np.zeros(np.shape(p), np.float32), params)
    state = _replicate_for_pmap(host_state, n_local)
    state['rng'] = jax.pmap(lambda k: k, axis_name='batch')(np.stack([np.asarray(jax.random.PRNGKey(s)) for s in rank_seeds]))

    def _zero_bins():
        z = np.zeros((n_local, n_bins + 2), np.float32)
        return jax.pmap(lambda s: s, axis_name='batch')({'sum': z, 'cnt': z.copy()})

    if log_bins:
        state['bins'] = _zero_bins()

    grad_part, update_part, fused = make_step_fns(
        unet,
        diffusion,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        accum_iter=accum_iter,
        grad_clip_norm=grad_clip_norm,
        ema_decays=ema_decays,
        bin_cfg=bin_cfg,
        betas=tuple(opt.param_groups[0]['betas']),
        eps=opt.param_groups[0]['eps'],
        weight_decay=opt.param_groups[0]['weight_decay'],
    )
    donate = (0,) if jax.default_backend() != 'cpu' else ()
    _step_cache = {}

    def get_step_fn(first, update):
        if (first, update) not in _step_cache:
            if grad_sync == 'nccl':

                def _f(state_, batch_, lr_):
                    return fused(state_, batch_, lr_, first, update)

                _step_cache[(first, update)] = jax.pmap(_f, axis_name='batch', donate_argnums=donate)
            else:

                def _g(state_, batch_):
                    return grad_part(state_, batch_, first)

                _step_cache[(first, update)] = jax.pmap(_g, axis_name='batch', donate_argnums=donate)
        return _step_cache[(first, update)]

    if grad_sync in ('host', 'split'):
        update_fn = jax.pmap(lambda state_, g_, lr_: update_part(state_, g_, lr_), axis_name='batch', donate_argnums=donate)

    if grad_sync == 'split':
        allreduce_fn = jax.pmap(lambda g_: lax.pmean(g_, axis_name='batch'), axis_name='batch')

    if grad_sync == 'host':

        def host_mean_replicated(accum):
            def one(a):
                m = np.asarray(a).mean(axis=0)
                return np.stack([m] * n_local)

            return jax.tree_util.tree_map(one, accum)

    ckpt_writer = _CheckpointWriter()
    ckpt_config = {
        'architecture': ref_arch_,
        'training': train_cfg,
        'diffusion': CFG.get('diffusion', {}),
        'data': data_cfg,
        'conditioning': COND,
    }

    def save_checkpoint(paths, epoch, global_step_, must=False):
        paths = [paths] if isinstance(paths, str) else list(paths)
        if async_checkpoint and not must and ckpt_writer.busy():
            ckpt_writer.n_skipped += 1
            print(
                f'[checkpoint] the previous checkpoint is still being written (slow disk?) -> skipping the save at epoch {epoch}; the next one is taken '
                f'once it has finished ({ckpt_writer.n_skipped} skipped so far)'
            )
            return
        t0 = time.perf_counter()
        host = _replica0({k: state[k] for k in ('params', 'ema', 'ema_n', 'opt', 'scaler')})
        if use_amp:
            loss_scaler._scale = float(host['scaler']['scale'])
            loss_scaler._growth_tracker = int(host['scaler']['growth_tracker'])
        payload = make_checkpoint_payload(
            epoch,
            global_step_,
            host['params'],
            {'num_updates': host['ema_n'], 'shadow_params': host['ema'][0], 'tracks': list(host['ema']), 'decays': ema_decays},
            opt,
            host['opt'],
            lr_schedule,
            loss_scaler,
            ckpt_config,
            COND,
            CFG['facies_values'],
            param_order,
        )
        payload = _snapshot_payload(payload)
        t_dev = time.perf_counter() - t0
        info = f'epoch {epoch}, global_step {global_step_}'
        if async_checkpoint:
            ckpt_writer.submit(paths, payload, info, t_dev, label='checkpoint', must=must)
        else:
            t1 = time.perf_counter()
            _write_checkpoint_files(paths, payload)
            print(f'Saved checkpoint -> {paths[0]} ({info}; device->host {t_dev:.1f} s, pickle+write {time.perf_counter() - t1:.1f} s)')

    preview_rng = np.random.RandomState(rank_seeds[0] % (2**32))
    preview_key = jax.random.PRNGKey(rank_seeds[0])

    if is_main:
        if start_epoch >= total_epochs:
            print(f'Already reached total_epochs={total_epochs}. Nothing to do.')
        print(f'[pmap/{COND}] Training epochs [{start_epoch}, {end_epoch}) (total_epochs={total_epochs})')

    def _read_loss(l):
        v = np.asarray(l)
        if not np.all(np.isfinite(v)):
            raise ValueError(f'Loss is {v.tolist()}, stopping training')
        return float(v[0])

    for epoch in range(start_epoch, end_epoch):
        loader.set_epoch(epoch)
        running_loss, n_batches = 0.0, 0
        pending_loss = None
        batch_iter = iter(loader)
        steps_iter = itertools.islice(batch_iter, n_steps_epoch)
        iterator = tqdm(steps_iter, total=n_steps_epoch, desc=f'[pmap/{COND}] Epoch {epoch}') if is_main else steps_iter
        try:
            for data_iter_step, batch in enumerate(iterator):
                first = data_iter_step % accum_iter == 0
                apply_update = (data_iter_step + 1) % accum_iter == 0

                if use_porosity:
                    x_1, frac = batch
                    batch_dict = {'x_1': x_1, 'frac': frac}
                else:
                    batch_dict = {'x_1': batch}

                if apply_update and warmup_steps > 0 and global_step < warmup_steps:
                    set_lr(opt, lr_at_step(global_step, train_cfg['lr'], warmup_steps))

                lr_arr = np.full((n_local,), opt.param_groups[0]['lr'], dtype=np.float32)
                if grad_sync == 'nccl':
                    state, loss = get_step_fn(first, apply_update)(state, batch_dict, lr_arr)
                else:
                    state, loss, local_grads = get_step_fn(first, apply_update)(state, batch_dict)
                    if apply_update:
                        if grad_sync == 'split':
                            state = update_fn(state, allreduce_fn(local_grads), lr_arr)
                        else:
                            state = update_fn(state, host_mean_replicated(local_grads), lr_arr)
                    del local_grads

                if pending_loss is not None:
                    loss_value = _read_loss(pending_loss)
                    running_loss += loss_value
                    n_batches += 1
                    if is_main:
                        iterator.set_postfix(loss=f'{loss_value:.4f}', lr=f"{opt.param_groups[0]['lr']:.2e}")
                pending_loss = loss

                if apply_update:
                    global_step += 1
        finally:
            batch_iter.close()

        if pending_loss is not None:
            running_loss += _read_loss(pending_loss)
            n_batches += 1
            pending_loss = None

        lr_schedule.step()

        if log_bins:
            b_sum, b_cnt = _sum_local(state['bins']['sum']), _sum_local(state['bins']['cnt'])
            state['bins'] = _zero_bins()
            if is_main:
                sigma_log.record(epoch, b_sum, b_cnt)

        if is_main:
            avg_loss = running_loss / max(n_batches, 1)
            print(f'[pmap/{COND}] Epoch {epoch}: avg loss (rank 0) = {avg_loss:.4f}')

            _snap_due = bool(save_every and (epoch + 1) % save_every == 0)
            _is_last = epoch == end_epoch - 1
            if checkpoint_every <= 1 or (epoch + 1) % checkpoint_every == 0 or _is_last or _snap_due:
                _paths = [ckpt_path]
                if _snap_due:
                    _base, _ext = os.path.splitext(ckpt_path)
                    _paths.append(f'{_base}_ep{epoch + 1}{_ext or ".pkl"}')
                save_checkpoint(_paths, epoch, global_step, must=_is_last or _snap_due)

            if sample_every and ((epoch + 1) % sample_every == 0 or epoch == end_epoch - 1):
                run_preview(
                    unet,
                    diffusion,
                    _replica0(state['ema'][ema_sel]),
                    epoch,
                    global_step,
                    train_cfg['results_dir'],
                    CFG,
                    dataset.porosity,
                    jax.random.fold_in(preview_key, epoch),
                    preview_rng,
                    use_porosity,
                )

    ckpt_writer.wait()
    if is_main and ckpt_writer.n_skipped:
        print(
            f'[checkpoint] {ckpt_writer.n_skipped} routine save(s) were skipped because the previous write was still running -- the disk is slow; '
            f'raise checkpoint_every to write less often'
        )
    if is_main:
        print('Multi-device training session complete.')
    return {
        'checkpoint_path': ckpt_path,
        'global_step': global_step,
        'end_epoch': end_epoch,
        'ema_decays': ema_decays,
        'ema_for_sampling': ema_decays[ema_sel],
    }


def collective_selftest(n_params):
    n = jax.local_device_count()
    if n < 2:
        print('single device -> no cross-device collective needed')
        return True
    f = jax.pmap(lambda x: lax.psum(x, 'i'), axis_name='i')
    ok = True
    for size in (8, 1_000_000, max(n_params, 1_000_001)):
        try:
            out = np.asarray(f(np.ones((n, size), np.float32)))
            good = bool(np.all(out == n))
            print(f'psum over {n} devices, {size:>10,} float32: {"OK" if good else "WRONG RESULT"}')
            ok &= good
        except Exception as e:
            print(f'psum over {n} devices, {size:>10,} float32: FAILED -> {str(e)[:300]}')
            ok = False
    if not ok:
        print(
            '\nCollectives are failing on this machine. Options, in order:\n'
            '  1. use --grad-sync host (works without NCCL; slower per step), or\n'
            '  2. set the environment variable NCCL_P2P_DISABLE=1 (and look at the NCCL_DEBUG=WARN output) before starting, or\n'
            '  3. lower training.batch_size / set XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 so NCCL has GPU memory left for its buffers.'
        )
    return ok


def launch_training(CFG, volume=None, n_params=None, grad_sync=None):
    assert os.path.exists(CFG['patch_cache_path']), f"Porosity cache missing: {CFG['patch_cache_path']} -- run prepare_data first"
    with open(CFG['config_path'], 'w') as f:
        json.dump(CFG, f, indent=2, default=lambda o: np.asarray(o).tolist())
    print(
        f"Wrote FINAL config -> {CFG['config_path']} (conditioning={CFG['conditioning']}, in_channels={CFG['architecture']['in_channels']}, "
        f"fraction_dim={CFG['architecture']['fraction_dim']}, porosity_embedding={CFG['architecture']['porosity_embedding']}, dims={CFG['architecture']['dims']})"
    )
    print(f"  checkpoint: {CFG['training']['checkpoint_path']}")
    print(f"  results:    {CFG['training']['results_dir']}")

    n_dev_train = jax.device_count()
    print(
        f"batch_size is PER DEVICE: {CFG['training']['batch_size']} x {n_dev_train} device(s) -> global micro-batch {CFG['training']['batch_size'] * n_dev_train} x accum_iter {CFG['training']['accum_iter']} -> EFFECTIVE batch {CFG['training']['batch_size'] * n_dev_train * CFG['training']['accum_iter']}"
    )
    print(f'Launching data-parallel training across {n_dev_train} device(s)...')

    if grad_sync is None:
        collectives_ok = collective_selftest(n_params or 0)
        if jax.default_backend() == 'tpu':
            grad_sync = 'split' if collectives_ok else 'host'
        else:
            grad_sync = 'nccl' if collectives_ok else 'host'
    print('GRAD_SYNC =', grad_sync)
    return run_training(CFG, grad_sync=grad_sync, volume=volume)
