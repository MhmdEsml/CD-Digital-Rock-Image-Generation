import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from tqdm.auto import tqdm

from porous.core.nn import flatten_tree
from porous.core.checkpoint_io import _CheckpointWriter, _snapshot_payload, fill_tree, load_checkpoint_file
from porous.core.parallel import _replica0, _replicate_for_pmap
from porous.core.adamw import adamw_init, adamw_update
from porous.core.ema import ema_update
from porous.cd.config import amp_dtype, normalise_resume
from porous.cd.consistency import cast_tree, cd_distance, make_extra, make_teacher_wrapper, pair_probs, pair_weights, student_ctx
from porous.core.consistency_math import _bc, cm_denoise, karras_grid
from porous.cd.data import unpack_batch
from porous.cd.dataset import ShardedLoader
from porous.core.patch_data import DistributedSampler
from porous.cd.report import calls_report
from porous.core.parallel import _sum_local


def _flat_in_order(tree, param_order):
    fl = flatten_tree(tree)
    return {k: np.asarray(fl[k]) for k in param_order}


def make_cd_payload(epoch, global_step, host, TEACHER, CDCFG, ema_decays, data_spec):
    po = TEACHER['param_order']
    return {
        'format': 'cd_student_porous_v1',
        'epoch': int(epoch),
        'global_step': int(global_step),
        'net_state_dict': _flat_in_order(host['params'], po),
        'target_state_dict': _flat_in_order(host['target'], po),
        'opt': {
            'step': int(host['opt']['step']),
            'exp_avg': _flat_in_order(host['opt']['exp_avg'], po),
            'exp_avg_sq': _flat_in_order(host['opt']['exp_avg_sq'], po),
        },
        'ema_tracks': {
            'decays': [float(d) for d in ema_decays],
            'num_updates': int(host['ema_n']),
            'tracks': [_flat_in_order(t, po) for t in host['ema']],
        },
        'config': {'cd': CDCFG, 'architecture': TEACHER['student_arch'], 'diffusion': TEACHER['dcfg']},
        'conditioning': TEACHER['conditioning'],
        'data_spec': {k: data_spec[k] for k in ('dataset', 'volume_dims', 'patch_size', 'stride', 'seed', 'holdout_axis', 'holdout_size')},
        'teacher': {
            'epoch': TEACHER['ckpt_epoch'],
            'ema_used': TEACHER['ema_used'],
            'ckpt_net_hash': TEACHER['net_hash'],
            'ckpt_ema_hash': TEACHER['ema_hash'],
        },
    }


class PairLossLog:

    def __init__(self, results_dir, grid, resume, skip_k=1):
        self.dir, self.grid, self.k = results_dir, [float(g) for g in grid], int(skip_k)
        self.path = os.path.join(results_dir, 'loss_by_interval.json')
        self.epochs = {}
        if resume and os.path.exists(self.path):
            try:
                old = json.load(open(self.path))
                if np.allclose(old['grid'], self.grid, rtol=1e-6) and int(old.get('skip_k', 1)) == self.k:
                    self.epochs = old['epochs']
            except Exception as e:
                print('[pair log] could not read the old log:', repr(e))

    def record(self, epoch, s, c):
        self.epochs[str(epoch)] = {'mean': [float(a / b) if b > 0 else None for a, b in zip(s, c)], 'count': [float(v) for v in c]}
        with open(self.path + '.tmp', 'w') as f:
            json.dump({'grid': self.grid, 'skip_k': self.k, 'epochs': self.epochs}, f)
        os.replace(self.path + '.tmp', self.path)


def make_cd_programs(CDCFG, TEACHER):
    cd, tr = CDCFG['consistency'], CDCFG['training']
    sd = TEACHER['diffusion'].sigma_data
    N, smin, smax, rho = int(cd['num_scales']), float(cd['sigma_min']), float(cd['sigma_max']), float(cd['rho'])
    grid = karras_grid(N, smin, smax, rho)
    sk = int(cd.get('skip_k', 1))
    probs = pair_probs(grid, cd['n_sampling'], cd['n_P_mean'], cd['n_P_std'], k=sk)
    wts = pair_weights(grid, probs, cd['weighting'], sd, k=sk)
    mu = float(cd['target_ema'])
    loss_kind, huber_c = cd['loss'], float(cd['huber_c'])
    ema_decays = [float(d) for d in tr['ema_decays']]
    betas = tuple(tr['optimizer_betas'])
    wd, clip_norm = float(tr['weight_decay']), float(tr['grad_clip_norm'] or 0.0)
    unet = TEACHER['student_unet']
    sub, solver = int(CDCFG['teacher']['substeps']), CDCFG['teacher']['solver']
    t_amp = amp_dtype(CDCFG['teacher']['compute_dtype'])
    grid_j = jnp.asarray(grid, jnp.float32)
    logp_j = jnp.log(jnp.asarray(probs, jnp.float32))
    w_j = jnp.asarray(wts, jnp.float32)
    n_pairs = N - sk
    n_steps = sub * sk

    def p1(target, t_params, rng, batch):
        rng, k_n, k_z, k_drop = jax.random.split(rng, 4)
        x0 = batch['x_1'].astype(jnp.float32)
        B = x0.shape[0]
        n = jax.random.categorical(k_n, logp_j, shape=(B,))
        s_lo, s_hi = grid_j[n], grid_j[n + sk]
        z = jax.random.normal(k_z, x0.shape, jnp.float32)
        x_hi = x0 + _bc(s_hi, x0.ndim) * z
        wrapped = make_teacher_wrapper(TEACHER, CDCFG, t_params, batch.get('por'), amp=t_amp)
        x = x_hi
        for j in range(n_steps):
            a = s_hi * (s_lo / s_hi) ** (j / n_steps)
            b = s_hi * (s_lo / s_hi) ** ((j + 1) / n_steps)
            d1 = (x - wrapped(x, a)) / _bc(a, x.ndim)
            x_e = x + _bc(b - a, x.ndim) * d1
            if solver == 'heun':
                d2 = (x_e - wrapped(x_e, b)) / _bc(b, x.ndim)
                x = x + _bc(b - a, x.ndim) * (0.5 * d1 + 0.5 * d2)
            else:
                x = x_e
        y = cm_denoise(unet, target, x, s_lo, make_extra(batch), student_ctx(CDCFG), sd, smin)
        aux = {'x_hi': x_hi, 's_hi': s_hi, 'y': y, 'w': w_j[n], 'n': n, 'k_drop': k_drop}
        if 'por' in batch:
            aux['por'] = batch['por']
        return rng, aux

    def p2(params, aux, bins):
        def loss_fn(p):
            ctx = student_ctx(CDCFG, training=True, key=aux['k_drop'])
            pred = cm_denoise(unet, p, aux['x_hi'], aux['s_hi'], make_extra(aux), ctx, sd, smin)
            per = cd_distance(pred, aux['y'], loss_kind, huber_c)
            return jnp.mean(per * aux['w']), per

        (loss, per), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        ps = lax.stop_gradient(per)
        new_bins = {'sum': bins['sum'].at[aux['n']].add(ps), 'cnt': bins['cnt'].at[aux['n']].add(1.0)}
        return loss, new_bins, grads

    def p3(core, g, lr):
        leaves = jax.tree_util.tree_leaves(g)
        gnorm = jnp.sqrt(sum(jnp.sum(jnp.square(a.astype(jnp.float32))) for a in leaves))
        finite = jnp.isfinite(gnorm)
        scale = jnp.minimum(1.0, clip_norm / (gnorm + 1e-6)) if clip_norm > 0 else 1.0
        g = jax.tree_util.tree_map(lambda a: jnp.where(finite, a * scale, 0.0), g)
        new_params, new_opt = adamw_update(core['params'], g, core['opt'], lr, betas, 1e-8, wd)
        keep = lambda n_, o_: jnp.where(finite, n_, o_)
        new_params = jax.tree_util.tree_map(keep, new_params, core['params'])
        new_opt = jax.tree_util.tree_map(keep, new_opt, core['opt'])
        new_target = jax.tree_util.tree_map(lambda t, p: mu * t + (1.0 - mu) * p, core['target'], new_params)
        new_ema, ema_n = [], core['ema_n']
        for tree_, d_ in zip(core['ema'], ema_decays):
            t_new, ema_n = ema_update(tree_, new_params, core['ema_n'], d_)
            new_ema.append(t_new)
        return {'params': new_params, 'opt': new_opt, 'target': new_target, 'ema': new_ema, 'ema_n': ema_n}, gnorm

    return dict(
        p1=p1,
        p2=p2,
        p3=p3,
        grid=grid,
        probs=probs,
        wts=wts,
        sk=sk,
        N=N,
        n_pairs=n_pairs,
        n_steps=n_steps,
        mu=mu,
        smin=smin,
        smax=smax,
        rho=rho,
        loss_kind=loss_kind,
        huber_c=huber_c,
        ema_decays=ema_decays,
    )


def train_cd(CDCFG, TEACHER, DATA, paths, grad_sync='split'):
    if grad_sync not in ('split', 'host'):
        raise ValueError(f"grad_sync must be 'split' or 'host', got {grad_sync!r}")
    cd, tr = CDCFG['consistency'], CDCFG['training']
    accum = int(tr.get('accum_iter', 1) or 1)
    if accum < 1:
        raise ValueError(f'training.accum_iter must be >= 1, got {accum}')
    if tr.get('matmul_precision'):
        jax.config.update('jax_default_matmul_precision', tr['matmul_precision'])
    n_local, world, pidx = jax.local_device_count(), jax.device_count(), jax.process_index()
    ranks = [pidx * n_local + i for i in range(n_local)]
    is_main = pidx == 0
    has_por = TEACHER['has_por']
    progs = make_cd_programs(CDCFG, TEACHER)
    grid, probs, sk, N, n_pairs, n_steps = progs['grid'], progs['probs'], progs['sk'], progs['N'], progs['n_pairs'], progs['n_steps']
    ema_decays = progs['ema_decays']
    assert len(set(ema_decays)) == len(ema_decays) and all(0 < d < 1 for d in ema_decays), ema_decays
    template = TEACHER['template']
    ckpt_path = paths['ckpt_path']
    results_dir = paths['results_dir']
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    data_spec = DATA['spec']
    cr = calls_report(CDCFG, TEACHER)

    if is_main:
        print(
            f'CD: N={N} sigmas in [{progs["smin"]}, {progs["smax"]}] (rho {progs["rho"]}) | interval sampling = {cd["n_sampling"]} | loss = {progs["loss_kind"]} (c={progs["huber_c"]}) '
            f'| weighting = {cd["weighting"]} | skip_k = {sk} | target mu = {progs["mu"]} | teacher solver = {CDCFG["teacher"]["solver"]}, {n_steps} steps per pair '
            f'| porosity CFG scale = {CDCFG["teacher"]["porosity_cfg_scale"]} | teacher dtype {CDCFG["teacher"]["compute_dtype"]}, student dtype {CDCFG["student"]["compute_dtype"]}'
        )
        print(
            f"teacher network calls per training step (per sample): {cr['teacher_calls']} (+ {cr['target_calls']} target-network call, + student forward/backward)"
        )
        print('grid:', np.round(grid, 4).tolist())
        print('P(pair n):', np.round(probs, 3).tolist())

    donate = (0,) if jax.default_backend() != 'cpu' else ()
    p1_fn = jax.pmap(progs['p1'], axis_name='batch')
    p2_fn = jax.pmap(progs['p2'], axis_name='batch')
    p3_fn = jax.pmap(progs['p3'], axis_name='batch', donate_argnums=donate)
    allreduce_fn = jax.pmap(lambda g_: lax.pmean(g_, axis_name='batch'), axis_name='batch')
    inv_accum = 1.0 / accum
    scale_fn = jax.pmap(lambda g_: jax.tree_util.tree_map(lambda y: y * inv_accum, g_), axis_name='batch')
    add_fn = jax.pmap(
        lambda a_, g_: jax.tree_util.tree_map(lambda x, y: x + y * inv_accum, a_, g_), axis_name='batch', donate_argnums=donate
    )

    def host_mean_replicated(tree):
        def one(a):
            m = np.asarray(a).mean(axis=0)
            return np.stack([m] * n_local)

        return jax.tree_util.tree_map(one, tree)

    params = jax.tree_util.tree_map(lambda a: np.array(a, np.float32, copy=True), TEACHER['params'])
    target = jax.tree_util.tree_map(lambda a: np.array(a, copy=True), params)
    opt_state = adamw_init(params)
    ema_tracks = [jax.tree_util.tree_map(lambda a: np.array(a, copy=True), params) for _ in ema_decays]
    ema_n = np.int32(0)
    start_epoch, global_step = 0, 0
    resume_mode = normalise_resume(tr['resume'])
    if resume_mode != 'no' and os.path.exists(ckpt_path):
        ck = load_checkpoint_file(ckpt_path)
        if ck.get('format') != 'cd_student_porous_v1':
            raise ValueError(f'{ckpt_path} is not a porous-media CD student checkpoint (format {ck.get("format")!r})')
        if ck['teacher']['ckpt_ema_hash'] != TEACHER['ema_hash'] or ck['teacher']['ema_used'] != TEACHER['ema_used']:
            raise ValueError(
                f'{ckpt_path} was distilled from a different teacher / EMA track ({ck["teacher"]["ema_used"]}); change run_tag or set resume=false'
            )
        old_ds = ck.get('data_spec', {})
        norm = lambda v: [int(q) for q in np.ravel(v)] if not isinstance(v, str) else v
        dd = [
            f'{k}: checkpoint={old_ds[k]} vs now={data_spec[k]}'
            for k in ('dataset', 'patch_size', 'holdout_axis', 'holdout_size')
            if old_ds.get(k) is not None and norm(old_ds[k]) != norm(data_spec[k])
        ]
        if dd:
            raise ValueError('the checkpoint was distilled on different data -> ' + '; '.join(dd) + '. Change run_tag or set resume=false.')
        old_cd = ck['config']['cd']['consistency']
        diffs = [
            f'{k}: checkpoint={old_cd.get(k, 1 if k == "skip_k" else None)} vs config={cd.get(k, 1)}'
            for k in ('num_scales', 'sigma_min', 'sigma_max', 'rho', 'target_ema', 'skip_k')
            if old_cd.get(k, 1 if k == 'skip_k' else None) != cd.get(k, 1)
        ]
        if diffs:
            raise ValueError(
                'the checkpoint was trained with a different discretisation / target setup -> '
                + '; '.join(diffs)
                + '. Change run_tag or set resume=false.'
            )
        params = fill_tree(template, ck['net_state_dict'])
        target = fill_tree(template, ck['target_state_dict'])
        opt_state = {
            'step': np.int32(ck['opt']['step']),
            'exp_avg': fill_tree(template, ck['opt']['exp_avg']),
            'exp_avg_sq': fill_tree(template, ck['opt']['exp_avg_sq']),
        }
        old_dec = [float(d) for d in ck['ema_tracks']['decays']]
        ema_tracks = []
        for d_ in ema_decays:
            hit = [i for i, o in enumerate(old_dec) if abs(o - d_) < 1e-12]
            src = ck['ema_tracks']['tracks'][hit[0]] if hit else ck['net_state_dict']
            if not hit and is_main:
                print(f'[resume] EMA decay {d_} is new: its track starts from the live weights')
            ema_tracks.append(fill_tree(template, src))
        ema_n = np.int32(ck['ema_tracks']['num_updates'])
        start_epoch, global_step = int(ck['epoch']) + 1, int(ck['global_step'])
        del ck
        if is_main:
            print(f'Resumed from {ckpt_path}: starting at epoch {start_epoch}, global_step {global_step}')
    elif is_main:
        print(
            (f'resume={resume_mode} but no checkpoint at {ckpt_path} -> ' if resume_mode != 'no' else 'resume=no -> ')
            + f'starting FRESH from the teacher ({TEACHER["ema_used"]})'
        )

    host_core = {'params': params, 'target': target, 'opt': opt_state, 'ema': ema_tracks, 'ema_n': ema_n}
    core = _replicate_for_pmap(host_core, n_local)
    teacher_rep = _replicate_for_pmap(cast_tree(TEACHER['params'], CDCFG['teacher']['compute_dtype']), n_local)
    rank_seeds = [tr['seed'] + r + 1000 * start_epoch for r in ranks]
    rng = jax.pmap(lambda k: k, axis_name='batch')(np.stack([np.asarray(jax.random.PRNGKey(s)) for s in rank_seeds]))

    def _zero_bins():
        z = np.zeros((n_local, n_pairs), np.float32)
        return jax.pmap(lambda s: s, axis_name='batch')({'sum': z, 'cnt': z.copy()})

    bins = _zero_bins()

    dataset = DATA['dataset']
    samplers = [DistributedSampler(dataset, num_replicas=world, rank=r, shuffle=True, seed=tr['seed'], drop_last=True) for r in ranks]
    loader = ShardedLoader(
        dataset,
        samplers,
        ranks,
        batch_size=tr['batch_size'],
        num_workers=tr['num_workers'],
        base_seed=tr['seed'] + 1000 * start_epoch,
        steps_per_epoch=tr.get('steps_per_epoch'),
    )
    if len(loader) == 0:
        raise ValueError(f'0 batches per rank: {len(dataset)} patches / {world} ranks < per-device batch {tr["batch_size"]}')
    max_steps = tr.get('max_steps_per_epoch')
    end_epoch = min(start_epoch + int(tr['session_epochs']), int(tr['total_epochs']))
    n_it = len(loader) if not max_steps else min(len(loader), max_steps)
    n_it -= n_it % accum
    if n_it <= 0:
        raise ValueError(f'{len(loader)} micro-batches per epoch < accum_iter={accum}: lower accum_iter or steps_per_epoch / batch_size')
    loader.steps_per_epoch = n_it
    n_upd = n_it // accum
    if is_main:
        print(
            f"grad_sync={grad_sync} | devices {world} ({jax.default_backend()}) | training patches {len(dataset)} | per-device batch {tr['batch_size']} x {world} devices = {tr['batch_size'] * world} per micro-step "
            f"x accum_iter {accum} -> EFFECTIVE batch {tr['batch_size'] * world * accum} | micro-steps/epoch {n_it} = {n_upd} optimizer updates | epochs [{start_epoch}, {end_epoch}) of {tr['total_epochs']}"
        )
        print(f'EMA tracks {ema_decays}')

    pair_log = PairLossLog(results_dir, grid, resume=start_epoch > 0, skip_k=sk) if is_main else None
    writer = _CheckpointWriter()

    def save_checkpoint(out_paths, epoch, gstep, must=False):
        out_paths = [out_paths] if isinstance(out_paths, str) else list(out_paths)
        if not must and writer.busy():
            writer.n_skipped += 1
            print(f'[checkpoint] previous write still running -> skipping the save at epoch {epoch}')
            return
        t0 = time.perf_counter()
        host = _replica0({k: core[k] for k in ('params', 'target', 'opt', 'ema', 'ema_n')})
        payload = _snapshot_payload(make_cd_payload(epoch, gstep, host, TEACHER, CDCFG, ema_decays, data_spec))
        writer.submit(out_paths, payload, f'epoch {epoch}, global_step {gstep}', time.perf_counter() - t0, label='CD checkpoint', must=must)

    def _read(l):
        v = np.asarray(l)
        if not np.all(np.isfinite(v)):
            raise ValueError(f'loss is {v.tolist()}: stopping')
        return float(v[0])

    for epoch in range(start_epoch, end_epoch):
        loader.set_epoch(epoch)
        t_ep = time.time()
        run_loss, n_b, pend, acc, gn_list = 0.0, 0, None, None, []
        it = tqdm(loader, desc=f'[CD] epoch {epoch}', total=n_it) if is_main else loader
        for step_i, batch in enumerate(it):
            bd = unpack_batch(batch, has_por)
            lr = float(tr['lr']) * (
                min(1.0, (global_step + 1) / int(tr.get('lr_warmup_steps', 0))) if int(tr.get('lr_warmup_steps', 0)) > 0 else 1.0
            )
            rng, aux = p1_fn(core['target'], teacher_rep, rng, bd)
            loss, bins, grads = p2_fn(core['params'], aux, bins)
            acc = scale_fn(grads) if step_i % accum == 0 else add_fn(acc, grads)
            del aux, grads
            if (step_i + 1) % accum == 0:
                lr_arr = np.full((n_local,), lr, np.float32)
                g = allreduce_fn(acc) if grad_sync == 'split' else host_mean_replicated(acc)
                core, gn = p3_fn(core, g, lr_arr)
                del g, acc
                acc = None
                gn_list.append(gn)
                global_step += 1
            if pend is not None:
                lv = _read(pend)
                run_loss += lv
                n_b += 1
                if is_main:
                    it.set_postfix(loss=f'{lv:.4f}', lr=f'{lr:.1e}')
            pend = loss
        if pend is not None:
            run_loss += _read(pend)
            n_b += 1
        run_gn = sum(float(np.asarray(g_)[0]) for g_ in gn_list)
        b_sum, b_cnt = _sum_local(bins['sum']), _sum_local(bins['cnt'])
        bins = _zero_bins()

        if is_main:
            pair_log.record(epoch, b_sum, b_cnt)
            print(
                f'[CD] epoch {epoch}: loss {run_loss / max(n_b, 1):.5f} | grad-norm {run_gn / max(len(gn_list), 1):.3f} | {time.time() - t_ep:.0f} s | update {global_step}'
            )
            snap_due = bool(tr.get('save_every') and (epoch + 1) % tr['save_every'] == 0)
            last = epoch == end_epoch - 1
            ce = int(tr.get('checkpoint_every', 1) or 1)
            if ce <= 1 or (epoch + 1) % ce == 0 or last or snap_due:
                out_paths = [ckpt_path]
                if snap_due:
                    base, ext = os.path.splitext(ckpt_path)
                    out_paths.append(f'{base}_ep{epoch + 1}{ext or ".pkl"}')
                save_checkpoint(out_paths, epoch, global_step, must=last or snap_due)

    writer.wait()
    if is_main:
        print('CD training session complete.')
    return {'checkpoint_path': ckpt_path, 'global_step': global_step, 'end_epoch': end_epoch, 'ema_decays': ema_decays}
