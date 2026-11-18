import gc
import time

import numpy as np
import jax
from jax import lax

from porous.core.parallel import _replicate_for_pmap
from porous.core.adamw import adamw_init
from porous.cd.consistency import cast_tree
from porous.cd.data import unpack_batch
from porous.core.unet import count_params


def calls_report(CDCFG, TEACHER):
    t, c = CDCFG['teacher'], CDCFG['consistency']
    n_steps = int(t['substeps']) * int(c.get('skip_k', 1))
    per_eval = (
        1 if (not TEACHER['has_por'] or float(t['porosity_cfg_scale']) == 1.0) else (1 if float(t['porosity_cfg_scale']) == 0.0 else 2)
    )
    evals = n_steps * (2 if t['solver'] == 'heun' else 1)
    teacher_calls = evals * per_eval
    return dict(solver_steps=n_steps, teacher_calls=teacher_calls, target_calls=1, student_fwd_bwd=1, guidance_branches=per_eval)


def size_report(CDCFG, TEACHER):
    P = count_params(TEACHER['params'])
    n_ema = len(CDCFG['training']['ema_decays'])
    tdt = 4 if CDCFG['teacher']['compute_dtype'] == 'float32' else 2
    ckpt = P * 4 * (2 + 2 + n_ema)
    dev = P * 4 * (2 + 2 + n_ema) + P * tdt
    cr = calls_report(CDCFG, TEACHER)
    print(f'parameters: {P:,} ({P * 4 / 1e6:.0f} MB float32)')
    print(
        f'checkpoint size: {ckpt / 1e9:.2f} GB  = {P * 4 / 1e6:.0f} MB x (weights 1 + target 1 + AdamW moments 2 + EMA tracks {n_ema}); every permanent snapshot (save_every) adds one more'
    )
    print(
        f'parameter-like state resident on EACH device: {dev / 1e9:.2f} GB (incl. the frozen teacher in {CDCFG["teacher"]["compute_dtype"]}), without activations'
    )
    print(
        f"teacher network calls per training step (per sample): {cr['teacher_calls']} = {cr['solver_steps']} solver steps (substeps {CDCFG['teacher']['substeps']} x skip_k {CDCFG['consistency']['skip_k']}) "
        f"x {2 if CDCFG['teacher']['solver'] == 'heun' else 1} ({CDCFG['teacher']['solver']}) x {cr['guidance_branches']} guidance branch(es); "
        f"+ {cr['target_calls']} target-network call (no grad) + {cr['student_fwd_bwd']} student forward/backward"
        f"{' with recomputation (use_checkpoint)' if CDCFG['student']['use_checkpoint'] else ''}"
    )
    return cr


def probe_train_step(CDCFG, TEACHER, DATA, n_timed=3):
    tr = CDCFG['training']
    from porous.cd.train import make_cd_programs

    progs = make_cd_programs(CDCFG, TEACHER)
    dev = jax.local_devices()[0]
    B = int(tr['batch_size'])
    rs = np.random.RandomState(0)
    ds = DATA['dataset']
    items = [ds.__getitem__(int(i), rng=rs) for i in rs.choice(len(ds), size=B, replace=len(ds) < B)]
    batch = unpack_batch(tuple(np.stack([it[j] for it in items])[None] for j in range(2)), TEACHER['has_por'])
    p1, p2 = jax.pmap(progs['p1'], axis_name='batch'), jax.pmap(progs['p2'], axis_name='batch')
    p3 = jax.pmap(progs['p3'], axis_name='batch', donate_argnums=(0,) if jax.default_backend() != 'cpu' else ())
    allreduce = jax.pmap(lambda g_: lax.pmean(g_, axis_name='batch'), axis_name='batch')
    params = jax.tree_util.tree_map(lambda a: np.array(a, np.float32, copy=True), TEACHER['params'])
    ema_decays = [float(d) for d in tr['ema_decays']]
    core = _replicate_for_pmap(
        {
            'params': params,
            'target': jax.tree_util.tree_map(np.copy, params),
            'opt': adamw_init(params),
            'ema': [jax.tree_util.tree_map(np.copy, params) for _ in ema_decays],
            'ema_n': np.int32(0),
        },
        1,
    )
    t_rep = _replicate_for_pmap(cast_tree(TEACHER['params'], CDCFG['teacher']['compute_dtype']), 1)
    rng = jax.pmap(lambda k: k, axis_name='batch')(np.stack([np.asarray(jax.random.PRNGKey(0))]))
    z = np.zeros((1, progs['n_pairs']), np.float32)
    bins = jax.pmap(lambda s: s, axis_name='batch')({'sum': z, 'cnt': z.copy()})
    times = []
    for i in range(1 + n_timed):
        t0 = time.perf_counter()
        rng, aux = p1(core['target'], t_rep, rng, batch)
        loss, bins, grads = p2(core['params'], aux, bins)
        g = allreduce(grads)
        core, gn = p3(core, g, np.full((1,), float(tr['lr']), np.float32))
        lv = float(np.asarray(loss)[0])
        times.append(time.perf_counter() - t0)
        del aux, grads, g
        assert np.isfinite(lv), 'non-finite loss in the probe step'
    sec = float(np.mean(times[1:])) if n_timed else times[0]
    mem = None
    try:
        st = dev.memory_stats()
        mem = (st.get('peak_bytes_in_use', 0) / 1e9, st.get('bytes_limit', 0) / 1e9)
    except Exception:
        pass
    ge = tr.get('steps_per_epoch') or max(len(DATA['dataset']) // (B * jax.device_count()), 1)
    print(
        f"probe: batch {B}/device | first step (incl. compilation) {times[0]:.1f} s | {sec:.2f} s/step (mean of {n_timed}) | loss {lv:.4f}"
    )
    print(
        f"       device memory peak {mem[0]:.2f} GB of {mem[1]:.2f} GB"
        if mem and mem[1]
        else '       device memory statistics are not available on this backend'
    )
    print(
        f"estimate: {sec * ge / 60:.1f} min per epoch of {ge} steps | {sec * ge * int(tr['total_epochs']) / 3600:.1f} h for {tr['total_epochs']} epochs "
        f"(plus evaluation and checkpoint time; devices run in parallel, so the per-step time does not grow with their number)"
    )
    del core, t_rep, progs
    gc.collect()
    return dict(sec_per_step=sec, first_step=times[0], peak_gb=None if not mem else mem[0])
