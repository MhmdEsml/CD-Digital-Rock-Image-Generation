import copy
import math
import os
import re
import time

import numpy as np
import jax

from porous.sampling.checkpoint import build_student_params, load_cd_checkpoint, load_edm_checkpoint, resolve_checkpoint
from porous.sampling.config import parse_tags
from porous.sampling.data import ensure_real_patches, norm3, save_real_packed
from porous.sampling.outputs import README_FILES, save_generated_binary, write_manifest, write_sampling_info
from porous.sampling.samplers import build_edm_sample_fn, build_sample_fn, replicate_for_pmap, sample_all_devices
from porous.sampling.schedules import edm_schedules, is_excluded_schedule, ordinary_taus, safe_name
from porous.core.consistency_math import parse_schedule


def make_jobs(cfg):
    jobs = []
    for m in cfg.models:
        if m == "edm":
            jobs.append(("edm", None, None))
            continue
        students = [s.strip() for s in str(cfg.student).split(",") if s.strip()] if cfg.student else [None]
        tags = parse_tags(cfg.cd_run_tags)
        if len(students) > 1 and cfg.ckpt:
            raise ValueError("--ckpt names ONE checkpoint file")
        for i, s in enumerate(students):
            tag = (
                tags[i]
                if i < len(tags)
                else (re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.splitext(os.path.basename(s))[0]) if (s and len(students) > 1) else "")
            )
            jobs.append(("cd", tag, s))
    return jobs


def run_model(cfg):
    started = time.time()
    print(f"\n{'*' * 60}\nCheckpoint\n{'*' * 60}")
    ckpt_path, teacher_info_path = resolve_checkpoint(
        cfg.model,
        ckpt=cfg.ckpt,
        teacher_dir=cfg.teacher_dir,
        teacher_info=cfg.teacher_info,
        student=cfg.student,
        edm_output_dir=cfg.edm_output_dir,
        cd_output_dir=cfg.cd_output_dir,
    )
    print(f"  MODEL = {cfg.model!r} | using: {ckpt_path}")
    if cfg.model == "edm":
        info = load_edm_checkpoint(ckpt_path, teacher_info_path, cfg.weights, cfg.ema_decay)
    else:
        info = load_cd_checkpoint(ckpt_path, cfg.weights, cfg.ema_decay)

    ds = info["data_spec"]
    dataset_name = cfg.dataset or ds.get("dataset")
    if not dataset_name:
        raise ValueError("the checkpoint has no data_spec.dataset: pass --dataset")
    volume_dims = norm3(cfg.volume_dims if cfg.volume_dims is not None else ds.get("volume_dims", (1000, 1000, 1000)))
    patch_size = int(norm3(cfg.patch_size if cfg.patch_size is not None else ds.get("patch_size", 64))[0])
    cd_tag = getattr(cfg, "cd_run_tag", "")
    run_label = f"cd_{cd_tag}" if (cfg.model == "cd" and cd_tag) else cfg.model
    ckpt_tag = f"{run_label}-ep{info['epochs_done']}-gs{info['global_step']}-{info['weights_tag']}"
    run_dir = os.path.join(cfg.out_dir, dataset_name, cfg.model, ckpt_tag)
    os.makedirs(run_dir, exist_ok=True)
    print(f"  dataset = {dataset_name} | volume {volume_dims} | patch {patch_size}^3 | outputs -> {run_dir}")

    if cfg.model == "cd":
        available = {**info["schedules"], **cfg.cd_extra_schedules}
        dropped = [n for n, sp in available.items() if is_excluded_schedule(n, sp, tuple(cfg.cd_exclude))]
        for n in dropped:
            del available[n]
        if dropped:
            print(f"  removed schedules: {dropped}")
        ordinary_names = []
        for k in cfg.cd_steps:
            nm = f"{k}-step"
            ordinary_names.append(nm)
            if nm not in available:
                available[nm] = ordinary_taus(k, info["cd"])
        others = [n for n in available if n not in ordinary_names]
        default_names = ordinary_names + others
        names = default_names if cfg.schedules is None else list(cfg.schedules)
        bad = [n for n in names if n not in available]
        if bad:
            raise ValueError(f"unknown schedule(s) {bad}; available: {list(available)}")
        print(f"  schedules: {names}")
        for nm in names:
            print(f"    {nm:<14} taus = {list(parse_schedule(available[nm])[0])}  gamma = {parse_schedule(available[nm])[1]}")
    else:
        available = edm_schedules(cfg, info)
        names = list(available) if cfg.schedules is None else list(cfg.schedules)
        bad = [n for n in names if n not in available]
        if bad:
            raise ValueError(f"unknown schedule(s) {bad}; available: {list(available)}")
        print(f"  schedules: {names}")
        for nm in names:
            print(f"    {nm:<18} {available[nm]}")

    print(f"\n{'#' * 60}\nDataset: {dataset_name}\n{'#' * 60}")
    real_patches = ensure_real_patches(cfg, dataset_name, volume_dims, patch_size)
    n_samples = len(real_patches)
    target_por = (real_patches == 0).mean(axis=(1, 2, 3)).astype(np.float32)
    print(f"  real porosity: mean {target_por.mean():.4f}  min {target_por.min():.4f}  max {target_por.max():.4f}")
    shared_dir = os.path.join(cfg.out_dir, dataset_name, "shared")
    save_real_packed(shared_dir, real_patches)
    del real_patches

    print(f"\n{'#' * 60}\nSampling\n{'#' * 60}")
    devices = jax.local_devices()
    n_dev = len(devices)
    print(
        f"  JAX {jax.__version__} | backend {jax.default_backend()} | sampling on ALL {n_dev} local device(s): {[str(d) for d in devices]}"
    )
    if info["matmul_precision"]:
        jax.config.update("jax_default_matmul_precision", info["matmul_precision"])

    sched_dirs = {}
    todo = []
    for name in names:
        sdir = os.path.join(run_dir, safe_name(name))
        os.makedirs(sdir, exist_ok=True)
        sched_dirs[name] = sdir
        npy = os.path.join(sdir, "generated_samples.npy")
        ok = False
        if cfg.reuse and os.path.exists(npy):
            try:
                ok = np.load(npy, mmap_mode="r").shape[0] >= n_samples
            except Exception:
                ok = False
        if ok:
            print(f"  [{name}] reusing {npy}")
        else:
            todo.append(name)

    if todo:
        unet, params = build_student_params(info)
        params_rep = replicate_for_pmap(params, n_dev)
        del params
        per_dev_bs = max(1, min(cfg.per_device_batch, math.ceil(n_samples / n_dev)))
        print(
            f"  {n_samples} samples per schedule = {n_dev} device(s) x {per_dev_bs} volumes per call "
            f"x {math.ceil(n_samples / (n_dev * per_dev_bs))} call(s)"
        )
        ch = int(info["arch"].get("out_channels", 1))
        z = np.random.RandomState(cfg.sampling_seed).standard_normal((n_samples, ch) + (patch_size,) * 3).astype(np.float32)
        por_in = target_por[:, None] if info["has_por"] else np.zeros((n_samples, 1), np.float32)
        key = jax.random.PRNGKey(cfg.sampling_seed + 1)

        for name in todo:
            if cfg.model == "edm":
                fn, sched_meta = build_edm_sample_fn(unet, info, available[name])
            else:
                taus, gamma = parse_schedule(available[name])
                fn, taus_used = build_sample_fn(unet, info, taus, gamma)
                sched_meta = {"taus_used": list(taus_used), "gamma": gamma}
            fn_pmap = jax.pmap(fn)
            t0 = time.time()
            x = sample_all_devices(fn_pmap, params_rep, z, por_in, key, n_dev, per_dev_bs, desc=f"  [{name}] sampling")
            dt = time.time() - t0
            if not np.isfinite(x).all():
                print(f"  WARNING [{name}]: {int((~np.isfinite(x)).sum())} non-finite voxels (counted as solid)")
            gen = np.where(x <= 0.0, -1.0, 1.0).astype(np.float32).transpose(0, 2, 3, 4, 1)
            sdir = sched_dirs[name]
            np.save(os.path.join(sdir, "generated_samples.npy"), gen)
            np.save(os.path.join(sdir, "target_porosity.npy"), target_por)
            write_sampling_info(sdir, ckpt_path, info, name, sched_meta, cfg.model, n_samples, n_dev, per_dev_bs, cfg.sampling_seed, dt)
            save_generated_binary(sdir, gen)
            print(
                f"  [{name}] {n_samples} samples in {dt:.1f} s ({dt / n_samples:.2f} s/sample) | "
                f"generated porosity {float((gen[..., 0] < 0).mean()):.4f} vs target {target_por.mean():.4f}"
            )
            del gen, x
        del params_rep
    else:
        print("  all schedules already sampled - nothing to do")

    for name in names:
        sdir = sched_dirs[name]
        if not os.path.exists(os.path.join(sdir, "generated_binary.npz")):
            save_generated_binary(sdir, np.load(os.path.join(sdir, "generated_samples.npy"), mmap_mode="r"))
        if not cfg.keep_samples:
            npy = os.path.join(sdir, "generated_samples.npy")
            if os.path.exists(npy):
                os.remove(npy)
                print(f"    {npy} removed (--no-keep-samples)")

    with open(os.path.join(run_dir, "README_files.md"), "w") as f:
        f.write(README_FILES)
    write_manifest(run_dir, cfg, info, ckpt_path, available, names, dataset_name, volume_dims, patch_size, ckpt_tag, started)
    print(f"\nINFO: {cfg.model.upper()} sampling complete -> {run_dir}")
    return run_dir


def main(cfg):
    jobs = make_jobs(cfg)
    run_dirs = []
    for k, (m, tag, student) in enumerate(jobs, 1):
        print(f"\n{'#' * 70}\n#  {m.upper()}{' (run_tag ' + repr(tag) + ')' if tag is not None else ''}   ({k} of {len(jobs)})\n{'#' * 70}")
        cfg_m = copy.copy(cfg)
        cfg_m.model = m
        if m == "cd":
            cfg_m.cd_run_tag = tag
            cfg_m.student = student
        run_dirs.append(run_model(cfg_m))
    return run_dirs
