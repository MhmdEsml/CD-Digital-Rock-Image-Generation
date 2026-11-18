import argparse
import json

DEFAULTS = dict(
    model="both",
    weights="ema",
    ema_decay=None,
    schedules=None,
    per_device_batch=4,
    sampling_seed=123,
    reuse=True,
    keep_samples=True,
    ckpt=None,
    teacher_dir=None,
    teacher_info=None,
    student=None,
    edm_output_dir=None,
    cd_output_dir=None,
    cd_run_tags=None,
    cd_steps=[1, 2, 3, 4, 5, 6, 7, 8],
    cd_exclude=["g0.3"],
    cd_extra_schedules={},
    edm_steps=[1, 2, 3, 4, 5, 6, 7, 8, 16, 32],
    edm_sampler=None,
    edm_cfg_scale=None,
    edm_s_churn=None,
    dataset=None,
    volume=None,
    volume_dims=None,
    patch_size=None,
    n_samples=3000,
    seed=42,
    real_patches_npy=None,
    save_real_patches=False,
    out_dir="./sampling_results",
)


def parse_models(v):
    items = list(v) if isinstance(v, (list, tuple)) else str(v).replace(";", ",").split(",")
    out = []
    for it in items:
        it = str(it).strip().lower()
        for m in (["edm", "cd"] if it == "both" else [it]):
            if m not in ("cd", "edm"):
                raise ValueError(f"model must be 'cd', 'edm', 'both' or a list of those, got {v!r}")
            if m not in out:
                out.append(m)
    return out


def parse_tags(v):
    if v is None:
        return [""]
    items = list(v) if isinstance(v, (list, tuple)) else str(v).split(",")
    out = []
    for t in items:
        t = str(t).strip()
        if t not in out:
            out.append(t)
    return out or [""]


def _ints(s):
    return [int(v) for v in str(s).replace(" ", "").split(",") if v != ""]


def _strs(s):
    return [v.strip() for v in str(s).split(",") if v.strip()]


def build_parser():
    ap = argparse.ArgumentParser(
        description="Generate samples with the EDM teacher and / or the CD student on ALL local accelerators (no metrics)."
    )
    a = ap.add_argument
    a("--config", default=None, help="JSON file with default values for any option below (command line wins)")
    a("--model", default=None, help='"cd", "edm", "both" or a comma separated list such as "edm,cd"')
    a("--weights", default=None, choices=["ema", "live", "target"])
    a("--ema-decay", type=float, default=None)
    a("--schedules", default=None, help='comma separated, e.g. "1-step,2-step,8-step" (CD) or "heun-8,heun-32" (EDM)')
    a("--per-device-batch", type=int, default=None)
    a("--sampling-seed", type=int, default=None)
    a("--no-reuse", action="store_true", help="re-sample even if generated_samples.npy exists")
    a("--no-keep-samples", action="store_true", help="delete the float32 generated_samples.npy after generated_binary.npz is written")
    g = ap.add_argument_group("trained models (local files)")
    g.add_argument("--ckpt", default=None, help="ONE local checkpoint file (needs a single --model)")
    g.add_argument("--teacher-dir", default=None, help="folder with the exported EDM teacher: edm.pkl (+ teacher_info.json)")
    g.add_argument(
        "--teacher-info",
        default=None,
        help="teacher_info.json (default: <teacher-dir>/teacher_info.json, else taken from the checkpoint's config)",
    )
    g.add_argument(
        "--student", default=None, help="CD student file(s): cd_student.pkl (EMA export) or cd_train_state.pkl; comma separated for several"
    )
    g.add_argument(
        "--cd-run-tags",
        default=None,
        help='comma separated run tags (labels) of the CD students given with --student, e.g. "uniform,uniform_t"',
    )
    g.add_argument("--edm-output-dir", default=None, help="search the last edm_porous_jax_ckpt*.pkl in this folder")
    g.add_argument("--cd-output-dir", default=None, help="search the last cd_student*.pkl in this folder")
    g = ap.add_argument_group("CD student schedules")
    g.add_argument("--cd-steps", default=None, help='comma separated numbers of steps of the ordinary schedules, default "1,2,3,4,5,6,7,8"')
    g.add_argument("--cd-exclude", default=None, help='comma separated name patterns of schedules to drop, default "g0.3"')
    g.add_argument("--cd-extra-schedules", default=None, help='JSON dict of extra schedules, e.g. \'{"4-step": [8.0, 2.0, 0.5]}\'')
    g = ap.add_argument_group("EDM teacher sampling")
    g.add_argument("--edm-steps", default=None, help='comma separated numbers of sigma levels, e.g. "8,16,32"')
    g.add_argument("--edm-sampler", default=None, choices=["heun", "euler"])
    g.add_argument("--cfg-scale", type=float, default=None, help="porosity guidance scale (1.0 = none)")
    g.add_argument("--s-churn", type=float, default=None)
    g = ap.add_argument_group("real data (target porosities)")
    g.add_argument("--dataset", default=None, help="dataset name (default: from the checkpoint), used for folder names")
    g.add_argument("--volume", default=None, help="local <Name>.raw volume (uint8, 0 = pore, 1 = solid)")
    g.add_argument("--volume-dims", default=None, help='e.g. "1000,1000,1000" or "1000" (default: from the checkpoint)')
    g.add_argument("--patch-size", type=int, default=None)
    g.add_argument("--n-samples", type=int, default=None, help="number of real patches = generated samples per schedule (default 3000)")
    g.add_argument("--seed", type=int, default=None, help="seed of the real-patch extraction (default 42)")
    g.add_argument("--real-patches-npy", default=None)
    g.add_argument("--save-real-patches", action="store_true", help="also write real_patches.npy")
    g = ap.add_argument_group("output")
    g.add_argument("--out-dir", default=None)
    return ap


def parse_config(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    cfg = dict(DEFAULTS)
    if a.config:
        with open(a.config) as f:
            file_cfg = json.load(f)
        bad = [k for k in file_cfg if k not in DEFAULTS]
        if bad:
            raise ValueError(f"unknown option(s) in {a.config}: {bad}")
        cfg.update(file_cfg)
    cli = dict(
        model=a.model,
        weights=a.weights,
        ema_decay=a.ema_decay,
        schedules=_strs(a.schedules) if a.schedules else None,
        per_device_batch=a.per_device_batch,
        sampling_seed=a.sampling_seed,
        ckpt=a.ckpt,
        teacher_dir=a.teacher_dir,
        teacher_info=a.teacher_info,
        student=a.student,
        cd_run_tags=a.cd_run_tags,
        edm_output_dir=a.edm_output_dir,
        cd_output_dir=a.cd_output_dir,
        cd_steps=_ints(a.cd_steps) if a.cd_steps else None,
        cd_exclude=_strs(a.cd_exclude) if a.cd_exclude else None,
        cd_extra_schedules=json.loads(a.cd_extra_schedules) if a.cd_extra_schedules else None,
        edm_steps=_ints(a.edm_steps) if a.edm_steps else None,
        edm_sampler=a.edm_sampler,
        edm_cfg_scale=a.cfg_scale,
        edm_s_churn=a.s_churn,
        dataset=a.dataset,
        volume=a.volume,
        volume_dims=_ints(a.volume_dims) if a.volume_dims else None,
        patch_size=a.patch_size,
        n_samples=a.n_samples,
        seed=a.seed,
        real_patches_npy=a.real_patches_npy,
        out_dir=a.out_dir,
    )
    cfg.update({k: v for k, v in cli.items() if v is not None})
    if a.no_reuse:
        cfg["reuse"] = False
    if a.no_keep_samples:
        cfg["keep_samples"] = False
    if a.save_real_patches:
        cfg["save_real_patches"] = True
    if isinstance(cfg["schedules"], str):
        cfg["schedules"] = _strs(cfg["schedules"])
    cfg["models"] = parse_models(cfg["model"])
    cfg["model"] = cfg["models"][0]
    ns = argparse.Namespace(**cfg)
    if ns.ckpt and len(ns.models) > 1:
        raise ValueError(
            "--ckpt names ONE checkpoint file: with several models give --teacher-dir / --student (or the output dirs) instead"
        )
    return ns
