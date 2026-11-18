import json
import os
import time

import numpy as np

from porous.sampling.data import save_binary_npz

README_FILES = """# Files of this run

Layout: `<dataset>/<model>/<checkpoint tag>/<schedule>/...` plus `<dataset>/shared/` (real patches), see `run_manifest.json` for the full configuration.

Per schedule folder (one folder per sampling schedule):
- `generated_samples.npy`  the generated samples as float32 (N, D, H, W, 1), -1 = pore, +1 = solid
- `generated_binary.npz`   the same samples, bit-packed and LZMA-compressed, 1 = pore. `d = np.load(path); np.unpackbits(d["packed"], axis=1)[:, :D*H*W].reshape(d["shape"])`
- `target_porosity.npy`    porosity of real patch i = the conditioning of generated sample i
- `sampling_info.json`     checkpoint, schedule, timings

Run folder: `run_manifest.json`.
Shared folder: `real_samples.csv` (coordinates of the real patches the target porosities come from), `real_patches_binary.npz` (bit-packed + LZMA, 1 = pore),
`real_patches.npy` (uint8, 0 = pore; only written with --save-real-patches).
"""


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, range)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size < 1000 else f"<array {o.shape} {o.dtype}>"
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


def save_generated_binary(sdir, gen):
    path = os.path.join(sdir, "generated_binary.npz")
    save_binary_npz(path, gen[..., 0] < 0.0)
    return path


def write_sampling_info(sdir, ckpt_path, info, name, sched_meta, model, n_samples, n_dev, per_dev_bs, sampling_seed, seconds):
    with open(os.path.join(sdir, "sampling_info.json"), "w") as f:
        json.dump(
            {
                "checkpoint": ckpt_path,
                "weights": info["weights_tag"],
                "epochs": info["epochs_done"],
                "global_step": info["global_step"],
                "schedule": name,
                **sched_meta,
                "model": model,
                "n_samples": n_samples,
                "devices": n_dev,
                "per_device_batch": per_dev_bs,
                "sampling_seed": sampling_seed,
                "seconds": round(seconds, 1),
            },
            f,
            indent=1,
        )


def write_manifest(run_dir, cfg, info, ckpt_path, available, names, dataset_name, volume_dims, patch_size, ckpt_tag, started):
    import platform
    import jax

    try:
        devs = [f"{d.platform}:{getattr(d, 'device_kind', '')}" for d in jax.local_devices()]
    except Exception:
        devs = []
    cfg_dict = {k: v for k, v in vars(cfg).items()}
    man = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_seconds": round(time.time() - started, 1),
        "model": cfg.model,
        "dataset": dataset_name,
        "volume_dims": list(volume_dims),
        "patch_size": patch_size,
        "checkpoint_tag": ckpt_tag,
        "checkpoint_file": ckpt_path,
        "checkpoint_bytes": os.path.getsize(ckpt_path) if os.path.exists(ckpt_path) else None,
        "weights": info["weights_tag"],
        "ema_decay": info["ema_decay"],
        "epochs_done": info["epochs_done"],
        "global_step": info["global_step"],
        "checkpoint_format": info["format"],
        "architecture": info["arch"],
        "sigma_data": info["sigma_data"],
        "consistency": info.get("cd"),
        "diffusion": info.get("diffusion"),
        "student_compute_dtype": info.get("compute_dtype"),
        "conditioning": "porosity" if info["has_por"] else "none",
        "data_spec": info["data_spec"],
        "schedules": {n: available[n] for n in names},
        "n_samples_per_schedule": cfg.n_samples,
        "sampling_seed": cfg.sampling_seed,
        "real_patch_seed": cfg.seed,
        "config": cfg_dict,
        "devices": devs,
        "versions": {"python": platform.python_version(), "jax": jax.__version__, "numpy": np.__version__},
        "conventions": {
            "generated_samples.npy": "float32 (N, D, H, W, 1), -1 = pore, +1 = solid",
            "generated_binary.npz / real_patches_binary.npz": "bit-packed, 1 = pore",
            "real_patches.npy": "uint8 (N, D, H, W), 0 = pore, 1 = solid",
            "generated sample i": "conditioned on the porosity of real patch i (target_porosity)",
        },
    }
    path = os.path.join(run_dir, "run_manifest.json")
    with open(path, "w") as f:
        json.dump(jsonable(man), f, indent=1)
    return path
