import hashlib
import json
import os
import zipfile

import numpy as np
import jax

from porous.core.nn import KeyGen
from porous.core.unet import make_unet
from porous.core.checkpoint_io import fill_tree, load_checkpoint_file, save_checkpoint_file
from porous.edm.fingerprints import check_fingerprints
from porous.core.fingerprint import FP_C_NOISE, FP_N_VALUES, FP_POROSITIES, FP_RTOL_NORM, FP_SEED, FP_SPATIAL, fingerprints_for
from porous.core.checkpoint_io import hash_state


def file_hash(path):
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def export_teacher(out_dir, checkpoint_path, config_path, cache_path, allow_unfinished=False):
    for p in (checkpoint_path, config_path, cache_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"not found: {p}")
    os.makedirs(out_dir, exist_ok=True)
    edm_path, info_path, bundle_path = (os.path.join(out_dir, n) for n in ("edm.pkl", "teacher_info.json", "eval_bundle.zip"))

    raw = load_checkpoint_file(checkpoint_path)
    edm_cfg_disk = json.load(open(config_path))
    total_epochs = edm_cfg_disk["training"]["total_epochs"]
    ck_cfg = raw.get("config", {})
    arch = ck_cfg.get("architecture", {})
    for _blk in ("data", "architecture"):
        if _blk in ck_cfg and json.dumps(ck_cfg[_blk], sort_keys=True, default=str) != json.dumps(
            edm_cfg_disk.get(_blk), sort_keys=True, default=str
        ):
            raise RuntimeError(
                f"{config_path} has a different '{_blk}' block than the checkpoint's own config: it is stale (another run?). Point --work-dir at the right run."
            )
    cz = np.load(cache_path, allow_pickle=False)
    data = ck_cfg.get("data") or edm_cfg_disk["data"]
    tr_ema = raw.get("ema_tracks") or {}
    info = dict(
        teacher_kind="porosity" if raw.get("conditioning") == "porosity" else "none",
        conditioning=raw.get("conditioning"),
        epoch=int(raw["epoch"]),
        global_step=int(raw["global_step"]),
        total_epochs=int(total_epochs),
        mode=raw.get("mode"),
        facies_values=[int(v) for v in raw.get("facies_values", [])],
        dataset=str(cz["dataset"].item()),
        data_source=str(cz["source"].item()),
        volume_dims=[int(v) for v in cz["volume_dims"]],
        patch_size=int(cz["patch_size"]),
        stride=[int(v) for v in cz["stride"]],
        holdout_axis=int(cz["holdout_axis"]),
        holdout_size=int(cz["holdout_size"]),
        flip_prob=data.get("flip_prob"),
        seed=int(ck_cfg.get("training", {}).get("seed", edm_cfg_disk["training"]["seed"])),
        stats_seed=int(cz["stats_seed"]),
        stats_n_patches=int(cz["stats_n_patches"]),
        data_mean=float(cz["data_mean"]),
        data_std=float(cz["data_std"]),
        data_rms=float(cz["data_rms"]),
        porosity_train_min=float(cz["porosity_train_min"]),
        porosity_train_max=float(cz["porosity_train_max"]),
        porosity_train_mean=float(cz["porosity_train_mean"]),
        porosity_train_std=float(cz["porosity_train_std"]),
        n_origins=int(len(cz["origins"])),
        n_train_patches=int((cz["split"] == 0).sum()),
        n_holdout_patches=int((cz["split"] == 1).sum()),
        sigma_data=ck_cfg.get("diffusion", {}).get("sigma_data"),
        architecture=arch,
        diffusion=ck_cfg.get("diffusion"),
        compute_dtype=ck_cfg.get("training", {}).get("compute_dtype", "float32"),
        ema_decays=tr_ema.get("decays"),
        ema_for_sampling=ck_cfg.get("training", {}).get("ema_for_sampling"),
        ckpt_net_hash=hash_state(raw["net_state_dict"]),
        ckpt_ema_hash=hash_state(raw["ema_state_dict"]),
        config_hash=file_hash(config_path),
        porosity_cache_hash=file_hash(cache_path),
        volume_hash=str(cz["volume_hash"].item()),
    )
    _fp_unet = make_unet(arch)
    _fp_template = {"net": _fp_unet.init(KeyGen(jax.random.PRNGKey(0)))}
    _fp_live = fill_tree(_fp_template, {k: np.asarray(v, np.float32) for k, v in raw["net_state_dict"].items()})["net"]
    info.update(fingerprints_for(_fp_unet, _fp_live))
    info["fingerprint_spec"] = dict(
        weights="net_state_dict (live weights, not an EMA track)",
        matmul_precision="highest",
        porosities=list(FP_POROSITIES),
        c_noise=list(FP_C_NOISE),
        spatial=list(FP_SPATIAL),
        seed=FP_SEED,
        n_values=FP_N_VALUES,
        rtol_norm=FP_RTOL_NORM,
    )
    check_fingerprints(json.loads(json.dumps(info, default=lambda o: np.asarray(o).tolist())), _fp_unet, _fp_live)
    save_checkpoint_file(edm_path, raw)
    _chk = load_checkpoint_file(edm_path)
    assert (
        hash_state(_chk["net_state_dict"]) == info["ckpt_net_hash"] and hash_state(_chk["ema_state_dict"]) == info["ckpt_ema_hash"]
    ), "re-written checkpoint differs from the original"
    del raw, _chk
    print(json.dumps({k: v for k, v in info.items() if k not in ("architecture", "diffusion")}, indent=1))
    _is_por = info["conditioning"] == "porosity"
    assert (
        info["conditioning"] in ("porosity", "none") and ((arch.get("fraction_dim") or 0) > 0) == _is_por
    ), f"checkpoint (conditioning={info['conditioning']}, fraction_dim={arch.get('fraction_dim')}) is inconsistent"
    assert (
        allow_unfinished or info["epoch"] == total_epochs - 1
    ), f"training not finished (epoch {info['epoch']} of {total_epochs}); use allow_unfinished to export anyway"

    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(config_path, "edm_config.json")
        zf.write(cache_path, "porosity_cache.npz")
    assert os.path.getsize(bundle_path) < 200e6, "the bundle is suspiciously large -- it must not contain the volume or patches"
    print(f"bundle: edm_config.json + porosity_cache.npz -> {os.path.getsize(bundle_path) / 1e6:.2f} MB (no volume, no patches)")

    json.dump(info, open(info_path, "w"), indent=2, default=lambda o: np.asarray(o).tolist())
    print(f"teacher exported to {out_dir}: edm.pkl, teacher_info.json, eval_bundle.zip")
    return info
