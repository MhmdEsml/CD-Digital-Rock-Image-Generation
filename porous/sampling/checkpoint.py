import glob
import itertools
import json
import os
import pickle
import re
import time

import numpy as np
import jax

from porous.core.nn import KeyGen, flatten_tree
from porous.core.unet import make_unet, count_params


def _np_frombuffer_compat(buf, dtype, shape, order="C", *extra, **kw):
    arr = np.frombuffer(buf, dtype=np.dtype(dtype))
    return (arr.reshape(tuple(shape), order="F") if order == "F" else arr.reshape(tuple(shape))).copy()


def _np_reconstruct_compat(*args, **kw):
    subtype = np.ndarray if (not args or args[0] in (None,)) else args[0]
    shape = args[1] if len(args) > 1 else ()
    dtype = np.dtype(args[2]) if len(args) > 2 and args[2] is not None else np.dtype(np.float64)
    return np.zeros(tuple(shape), dtype=dtype) if subtype is np.ndarray else subtype.__new__(subtype, shape, dtype)


def _np_scalar_compat(dtype, data=None):
    if data is None:
        return np.asarray(dtype).reshape(()).item()
    if isinstance(data, (bytes, bytearray)):
        return np.frombuffer(data, dtype=np.dtype(dtype))[0]
    return np.asarray(data, dtype=np.dtype(dtype)).reshape(()).item()


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "_frombuffer" and module in ("numpy.core.multiarray", "numpy._core.numeric", "numpy._core.multiarray"):
            return _np_frombuffer_compat
        if name == "_reconstruct" and module in ("numpy.core.multiarray", "numpy._core.multiarray", "numpy._core.numeric"):
            return _np_reconstruct_compat
        if name == "scalar" and module in ("numpy.core.multiarray", "numpy._core.multiarray"):
            return _np_scalar_compat
        return super().find_class(module, name)


def _is_nd_marker(o):
    return (
        isinstance(o, (list, tuple))
        and len(o) == 4
        and o[0] == "__nd__"
        and isinstance(o[1], str)
        and isinstance(o[2], (tuple, list))
        and isinstance(o[3], (bytes, bytearray))
    )


def _decode_portable(o):
    if _is_nd_marker(o):
        return np.frombuffer(o[3], dtype=o[1]).reshape(tuple(o[2])).copy()
    if isinstance(o, dict):
        return {k: _decode_portable(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_decode_portable(v) for v in o]
    if isinstance(o, tuple):
        return tuple(_decode_portable(v) for v in o)
    return o


def load_pickle(path):
    try:
        with open(path, "rb") as f:
            obj = pickle.Unpickler(f).load()
    except TypeError as e:
        if "_frombuffer" not in str(e):
            raise
        with open(path, "rb") as f:
            obj = _CompatUnpickler(f).load()
    return _decode_portable(obj)


_SHAPES_REPAIRED = [0]


def _fix_shape(arr, expected_shape, key=""):
    arr = np.asarray(arr)
    exp = tuple(expected_shape)
    if arr.shape == exp:
        return arr
    if arr.size != int(np.prod(exp)):
        raise ValueError(f"{key}: size {arr.size} != {int(np.prod(exp))} (not repairable by permutation)")
    for p in itertools.permutations(range(len(exp))):
        if arr.shape == tuple(exp[i] for i in p):
            _SHAPES_REPAIRED[0] += 1
            return np.ascontiguousarray(arr.transpose(np.argsort(p)))
    raise ValueError(f"{key}: shape {arr.shape} is not a permutation of {exp}")


def fill_tree(template, flat, prefix=""):
    if isinstance(template, dict):
        return {k: fill_tree(v, flat, f"{prefix}.{k}" if prefix else str(k)) for k, v in template.items()}
    if prefix not in flat:
        raise KeyError(f"missing key {prefix!r} in checkpoint state dict")
    arr = np.asarray(flat[prefix], np.float32)
    exp = tuple(np.shape(template))
    return arr if arr.shape == exp else _fix_shape(arr, exp, prefix)


def explain_key_mismatch(expected, got, what="checkpoint"):
    miss, extra = sorted(set(expected) - set(got)), sorted(set(got) - set(expected))
    if not miss and not extra:
        return
    raise AssertionError(
        f"the {what} parameter names do not match the rebuilt U-Net.\n"
        f"  in the model but not in the {what} ({len(miss)}): {miss[:8]}\n"
        f"  in the {what} but not in the model ({len(extra)}): {extra[:8]}"
    )


def find_last_checkpoint(root):
    hits = glob.glob(os.path.join(root, "**", "cd_student*.pkl"), recursive=True)
    hits = [p for p in hits if not p.endswith((".tmp", ".repaired"))]
    if not hits:
        raise FileNotFoundError(
            f"no cd_student*.pkl under {root}: pass the student checkpoint with --student PATH (or --ckpt PATH) "
            f"or point --cd-output-dir to the folder written by the CD training script."
        )
    main_, snaps, exports = [], [], []
    for p in hits:
        b = os.path.basename(p)
        m = re.search(r"_ep(\d+)\.pkl$", b)
        if "inference" in b:
            exports.append(p)
        elif m:
            snaps.append((int(m.group(1)), p))
        else:
            main_.append(p)
    print("  checkpoint candidates:")
    for p in sorted(hits):
        print(f"    {os.path.getsize(p) / 1e6:9.1f} MB  {p}")
    if main_:
        return max(main_, key=os.path.getmtime)
    if snaps:
        return max(snaps)[1]
    return max(exports, key=os.path.getmtime)


def load_cd_checkpoint(path, weights="ema", ema_decay=None):
    t0 = time.time()
    print(f"  loading {path} ({os.path.getsize(path) / 1e6:.0f} MB) ...")
    ck = load_pickle(path)
    fmt = ck.get("format")
    if fmt == "cd_student_porous_v1":
        cfg_all = ck["config"]
        cdcfg = cfg_all["cd"]
        if weights == "live":
            flat, wtag, decay = ck["net_state_dict"], "live", None
        elif weights == "target":
            flat, wtag, decay = ck["target_state_dict"], "target", None
        elif weights == "ema":
            decays = [float(d) for d in ck["ema_tracks"]["decays"]]
            if ema_decay is None:
                j = len(decays) - 1
            else:
                hit = [i for i, d in enumerate(decays) if abs(d - float(ema_decay)) <= 1e-12]
                if not hit:
                    raise ValueError(f"--ema-decay={ema_decay} is not a track of this checkpoint; available: {decays}")
                j = hit[0]
            flat, decay = ck["ema_tracks"]["tracks"][j], decays[j]
            wtag = f"ema{decay:g}"
        else:
            raise ValueError(f"--weights must be 'ema', 'live' or 'target', got {weights!r}")
        info = dict(
            flat=flat,
            arch=cfg_all["architecture"],
            sigma_data=float(cfg_all["diffusion"]["sigma_data"]),
            cd=dict(cdcfg["consistency"]),
            schedules=dict(cdcfg["evaluation"]["schedules"]),
            compute_dtype=cdcfg["student"]["compute_dtype"],
            matmul_precision=(cdcfg.get("training") or {}).get("matmul_precision"),
            data_spec=dict(ck.get("data_spec") or {}),
            epochs_done=int(ck["epoch"]) + 1,
            global_step=int(ck["global_step"]),
            weights_tag=wtag,
            ema_decay=decay,
            format=fmt,
        )
        print(
            f"  training-state checkpoint: {info['epochs_done']} epochs, global_step {info['global_step']}, "
            f"weights = {wtag}   (EMA tracks available: {[float(d) for d in ck['ema_tracks']['decays']]})"
        )
    elif fmt == "cd_student_porous_inference_v1":
        if weights != "ema":
            print(f"  NOTE: an exported inference file only holds the EMA weights; ignoring --weights={weights!r}")
        decay = float(ck["ema_decay"])
        info = dict(
            flat=ck["params"],
            arch=ck["architecture"],
            sigma_data=float(ck["diffusion"]["sigma_data"]),
            cd=dict(ck["consistency"]),
            schedules=dict(ck["recommended_schedules"]),
            compute_dtype=ck.get("student_compute_dtype", "float32"),
            matmul_precision=None,
            data_spec=dict(ck.get("data_spec") or {}),
            epochs_done=int(ck.get("distilled_epochs", 0)),
            global_step=int(ck.get("global_step", 0)),
            weights_tag=f"ema{decay:g}",
            ema_decay=decay,
            format=fmt,
        )
        print(f"  exported student: {info['epochs_done']} epochs, global_step {info['global_step']}, EMA {decay}")
    else:
        raise ValueError(
            f"{path}: unknown checkpoint format {fmt!r} " f"(expected 'cd_student_porous_v1' or 'cd_student_porous_inference_v1')"
        )
    info["has_por"] = int(info["arch"].get("fraction_dim") or 0) > 0
    info["flat"] = {k: np.asarray(v) for k, v in info["flat"].items()}
    info["path"] = path
    del ck
    print(
        f"  loaded in {time.time() - t0:.0f} s | conditioning = {'porosity' if info['has_por'] else 'none'} | "
        f"sigma_data = {info['sigma_data']} | student dtype = {info['compute_dtype']}"
    )
    return info


def find_last_edm_checkpoint(root):
    hits = [
        p for p in glob.glob(os.path.join(root, "**", "edm_porous_jax_ckpt*.pkl"), recursive=True) if not p.endswith((".tmp", ".repaired"))
    ]
    if not hits:
        raise FileNotFoundError(
            f"no edm_porous_jax_ckpt*.pkl under {root}: pass the teacher with --teacher-dir DIR (edm.pkl) or --ckpt PATH, or point --edm-output-dir to the folder written by the EDM training script"
        )
    print("  checkpoint candidates:")
    for p in sorted(hits):
        print(f"    {os.path.getsize(p) / 1e6:9.1f} MB  {p}")
    main_ = [p for p in hits if not re.search(r"_ep\d+\.pkl$", p)]
    if main_:
        return max(main_, key=os.path.getmtime)
    return max(hits, key=lambda p: int(re.search(r"_ep(\d+)\.pkl$", p).group(1)))


def load_edm_checkpoint(path, info_path=None, weights="ema", ema_decay=None):
    t0 = time.time()
    print(f"  loading {path} ({os.path.getsize(path) / 1e6:.0f} MB) ...")
    ck = load_pickle(path)
    if ck.get("format") is not None or "net_state_dict" not in ck or "ema_state_dict" not in ck:
        raise ValueError(f"{path} is not an EDM teacher checkpoint (format = {ck.get('format')!r}). For a CD student use --model cd.")
    cfg_all = ck["config"]
    dcfg, tr = dict(cfg_all["diffusion"]), dict(cfg_all.get("training") or {})
    if isinstance(dcfg.get("sigma_data"), str):
        raise RuntimeError(f"the teacher checkpoint stores an unresolved sigma_data={dcfg['sigma_data']!r}")
    order = list(ck["net_state_dict"])
    tracks = ck.get("ema_tracks")
    if weights == "live":
        flat, wtag, decay = ck["net_state_dict"], "live", None
    elif weights == "ema":
        if tracks:
            decays = [float(d) for d in tracks["decays"]]
            want = ema_decay if ema_decay is not None else tr.get("ema_for_sampling")
            hit = [i for i, d in enumerate(decays) if want is not None and abs(d - float(want)) <= 1e-12]
            if ema_decay is not None and not hit:
                raise ValueError(f"--ema-decay={ema_decay} is not a track of this checkpoint; available: {decays}")
            j = hit[0] if hit else len(decays) - 1
            flat, decay = tracks["tracks"][j], decays[j]
        else:
            flat = {k: ck["ema_state_dict"][f"shadow_params.{i}"] for i, k in enumerate(order)}
            decay = tr.get("ema_decay")
        wtag = f"ema{float(decay):g}" if decay is not None else "ema"
    else:
        raise ValueError(f"--weights must be 'ema' or 'live' for the EDM teacher, got {weights!r}")
    ds = {}
    if info_path:
        with open(info_path) as f:
            ti = json.load(f)
        ds = {k: ti[k] for k in ("dataset", "volume_dims", "patch_size", "stride") if ti.get(k) is not None}
    dsc = cfg_all.get("data") or {}
    for k in ("dataset", "volume_dims", "patch_size", "stride"):
        if k not in ds and dsc.get(k) is not None:
            ds[k] = dsc[k]
    info = dict(
        flat=flat,
        arch=cfg_all["architecture"],
        sigma_data=float(dcfg["sigma_data"]),
        diffusion=dcfg,
        sampling=dict(cfg_all.get("sampling") or {}),
        compute_dtype="float32",
        matmul_precision=tr.get("matmul_precision"),
        data_spec=ds,
        epochs_done=int(ck["epoch"]) + 1,
        global_step=int(ck["global_step"]),
        weights_tag=wtag,
        ema_decay=decay,
        format="edm_teacher_v1",
    )
    print(
        f"  EDM teacher checkpoint: {info['epochs_done']} epochs, global_step {info['global_step']}, weights = {wtag}"
        + (f"   (EMA tracks available: {[float(d) for d in tracks['decays']]})" if tracks else "")
    )
    info["has_por"] = int(info["arch"].get("fraction_dim") or 0) > 0
    info["flat"] = {k: np.asarray(v) for k, v in info["flat"].items()}
    info["path"] = path
    del ck
    print(
        f"  loaded in {time.time() - t0:.0f} s | conditioning = {'porosity' if info['has_por'] else 'none'} | sigma_data = {info['sigma_data']} | "
        f"sigma range [{dcfg.get('sigma_min')}, {dcfg.get('sigma_max')}], rho {dcfg.get('rho')}"
    )
    return info


def build_student_params(info):
    unet = make_unet(info["arch"])
    _SHAPES_REPAIRED[0] = 0
    try:
        cpu = jax.devices("cpu")[0]
    except Exception:
        cpu = None
    if cpu is not None:
        with jax.default_device(cpu):
            template = {"net": unet.init(KeyGen(jax.random.PRNGKey(0)))}
    else:
        template = {"net": unet.init(KeyGen(jax.random.PRNGKey(0)))}
    explain_key_mismatch(set(flatten_tree(template)), set(info["flat"]), "checkpoint")
    params = fill_tree(template, info["flat"])
    if _SHAPES_REPAIRED[0]:
        print(f"  NOTE: {_SHAPES_REPAIRED[0]} weight tensors had permuted shapes and were transposed back")
    print(f"  {'teacher' if info.get('format') == 'edm_teacher_v1' else 'student'}: {count_params(params):,} parameters")
    return unet, params


def resolve_checkpoint(model, ckpt=None, teacher_dir=None, teacher_info=None, student=None, edm_output_dir=None, cd_output_dir=None):
    if model not in ("cd", "edm"):
        raise ValueError(f"model must be 'cd' or 'edm', got {model!r}")
    if ckpt:
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(ckpt)
        return ckpt, teacher_info
    if model == "edm":
        if teacher_dir:
            teacher_pkl = os.path.join(teacher_dir, "edm.pkl")
            if not os.path.isfile(teacher_pkl):
                raise FileNotFoundError(f"{teacher_pkl} not found (--teacher-dir must contain edm.pkl, written by the EDM training script)")
            info = teacher_info or os.path.join(teacher_dir, "teacher_info.json")
            return teacher_pkl, (info if os.path.isfile(info) else None)
        if edm_output_dir:
            return find_last_edm_checkpoint(edm_output_dir), teacher_info
        raise ValueError("EDM sampling needs --teacher-dir DIR (with edm.pkl), --ckpt PATH or --edm-output-dir DIR")
    if student:
        if not os.path.isfile(student):
            raise FileNotFoundError(student)
        return student, None
    if cd_output_dir:
        return find_last_checkpoint(cd_output_dir), None
    raise ValueError("CD sampling needs --student PATH (cd_student.pkl), --ckpt PATH or --cd-output-dir DIR")
