import json
import os
import re

import jax.numpy as jnp


def _parse_value(text):
    try:
        return json.loads(text)
    except ValueError:
        return text


def apply_overrides(cfg, overrides):
    for item in overrides or ():
        if '=' not in item:
            raise ValueError(f"--set expects key.sub=value, got {item!r}")
        path, text = item.split('=', 1)
        keys = path.strip().split('.')
        node = cfg
        for k in keys[:-1]:
            if not isinstance(node.get(k), dict):
                raise KeyError(f"--set {path}: {k!r} is not a section of the config")
            node = node[k]
        if keys[-1] not in node:
            raise KeyError(f"--set {path}: unknown key {keys[-1]!r}")
        node[keys[-1]] = _parse_value(text)
    return cfg


def validate_config(CDCFG):
    assert CDCFG['consistency']['n_sampling'] in ('uniform', 'lognormal', 'uniform_t')
    assert CDCFG['consistency']['loss'] in ('pseudo_huber', 'l2', 'l1')
    assert CDCFG['consistency']['weighting'] in ('none', 'inv_delta', 'edm')
    assert CDCFG['teacher']['solver'] in ('heun', 'euler') and int(CDCFG['teacher']['substeps']) >= 1
    assert CDCFG['teacher']['compute_dtype'] in ('float32', 'bfloat16') and CDCFG['student']['compute_dtype'] in ('float32', 'bfloat16')
    assert float(CDCFG['teacher']['porosity_cfg_scale']) >= 0.0
    assert 0.0 <= CDCFG['consistency']['target_ema'] < 1.0
    assert 1 <= int(CDCFG['consistency'].get('skip_k', 1)) <= int(CDCFG['consistency']['num_scales']) - 1


def load_config(path, overrides=()):
    with open(path) as f:
        CDCFG = json.load(f)
    apply_overrides(CDCFG, overrides)
    CDCFG['run_tag'] = re.sub(r'[^A-Za-z0-9._-]+', '_', str(CDCFG.get('run_tag', '') or '').strip())
    validate_config(CDCFG)
    return CDCFG


def derive_paths(CDCFG, work_dir):
    tag = CDCFG['run_tag']
    suffix = f'_{tag}' if tag else ''
    work_dir = os.path.abspath(work_dir)
    ckpt_dir = os.path.join(work_dir, 'CD_out', 'ckpt')
    return dict(
        work_dir=work_dir,
        tag_suffix=suffix,
        ckpt_dir=ckpt_dir,
        ckpt_path=os.path.join(ckpt_dir, f'cd_student{suffix}.pkl'),
        results_dir=os.path.join(work_dir, 'CD_out', 'results', f'cd{suffix}'),
        student_dir=os.path.join(work_dir, f'student{suffix}'),
        config_path=os.path.join(work_dir, 'CD_out', f'cd_config{suffix}.json'),
    )


def normalise_resume(v):
    if v is True:
        return 'from_disk'
    if v is False or v is None:
        return 'no'
    t = str(v).strip().lower()
    if t in ('from_disk', 'disk', 'true'):
        return 'from_disk'
    if t in ('no', 'false', 'none', ''):
        return 'no'
    raise ValueError(f"training.resume must be 'no' or 'from_disk' (or true / false); got {v!r}")


def amp_dtype(name):
    return None if name == 'float32' else jnp.bfloat16
