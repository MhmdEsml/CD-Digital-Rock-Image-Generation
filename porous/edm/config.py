import json
import os
import re


def _parse_value(v):
    try:
        return json.loads(v)
    except Exception:
        return v


def apply_overrides(CFG, overrides):
    for item in overrides or []:
        key, _, val = item.partition('=')
        if not _:
            raise ValueError(f"override {item!r} must look like key.sub=value")
        node = CFG
        parts = key.split('.')
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(val)
    return CFG


def load_config(path, overrides=None):
    with open(path) as f:
        CFG = json.load(f)
    return apply_overrides(CFG, overrides)


def get_stride(data_cfg):
    s = data_cfg['stride']
    s = [int(s)] * 3 if isinstance(s, (int, float)) else [int(v) for v in s]
    assert len(s) == 3 and all(v > 0 for v in s), s
    return tuple(s)


def resolve_sigma_data(dcfg, data_rms=None):
    sd = dcfg.get('sigma_data', 0.5)
    if isinstance(sd, str):
        if sd.strip().lower() != 'auto':
            raise ValueError(f"diffusion.sigma_data must be a number or 'auto', got {sd!r}")
        if data_rms is None:
            raise ValueError(
                "diffusion.sigma_data='auto' but no data_rms is available: run the data-cache cell (it measures it) "
                "or pass data_rms to resolve_sigma_data"
            )
        sd = round(float(data_rms), 4)
        dcfg['sigma_data'] = sd
        print(f"sigma_data='auto' -> {sd} (= data_rms rounded to 4 decimals)")
    return float(sd)


def derive_config(CFG, work_dir):
    work_dir = os.path.abspath(work_dir)
    CFG['work_dir'] = work_dir
    CFG['config_path'] = f'{work_dir}/edm_config.json'
    CFG['patch_cache_path'] = f'{work_dir}/EDM_out/porosity_cache.npz'
    os.makedirs(f'{work_dir}/EDM_out/ckpt', exist_ok=True)

    CONDITIONING = CFG.get('conditioning', 'porosity')
    assert CONDITIONING in ('porosity', 'none'), f'conditioning must be "porosity" or "none", got {CONDITIONING!r}'
    USE_POROSITY = CONDITIONING == 'porosity'
    CFG['conditioning'] = CONDITIONING
    RUN_TAG = re.sub(r'[^A-Za-z0-9._-]+', '_', str(CFG.get('run_tag', '') or '').strip())
    CFG['run_tag'] = RUN_TAG
    _TAG_SUFFIX = f'_{RUN_TAG}' if RUN_TAG else ''
    CFG['training']['results_dir'] = f"{work_dir}/EDM_out/results/porous_edm{_TAG_SUFFIX}"
    CFG['training']['checkpoint_path'] = f"{work_dir}/EDM_out/ckpt/edm_porous_jax_ckpt{_TAG_SUFFIX}.pkl"
    CFG['facies_values'] = [int(v) for v in CFG.get('facies_values', [0, 1])]

    _arch, _data = CFG['architecture'], CFG['data']
    assert _arch['dims'] == 3, 'this code trains a true 3-D U-Net: architecture.dims must be 3'
    assert _arch['porosity_embedding'] in ('linear', 'fourier'), _arch['porosity_embedding']
    _arch['out_channels'] = CFG['image_channels']
    _arch['in_channels'] = CFG['image_channels']
    _arch['fraction_dim'] = 1 if USE_POROSITY else 0
    PATCH = int(_data['patch_size'])
    CFG['image_size'] = PATCH
    _arch['image_size'] = PATCH
    _levels = len(_arch['channel_mult'])
    assert (
        PATCH % (2 ** (_levels - 1)) == 0
    ), f"patch_size={PATCH} must be divisible by 2^(len(channel_mult)-1) = {2 ** (_levels - 1)} (isotropic down/up-sampling)"
    assert _data['source'] == 'local', _data['source']
    if not _data.get('volume_path'):
        raise ValueError("data.volume_path is not set: pass --volume /path/to/volume.raw (or set data.volume_path in the config)")

    print(
        f"conditioning: {CONDITIONING} | run_tag: {RUN_TAG!r} | patch {PATCH}^3 | stride {get_stride(_data)} | "
        f"hold-out: axis {_data['holdout_axis']}, size {_data['holdout_size']} | dataset: {_data['dataset']} ({_data['source']})"
    )
    print(
        f"Diffusion: EDM sigma_data={CFG['diffusion'].get('sigma_data')} sigma_dist={CFG['diffusion'].get('sigma_dist')} "
        f"-> architecture.out_channels = {_arch['out_channels']}, fraction_dim (porosity) = {_arch['fraction_dim']}"
    )
    return CFG
