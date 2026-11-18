import copy
import json
import os

import numpy as np
import jax
import jax.numpy as jnp

from porous.core.nn import KeyGen, flatten_tree
from porous.core.edm import ModelWrapper, build_diffusion
from porous.core.unet import make_unet
from porous.core.checkpoint_io import fill_tree, load_pickle_file
from porous.core.checkpoint_io import hash_state
from porous.core.fingerprint import fp_compare, network_fingerprint, porosity_fingerprint


def _resolve_conditioning(ck, info, arch):
    fd = int(arch.get('fraction_dim') or 0)
    if fd not in (0, 1):
        raise ValueError(
            f'architecture.fraction_dim = {fd}: this code distils a porosity-conditioned (fraction_dim = 1) or unconditioned (0) teacher'
        )
    cond = 'porosity' if fd == 1 else 'none'
    for where, v in (
        ('checkpoint.conditioning', ck.get('conditioning')),
        ('checkpoint.config.conditioning', (ck.get('config') or {}).get('conditioning')),
        ('teacher_info.conditioning', info.get('conditioning')),
    ):
        if v is not None and str(v).lower() != cond:
            raise ValueError(
                f"{where} = {v!r} disagrees with architecture.fraction_dim = {fd} (-> {cond!r}): the checkpoint and teacher_info.json do not belong together"
            )
    if ck.get('mode') not in (None, 'unconditional'):
        print(
            f"[warn] checkpoint.mode = {ck.get('mode')!r}: the EDM teacher training code always writes 'unconditional' (informational only, ignored)"
        )
    return cond, 'architecture.fraction_dim'


def check_fingerprints(info, unet, net_params, allow_missing=False):
    need = ['network_fingerprint'] + (['porosity_fingerprint'] if getattr(unet, 'fraction_dim', 0) else [])
    missing = [k for k in need if info.get(k) is None]
    if missing:
        msg = (
            f'teacher_info.json has no {missing}: the U-Net of this code cannot be compared numerically with the teacher\'s, so a same-shape / different-function '
            'mismatch (another Fourier frequency base, the embedding injected elsewhere, anisotropic sampling, ...) would load silently. Export the teacher again with the EDM '
            'code of this repository (it writes both fingerprints), or set teacher.allow_missing_fingerprint = true in the config '
            'to continue without this safeguard.'
        )
        if not allow_missing:
            raise RuntimeError(msg)
        print('[WARN] ' + msg)
    res = {}
    fp = info.get('porosity_fingerprint')
    if fp is not None and need[-1] == 'porosity_fingerprint':
        got = porosity_fingerprint(unet, net_params, fp['porosities'])
        c1, c2 = fp_compare(fp['embedding'], got['embedding']), fp_compare(fp['null'], got['null'])
        res['porosity'] = (c1, c2)
        print(
            f"porosity fingerprint: embedding rel. error {c1['norm_rel']:.2e} (max |diff| {c1['max_abs']:.2e}), null rel. error {c2['norm_rel']:.2e} (max |diff| {c2['max_abs']:.2e})"
        )
        if not (c1['ok'] and c2['ok']):
            raise RuntimeError(
                'the porosity branch of this code computes something different from the teacher\'s although the parameter names and shapes match: the embedding code '
                f'(Fourier frequency base / levels / layout / null embedding) differs. embedding {c1}, null {c2}. Use the U-Net code of the EDM teacher training verbatim.'
            )
    nf = info.get('network_fingerprint')
    if nf is not None:
        got = network_fingerprint(unet, net_params, nf['shape'], nf['c_noise'], nf['porosities'], nf['seed'], nf['n_values'])
        c = fp_compare(nf['forward'], got['forward'])
        res['network'] = c
        print(
            f"network fingerprint: relative error {c['norm_rel']:.2e} (max |diff| {c['max_abs']:.2e}) over {len(nf['forward'])} forward passes"
        )
        if not c['ok']:
            raise RuntimeError(
                'the FULL network output of this code differs from the teacher\'s on the fixed fingerprint input although the parameter names and shapes match '
                f'(and the porosity branch itself may agree): {c}. The way the embedding is combined, the down/up-sampling layers or the normalisation differ. '
                'Use the U-Net code of the EDM teacher training verbatim.'
            )
    return res


def explain_key_mismatch(expected, got, what='checkpoint'):
    miss, extra = sorted(set(expected) - set(got)), sorted(set(got) - set(expected))
    if not miss and not extra:
        return
    raise AssertionError(
        f'the {what} parameter names do not match the rebuilt U-Net.\n  in the model but not in the {what} ({len(miss)}): {miss[:8]}\n'
        f'  in the {what} but not in the model ({len(extra)}): {extra[:8]}\n'
        'The U-Net of the EDM teacher training must be used unchanged (architecture keys and parameter names must match).'
    )


def build_teacher(CDCFG, info, ck):
    net_hash, ema_hash = hash_state(ck['net_state_dict']), hash_state(ck['ema_state_dict'])
    if CDCFG['teacher'].get('verify_hash', True):
        for name, got, key in (('net_state_dict', net_hash, 'ckpt_net_hash'), ('ema_state_dict', ema_hash, 'ckpt_ema_hash')):
            if key not in info:
                print(f'[warn] teacher_info.json has no {key}: cannot verify {name}')
            elif info[key] != got:
                raise RuntimeError(
                    f'the teacher checkpoint does not match teacher_info.json ({name}: {got} != {info[key]}): corrupt or mismatched export'
                )
        print('teacher hashes verified against teacher_info.json')

    arch = copy.deepcopy(ck['config']['architecture'])
    dcfg = copy.deepcopy(ck['config']['diffusion'])
    if isinstance(dcfg.get('sigma_data'), str):
        raise RuntimeError(f"the teacher checkpoint stores an unresolved sigma_data={dcfg['sigma_data']!r}")
    diffusion = build_diffusion(dcfg)
    conditioning, cond_src = _resolve_conditioning(ck, info, arch)
    print(
        f"teacher: epoch {ck.get('epoch')} (info says {info.get('epoch')} / total {info.get('total_epochs')}), global_step {ck.get('global_step')}, "
        f"conditioning = {conditioning} ({cond_src}), dims = {arch.get('dims')}, porosity embedding = {arch.get('porosity_embedding')} (levels {arch.get('porosity_fourier_levels')})"
    )
    if info.get('total_epochs') is not None and ck.get('epoch') is not None and int(ck['epoch']) != int(info['total_epochs']) - 1:
        print(f"[warn] the teacher is not the final epoch of its run (epoch {ck['epoch']} of {info['total_epochs']})")

    assert int(arch.get('dims', 3)) == 3, f"architecture.dims = {arch.get('dims')}: this code distils a 3-D teacher"
    arch['dims'] = 3
    assert int(arch['in_channels']) == int(
        arch.get('out_channels', 1)
    ), f"in_channels ({arch['in_channels']}) != out_channels ({arch.get('out_channels')}): the teacher has extra input channels, which the student does not feed"
    assert (int(arch.get('fraction_dim') or 0) > 0) == (
        conditioning == 'porosity'
    ), f"architecture.fraction_dim={arch.get('fraction_dim')} disagrees with the conditioning {conditioning!r}"
    assert not int(arch.get('num_classes') or 0) and not arch.get(
        'with_fourier_features'
    ), 'unsupported teacher architecture (class labels / extra Fourier input channels)'
    image_channels = int(arch['out_channels'])

    names = list(ck['net_state_dict'])
    want = CDCFG['teacher'].get('ema_decay')
    tracks = ck.get('ema_tracks')
    if tracks:
        decays = [float(d) for d in tracks['decays']]
        if want is None:
            j = 0
        else:
            hit = [i for i, d in enumerate(decays) if abs(d - float(want)) <= 1e-12]
            if not hit:
                raise ValueError(f'teacher.ema_decay={want} is not an EMA track of the teacher; available: {decays}')
            j = hit[0]
        flat = {k: np.asarray(v, np.float32) for k, v in tracks['tracks'][j].items()}
        used = f'ema_tracks[{j}] (decay {decays[j]})'
    else:
        sd_ = ck['ema_state_dict']
        flat = {k: np.asarray(sd_[f'shadow_params.{i}'], np.float32) for i, k in enumerate(names)}
        used = 'ema_state_dict (single EMA track)'
    print('teacher weights =', used)

    unet = make_unet(arch)
    template = {'net': unet.init(KeyGen(jax.random.PRNGKey(0)))}
    param_order = list(flatten_tree(template))
    explain_key_mismatch(set(param_order), set(names), 'checkpoint (net_state_dict)')
    explain_key_mismatch(set(param_order), set(flat), 'checkpoint (EMA track)')
    params = fill_tree(template, flat)
    check_fingerprints(
        info,
        unet,
        fill_tree(template, {k: np.asarray(v, np.float32) for k, v in ck['net_state_dict'].items()})['net'],
        allow_missing=bool(CDCFG['teacher'].get('allow_missing_fingerprint', False)),
    )
    t_smin, t_smax = dcfg.get('sigma_min'), dcfg.get('sigma_max')
    c_smin, c_smax = float(CDCFG['consistency']['sigma_min']), float(CDCFG['consistency']['sigma_max'])
    if t_smin is not None and t_smax is not None and (abs(float(t_smin) - c_smin) > 1e-12 or abs(float(t_smax) - c_smax) > 1e-12):
        print(
            f'[WARN] the CD sigma range [{c_smin}, {c_smax}] differs from the range the teacher was trained / sampled on [{t_smin}, {t_smax}]: outside it the teacher denoiser is '
            'extrapolating, which makes the distillation targets unreliable there. Use the teacher\'s range unless you have a reason not to.'
        )

    s_arch = dict(arch, dropout=0.0, use_checkpoint=bool(CDCFG['student']['use_checkpoint']))
    s_arch['porosity_dropout'] = 0.0
    student_unet = make_unet(s_arch)
    explain_key_mismatch(set(param_order), set(flatten_tree({'net': student_unet.init(KeyGen(jax.random.PRNGKey(1)))})), 'student')

    return dict(
        info=info,
        ckpt_epoch=int(ck.get('epoch', -1)),
        arch=arch,
        student_arch=s_arch,
        dcfg=dcfg,
        diffusion=diffusion,
        conditioning=conditioning,
        has_por=conditioning == 'porosity',
        image_channels=image_channels,
        unet=unet,
        student_unet=student_unet,
        template=template,
        param_order=param_order,
        params=params,
        ema_used=used,
        net_hash=net_hash,
        ema_hash=ema_hash,
        ck_config=copy.deepcopy(ck.get('config', {})),
    )


def teacher_files(teacher_dir):
    files = {}
    for name in ('teacher_info.json', 'edm.pkl', 'eval_bundle.zip'):
        path = os.path.join(teacher_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f'{path} not found: --teacher-dir must contain edm.pkl, teacher_info.json and eval_bundle.zip (written by the EDM teacher export)'
            )
        files[name] = path
    return files


def load_teacher(CDCFG, files):
    with open(files['teacher_info.json']) as f:
        info = json.load(f)
    with open(files['edm.pkl'], 'rb') as f:
        ck = load_pickle_file(f)
    t = build_teacher(CDCFG, info, ck)
    del ck
    return t


class EMAOutputWrapper(ModelWrapper):

    def __init__(self, model, mean_channels, porosity=None, cfg_scale=1.0):
        super().__init__(model)
        self.mean_channels = mean_channels
        self.porosity = porosity
        self.cfg_scale = float(cfg_scale)

    def forward(self, x, t, **extras):
        if t.ndim == 0:
            t = jnp.broadcast_to(t, (x.shape[0],))
        has = self.porosity is not None
        s = self.cfg_scale if has else 1.0
        run = lambda use: self.model(x, t, extra=({'fraction': self.porosity} if (use and has) else {}))
        if s == 1.0:
            return run(True)
        if s == 0.0:
            return run(False)
        d_c, d_u = run(True), run(False)
        return d_u + s * (d_c - d_u)

    def __call__(self, x, t, **extras):
        return self.forward(x, t, **extras)
