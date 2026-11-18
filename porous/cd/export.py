import json
import os
import pickle
import shutil


from porous.core.checkpoint_io import load_checkpoint_file


def export_student(CDCFG, TEACHER, paths):
    EXP = CDCFG['export']
    ck = load_checkpoint_file(paths['ckpt_path'])
    decays = [float(d) for d in ck['ema_tracks']['decays']]
    j = len(decays) - 1 if EXP.get('ema_decay') is None else [abs(d - float(EXP['ema_decay'])) < 1e-12 for d in decays].index(True)
    export = {
        'format': 'cd_student_porous_inference_v1',
        'params': ck['ema_tracks']['tracks'][j],
        'ema_decay': decays[j],
        'architecture': TEACHER['student_arch'],
        'diffusion': TEACHER['dcfg'],
        'consistency': {k: CDCFG['consistency'][k] for k in ('num_scales', 'sigma_min', 'sigma_max', 'rho')},
        'recommended_schedules': CDCFG['evaluation']['schedules'],
        'conditioning': TEACHER['conditioning'],
        'student_compute_dtype': CDCFG['student']['compute_dtype'],
        'data_spec': ck['data_spec'],
        'distilled_epochs': int(ck['epoch']) + 1,
        'global_step': int(ck['global_step']),
        'teacher': ck['teacher'],
        'teacher_guidance': {'porosity_cfg_scale': CDCFG['teacher']['porosity_cfg_scale']},
    }
    out_dir = paths['student_dir']
    os.makedirs(out_dir, exist_ok=True)
    export_path = os.path.join(out_dir, 'cd_student.pkl')
    with open(export_path + '.tmp', 'wb') as f:
        pickle.dump(export, f, protocol=4)
    os.replace(export_path + '.tmp', export_path)
    print(f'wrote {export_path} ({os.path.getsize(export_path) / 1e6:.0f} MB; EMA decay {decays[j]}, epoch {ck["epoch"]})')
    del ck
    shutil.copyfile(paths['ckpt_path'], os.path.join(out_dir, 'cd_train_state.pkl'))
    with open(os.path.join(out_dir, 'cd_config.json'), 'w') as f:
        json.dump(CDCFG, f, indent=2)
    print(f'wrote {out_dir}/cd_train_state.pkl and cd_config.json')
    return export_path
