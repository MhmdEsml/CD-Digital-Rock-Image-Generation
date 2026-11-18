import argparse
import json
import os
import sys

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'configs', 'cd.json')


def build_parser():
    p = argparse.ArgumentParser(description='Consistency-distillation (CD) student training from a local EDM teacher and a local volume.')
    p.add_argument('--config', default=DEFAULT_CONFIG, help='JSON config (default: configs/cd.json)')
    p.add_argument(
        '--teacher-dir',
        default=None,
        help='directory with edm.pkl, teacher_info.json and eval_bundle.zip (written by the EDM teacher export)',
    )
    p.add_argument('--volume', default=None, help='local raw volume file (uint8, 0 = pore, 1 = grain, C order) the teacher was trained on')
    p.add_argument(
        '--volume-dims',
        type=int,
        nargs=3,
        default=None,
        metavar=('D', 'H', 'W'),
        help='volume dimensions; default: read from teacher_info.json (giving them replaces the teacher\'s data definition, with a warning)',
    )
    p.add_argument('--work-dir', default='./work_cd', help='output directory (checkpoints, results, student export in <work-dir>/student/)')
    p.add_argument('--resume', choices=['no', 'from_disk'], default=None, help='override training.resume')
    p.add_argument('--epochs', type=int, default=None, help='override training.total_epochs (and training.session_epochs)')
    p.add_argument('--batch-size', type=int, default=None, help='override training.batch_size (per device)')
    p.add_argument('--seed', type=int, default=None, help='override training.seed')
    p.add_argument('--run-tag', default=None, help='override run_tag (suffix of checkpoint / results / student folder names)')
    p.add_argument(
        '--grad-sync',
        choices=['split', 'host'],
        default='split',
        help="gradient all-reduce: 'split' = on-device pmean (default) | 'host' = averaged on the host (use if the device collective fails)",
    )
    p.add_argument(
        '--set',
        dest='overrides',
        action='append',
        default=[],
        metavar='KEY.SUB=VALUE',
        help='generic config override (value parsed as JSON), repeatable',
    )
    p.add_argument(
        '--probe', action='store_true', help='before training: run one full step at the configured batch size and report time / memory'
    )
    p.add_argument('--no-export', action='store_true', help='train only; do not write the student export afterwards')
    p.add_argument('--export-only', action='store_true', help='skip training: export the student from the existing rolling checkpoint')
    return p


def _collect_overrides(a):
    ov = list(a.overrides)
    if a.resume is not None:
        ov.append(f'training.resume="{a.resume}"')
    if a.epochs is not None:
        ov += [f'training.total_epochs={a.epochs}', f'training.session_epochs={a.epochs}']
    if a.batch_size is not None:
        ov.append(f'training.batch_size={a.batch_size}')
    if a.seed is not None:
        ov.append(f'training.seed={a.seed}')
    if a.run_tag is not None:
        ov.append(f'run_tag={json.dumps(a.run_tag)}')
    if a.volume_dims is not None:
        ov.append('data.override=' + json.dumps({'volume_dims': list(a.volume_dims)}))
    return ov


def main(argv=None):
    a = build_parser().parse_args(argv)
    import jax
    from porous.cd.config import derive_paths, load_config
    from porous.cd.data import build_data, extract_bundle, local_volume_path, resolve_data_spec
    from porous.cd.export import export_student
    from porous.cd.report import probe_train_step, size_report
    from porous.cd.teacher import load_teacher, teacher_files
    from porous.cd.train import train_cd
    from porous.core.unet import count_params

    CDCFG = load_config(a.config, _collect_overrides(a))
    print(
        'JAX',
        jax.__version__,
        '| backend:',
        jax.default_backend(),
        '| local devices:',
        jax.local_device_count(),
        '| total devices:',
        jax.device_count(),
    )
    if not a.teacher_dir or not a.volume:
        raise SystemExit('--teacher-dir and --volume are required')
    paths = derive_paths(CDCFG, a.work_dir)
    os.makedirs(paths['ckpt_dir'], exist_ok=True)
    os.makedirs(paths['results_dir'], exist_ok=True)
    with open(paths['config_path'], 'w') as f:
        json.dump(CDCFG, f, indent=2)
    print(
        f"run_tag={CDCFG['run_tag']!r} | checkpoint: {paths['ckpt_path']} | results: {paths['results_dir']} | student: {paths['student_dir']}"
    )

    files = teacher_files(a.teacher_dir)
    bundle = extract_bundle(files['eval_bundle.zip'], os.path.join(paths['work_dir'], 'CD_out', 'teacher_bundle'))
    with open(bundle['config']) as f:
        bundle['config_dict'] = json.load(f)
    print(f"bundle: edm_config.json + {len(bundle['npz'])} npz file(s): {[os.path.basename(p) for p in bundle['npz']]}")
    TEACHER = load_teacher(CDCFG, files)
    print(TEACHER['diffusion'].describe())
    print(
        f"parameters: {count_params(TEACHER['params']):,} | student: dropout 0, use_checkpoint={TEACHER['student_arch']['use_checkpoint']}"
    )

    spec = resolve_data_spec(CDCFG, TEACHER, bundle['config_dict'])
    DATA = build_data(CDCFG, TEACHER, bundle, local_volume_path(a.volume), spec=spec)
    print(
        f"training dataset: {len(DATA['dataset'])} patches of {DATA['patch_size']} | flips: 8 equiprobable states, from the per-(rank, epoch, batch) RandomState"
    )

    size_report(CDCFG, TEACHER)
    if a.probe:
        probe_train_step(CDCFG, TEACHER, DATA)

    if not a.export_only:
        print(train_cd(CDCFG, TEACHER, DATA, paths, grad_sync=a.grad_sync))

    if not a.no_export:
        export_student(CDCFG, TEACHER, paths)
    return 0


if __name__ == '__main__':
    sys.exit(main())
