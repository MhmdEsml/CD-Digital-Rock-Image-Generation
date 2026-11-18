#!/usr/bin/env python
import argparse
import os
import sys

os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_triton_gemm=false").strip()
os.environ.setdefault('NCCL_DEBUG', 'WARN')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=os.path.join(HERE, 'configs', 'edm.json'), help='JSON config (default: configs/edm.json)')
    ap.add_argument('--work-dir', default='./work_edm', help='all outputs (checkpoints, cache, results, teacher export)')
    ap.add_argument(
        '--volume', default=None, help='LOCAL binary volume: raw uint8 file (needs --volume-dims) or .npy (0 = pore, 1 = grain)'
    )
    ap.add_argument('--volume-dims', type=int, nargs=3, metavar=('D', 'H', 'W'), default=None, help='shape of a raw volume')
    ap.add_argument(
        '--resume', choices=['no', 'from_disk'], default=None, help='continue from the latest local checkpoint (default: from the config)'
    )
    ap.add_argument('--epochs', type=int, default=None, help='training.total_epochs (also session_epochs)')
    ap.add_argument('--batch-size', type=int, default=None, help='training.batch_size (PER DEVICE)')
    ap.add_argument('--seed', type=int, default=None, help='training.seed')
    ap.add_argument('--grad-sync', choices=['nccl', 'host', 'split'], default=None, help='gradient averaging (default: automatic)')
    ap.add_argument(
        '--set',
        dest='overrides',
        action='append',
        default=[],
        metavar='KEY.SUB=VALUE',
        help='generic config override (JSON value); repeatable',
    )
    ap.add_argument(
        '--no-export', action='store_true', help='do not write teacher/ (edm.pkl, teacher_info.json, eval_bundle.zip) after training'
    )
    ap.add_argument('--allow-unfinished-export', action='store_true', help='export even if training has not reached total_epochs')
    ap.add_argument('--export-only', action='store_true', help='skip training: export the teacher from --checkpoint')
    ap.add_argument('--checkpoint', default=None, help='training checkpoint to export (with --export-only)')
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    work_dir = os.path.abspath(args.work_dir)

    if args.export_only:
        from porous.edm.export import export_teacher

        if not args.checkpoint:
            raise SystemExit('--export-only needs --checkpoint PATH')
        export_teacher(
            os.path.join(work_dir, 'teacher'),
            args.checkpoint,
            os.path.join(work_dir, 'edm_config.json'),
            os.path.join(work_dir, 'EDM_out', 'porosity_cache.npz'),
            allow_unfinished=args.allow_unfinished_export,
        )
        return

    from porous.edm.config import apply_overrides, derive_config, load_config

    CFG = load_config(args.config)
    if args.volume:
        CFG['data']['volume_path'] = args.volume
    if args.volume_dims:
        CFG['data']['volume_dims'] = list(args.volume_dims)
    if args.resume:
        CFG['training']['resume'] = args.resume
    if args.epochs is not None:
        CFG['training']['total_epochs'] = args.epochs
        CFG['training']['session_epochs'] = args.epochs
    if args.batch_size is not None:
        CFG['training']['batch_size'] = args.batch_size
    if args.seed is not None:
        CFG['training']['seed'] = args.seed
    apply_overrides(CFG, args.overrides)
    derive_config(CFG, work_dir)

    from porous.edm.data import prepare_data
    from porous.edm.model import build_model
    from porous.edm.train import launch_training

    volume, patch_cache = prepare_data(CFG)
    unet, params, param_order, n_params = build_model(CFG)
    del unet, params, param_order
    result = launch_training(CFG, volume=volume, n_params=n_params, grad_sync=args.grad_sync)

    if args.no_export:
        return
    finished = result['end_epoch'] >= CFG['training']['total_epochs']
    if not finished and not args.allow_unfinished_export:
        print(
            f"Training stopped at epoch {result['end_epoch']} of {CFG['training']['total_epochs']}: teacher NOT exported. "
            "Re-run with --resume from_disk to continue, then it is exported; or use --export-only --allow-unfinished-export."
        )
        return
    from porous.edm.export import export_teacher

    export_teacher(
        os.path.join(work_dir, 'teacher'),
        CFG['training']['checkpoint_path'],
        CFG['config_path'],
        CFG['patch_cache_path'],
        allow_unfinished=args.allow_unfinished_export,
    )


if __name__ == '__main__':
    main()
