import os

os.environ['CUDNN_CONV_USE_FFT'] = '0'
os.environ['CUDNN_WORKSPACE_LIMIT'] = '512'
os.environ['CUDNN_CONV_WSCAP_DBL'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import argparse
import copy
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False


def parse_args():
    parser = argparse.ArgumentParser(description='Train Enhanced Video SegFormer')
    parser.add_argument('--config',
                        default='',
                        help='Path to training config file')
    parser.add_argument('--work-dir',
                        default='',
                        help='Working directory')
    parser.add_argument('--load-from', help='Path to pretrained model')
    parser.add_argument('--resume-from', help='Path to checkpoint to resume from')
    parser.add_argument('--no-validate', action='store_true', help='Disable validation during training')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--gpu-ids', type=int, nargs='+', help='GPU IDs to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--deterministic', action='store_true', help='Enable deterministic training')
    parser.add_argument('--options', nargs='+', action=DictAction, help='Override config options')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none', help='Job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_environment():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    warnings.filterwarnings('ignore')

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}, "
              f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        raise RuntimeError("CUDA is not available. Please check the GPU environment.")


def validate_config(cfg):
    required_keys = ['model', 'data', 'optimizer', 'lr_config']
    for key in required_keys:
        if not hasattr(cfg, key):
            raise ValueError(f"Missing required config key: {key}")

    if cfg.model.type != 'VideoSegFormer':
        raise ValueError(f"Model type must be VideoSegFormer, got: {cfg.model.type}")

    if cfg.model.decode_head.num_classes != 41:
        print(f"Warning: decode_head num_classes={cfg.model.decode_head.num_classes}, expected 41")

    print("Config validated: 41-class video segmentation model")
    return True


def main():
    setup_environment()

    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    print(f"Config loaded: {args.config}")

    validate_config(cfg)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    if args.gpus is not None:
        cfg.gpu_ids = range(args.gpus)
    else:
        cfg.gpu_ids = range(1)

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    print(f"Work directory: {cfg.work_dir}")

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed

    meta = dict()
    meta['env_info'] = env_info
    meta['seed'] = args.seed
    meta['exp_name'] = osp.splitext(osp.basename(args.config))[0]

    logger.info('Building model...')
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    logger.info(f'Model:\n{model}')
    print("Model built successfully.")

    logger.info('Building datasets...')
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    print(f"Train samples: {len(datasets[0])}")
    if len(datasets) > 1:
        print(f"Val samples: {len(datasets[1])}")

    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmseg_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)

    model.CLASSES = datasets[0].CLASSES
    model.PALETTE = datasets[0].PALETTE

    print(f"Training started — timestamp: {timestamp}, "
          f"max_epochs: {cfg.runner.max_epochs}, "
          f"batch_size: {cfg.data.samples_per_gpu}")

    logger.info('Starting training...')
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)

    print("Training complete.")


if __name__ == '__main__':
    main()