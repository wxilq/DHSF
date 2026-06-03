import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor


def parse_args():
    parser = argparse.ArgumentParser(
        description='SegFormer test script for VOC dataset')
    parser.add_argument('--config', default='rs128/segformer.b0.512x512.ade.160k.py',
                        help='test config file path')
    parser.add_argument('--checkpoint', default='rs128/latest.pth',
                        help='checkpoint file')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', default='rs128_val/res.pkl',
                        help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
             'useful when you want to format the result to a specific format and '
             'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default=['mIoU'],
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
             ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', default='rs128_val/vis_results',
        help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
             'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')

    # VOC数据集特定参数
    parser.add_argument('--data-root', default='data/VOCdevkit/VOC2012',
                        help='VOC dataset root path')
    parser.add_argument('--val-list', default='data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt',
                        help='validation list file')
    parser.add_argument('--img-dir', default='data/VOCdevkit/VOC2012/JPEGImages',
                        help='images directory')
    parser.add_argument('--ann-dir', default='data/VOCdevkit/VOC2012/SegmentationClass',
                        help='annotations directory')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def setup_voc_dataset_config(cfg, args):
    """设置VOC数据集配置"""
    # 修改测试数据集配置
    if hasattr(cfg.data, 'test'):
        # 设置数据集类型为PascalVOCDataset
        cfg.data.test.type = 'PascalVOCDataset'
        cfg.data.test.data_root = args.data_root
        cfg.data.test.img_dir = os.path.basename(args.img_dir)
        cfg.data.test.ann_dir = os.path.basename(args.ann_dir)

        # 修复：正确设置split路径 - 相对于data_root的路径
        val_list_relative = os.path.relpath(args.val_list, args.data_root)
        cfg.data.test.split = val_list_relative

        print(f"Debug: val_list_relative = {val_list_relative}")
        print(f"Debug: Full split path will be: {os.path.join(args.data_root, val_list_relative)}")

        # 确保img_norm_cfg存在
        if not hasattr(cfg, 'img_norm_cfg'):
            cfg.img_norm_cfg = dict(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True
            )

        # 设置pipeline
        cfg.data.test.pipeline = [
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', **cfg.img_norm_cfg),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img']),
                ])
        ]

    return cfg


def main():
    args = parse_args()

    # 创建输出目录
    output_dir = 'rs128_val'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'Created output directory: {output_dir}')

    # 创建可视化目录
    if args.show_dir and not os.path.exists(args.show_dir):
        os.makedirs(args.show_dir)
        print(f'Created visualization directory: {args.show_dir}')

    # 检查必要的文件是否存在
    if not os.path.exists(args.config):
        raise FileNotFoundError(f'Config file not found: {args.config}')
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f'Checkpoint file not found: {args.checkpoint}')
    if not os.path.exists(args.val_list):
        raise FileNotFoundError(f'Validation list not found: {args.val_list}')
    if not os.path.exists(args.img_dir):
        raise FileNotFoundError(f'Image directory not found: {args.img_dir}')
    if not os.path.exists(args.ann_dir):
        raise FileNotFoundError(f'Annotation directory not found: {args.ann_dir}')

    print(f"Using config: {args.config}")
    print(f"Using checkpoint: {args.checkpoint}")
    print(f"Data root: {args.data_root}")
    print(f"Validation list: {args.val_list}")
    print(f"Images directory: {args.img_dir}")
    print(f"Annotations directory: {args.ann_dir}")
    print(f"GPU device: {args.gpu}")

    assert args.out or args.eval or args.format_only or args.show \
           or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if 'None' in args.eval:
        args.eval = None
    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    # 加载配置文件
    cfg = mmcv.Config.fromfile(args.config)

    # 设置VOC数据集配置
    cfg = setup_voc_dataset_config(cfg, args)

    if args.options is not None:
        cfg.merge_from_dict(args.options)

    # 设置cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # 设置数据增强
    if args.aug_test:
        # 为VOC数据集设置多尺度测试
        cfg.data.test.pipeline[1].img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        cfg.data.test.pipeline[1].flip = True

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # 初始化分布式环境
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # 构建数据加载器
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    print(f"Dataset built successfully. Number of samples: {len(dataset)}")

    # 构建模型并加载检查点
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))

    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # 设置类别信息
    if 'meta' in checkpoint and 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']
        print(f"Number of classes: {len(model.CLASSES)}")
    else:
        print("Warning: No class information found in checkpoint")

    efficient_test = True
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    # 设置设备
    if not distributed:
        model = MMDataParallel(model, device_ids=[args.gpu])
        print(f"Testing on single GPU: {args.gpu}")
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  efficient_test)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, efficient_test)

    rank, _ = get_dist_info()
    if rank == 0:
        # 保存预测结果
        if args.out:
            print(f'\nSaving prediction results to: {args.out}')
            mmcv.dump(outputs, args.out)

        kwargs = {} if args.eval_options is None else args.eval_options

        if args.format_only:
            dataset.format_results(outputs, **kwargs)

        if args.eval:
            print("Starting evaluation...")
            eval_results = dataset.evaluate(outputs, args.eval, **kwargs)

            # 保存评估结果到文件
            eval_results_file = os.path.join(output_dir, 'eval_results.txt')
            with open(eval_results_file, 'w') as f:
                f.write("SegFormer VOC Evaluation Results\n")
                f.write("=" * 50 + "\n")
                f.write(f"Config: {args.config}\n")
                f.write(f"Checkpoint: {args.checkpoint}\n")
                f.write(f"Dataset: {args.data_root}\n")
                f.write(f"Number of test samples: {len(dataset)}\n")
                f.write("-" * 50 + "\n")
                for key, value in eval_results.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")

            print(f'\nEvaluation results saved to: {eval_results_file}')
            print("Evaluation Summary:")
            print("-" * 30)
            for key, value in eval_results.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")

        print(f"\nAll results saved to: {output_dir}")
        if args.show_dir:
            print(f"Visualization results saved to: {args.show_dir}")


if __name__ == '__main__':
    main()