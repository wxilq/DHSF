ENABLE_SDSM = True
ENABLE_ENHANCED_CHSM = True
ENABLE_HIERARCHICAL_PREDICTION = True

ENABLE_ADAPTIVE_PATCH = False
ENABLE_TEMPORAL_PROCESSING = False

HIERARCHICAL_PREDICTION_CONFIG = {
    'level_weights': [0.02, 0.015, 0.01, 0.005],
    'target_frame': 2,
    'main_loss_weight': 1.0,
    'aux_loss_total_weight': 0.5
}

USE_FILTERED_LABELS = False
USE_UINT8_FORMAT = True
TEMPORAL_LENGTH = 4
TEMPORAL_STRIDE = 1
TEMPORAL_OVERLAP = 1
TEMPORAL_WINDOW_SIZE = 3

IMAGE_SIZE_MODE = '512'
IMAGE_SIZE_CONFIGS = {
    '512': (512, 512),
    '640': (640, 640),
    '768': (768, 768),
    '1024': (1024, 1024)
}

IMG_SIZE = IMAGE_SIZE_CONFIGS[IMAGE_SIZE_MODE]

assert IMG_SIZE == (512, 512), f"Image size must be (512, 512), got {IMG_SIZE}"

ADAPTIVE_PATCH_TYPE = 'multi_scale'

ENABLE_ATTENTION_VISUALIZATION = True
ENABLE_EXPERIMENT_ANALYSIS = True
ENABLE_TRAINING_MONITORING = True

MEMORY_OPTIMIZATION = {
    'use_gradient_checkpointing': True,
    'use_fp16': True,
    'batch_size': 1,
    'accumulate_grad_batches': 4,
    'temporal_length': 4,
    'lightweight_neck': True,
    'lightweight_loss': True,
}

dataset_type = 'ScanNetVideoDataset'
data_root = ''

if USE_FILTERED_LABELS:
    label_dir = 'label-filt_uint8' if USE_UINT8_FORMAT else 'label-filt'
    num_classes = 41
else:
    label_dir = 'label_uint8' if USE_UINT8_FORMAT else 'label'
    num_classes = 41

class_names = [
    'wall',
    'ceiling',
    'floor',
    'room',
    'window',
    'decoration',
    'cabinet',
    'door',
    'sofa',
    'table',
    'countertop',
    'chair',
    'bathtub',
    'curtain',
    'bed',
    'fridge',
    'shelf',
    'television',
    'light',
    'baseboard',
    'sink',
    'stove',
    'decals',
    'garbagecan',
    'toilet',
    'carpet',
    'dresser',
    'laptop',
    'towel',
    'box',
    'fireplace',
    'microwave',
    'coffeemachine',
    'paper',
    'stairs',
    'dishwasher',
    'pot',
    'food',
    'instrument',
    'bottle',
    'bowl'
]

class_palette = [
    [244, 35, 232],
    [70, 70, 70],
    [128, 64, 128],
    [110, 110, 110],
    [100, 170, 200],
    [200, 100, 100],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [190, 153, 153],
    [180, 165, 180],
    [102, 102, 156],
    [255, 0, 0],
    [70, 130, 180],
    [107, 142, 35],
    [70, 130, 180],
    [220, 20, 60],
    [119, 11, 32],
    [0, 0, 142],
    [0, 60, 100],
    [0, 0, 70],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 142],
    [0, 60, 100],
    [0, 0, 70],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 142],
    [0, 60, 100],
    [0, 0, 70],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 142],
    [0, 60, 100],
    [0, 0, 70],
    [0, 80, 100],
    [0, 0, 230]
]

id_mapping = {
    2: 0,
    3: 1,
    8: 2,
    67: 3,
    21: 4,
    15: 5,
    1: 6,
    18: 6,
    6: 7,
    52: 7,
    51: 7,
    34: 7,
    76: 7,
    75: 7,
    56: 8,
    44: 9,
    9: 10,
    31: 11,
    88: 12,
    59: 13,
    81: 14,
    4: 15,
    14: 16,
    58: 17,
    7: 18,
    47: 19,
    54: 19,
    27: 20,
    23: 21,
    37: 22,
    5: 23,
    89: 24,
    69: 25,
    71: 26,
    60: 27,
    32: 28,
    55: 29,
    66: 30,
    25: 31,
    29: 32,
    10: 33,
    65: 34,
    26: 35,
    20: 36,
    13: 37,
    73: 38,
    17: 39,
    16: 40,
    0: 255,
}

complete_id_mapping = {}
for i in range(256):
    if i in id_mapping:
        complete_id_mapping[i] = id_mapping[i]
    else:
        complete_id_mapping[i] = 255

assert len(id_mapping) == 49, f"ID mapping count error: expected 49, got {len(id_mapping)}"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

backbone_config = {
    'type': 'mit_b1',
    'style': 'pytorch'
}

neck_config = None

decode_head_config = {
    'type': 'SegFormerHead',
    'in_channels': [64, 128, 320, 512],
    'in_index': [0, 1, 2, 3],
    'feature_strides': [4, 8, 16, 32],
    'channels': 256,
    'dropout_ratio': 0.1,
    'num_classes': 41,
    'norm_cfg': dict(type='SyncBN', requires_grad=True),
    'align_corners': False,
    'decoder_params': dict(embed_dim=256),
    'loss_decode': dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0
    )
}

model = dict(
    type='VideoSegFormer',
    backbone=backbone_config,
    neck=neck_config,
    decode_head=decode_head_config,
    temporal_length=TEMPORAL_LENGTH,
    enable_semantic_decomposition=False,
    enable_hierarchical_prediction=ENABLE_HIERARCHICAL_PREDICTION,
    hierarchical_config=HIERARCHICAL_PREDICTION_CONFIG,
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

workflow = [('train', 1)]

cudnn_benchmark = False

fp16 = dict(
    loss_scale='dynamic',
    initial_scale=2.**10,
    growth_interval=2000,
    backoff_factor=0.5,
    growth_factor=2.0
)

optimizer_config = dict(
    grad_clip=dict(max_norm=1.0, norm_type=2)
)

optimizer = dict(
    type='AdamW',
    lr=0.00002,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.),
            'sdsm': dict(lr_mult=0.8),
            'enhanced_chsm': dict(lr_mult=1.2),
            'adaptive_patch': dict(lr_mult=0.5),
        }
    )
)

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)

runner = dict(
    type='EpochBasedRunner',
    max_epochs=100
)

train_pipeline = []
val_pipeline = []

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    persistent_workers=False,
    pin_memory=False,
    prefetch_factor=None,
    shuffle=False,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        split='train',
        img_dir='train',
        ann_dir=label_dir,
        temporal_length=TEMPORAL_LENGTH,
        temporal_stride=TEMPORAL_STRIDE,
        temporal_overlap=TEMPORAL_OVERLAP,
        id_mapping=complete_id_mapping,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        split='val',
        img_dir='val',
        ann_dir=label_dir,
        temporal_length=TEMPORAL_LENGTH,
        temporal_stride=TEMPORAL_STRIDE,
        temporal_overlap=TEMPORAL_OVERLAP,
        id_mapping=complete_id_mapping,
        pipeline=val_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        split='test',
        img_dir='test',
        ann_dir=label_dir,
        temporal_length=TEMPORAL_LENGTH,
        temporal_stride=TEMPORAL_STRIDE,
        temporal_overlap=TEMPORAL_OVERLAP,
        pipeline=val_pipeline
    )
)

custom_hooks = []

evaluation = dict(
    interval=1,
    metric='mIoU',
    pre_eval=True,
    save_best='mIoU',
    classwise=True,
    rule='greater'
)

checkpoint_config = dict(
    by_epoch=True,
    interval=5,
    max_keep_ckpts=3,
)

log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ]
)

log_level = 'INFO'

work_dir = ''

resume_from = None
load_from = None

seed = 0
deterministic = False
device = 'cuda'

dist_params = dict(backend='nccl')

print(f"Config loaded. Work dir: {work_dir}")