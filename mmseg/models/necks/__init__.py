# Copyright (c) OpenMMLab. All rights reserved.
# from .fpn import FPN  # 已删除
from .segformer_temporal_neck import (
    SegFormerTemporalNeck,
    SegFormerMambaNeck,
    SegFormerB0MambaNeck,
    SegFormerB1MambaNeck,
    SegFormerB2MambaNeck,
    SegFormerB3MambaNeck,
    SegFormerB4MambaNeck,
    SegFormerB5MambaNeck,
    VideoSegFormerNeck,  # 兼容性别名
    LightweightSegFormerMambaNeck,
    SegFormerB1LightweightMambaNeck
)
# SimpleTemporalNeck已删除，直接在VideoSegFormer中集成SDSM和CHSM
# AMSSM neck modules removed to save memory

__all__ = [
    'SegFormerTemporalNeck', 'SegFormerMambaNeck',
    'SegFormerB0MambaNeck', 'SegFormerB1MambaNeck', 'SegFormerB2MambaNeck',
    'SegFormerB3MambaNeck', 'SegFormerB4MambaNeck', 'SegFormerB5MambaNeck',
    'VideoSegFormerNeck', 'LightweightSegFormerMambaNeck', 'SegFormerB1LightweightMambaNeck'
]