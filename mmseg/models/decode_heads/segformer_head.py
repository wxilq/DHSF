# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
import attr

from IPython import embed

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


@HEADS.register_module()
class SegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        # 处理可能的5维张量 (B, T, C, H, W) -> (B*T, C, H, W)
        if len(c4.shape) == 5:
            B, T, C, H, W = c4.shape
            n = B * T
            # 重塑所有特征为4维
            c1 = c1.view(B * T, *c1.shape[2:])
            c2 = c2.view(B * T, *c2.shape[2:])
            c3 = c3.view(B * T, *c3.shape[2:])
            c4 = c4.view(B * T, *c4.shape[2:])
        else:
            n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        # 确保所有特征图尺寸完全一致
        target_size = _c1.size()[2:]
        _c4 = resize(_c4, size=target_size, mode='bilinear', align_corners=False)
        _c3 = resize(_c3, size=target_size, mode='bilinear', align_corners=False)
        _c2 = resize(_c2, size=target_size, mode='bilinear', align_corners=False)

        # 最终检查：确保所有特征图尺寸完全一致
        features_to_cat = [_c4, _c3, _c2, _c1]
        target_h, target_w = _c1.size()[2:]

        aligned_features = []
        for feat in features_to_cat:  # 移除未使用的索引变量i
            if feat.size()[2:] != (target_h, target_w):
                feat = resize(feat, size=(target_h, target_w), mode='bilinear', align_corners=False)
            aligned_features.append(feat)

        _c = self.linear_fuse(torch.cat(aligned_features, dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        # 🔥 修复2：添加最终上采样到原图尺寸（GitHub标准SegFormer实现）
        # 当前x的尺寸是[B, num_classes, 128, 128]，需要上采样到[B, num_classes, 512, 512]
        x = resize(
            input=x,
            size=(512, 512),  # 上采样到原图尺寸
            mode='bilinear',
            align_corners=False  # SegFormer标准设置
        )

        return x
