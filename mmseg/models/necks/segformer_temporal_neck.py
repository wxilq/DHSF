"""
SegFormer + Mamba Temporal Neck for Video Segmentation
Compatible with PyTorch 1.7.0+cu110
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from ..builder import NECKS

try:
    from torch.utils.checkpoint import checkpoint
except ImportError:
    checkpoint = None


class MambaBlock(nn.Module):
    """Simplified Mamba Block for PyTorch 1.7.0+cu110"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.in_proj = nn.Linear(d_model, d_model * expand)
        self.out_proj = nn.Linear(d_model * expand, d_model)
        # 手动计算padding以保持序列长度
        padding = (d_conv - 1) // 2
        self.conv1d = nn.Conv1d(d_model * expand, d_model * expand, 
                               kernel_size=d_conv, padding=padding, groups=d_model * expand)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, L, D = x.shape
        x = x.view(B * T, L, D)
        x = self.in_proj(x)  # [B*T, L, D*expand]
        x = x.transpose(1, 2)  # [B*T, D*expand, L]
        x = self.conv1d(x)  # [B*T, D*expand, L_new] - 可能改变长度
        x = x.transpose(1, 2)  # [B*T, L_new, D*expand]
        x = self.out_proj(x)  # [B*T, L_new, D]
        x = self.norm(x)
        x = self.dropout(x)
        L_new = x.shape[1]  # 获取实际的序列长度
        x = x.view(B, T, L_new, D)  # 使用实际长度
        return x


@NECKS.register_module()
class SegFormerTemporalNeck(BaseModule):
    """SegFormer + Mamba Temporal Neck for 5D video inputs"""
    
    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 temporal_length: int = 8,
                 feature_dim: int = 256,
                 num_stages: int = 4,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 dropout: float = 0.1,
                 fusion_type: str = 'mamba',
                 use_gradient_checkpointing: bool = True,
                 enable_sdsm: bool = False,
                 sdsm_config: dict = None,
                 enable_chsm: bool = False,
                 chsm_config: dict = None,
                 init_cfg: Optional[Dict] = None):
        super().__init__(init_cfg)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temporal_length = temporal_length
        self.feature_dim = feature_dim
        self.num_stages = num_stages
        self.fusion_type = fusion_type
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # 创新模块配置
        self.enable_sdsm = enable_sdsm
        self.sdsm_config = sdsm_config or {}
        self.enable_chsm = enable_chsm
        self.chsm_config = chsm_config or {}
        
        if use_gradient_checkpointing and checkpoint is None:
            print("Warning: Gradient checkpointing not available")
            self.use_gradient_checkpointing = False
        
        # Input projections for 5D inputs
        self.input_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for ch in in_channels
        ])
        
        # Feature projection layers for each scale
        self.feature_projections = nn.ModuleList([
            nn.Linear(out_channels, feature_dim) if out_channels != feature_dim else nn.Identity()
            for _ in range(num_stages)
        ])
        
        # Temporal modules
        self.temporal_modules = nn.ModuleList([
            self._create_temporal_module(fusion_type, feature_dim, d_state, d_conv, expand, dropout)
            for _ in range(num_stages)
        ])
        
        # Cross-stage fusion - 轻量化版本
        self.cross_stage_fusion = nn.ModuleList([
            nn.MultiheadAttention(feature_dim, num_heads=4, dropout=dropout)  # 减少注意力头数
            for _ in range(num_stages)
        ])
        
        # Output projections - 简化版本
        self.output_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, out_channels),
                nn.ReLU(inplace=True)  # 移除LayerNorm以减少参数
            ) for _ in range(num_stages)
        ])
        
        # Spatial reconstruction - 轻量化版本
        self.spatial_reconstruction = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=1)  # 使用1x1卷积减少参数
            for _ in range(num_stages)
        ])
        
    def _create_temporal_module(self, fusion_type, feature_dim, d_state, d_conv, expand, dropout):
        if fusion_type == 'mamba':
            return MambaBlock(d_model=feature_dim, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout)
        elif fusion_type == 'attention':
            return nn.MultiheadAttention(embed_dim=feature_dim, num_heads=8, dropout=dropout)
        elif fusion_type == 'lstm':
            return nn.LSTM(input_size=feature_dim, hidden_size=feature_dim, num_layers=2, dropout=dropout, batch_first=True, bidirectional=True)
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
    
    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass for temporal neck
        Args:
            inputs: List of tensors with shape [B, T, C, H, W] for each scale
        Returns:
            List of tensors with shape [B, C, H, W] for each scale
        """
        assert len(inputs) == len(self.in_channels)
        
        # Input projections - handle 5D temporal inputs [B, T, C, H, W]
        projected_inputs = []
        for i, (input_feat, proj) in enumerate(zip(inputs, self.input_projections)):
            B, T, C, H, W = input_feat.shape
            
            # Reshape to [B*T, C, H, W] for 2D convolution
            input_feat_reshaped = input_feat.view(B * T, C, H, W)
            projected_feat = proj(input_feat_reshaped)  # [B*T, out_channels, H, W]
            
            # Reshape back to [B, T, out_channels, H, W]
            projected_feat = projected_feat.view(B, T, self.out_channels, H, W)
            projected_inputs.append(projected_feat)
        
        # Temporal modeling
        temporal_outputs = []
        for i, (projected_feat, temporal_module, feature_proj) in enumerate(
            zip(projected_inputs, self.temporal_modules, self.feature_projections)):
            
            B, T, C, H, W = projected_feat.shape
            
            # Reshape to [B, T, H*W, C] for temporal processing
            seq_feat = projected_feat.permute(0, 1, 3, 4, 2).contiguous()  # [B, T, H, W, C]
            seq_feat = seq_feat.view(B, T, H * W, C)  # [B, T, H*W, C]
            
            # Project to feature dimension
            seq_feat = feature_proj(seq_feat)  # [B, T, H*W, feature_dim]
            
            # Apply temporal modeling
            if self.use_gradient_checkpointing and self.training and checkpoint is not None:
                if self.fusion_type == 'mamba':
                    temp_out = checkpoint(temporal_module, seq_feat)
                elif self.fusion_type == 'attention':
                    temp_out, _ = checkpoint(temporal_module, seq_feat, seq_feat, seq_feat)
                elif self.fusion_type == 'lstm':
                    temp_out, _ = checkpoint(temporal_module, seq_feat)
            else:
                if self.fusion_type == 'mamba':
                    temp_out = temporal_module(seq_feat)
                elif self.fusion_type == 'attention':
                    temp_out, _ = temporal_module(seq_feat, seq_feat, seq_feat)
                elif self.fusion_type == 'lstm':
                    temp_out, _ = temporal_module(seq_feat)
            
            # 获取实际的序列长度（可能被temporal_module改变）
            L_new = temp_out.shape[2]
            
            temporal_outputs.append({
                'features': temp_out,
                'spatial_shape': (H, W),
                'sequence_length': L_new,
                'stage_idx': i
            })
        
        # Cross-stage fusion
        fused_outputs = self._cross_stage_fusion(temporal_outputs)
        
        # Output projections and spatial reconstruction
        final_outputs = []
        for i, (fused_feat, output_proj, spatial_conv) in enumerate(
            zip(fused_outputs, self.output_projections, self.spatial_reconstruction)):
            
            B, T, L_new, D = fused_feat.shape
            projected_feat = output_proj(fused_feat.view(B * T * L_new, D))
            projected_feat = projected_feat.view(B, T, L_new, self.out_channels)
            
            H, W = temporal_outputs[i]['spatial_shape']
            
            # 重新计算空间尺寸（如果序列长度改变了）
            if L_new != H * W:
                # 如果序列长度改变了，我们需要重新计算空间尺寸
                # 假设是正方形特征图
                new_size = int((L_new) ** 0.5)
                H_new, W_new = new_size, new_size
                # 如果计算出的尺寸不匹配，使用原始尺寸并截断或填充
                if H_new * W_new != L_new:
                    H_new, W_new = H, W
                    # 截断或填充到原始尺寸
                    if L_new > H * W:
                        projected_feat = projected_feat[:, :, :H*W, :]
                    else:
                        # 填充到原始尺寸
                        padding = torch.zeros(B, T, H*W - L_new, self.out_channels, 
                                            device=projected_feat.device, dtype=projected_feat.dtype)
                        projected_feat = torch.cat([projected_feat, padding], dim=2)
                    L_new = H * W
            else:
                H_new, W_new = H, W
            
            spatial_feat = projected_feat.view(B * T, L_new, self.out_channels)
            spatial_feat = spatial_feat.transpose(1, 2).view(B * T, self.out_channels, H_new, W_new)
            
            spatial_feat = spatial_conv(spatial_feat)
            spatial_feat = spatial_feat.view(B, T, self.out_channels, H_new, W_new).mean(dim=1)
            
            final_outputs.append(spatial_feat)
        
        return final_outputs
    
    def _cross_stage_fusion(self, temporal_outputs):
        fused_outputs = []
        for i, temp_out in enumerate(temporal_outputs):
            features = temp_out['features']
            B, T, L, D = features.shape
            features_reshaped = features.view(B * T, L, D)
            fused_feat, _ = self.cross_stage_fusion[i](features_reshaped, features_reshaped, features_reshaped)
            fused_feat = fused_feat.view(B, T, L, D)
            fused_outputs.append(fused_feat)
        return fused_outputs


class VideoSegFormerNeck(SegFormerTemporalNeck):
    """Specialized neck for Video SegFormer"""
    
    def __init__(self, in_channels: List[int], out_channels: int = 256, temporal_length: int = 8, fusion_type: str = 'mamba', **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            temporal_length=temporal_length,
            feature_dim=out_channels,
            num_stages=4,
            d_state=16,
            d_conv=4,
            expand=2,
            dropout=0.1,
            fusion_type=fusion_type,
            use_gradient_checkpointing=True,
            **kwargs
        )


@NECKS.register_module()
class SegFormerMambaNeck(SegFormerTemporalNeck):
    """SegFormer + Mamba Neck with optimized configuration"""

    def __init__(self, in_channels: List[int], out_channels: int = 256, temporal_length: int = 8, **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            temporal_length=temporal_length,
            feature_dim=out_channels,
            num_stages=4,
            d_state=16,
            d_conv=4,
            expand=2,
            dropout=0.1,
            fusion_type='mamba',
            use_gradient_checkpointing=True,
            **kwargs
        )


# SegFormer variant-specific neck classes
@NECKS.register_module()
class SegFormerB0MambaNeck(SegFormerMambaNeck):
    """SegFormer B0 + Mamba Neck"""
    def __init__(self, **kwargs):
        super().__init__(
            in_channels=[32, 64, 160, 256],
            out_channels=128,
            **kwargs
        )


@NECKS.register_module()
class SegFormerB1MambaNeck(SegFormerMambaNeck):
    """SegFormer B1 + Mamba Neck"""
    def __init__(self, **kwargs):
        super().__init__(
            in_channels=[64, 128, 320, 512],
            out_channels=256,
            **kwargs
        )


@NECKS.register_module()
class SegFormerB2MambaNeck(SegFormerMambaNeck):
    """SegFormer B2 + Mamba Neck"""
    def __init__(self, **kwargs):
        super().__init__(
            in_channels=[64, 128, 320, 512],
            out_channels=256,
            **kwargs
        )


@NECKS.register_module()
class SegFormerB3MambaNeck(SegFormerMambaNeck):
    """SegFormer B3 + Mamba Neck"""
    def __init__(self, **kwargs):
        super().__init__(
            in_channels=[64, 128, 320, 512],
            out_channels=256,
            **kwargs
        )


@NECKS.register_module()
class SegFormerB4MambaNeck(SegFormerMambaNeck):
    """SegFormer B4 + Mamba Neck"""
    def __init__(self, **kwargs):
        super().__init__(
            in_channels=[64, 128, 320, 512],
            out_channels=256,
            **kwargs
        )


@NECKS.register_module()
class SegFormerB5MambaNeck(SegFormerMambaNeck):
    """SegFormer B5 + Mamba Neck"""
    def __init__(self, **kwargs):
        super().__init__(
            in_channels=[64, 128, 320, 512],
            out_channels=256,
            **kwargs
        )


@NECKS.register_module()
class LightweightSegFormerMambaNeck(SegFormerTemporalNeck):
    """轻量级SegFormer + Mamba Neck，专门用于显存优化"""

    def __init__(self, in_channels: List[int], out_channels: int = 256, temporal_length: int = 3,
                 enable_sdsm: bool = False, sdsm_config: dict = None,
                 enable_chsm: bool = False, chsm_config: dict = None, **kwargs):
        # 使用更小的参数配置，但保持feature_dim与out_channels一致
        lightweight_kwargs = {
            'feature_dim': out_channels,  # 与out_channels保持一致
            'd_state': 8,            # 减少状态维度
            'd_conv': 3,             # 减少卷积核大小
            'expand': 1,             # 减少扩展倍数
            'dropout': 0.1,
            'fusion_type': 'mamba',
            'use_gradient_checkpointing': True,
            # 传递创新模块配置
            'enable_sdsm': enable_sdsm,
            'sdsm_config': sdsm_config,
            'enable_chsm': enable_chsm,
            'chsm_config': chsm_config,
        }
        lightweight_kwargs.update(kwargs)

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            temporal_length=temporal_length,
            num_stages=4,
            **lightweight_kwargs
        )
        # LightweightSegFormerMambaNeck初始化完成


@NECKS.register_module()
class SegFormerB1LightweightMambaNeck(LightweightSegFormerMambaNeck):
    """SegFormer B1 + 轻量级Mamba Neck"""
    def __init__(self, **kwargs):
        # 确保输入通道数与mit_b2的输出匹配
        super().__init__(
            in_channels=[64, 128, 320, 512],  # mit_b2的输出通道
            out_channels=256,  # 恢复到256以匹配SegFormerHead期望
            **kwargs
        )