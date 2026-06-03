"""
Cross-Level Fusion Module

This module implements fusion strategies for combining features
from different semantic hierarchy levels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple


class CrossLevelFusion(nn.Module):
    """
    Cross-Level Fusion Module for Hierarchical Temporal Features
    Compatible with PyTorch 1.7.0+cu110
    """
    
    def __init__(self, 
                 feature_dim: int = 256,
                 num_levels: int = 4,
                 fusion_type: str = 'attention'):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_levels = num_levels
        self.fusion_type = fusion_type
        
        if fusion_type == 'attention':
            # Multi-head attention for cross-level fusion
            self.cross_level_attention = nn.ModuleList([
                nn.MultiheadAttention(feature_dim, num_heads=8)
                for _ in range(num_levels)
            ])
        elif fusion_type == 'convolution':
            # Convolutional fusion
            self.cross_level_conv = nn.ModuleList([
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)
                for _ in range(num_levels)
            ])
        
        # Level-specific projections
        self.level_projections = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim)
            for _ in range(num_levels)
        ])
        
        # Fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(num_levels) / num_levels)
        
        # Output projection
        self.output_proj = nn.Linear(feature_dim, feature_dim)
    
    def forward(self, level_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            level_features: List of features from different levels [B, T, L_i, D]
        Returns:
            Fused features [B, T, L, D]
        """
        B, T, _, D = level_features[0].shape
        
        if self.fusion_type == 'attention':
            return self._attention_fusion(level_features)
        elif self.fusion_type == 'convolution':
            return self._convolution_fusion(level_features)
        else:
            return self._weighted_fusion(level_features)
    
    def _attention_fusion(self, level_features: List[torch.Tensor]) -> torch.Tensor:
        """Attention-based cross-level fusion (fallback to weighted fusion for compatibility)"""
        return self._weighted_fusion(level_features)
    
    def _convolution_fusion(self, level_features: List[torch.Tensor]) -> torch.Tensor:
        """Convolutional cross-level fusion"""
        # Reshape to spatial format for convolution
        spatial_features = []
        for level_feat in level_features:
            B, T, L, D = level_feat.shape
            # Assume L is a perfect square
            H = W = int(math.sqrt(L))
            if H * W == L:
                spatial_feat = level_feat.view(B, T, H, W, D)
                spatial_features.append(spatial_feat)
        
        if not spatial_features:
            return self._weighted_fusion(level_features)
        
        # Apply convolutional fusion
        fused_spatial = []
        for i in range(len(spatial_features)):
            if i < len(self.cross_level_conv):
                spatial_feat = spatial_features[i]
                conv = self.cross_level_conv[i]

                # Apply convolution: [B, T, H, W, D] -> [B*T, D, H, W]
                B, T, H, W, D = spatial_feat.shape
                spatial_conv = spatial_feat.view(B * T, H, W, D).permute(0, 3, 1, 2)
                spatial_conv = conv(spatial_conv)
                spatial_conv = spatial_conv.permute(0, 2, 3, 1).view(B, T, H, W, D)
                fused_spatial.append(spatial_conv)

        # Combine and reshape back
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = sum(w * feat for w, feat in zip(weights, fused_spatial))
        # Reshape back to temporal format
        if isinstance(fused, torch.Tensor) and len(fused.shape) == 5:
            B, T, H, W, D = fused.shape
            fused = fused.reshape(B, T, H * W, D)  # [B, T, L, D]
        else:
            # Fallback to weighted fusion
            return self._weighted_fusion(level_features)
        
        return self.output_proj(fused)
    
    def _weighted_fusion(self, level_features: List[torch.Tensor]) -> torch.Tensor:
        target_length = level_features[0].size(2)
        B, T, _, D = level_features[0].shape
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = None
        for i, (level_feat, w) in enumerate(zip(level_features, weights)):
            if level_feat.size(2) != target_length:
                # [B, T, L, D] -> [B*T, D, L]
                feat_reshaped = level_feat.reshape(B * T, level_feat.size(2), D).transpose(1, 2)
                feat_interp = F.interpolate(
                    feat_reshaped, size=target_length, mode='linear', align_corners=False
                )
                level_feat = feat_interp.transpose(1, 2).reshape(B, T, target_length, D)
            if fused is None:
                fused = w * level_feat
            else:
                fused = fused + w * level_feat
        if fused is None:
            # fallback: return zeros
            fused = torch.zeros_like(level_features[0])
        fused_reshaped = fused.view(B * T * target_length, D)
        fused_proj = self.output_proj(fused_reshaped)
        return fused_proj.view(B, T, target_length, D)


class HierarchicalFusion(nn.Module):
    """
    Hierarchical Fusion with progressive refinement
    """
    
    def __init__(self, 
                 feature_dim: int = 256,
                 num_levels: int = 4):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_levels = num_levels
        
        # Progressive fusion layers
        self.fusion_layers = nn.ModuleList([
            CrossLevelFusion(feature_dim, 2, 'attention')  # Fuse 2 levels at a time
            for _ in range(num_levels - 1)
        ])
        
        # Final fusion
        self.final_fusion = CrossLevelFusion(feature_dim, num_levels, 'attention')
    
    def forward(self, level_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Progressive hierarchical fusion
        """
        if len(level_features) == 1:
            return level_features[0]
        
        # Progressive fusion
        current_features = level_features.copy()
        
        for fusion_layer in self.fusion_layers:
            if len(current_features) >= 2:
                # Fuse first two levels
                fused = fusion_layer(current_features[:2])
                current_features = [fused] + current_features[2:]
        
        # Final fusion of all remaining levels
        if len(current_features) > 1:
            return self.final_fusion(current_features)
        else:
            return current_features[0] 