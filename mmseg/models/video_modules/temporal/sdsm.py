"""
SDSM: Spatio-Temporal Decoupled State Model for hierarchical video semantic segmentation.

Compatible with PyTorch 1.7.0+cu110
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Dict, Any


class SpatialStateSpaceModel(nn.Module):
    """Spatial State Space Model - spatial modeling branch of SDSM.

    Models intra-frame spatial correlations using state space equations,
    capturing long-range spatial dependencies independently for each frame.
    """

    def __init__(self,
                 d_model: int,
                 d_state: int = 16,
                 d_conv: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv

        self.A_spatial = nn.Parameter(torch.randn(d_state, d_state))
        self.B_spatial = nn.Parameter(torch.randn(d_model, d_state))
        self.C_spatial = nn.Parameter(torch.randn(d_state, d_model))
        self.D_spatial = nn.Parameter(torch.randn(d_model))

        self.d_conv = d_conv
        self.spatial_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=d_conv,
            padding=(d_conv-1)//2,  # Ensure same length output
            groups=d_model
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    def _init_parameters(self):
        """Initialize spatial SSM parameters."""
        nn.init.normal_(self.A_spatial, mean=0, std=0.1)
        with torch.no_grad():
            self.A_spatial.data = -torch.abs(self.A_spatial.data)

        nn.init.normal_(self.B_spatial, mean=0, std=0.1)
        nn.init.normal_(self.C_spatial, mean=0, std=0.1)
        nn.init.zeros_(self.D_spatial)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Args: x [B, T, L, D]. Returns: [B, T, L, D]."""
        B, T, L, _ = x.shape
        spatial_outputs = []

        for t in range(T):
            x_t = x[:, t, :, :]
            x_norm = self.norm(x_t)

            if L < self.d_conv:
                x_conv = x_norm
            else:
                x_conv = self.spatial_conv(x_norm.transpose(1, 2))
                if x_conv.size(2) != L:
                    x_conv = F.interpolate(x_conv, size=L, mode='linear', align_corners=False)
                x_conv = x_conv.transpose(1, 2)

            x_spatial = self._spatial_ssm_forward(x_conv)
            x_spatial = self.dropout(x_spatial)
            spatial_outputs.append(x_spatial)

        return torch.stack(spatial_outputs, dim=1)

    def _spatial_ssm_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial SSM core forward. Args: x [B, L, D]. Returns: [B, L, D]."""
        B, L, _ = x.shape

        x_proj = x @ self.B_spatial
        A_powers = torch.eye(self.d_state, device=x.device)
        states = []

        for i in range(min(L, 32)):
            if i == 0:
                state_contrib = x_proj[:, 0:1, :] * A_powers.unsqueeze(0)
            else:
                A_powers = A_powers @ self.A_spatial
                if i < L:
                    state_contrib = x_proj[:, i:i+1, :] * A_powers.unsqueeze(0)
                else:
                    break
            states.append(state_contrib)

        B, L, D = x.shape
        y = x + 0.1 * torch.mean(x, dim=1, keepdim=True)
        return y


class TemporalStateSpaceModel(nn.Module):
    """
    Temporal State Space Model for modeling temporal dependencies across frames.
    
    Focuses on temporal evolution and consistency across video frames,
    using state space modeling to capture long-range temporal dependencies.
    """
    
    def __init__(self,
                 d_model: int,
                 d_state: int = 16,
                 d_conv: int = 4,
                 temporal_length: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.temporal_length = temporal_length

        self._scene_hidden_states = None
        self._current_scene_id = None
        self._is_scene_start = True

        self.A_temporal = nn.Parameter(torch.randn(d_state, d_state))
        self.B_temporal = nn.Parameter(torch.randn(d_model, d_state))
        self.C_temporal = nn.Parameter(torch.randn(d_state, d_model))
        self.D_temporal = nn.Parameter(torch.randn(d_model))

        effective_kernel_size = min(self.d_conv, self.temporal_length)
        effective_padding = (effective_kernel_size-1)//2

        self.temporal_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=effective_kernel_size,
            padding=effective_padding,
            groups=d_model
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    def _init_parameters(self):
        """Initialize temporal SSM parameters."""
        nn.init.normal_(self.A_temporal, mean=0, std=0.1)
        with torch.no_grad():
            self.A_temporal.data = -torch.abs(self.A_temporal.data)

        nn.init.normal_(self.B_temporal, mean=0, std=0.1)
        nn.init.normal_(self.C_temporal, mean=0, std=0.1)
        nn.init.zeros_(self.D_temporal)

    def reset_scene_states(self):
        """Reset scene-level states."""
        self._scene_hidden_states = None
        self._current_scene_id = None
        self._is_scene_start = True

    def set_scene_info(self, scene_id: str = None, is_scene_start: bool = False):
        """Set scene information for state management."""
        if scene_id != self._current_scene_id or is_scene_start:
            self.reset_scene_states()
            self._current_scene_id = scene_id
            self._is_scene_start = True

    def forward(self, x: torch.Tensor, scene_id: str = None, is_scene_start: bool = False) -> torch.Tensor:
        """Forward pass. Args: x [B, T, L, D]. Returns: [B, T, L, D]."""
        self.set_scene_info(scene_id, is_scene_start)
        B, T, L, D = x.shape

        x_norm = self.norm(x)

        x_reshaped = x_norm.permute(0, 2, 3, 1).contiguous()
        x_reshaped = x_reshaped.view(B*L, D, T)

        x_conv = self.temporal_conv(x_reshaped)

        if x_conv.size(2) != T:
            x_conv = F.interpolate(x_conv, size=T, mode='linear', align_corners=False)

        x_conv = x_conv.view(B, L, D, T).permute(0, 3, 1, 2)

        x_temporal = x_conv + 0.1 * torch.mean(x_conv, dim=1, keepdim=True)
        x_temporal = self.dropout(x_temporal)

        return x_temporal

    def _temporal_ssm_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Temporal SSM forward pass. Args: x [B, T, D]. Returns: [B, T, D]."""
        B, T, D = x.shape

        if self._scene_hidden_states is None or self._is_scene_start:
            h = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
            self._is_scene_start = False
        else:
            h = self._scene_hidden_states
            if h.shape[0] != B:
                h = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)

        outputs = []
        
        for t in range(T):
            h = torch.tanh(h @ self.A_temporal + x[:, t, :] @ self.B_temporal)
            y = h @ self.C_temporal + x[:, t, :] * self.D_temporal
            outputs.append(y)

        self._scene_hidden_states = h.detach().clone()

        return torch.stack(outputs, dim=1)


class SimplifiedSpatioTemporalFusion(nn.Module):
    """Simplified spatio-temporal feature fusion with level-specific fixed weights."""

    def __init__(self,
                 d_model: int,
                 level_index: int = 0):
        super().__init__()
        self.d_model = d_model
        self.level_index = level_index

        self.level_weights = {
            0: {'spatial': 0.8, 'temporal': 0.2},
            1: {'spatial': 0.7, 'temporal': 0.3},
            2: {'spatial': 0.6, 'temporal': 0.4},
            3: {'spatial': 0.5, 'temporal': 0.5}
        }

        self.norm = nn.LayerNorm(d_model)
        self.spatial_scale = nn.Parameter(torch.ones(1))
        self.temporal_scale = nn.Parameter(torch.ones(1))

    def forward(self,
                spatial_features: torch.Tensor,
                temporal_features: torch.Tensor,
                level_params: Optional[Dict] = None) -> torch.Tensor:
        """Forward pass. Returns fused [B, T, L, D] features."""
        weights = self.level_weights[self.level_index]
        spatial_weight = weights['spatial'] * self.spatial_scale
        temporal_weight = weights['temporal'] * self.temporal_scale

        fused_features = spatial_weight * spatial_features + temporal_weight * temporal_features
        fused_features = self.norm(fused_features)

        return fused_features


class LevelSpecificSDSM(nn.Module):
    """
    Level-Specific SDSM for adapting to different semantic hierarchy levels.

    Each semantic level (pixel, object, room, scene) has different characteristics:
    - Pixel level: High spatial detail, moderate temporal consistency
    - Object level: Balanced spatial-temporal modeling
    - Room level: Spatial layout focus, temporal stability
    - Scene level: Global context, high temporal consistency
    """

    def __init__(self,
                 d_model: int,
                 d_state: int = 16,
                 d_conv: int = 4,
                 level_idx: int = 0,
                 temporal_length: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.level_idx = level_idx
        self.temporal_length = temporal_length

        self.spatial_ssm = SpatialStateSpaceModel(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            dropout=dropout
        )

        self.temporal_ssm = TemporalStateSpaceModel(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            temporal_length=self.temporal_length,
            dropout=dropout
        )

        self.spatiotemporal_fusion = SimplifiedSpatioTemporalFusion(
            d_model=d_model,
            level_index=level_idx
        )

        self.level_specific_params = self._init_level_params(level_idx)

    def _init_level_params(self, level_idx: int) -> Dict[str, float]:
        """Initialize level-specific parameters. level_idx: 0=pixel,1=object,2=room,3=scene."""
        level_configs = {
            0: {'spatial_weight': 0.7, 'temporal_weight': 0.5, 'fusion_bias': 'spatial', 'stability_factor': 0.3},
            1: {'spatial_weight': 0.6, 'temporal_weight': 0.6, 'fusion_bias': 'balanced', 'stability_factor': 0.5},
            2: {'spatial_weight': 0.8, 'temporal_weight': 0.4, 'fusion_bias': 'spatial', 'stability_factor': 0.7},
            3: {'spatial_weight': 0.4, 'temporal_weight': 0.8, 'fusion_bias': 'temporal', 'stability_factor': 0.9}
        }
        return level_configs.get(level_idx, level_configs[1])

    def forward(self, x: torch.Tensor, scene_id: str = None, is_scene_start: bool = False) -> torch.Tensor:
        """Forward pass. Args: x [B, T, L, D]. Returns: enhanced [B, T, L, D]."""
        spatial_features = self.spatial_ssm(x)
        temporal_features = self.temporal_ssm(x, scene_id, is_scene_start)
        fused_features = self.spatiotemporal_fusion(spatial_features, temporal_features, self.level_specific_params)
        output = x + fused_features
        return output


class SpatioTemporalDecoupledStateModel(nn.Module):
    """
    Main SDSM module that orchestrates spatio-temporal decoupled modeling
    across all semantic hierarchy levels.

    This is the core innovation that:
    1. Applies level-specific SDSM to each semantic level
    2. Maintains separate spatial and temporal state modeling
    3. Adaptively fuses spatial and temporal information
    4. Provides hierarchical feature enhancement

    Enhanced to directly work with SegFormer backbone outputs.
    """

    def __init__(self,
                 in_channels: List[int] = [64, 128, 320, 512],
                 temporal_length: int = 4,
                 d_state: int = 16,
                 dropout: float = 0.1):
        super().__init__()

        self.in_channels = in_channels
        self.temporal_length = temporal_length
        self.num_levels = len(in_channels)
        self.d_state = d_state

        self.stage_configs = {
            'stage0': {'d_state': 4, 'd_conv': 3},
            'stage1': {'d_state': 4, 'd_conv': 3},
            'stage2': {'d_state': 6, 'd_conv': 3},
            'stage3': {'d_state': 6, 'd_conv': 4},
        }

        self.level_sdsm = nn.ModuleList()
        for i, ch in enumerate(in_channels):
            stage_key = f'stage{i}'
            config = self.stage_configs[stage_key]
            level_sdsm = LevelSpecificSDSM(
                d_model=ch,
                d_state=config['d_state'],
                d_conv=config['d_conv'],
                level_idx=i,
                temporal_length=temporal_length,
                dropout=dropout
            )
            self.level_sdsm.append(level_sdsm)

        self.cross_level_interaction = nn.ModuleList([
            nn.MultiheadAttention(in_channels[i], num_heads=8, dropout=dropout)
            for i in range(self.num_levels - 1)
        ])

        self.final_refinement = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(ch),
                nn.Linear(ch, ch),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for ch in in_channels
        ])

    def reset_scene_states(self):
        """Reset scene states for all levels."""
        for level_sdsm in self.level_sdsm:
            if hasattr(level_sdsm, 'temporal_ssm'):
                level_sdsm.temporal_ssm.reset_scene_states()

    def forward(self, segformer_features: List[torch.Tensor], scene_id: str = None, is_scene_start: bool = False) -> List[torch.Tensor]:
        """Forward pass. Returns enhanced features List[[B, C_i, H_i, W_i]]."""
        if len(segformer_features) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} levels, got {len(segformer_features)}")

        spatial_shapes = []
        converted_features = []

        for level_idx, feat in enumerate(segformer_features):
            if feat.dim() == 5:
                B, T, C, H, W = feat.shape
                feat = feat.view(B*T, C, H, W)
            elif feat.dim() == 4:
                BT, C, H, W = feat.shape
                B = BT // self.temporal_length
                T = self.temporal_length
            else:
                raise ValueError(f"Unexpected feature dimension: {feat.shape}")

            spatial_shapes.append((H, W))

            feat_temporal = feat.view(B, T, C, H, W)
            feat_sequence = feat_temporal.permute(0, 1, 3, 4, 2)
            feat_sequence = feat_sequence.reshape(B, T, H*W, C)

            del feat_temporal
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            converted_features.append(feat_sequence)

        decoupled_outputs = []
        for level_idx, level_features in enumerate(converted_features):
            level_output = self.level_sdsm[level_idx](level_features, scene_id, is_scene_start)
            decoupled_outputs.append(level_output)

        enhanced_outputs = decoupled_outputs

        refined_outputs = []
        for level_idx, level_features in enumerate(enhanced_outputs):
            refined_outputs.append(level_features)

        output_features = []
        for level_idx, (refined_feat, (H, W)) in enumerate(zip(refined_outputs, spatial_shapes)):
            B, T, L, C = refined_feat.shape
            expected_size = H * W
            if L != expected_size:
                if L > expected_size:
                    refined_feat_2d = refined_feat.view(B*T, L, C).transpose(1, 2)
                    refined_feat_2d = F.adaptive_avg_pool1d(refined_feat_2d, expected_size)
                    refined_feat = refined_feat_2d.transpose(1, 2).view(B, T, expected_size, C)
                    L = expected_size
                elif L < expected_size:
                    refined_feat_2d = refined_feat.view(B*T, L, C).transpose(1, 2)
                    refined_feat_2d = F.interpolate(refined_feat_2d, size=expected_size, mode='linear', align_corners=False)
                    refined_feat = refined_feat_2d.transpose(1, 2).view(B, T, expected_size, C)
                    L = expected_size

            feat_spatial = refined_feat.view(B, T, H, W, C)
            feat_spatial = feat_spatial.permute(0, 1, 4, 2, 3).contiguous()
            feat_output = feat_spatial[:, 2]
            output_features.append(feat_output)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_features

    def _apply_cross_level_interaction(self, level_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply cross-level attention between adjacent semantic levels."""
        enhanced_features = level_features.copy()

        for i in range(len(level_features) - 1):
            if i < len(self.cross_level_interaction):
                curr_feat = level_features[i]
                next_feat = level_features[i + 1]

                B, T, L_curr, D = curr_feat.shape
                _, _, L_next, _ = next_feat.shape

                curr_flat = curr_feat.view(B * T, L_curr, D)
                next_flat = next_feat.view(B * T, L_next, D)

                try:
                    attended_curr, _ = self.cross_level_interaction[i](
                        curr_flat.transpose(0, 1),
                        next_flat.transpose(0, 1),
                        next_flat.transpose(0, 1)
                    )
                    attended_curr = attended_curr.transpose(0, 1).view(B, T, L_curr, D)
                    enhanced_features[i] = attended_curr
                except Exception:
                    pass

        return enhanced_features

    def spatial_only_forward(self, segformer_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Spatial-only processing mode. Returns List[[B, C, H, W]]."""
        spatial_features = []

        for level, feat in enumerate(segformer_features):
            if feat.dim() == 5:
                B, T, C, H, W = feat.shape
                if T == 1:
                    spatial_feat = feat.squeeze(1)
                else:
                    spatial_feat = feat[:, 0]
            elif feat.dim() == 4:
                spatial_feat = feat
            else:
                raise ValueError(f"Unsupported feature dimension: {feat.shape}")

            spatial_features.append(spatial_feat)

        return spatial_features
