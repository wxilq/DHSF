

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor


class LightweightHierarchicalPredictor(nn.Module):
    """Lightweight hierarchical predictor with one 1x1 conv per semantic level."""

    def __init__(self, in_channels=[64, 128, 320, 512], num_classes=14):
        super().__init__()

        self.predictors = nn.ModuleList([
            nn.Conv2d(in_ch, num_classes, kernel_size=1, bias=False)
            for in_ch in in_channels
        ])

        self.level_weights = nn.Parameter(torch.tensor([0.4, 0.3, 0.2, 0.1]))

    def forward(self, chsm_features):
        """Forward pass producing hierarchical predictions and fused prediction."""
        hierarchical_preds = []

        for i, (feat, predictor) in enumerate(zip(chsm_features, self.predictors)):
            pred = predictor(feat)
            hierarchical_preds.append(pred)

        target_size = hierarchical_preds[0].shape[-2:]

        weighted_preds = []
        for i, pred in enumerate(hierarchical_preds):
            if pred.shape[-2:] != target_size:
                pred_upsampled = F.interpolate(
                    pred,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
            else:
                pred_upsampled = pred

            weighted_pred = pred_upsampled * self.level_weights[i]
            weighted_preds.append(weighted_pred)

        final_pred = sum(weighted_preds)

        return hierarchical_preds, final_pred


@SEGMENTORS.register_module()
class VideoSegFormer(BaseSegmentor):
    """Enhanced Video SegFormer: Backbone -> SDSM -> CHSM -> Hierarchical Predictor -> Decode Head."""

    def __init__(self,
                 backbone: dict,
                 decode_head: dict,
                 neck: Optional[dict] = None,
                 auxiliary_head: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 init_cfg: Optional[dict] = None,
                 **kwargs):

        self.temporal_length = kwargs.pop('temporal_length', 4)
        self.semantic_levels = kwargs.pop('semantic_levels', 4)
        self.enable_semantic_decomposition = kwargs.pop('enable_semantic_decomposition', True)
        self.memory_efficient = kwargs.pop('memory_efficient', True)
        self.enable_hierarchical_prediction = kwargs.pop('enable_hierarchical_prediction', True)
        self.hierarchical_config = kwargs.pop('hierarchical_config', {})
        self.current_training_stage = None

        super(VideoSegFormer, self).__init__()

        self.backbone = builder.build_backbone(backbone)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.pretrained = pretrained
        self.init_cfg = init_cfg

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            self.neck = None

        self.semantic_decomposer = None

        if self.temporal_length > 1:
            from ..video_modules.temporal.sdsm import SpatioTemporalDecoupledStateModel
            self.sdsm = SpatioTemporalDecoupledStateModel(
                in_channels=self._get_backbone_channels(),
                temporal_length=self.temporal_length,
                d_state=8,
                dropout=0.1
            )
        else:
            self.sdsm = None

        if self.temporal_length > 1:
            from ..video_modules.fusion.enhanced_chsm import SimplifiedCrossHierarchicalFusion
            self.chsm = SimplifiedCrossHierarchicalFusion(
                embed_dims=self._get_backbone_channels()
            )
        else:
            self.chsm = None

        if self.enable_hierarchical_prediction:
            self.hierarchical_predictor = LightweightHierarchicalPredictor(
                in_channels=self._get_backbone_channels(),
                num_classes=self.num_classes
            )
            self.hierarchical_loss = None
        else:
            self.hierarchical_predictor = None
            self.hierarchical_loss = None

    def _safe_label_preprocessing(self, labels: torch.Tensor) -> torch.Tensor:
        """Clamp labels to valid range and set invalid values to ignore index (255)."""
        if labels.dtype != torch.long:
            labels = labels.long()
        if not labels.is_contiguous():
            labels = labels.contiguous()
        labels = torch.clamp(labels, min=0, max=255)
        invalid_mask = (labels > 24) & (labels < 255)
        labels[invalid_mask] = 255
        return labels

    def set_training_stage(self, stage_name: str):
        """Set current training stage for progressive training control."""
        self.current_training_stage = stage_name

    def _get_backbone_channels(self) -> List[int]:
        """Return backbone output channel dimensions."""
        if hasattr(self.backbone, 'embed_dims'):
            return self.backbone.embed_dims
        else:
            return [64, 128, 320, 512]
    
    def _init_decode_head(self, decode_head: dict):
        """Initialize decode head"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        
    def _init_auxiliary_head(self, auxiliary_head: Optional[dict]):
        """Initialize auxiliary head"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)
        else:
            self.auxiliary_head = None



    def extract_feat(self, img: torch.Tensor) -> List[torch.Tensor]:
        """Extract SDSM frame-3 features from temporal input [B, T, C, H, W]."""
        if img.dim() == 4:
            img = img.unsqueeze(1)

        _, T, _, _, _ = img.shape

        if self.memory_efficient and T > 4:
            temporal_features = self._extract_features_memory_efficient(img)
        else:
            temporal_features = self._extract_features_standard(img)

        if self.sdsm is not None:
            scene_id, is_scene_start = self._get_current_scene_state()
            sdsm_features = self.sdsm(temporal_features, scene_id, is_scene_start)
        else:
            sdsm_features = []
            for level_feat in temporal_features:
                frame3_feat = level_feat[:, 2]
                sdsm_features.append(frame3_feat)

        self._current_temporal_features = temporal_features

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return sdsm_features


    def _get_hierarchical_features(self):
        """Get CHSM-enhanced backbone frame-3 features for hierarchical predictor."""
        if not hasattr(self, '_current_temporal_features') or self._current_temporal_features is None:
            return None

        temporal_features = self._current_temporal_features

        if self.semantic_decomposer is not None:
            temporal_features = self.semantic_decomposer(temporal_features)

        backbone_features_frame3 = []
        for level_feat in temporal_features:
            frame3_feat = level_feat[:, 2]
            backbone_features_frame3.append(frame3_feat)

        if self.chsm is not None:
            enhanced_features = self.chsm(backbone_features_frame3)
        else:
            enhanced_features = backbone_features_frame3

        if self.neck is not None:
            enhanced_features = self.neck(enhanced_features)

        if hasattr(self, '_current_temporal_features'):
            del self._current_temporal_features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return enhanced_features




    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run decode head forward training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas, gt_semantic_seg, self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _extract_features_standard(self, img):
        """Extract multi-level features from temporal input [B, T, C, H, W]."""
        if not self.training and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        B, T, C, H, W = img.shape
        img_reshaped = img.view(B * T, C, H, W)
        features = self.backbone(img_reshaped)

        final_features = []
        for level_feat in features:
            _, C_i, H_i, W_i = level_feat.shape
            level_feat_temporal = level_feat.view(B, T, C_i, H_i, W_i)
            final_features.append(level_feat_temporal)

        return final_features

    def _extract_features_memory_efficient(self, img):
        """Memory-efficient feature extraction (falls back to standard)."""
        return self._extract_features_standard(img)

    def train_step(self, data_batch, optimizer, **kwargs):
        """Training step compatible with MMSegmentation."""
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img'].data)
        )
        return outputs

    def encode_decode(self, img: torch.Tensor, img_metas: List[Dict]) -> torch.Tensor:
        """Encode input and decode to segmentation output."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[-2:],
            mode='bilinear',
            align_corners=self.align_corners
        )
        return out

    def _decode_head_forward_test(self, x: List[torch.Tensor], img_metas: List[Dict]) -> torch.Tensor:
        """Run forward function for decode head in test mode."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x: List[torch.Tensor], img_metas: List[Dict], gt_semantic_seg: torch.Tensor) -> Dict:
        """Run forward function and calculate loss for auxiliary head in training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas, gt_semantic_seg, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            if self.auxiliary_head is not None:
                loss_aux = self.auxiliary_head.forward_train(x, img_metas, gt_semantic_seg, self.train_cfg)
                losses.update(add_prefix(loss_aux, 'aux'))
        return losses

    def forward_dummy(self, img: torch.Tensor) -> torch.Tensor:
        """Dummy forward function."""
        dummy_img_metas = [{'ori_shape': img.shape[-2:], 'img_shape': img.shape[-2:], 'pad_shape': img.shape[-2:]}]
        seg_logit = self.encode_decode(img, dummy_img_metas)
        return seg_logit

    def forward_train(self, img: torch.Tensor, img_metas=None, gt_semantic_seg: torch.Tensor = None) -> Dict:
        """Forward training: handles temporal data and computes losses."""
        if img.dim() == 4 and img.shape[0] == 4:
            img = img.unsqueeze(0)
        elif img.dim() == 4 and img.shape[-1] == 3:
            img = img.permute(0, 3, 1, 2).unsqueeze(0)
        elif img.dim() == 4:
            img = img.unsqueeze(1)

        if gt_semantic_seg is not None:
            if gt_semantic_seg.dim() == 3 and gt_semantic_seg.shape[0] == 4:
                gt_semantic_seg = gt_semantic_seg.unsqueeze(0)
            elif gt_semantic_seg.dim() == 4 and gt_semantic_seg.shape[0] != 1:
                gt_semantic_seg = gt_semantic_seg.unsqueeze(0)

        x = self.extract_feat(img)

        target_frame_idx = 2
        if gt_semantic_seg is not None and gt_semantic_seg.dim() == 4:
            gt_semantic_seg_target_frame = gt_semantic_seg[:, target_frame_idx]
        else:
            gt_semantic_seg_target_frame = gt_semantic_seg

        if img_metas is None:
            img_metas = [{}]
        decode_losses = self._decode_head_forward_train(x, img_metas, gt_semantic_seg_target_frame)
        main_loss = decode_losses['decode.loss_seg']

        hierarchical_loss = 0.0
        if self.enable_hierarchical_prediction and hasattr(self, 'hierarchical_predictor') and self.hierarchical_predictor is not None:
            enhanced_features = self._get_hierarchical_features()
            if enhanced_features is not None:
                hierarchical_preds, _ = self.hierarchical_predictor(enhanced_features)
                hierarchical_losses = self._compute_hierarchical_losses_simple(hierarchical_preds, gt_semantic_seg_target_frame)
                hierarchical_loss = hierarchical_losses['aux.loss_hierarchical']

        total_loss = 1.0 * main_loss + 1.0 * hierarchical_loss

        return {
            'decode.loss_seg': main_loss,
            'aux.loss_hierarchical': hierarchical_loss,
            'loss': total_loss
        }

    def _compute_hierarchical_losses_simple(self, hierarchical_preds, gt_semantic_seg_target_frame):
        """Compute weighted cross-entropy losses at each hierarchy level."""
        import torch.nn.functional as F

        level_weights = [0.4, 0.3, 0.2, 0.1]
        total_aux_loss = 0.0

        for pred, weight in zip(hierarchical_preds, level_weights):
            target_resized = F.interpolate(
                gt_semantic_seg_target_frame.unsqueeze(1).float(),
                size=pred.shape[2:],
                mode='nearest'
            ).squeeze(1).long()
            level_loss = F.cross_entropy(pred, target_resized, ignore_index=255)
            total_aux_loss += level_loss * weight

        return {'aux.loss_hierarchical': total_aux_loss}

    def _get_current_scene_state(self):
        """Return current scene id and whether it is a scene boundary."""
        scene_id = getattr(self, '_current_scene_id', None)
        is_scene_start = getattr(self, '_current_is_scene_start', False)
        return scene_id, is_scene_start

    def forward_test(self, imgs, img_metas=None, **kwargs):
        """Forward test with simplified data structure."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if isinstance(imgs, list) and len(imgs) > 0:
            img = imgs[0]
        else:
            img = imgs

        if img.dim() == 4 and img.shape[0] == 4:
            img = img.unsqueeze(0)
        elif img.dim() == 4 and img.shape[-1] == 3:
            img = img.permute(0, 3, 1, 2).unsqueeze(0)
        elif img.dim() == 4:
            img = img.unsqueeze(1)

        result = self.encode_decode(img, img_metas)

        result_pred = result.argmax(dim=1)
        result_pred = result_pred.cpu().numpy()
        result_pred = list(result_pred)
        return result_pred

    def simple_test(self, img: torch.Tensor, img_meta=None, rescale: bool = True) -> List:
        """Simple test with single image"""
        seg_logit = self.encode_decode(img, img_meta)
        if rescale and img_meta is not None:
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2] if isinstance(img_meta, list) and len(img_meta) > 0 else (512, 512)
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False
            )
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            return [seg_pred]
        seg_pred = seg_pred.cpu().numpy()
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas=None, rescale: bool = True) -> List:
        """Test with augmentations"""
        if isinstance(imgs, list):
            return self.simple_test(imgs[0], img_metas[0] if img_metas else None, rescale)
        else:
            return self.simple_test(imgs, img_metas, rescale)


