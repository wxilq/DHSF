import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.segformer_head import SegFormerHead
from mmseg.ops import resize
from mmseg.models.losses import accuracy


@HEADS.register_module()
class HierarchicalSegformerHead(SegFormerHead):
    """Hierarchical SegFormer Head with multi-level semantic predictions.

    This head extends the standard SegFormer head to support hierarchical
    semantic segmentation with different levels of granularity.

    Args:
        enable_hierarchical_predictions (bool): Enable hierarchical predictions.
        hierarchical_channels (list): Channel numbers for each hierarchical level.
        **kwargs: Arguments passed to parent SegFormerHead.
    """

    def __init__(self,
                 enable_hierarchical_predictions=True,
                 hierarchical_channels=[64, 128, 256, 512],
                 **kwargs):

        super(HierarchicalSegformerHead, self).__init__(**kwargs)

        self.enable_hierarchical_predictions = enable_hierarchical_predictions
        self.hierarchical_channels = hierarchical_channels

    def forward(self, inputs):
        """Forward function."""
        seg_logits = super(HierarchicalSegformerHead, self).forward(inputs)
        return seg_logits

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training."""
        seg_logits = self.forward(inputs)

        if self.enable_hierarchical_predictions:
            B, C, H, W = seg_logits.shape

            hierarchical_preds = {
                'pixel': seg_logits,
                'object': F.avg_pool2d(seg_logits, kernel_size=2, stride=2),
                'room': F.avg_pool2d(seg_logits, kernel_size=4, stride=4),
                'scene': F.avg_pool2d(seg_logits, kernel_size=8, stride=8)
            }

            losses = self.losses(hierarchical_preds, gt_semantic_seg)
            return losses
        else:
            losses = self.losses(seg_logits, gt_semantic_seg)
            return losses

    def losses(self, seg_logit, seg_label):
        """Compute segmentation losses."""
        if isinstance(seg_logit, dict):
            loss = dict()

            if len(seg_label.shape) == 4:
                hierarchical_targets = seg_label.squeeze(1)
            elif len(seg_label.shape) == 3:
                hierarchical_targets = seg_label
            else:
                raise ValueError(f"Unexpected seg_label shape: {seg_label.shape}")

            if hasattr(self.loss_decode, '__class__') and 'hierarchical' in self.loss_decode.__class__.__name__.lower():
                hierarchical_loss = self.loss_decode(seg_logit, hierarchical_targets)
                if isinstance(hierarchical_loss, dict):
                    loss.update(hierarchical_loss)
                else:
                    loss['loss_hierarchical'] = hierarchical_loss
            else:
                pixel_logit = seg_logit['pixel']
                if pixel_logit.shape[2:] != hierarchical_targets.shape[-2:]:
                    pixel_logit = resize(
                        input=pixel_logit,
                        size=hierarchical_targets.shape[-2:],
                        mode='bilinear',
                        align_corners=self.align_corners)

                loss['loss_seg'] = self.loss_decode(
                    pixel_logit,
                    hierarchical_targets,
                    ignore_index=self.ignore_index)

            pixel_logit = seg_logit['pixel']
            if pixel_logit.shape[2:] != hierarchical_targets.shape[-2:]:
                pixel_logit = resize(
                    input=pixel_logit,
                    size=hierarchical_targets.shape[-2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
            loss['acc_seg'] = accuracy(pixel_logit, hierarchical_targets)

            return loss
        else:
            return super(HierarchicalSegformerHead, self).losses(seg_logit, seg_label)
