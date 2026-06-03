import os
import os.path as osp
import mmcv
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple, Union
import gc

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ScanNetVideoDataset(CustomDataset):
    """ScanNet Video Dataset for temporal segmentation.

    Expected data structure:
      data_root/{train,val,test}/scene_id/color/*.jpg
      data_root/{train,val,test}/scene_id/label_uint8/*.png
    """

    CLASSES = [
    'wall', 'ceiling', 'floor', 'room', 'window', 'decoration',
    'cabinet', 'door', 'sofa', 'table', 'countertop', 'chair',
    'bathtub', 'curtain', 'bed', 'fridge', 'shelf', 'television',
    'light', 'baseboard', 'sink', 'stove', 'decals', 'garbagecan',
    'toilet', 'carpet', 'dresser', 'laptop', 'towel', 'box',
    'fireplace', 'microwave', 'coffeemachine', 'paper', 'stairs',
    'dishwasher', 'pot', 'food', 'instrument', 'bottle', 'bowl'
]
    
    PALETTE = [
    [244, 35, 232], [70, 70, 70], [128, 64, 128], [110, 110, 110],
    [100, 170, 200], [200, 100, 100], [153, 153, 153], [250, 170, 30],
    [220, 220, 0], [190, 153, 153], [180, 165, 180], [102, 102, 156],
    [255, 0, 0], [70, 130, 180], [107, 142, 35], [70, 130, 180],
    [220, 20, 60], [119, 11, 32], [0, 0, 142], [0, 60, 100],
    [0, 0, 70], [0, 80, 100], [0, 0, 230], [119, 11, 32],
    [0, 0, 142], [0, 60, 100], [0, 0, 70], [0, 80, 100],
    [0, 0, 230], [119, 11, 32], [0, 0, 142], [0, 60, 100],
    [0, 0, 70], [0, 80, 100], [0, 0, 230], [119, 11, 32],
    [0, 0, 142], [0, 60, 100], [0, 0, 70], [0, 80, 100], [0, 0, 230]
    ]

    def __init__(self,
                 temporal_length: int = 4,
                 temporal_stride: int = 1,
                 temporal_overlap: int = 2,
                 frame_sampling: str = 'uniform',
                 id_mapping: dict = None,
                 **kwargs):
        self.temporal_length = temporal_length
        self.temporal_stride = temporal_stride
        self.temporal_overlap = temporal_overlap
        self.frame_sampling = frame_sampling
        self.id_mapping = id_mapping

        super(ScanNetVideoDataset, self).__init__(**kwargs)

        self._current_scene_id = None
        self._current_is_scene_start = True

        self.video_infos = self._build_video_infos()
        self.video_infos = self._apply_scene_aware_sampling(self.video_infos)
        
    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        """Override to bypass parent annotation loading; handled in _build_video_infos."""
        return []

    def _build_video_infos(self) -> List[Dict]:
        """Build video sequence info list adapted to ScanNet directory structure."""
        video_infos = []
        split_dir = self.img_dir

        if not osp.exists(split_dir):
            return video_infos

        try:
            dir_contents = os.listdir(split_dir)
        except Exception:
            return video_infos

        scene_dirs = [d for d in dir_contents
                     if osp.isdir(osp.join(split_dir, d)) and d.startswith('scene')]

        for scene_name in sorted(scene_dirs):
            scene_path = osp.join(split_dir, scene_name)
            color_dir = osp.join(scene_path, 'color')
            label_dir = osp.join(scene_path, 'label_uint8')

            if not osp.exists(color_dir) or not osp.exists(label_dir):
                continue

            try:
                img_files = [f for f in os.listdir(color_dir)
                             if f.endswith(('.jpg', '.jpeg', '.png'))]

                if len(img_files) == 0:
                    continue

                def extract_frame_number(filename):
                    try:
                        return int(osp.splitext(filename)[0])
                    except ValueError:
                        return 0

                img_files.sort(key=extract_frame_number)

                if len(img_files) < self.temporal_length:
                    continue

                valid_sequences = self._find_valid_sequences(img_files, scene_name, label_dir)
                video_infos.extend(valid_sequences)

            except Exception:
                continue
        return video_infos

    def _apply_scene_aware_sampling(self, video_infos: List[Dict]) -> List[Dict]:
        """Shuffle scenes randomly but preserve temporal order within each scene."""
        import random

        scene_groups = {}
        for video_info in video_infos:
            scene_name = video_info['scene_name']
            if scene_name not in scene_groups:
                scene_groups[scene_name] = []
            scene_groups[scene_name].append(video_info)

        scene_names = list(scene_groups.keys())
        random.shuffle(scene_names)

        reordered_video_infos = []
        for scene_name in scene_names:
            reordered_video_infos.extend(scene_groups[scene_name])

        return reordered_video_infos

    def _find_valid_sequences(self, frame_files: List[str], scene_name: str, label_dir: str) -> List[Dict]:
        """Find valid frame sequences that have at least half labels available."""
        sequences = []

        label_files_set = set()
        if osp.exists(label_dir):
            try:
                for f in os.listdir(label_dir):
                    if f.endswith('.png'):
                        label_files_set.add(osp.splitext(f)[0])
            except Exception:
                pass

        step = max(1, self.temporal_length - self.temporal_overlap)
        max_start_idx = len(frame_files) - self.temporal_length
        if max_start_idx < 0:
            return sequences

        for start_idx in range(0, max_start_idx + 1, step):
            end_idx = start_idx + self.temporal_length

            if self.frame_sampling == 'uniform':
                if self.temporal_stride == 1:
                    indices = list(range(start_idx, end_idx))
                else:
                    available_range = end_idx - start_idx
                    if available_range >= self.temporal_length * self.temporal_stride:
                        indices = list(range(start_idx, start_idx + self.temporal_length * self.temporal_stride, self.temporal_stride))
                    else:
                        indices = list(range(start_idx, end_idx))
            elif self.frame_sampling == 'random':
                available_indices = list(range(start_idx, end_idx))
                if len(available_indices) >= self.temporal_length:
                    indices = sorted(np.random.choice(available_indices, size=self.temporal_length, replace=False))
                else:
                    indices = available_indices
            else:
                indices = list(range(start_idx, end_idx))

            if len(indices) != self.temporal_length or any(idx >= len(frame_files) for idx in indices):
                continue
            
            frame_files_subset = [frame_files[i] for i in indices]
            valid_labels = sum(1 for frame_file in frame_files_subset
                             if osp.splitext(frame_file)[0] in label_files_set)

            if valid_labels >= max(1, len(frame_files_subset) // 2):
                sequence_info = {
                    'scene_name': scene_name,
                    'frame_indices': indices,
                    'frame_files': frame_files_subset,
                    'sequence_length': len(indices),
                    'valid_labels': valid_labels
                }
                sequences.append(sequence_info)

        return sequences

    def __len__(self) -> int:
        return len(self.video_infos)

    def __getitem__(self, idx: int) -> Dict:
        video_info = self.video_infos[idx]

        scene_id = video_info['scene_name']
        is_scene_start = self._is_scene_start(idx, scene_id)

        imgs = []
        gt_semantic_segs = []

        for i, frame_file in enumerate(video_info['frame_files']):
            img_path = osp.join(self.img_dir, video_info['scene_name'], 'color', frame_file)
            label_file = osp.splitext(frame_file)[0] + '.png'
            seg_path = osp.join(self.img_dir, video_info['scene_name'], 'label_uint8', label_file)

            if osp.exists(img_path):
                img = mmcv.imread(img_path)
                imgs.append(img)
            else:
                if len(imgs) > 0:
                    img = imgs[-1].copy()
                else:
                    img = self._find_nearest_valid_frame(video_info, i, 'img')
                    if img is None:
                        img = np.zeros((512, 512, 3), dtype=np.uint8)
                imgs.append(img)

            if osp.exists(seg_path):
                gt_semantic_seg = mmcv.imread(seg_path, flag='unchanged', backend='pillow')
                if gt_semantic_seg.ndim == 3:
                    gt_semantic_seg = gt_semantic_seg[:, :, 0]
                if self.id_mapping is not None:
                    gt_semantic_seg = self._apply_label_mapping(gt_semantic_seg)
                gt_semantic_segs.append(gt_semantic_seg)
            else:
                if len(gt_semantic_segs) > 0:
                    gt_semantic_seg = gt_semantic_segs[-1].copy()
                else:
                    gt_semantic_seg = self._find_nearest_valid_frame(video_info, i, 'seg')
                    if gt_semantic_seg is None:
                        gt_semantic_seg = np.zeros((512, 512), dtype=np.uint8)
                if self.id_mapping is not None:
                    gt_semantic_seg = self._apply_label_mapping(gt_semantic_seg)
                gt_semantic_segs.append(gt_semantic_seg)

        if len(imgs) != self.temporal_length or len(gt_semantic_segs) != self.temporal_length:
            raise ValueError("Sequence length mismatch.")

        results = {
            'img': np.stack(imgs, axis=0),
            'gt_semantic_seg': np.stack(gt_semantic_segs, axis=0),
        }

        if isinstance(results['img'], np.ndarray):
            if results['img'].dtype != np.uint8:
                results['img'] = results['img'].astype(np.uint8)
            if results['img'].ndim != 4:
                raise ValueError(f"Expected 4D image array, got {results['img'].ndim}D")

        if isinstance(results['gt_semantic_seg'], np.ndarray):
            if results['gt_semantic_seg'].dtype != np.uint8:
                results['gt_semantic_seg'] = results['gt_semantic_seg'].astype(np.uint8)
            if results['gt_semantic_seg'].ndim == 4:
                results['gt_semantic_seg'] = results['gt_semantic_seg'][:, :, :, 0]
            elif results['gt_semantic_seg'].ndim != 3:
                raise ValueError(f"Expected 3D label array, got {results['gt_semantic_seg'].ndim}D")

        processed_imgs = []
        processed_segs = []

        is_training = not getattr(self, 'test_mode', False)



        for t in range(self.temporal_length):
            img_t = results['img'][t]
            seg_t = results['gt_semantic_seg'][t]

            import cv2
            h, w = img_t.shape[:2]

            if h == 512 and w == 512:
                img_normalized = (img_t.astype(np.float32) - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
                processed_imgs.append(img_normalized)
                processed_segs.append(seg_t)
            else:
                scale = min(512/h, 512/w)
                new_h, new_w = int(h * scale), int(w * scale)

                img_resized = cv2.resize(img_t, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                if is_training:
                    seg_resized = cv2.resize(seg_t, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                else:
                    seg_resized = seg_t

                pad_h = (512 - new_h) // 2
                pad_w = (512 - new_w) // 2
                img_padded = np.pad(img_resized,
                               ((pad_h, 512-new_h-pad_h), (pad_w, 512-new_w-pad_w), (0, 0)),
                               mode='constant', constant_values=0)

                if is_training:
                    seg_padded = np.pad(seg_resized,
                                   ((pad_h, 512-new_h-pad_h), (pad_w, 512-new_w-pad_w)),
                                   mode='constant', constant_values=255)
                else:
                    seg_padded = seg_resized

                img_normalized = (img_padded.astype(np.float32) - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
                processed_imgs.append(img_normalized)
                processed_segs.append(seg_padded)

        processed_imgs_tensor = []
        for img in processed_imgs:
            if img.shape[2] == 3:
                img_tensor = img.transpose(2, 0, 1)
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}")
            processed_imgs_tensor.append(img_tensor)

        img_tensor = np.stack(processed_imgs_tensor, axis=0).astype(np.float32)
        seg_tensor = np.stack(processed_segs, axis=0).astype(np.int64)

        self._update_global_scene_state(scene_id, is_scene_start)

        return {
            'img': img_tensor,
            'gt_semantic_seg': seg_tensor,
            'img_metas': [{}]
        }

    def _apply_label_mapping(self, label: np.ndarray) -> np.ndarray:
        """Vectorized label mapping using a lookup table."""
        if label.dtype != np.uint8:
            label = label.astype(np.uint8)

        lookup_table = np.full(256, 255, dtype=np.uint8)
        for old_id, new_id in self.id_mapping.items():
            if isinstance(old_id, int) and isinstance(new_id, int):
                if 0 <= old_id <= 255:
                    lookup_table[old_id] = new_id

        return lookup_table[label]

    def _find_nearest_valid_frame(self, video_info, current_idx, frame_type):
        """Search forward then backward for the nearest valid frame or label."""
        for direction in [
            range(current_idx + 1, len(video_info['frame_files'])),
            range(current_idx - 1, -1, -1)
        ]:
            for j in direction:
                frame_file = video_info['frame_files'][j]
                if frame_type == 'img':
                    path = osp.join(self.img_dir, video_info['scene_name'], 'color', frame_file)
                    if osp.exists(path):
                        return mmcv.imread(path)
                else:
                    label_file = osp.splitext(frame_file)[0] + '.png'
                    path = osp.join(self.img_dir, video_info['scene_name'], 'label_uint8', label_file)
                    if osp.exists(path):
                        seg = mmcv.imread(path, flag='unchanged', backend='pillow')
                        if seg.ndim == 3:
                            seg = seg[:, :, 0]
                        if self.id_mapping is not None:
                            seg = self._apply_label_mapping(seg)
                        return seg
        return None

    def _is_scene_start(self, idx: int, current_scene: str) -> bool:
        """Detect scene boundary based on global state."""
        if idx == 0:
            return True
        return current_scene != self._current_scene_id

    def _update_global_scene_state(self, scene_id: str, is_scene_start: bool):
        """Update global scene state for SSM state management."""
        if scene_id != self._current_scene_id or is_scene_start:
            self._current_scene_id = scene_id
            self._current_is_scene_start = True
        else:
            self._current_is_scene_start = False

    def get_current_scene_state(self):
        """Return current scene ID and whether this sequence is a scene start."""
        return self._current_scene_id, self._current_is_scene_start

    def get_gt_seg_maps(self, efficient_test: bool = None) -> List[str]:
        """Return GT label paths for the target frame (index 2) of each sequence."""
        gt_seg_maps = []
        target_frame_idx = 2

        for video_info in self.video_infos:
            target_frame_file = video_info['frame_files'][target_frame_idx]
            label_file = osp.splitext(target_frame_file)[0] + '.png'
            seg_path = osp.join(video_info['scene_name'], 'label_uint8', label_file)
            gt_seg_maps.append(seg_path)
        return gt_seg_maps

    def get_classes_and_palette(self, classes: Optional[List[str]] = None, palette: Optional[List[List[int]]] = None) -> Tuple[List[str], List[List[int]]]:
        if classes is None:
            classes = self.CLASSES
        if palette is None:
            palette = self.PALETTE
        return classes, palette

    def evaluate(self, results: List, metric: str = 'mIoU', logger=None, **kwargs) -> Dict:
        """Evaluate predictions against original-resolution GT labels."""
        import mmcv
        import cv2
        from mmseg.core.evaluation import eval_metrics

        single_frame_results = []
        for result in results:
            if isinstance(result, list) and len(result) > 0:
                single_frame_results.append(result[0])
            else:
                single_frame_results.append(result)
        del results

        gt_seg_maps = self.get_gt_seg_maps()

        original_gt_labels = []
        original_size = None

        for gt_path in gt_seg_maps:
            full_gt_path = osp.join(self.img_dir, gt_path)
            if osp.exists(full_gt_path):
                gt_label = mmcv.imread(full_gt_path, flag='unchanged', backend='pillow')
                if gt_label.ndim == 3:
                    gt_label = gt_label[:, :, 0]
                if original_size is None:
                    original_size = gt_label.shape[:2]
                if self.id_mapping is not None:
                    gt_label = self._apply_label_mapping(gt_label)
                original_gt_labels.append(gt_label)
            else:
                if original_size is None:
                    original_size = (968, 1296)
                dummy_label = np.zeros(original_size, dtype=np.uint8)
                if self.id_mapping is not None:
                    dummy_label = self._apply_label_mapping(dummy_label)
                original_gt_labels.append(dummy_label)
        del gt_seg_maps

        original_size_results = []
        for result in single_frame_results:
            if result.shape[:2] != original_size:
                scale = min(512/original_size[0], 512/original_size[1])
                new_h = int(original_size[0] * scale)
                new_w = int(original_size[1] * scale)

                pad_top = (512 - new_h) // 2
                pad_bottom = 512 - new_h - pad_top
                pad_left = (512 - new_w) // 2
                pad_right = 512 - new_w - pad_left

                if pad_bottom > 0:
                    result_cropped = result[pad_top:-pad_bottom, pad_left:-pad_right if pad_right > 0 else None]
                else:
                    result_cropped = result[pad_top:, pad_left:-pad_right if pad_right > 0 else None]

                if isinstance(result_cropped, torch.Tensor):
                    result_cropped_np = result_cropped.cpu().numpy().astype(np.uint8)
                else:
                    result_cropped_np = result_cropped.astype(np.uint8)

                resized_result = cv2.resize(
                    result_cropped_np,
                    (original_size[1], original_size[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                original_size_results.append(resized_result)
            else:
                original_size_results.append(result)
        del single_frame_results

        if isinstance(metric, str):
            metric = [metric]

        eval_results = eval_metrics(
            original_size_results,
            original_gt_labels,
            num_classes=41,
            ignore_index=255,
            metrics=metric,
            nan_to_num=None,
            label_map=None,
            reduce_zero_label=False
        )

        if isinstance(eval_results, list):
            all_acc, acc, iou = eval_results[0], eval_results[1], eval_results[2]
            detailed_results = {
                'mIoU': np.nanmean(iou),
                'mAcc': np.nanmean(acc),
                'aAcc': all_acc
            }
            for i, class_name in enumerate(self.CLASSES):
                detailed_results[f'IoU.{class_name}'] = iou[i]
                detailed_results[f'Acc.{class_name}'] = acc[i]
            eval_results = detailed_results

        del original_size_results, original_gt_labels
        return eval_results

    def format_results(self, results: List, **kwargs) -> List:
        """Format results by extracting only the first frame per sequence."""
        single_frame_results = []
        for result in results:
            if isinstance(result, list) and len(result) > 0:
                single_frame_results.append(result[0])
            else:
                single_frame_results.append(result)
        return super(ScanNetVideoDataset, self).format_results(single_frame_results, **kwargs)