"""
自定义数据加载变换 - 支持预映射标签
🎯 核心功能：解决训练时标签映射被覆盖的问题
"""

import os.path as osp
import mmcv
import numpy as np
from ..builder import PIPELINES
from .loading import LoadAnnotations


@PIPELINES.register_module()
class LoadAnnotationsWithMapping(LoadAnnotations):
    """
    自定义LoadAnnotations，支持使用数据集中已映射的标签
    
    🎯 核心功能：
    - 如果标签已经在数据集中映射，直接使用，避免重新加载覆盖映射结果
    - 否则按照标准流程从文件加载
    - 这是解决训练时CUDA断言错误的关键组件
    
    🔧 工作原理：
    1. 检查数据字典中的_labels_mapped标记
    2. 如果为True，跳过文件加载，直接使用已映射的标签
    3. 如果为False，使用标准LoadAnnotations流程
    """

    def __call__(self, results):
        """
        加载标注信息 - 强化保护机制，确保已映射标签不被覆盖

        Args:
            results (dict): 包含图像和标注信息的结果字典

        Returns:
            dict: 更新后的结果字典
        """
        # 🔧 关键修复：强化检查标签是否已经映射
        # 多重保护机制，确保已映射的标签绝不被覆盖
        if results.get('_labels_mapped', False) and 'gt_semantic_seg' in results:
            # 标签已经在数据集中映射，直接使用，绝不重新加载
            # 这是解决标签映射被覆盖问题的核心保护机制
            gt_semantic_seg = results['gt_semantic_seg']

            # 确保标签是正确的数据类型和格式
            if isinstance(gt_semantic_seg, np.ndarray):
                if gt_semantic_seg.dtype != np.uint8:
                    gt_semantic_seg = gt_semantic_seg.astype(np.uint8)

                # 确保标签是2D数组
                if gt_semantic_seg.ndim == 3:
                    gt_semantic_seg = gt_semantic_seg[:, :, 0]

                results['gt_semantic_seg'] = gt_semantic_seg
                results['seg_fields'] = results.get('seg_fields', [])
                if 'gt_semantic_seg' not in results['seg_fields']:
                    results['seg_fields'].append('gt_semantic_seg')

            # 🔧 强化保护：确保不会调用父类的加载方法
            # 直接返回，绝不进入标准加载流程
            return results

        # 标签未映射，使用标准加载流程
        return super(LoadAnnotationsWithMapping, self).__call__(results)

    def _load_semantic_seg(self, results):
        """
        加载语义分割标注 - 强化保护机制

        Args:
            results (dict): 结果字典

        Returns:
            dict: 更新后的结果字典
        """
        # 🔧 关键修复：强化保护，如果标签已映射，绝不重新加载
        # 多重检查确保已映射标签不被覆盖
        if results.get('_labels_mapped', False) and 'gt_semantic_seg' in results:
            # 标签已经映射且存在，直接返回，绝不重新加载
            return results

        # 标签未映射，使用父类的标准加载方法
        return super(LoadAnnotationsWithMapping, self)._load_semantic_seg(results)
