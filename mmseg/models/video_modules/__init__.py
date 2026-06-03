"""
Video Modules for Hierarchical Temporal Mamba-based Video Semantic Segmentation
Compatible with PyTorch 1.7.0+cu110
"""

# Temporal modules
from .temporal.semantic_levels import SemanticLevelDecomposition, SemanticLevels
from .temporal.sdsm import (
    SpatioTemporalDecoupledStateModel,
    LevelSpecificSDSM,
    SpatialStateSpaceModel,
    TemporalStateSpaceModel,
    SimplifiedSpatioTemporalFusion
)

# Adapter modules - 修复导入错误，暂时注释所有不存在的模块
# 注意：这些模块在当前项目中不存在，为避免导入错误暂时注释
# 如果需要这些功能，可以后续实现或使用替代方案
# from .adapters.temporal_adapter import TemporalAdapter, CrossStageTemporalAdapter  # 文件不存在
# from .adapters.information_theory_patch_embed import InformationTheoryPatchEmbed  # 文件不存在
# from .adapters.variational_patch_embed import VariationalPatchEmbed  # 文件不存在
# from .adapters.pareto_patch_embed import ParetoPatchEmbed, ParetoFrontSolver  # 文件不存在

# Fusion modules
from .fusion.cross_level_fusion import CrossLevelFusion, HierarchicalFusion
from .fusion.enhanced_chsm import (
    EnhancedCrossHierarchicalFusion,
    StateTransferChannel,
    InterLevelAttention,
    StateFusionModule
)

# Pipeline modules已删除，直接在VideoSegFormer中集成

__all__ = [
    # Temporal modules
    'SemanticLevelDecomposition',
    'SemanticLevels',
    'SpatioTemporalDecoupledStateModel',
    'LevelSpecificSDSM',
    'SpatialStateSpaceModel',
    'TemporalStateSpaceModel',
    'SimplifiedSpatioTemporalFusion',

    # Adapter modules - 暂时注释所有不存在的模块
    # 注意：这些模块暂时不可用，避免导入错误
    # 'TemporalAdapter',  # 文件不存在
    # 'CrossStageTemporalAdapter',  # 文件不存在
    # 'InformationTheoryPatchEmbed',  # 文件不存在
    # 'VariationalPatchEmbed',  # 文件不存在
    # 'ParetoPatchEmbed',  # 文件不存在
    # 'ParetoFrontSolver',  # 文件不存在

    # Fusion modules
    'CrossLevelFusion',
    'HierarchicalFusion',
    'EnhancedCrossHierarchicalFusion',
    'StateTransferChannel',
    'InterLevelAttention',
    'StateFusionModule',

    # Pipeline modules已删除
]