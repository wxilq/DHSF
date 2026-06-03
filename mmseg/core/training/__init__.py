"""
Progressive Training Module for Video Semantic Segmentation

This module provides comprehensive progressive training strategies for
hierarchical video semantic segmentation models.

Components:
- ProgressiveTrainingScheduler: Multi-stage training coordination
- AdaptiveLearningRateScheduler: Dynamic learning rate adjustment
- DynamicLossWeightAdjuster: Automatic loss weight optimization
- TrainingStateMonitor: Training health monitoring and anomaly detection
- ProgressiveTrainingCoordinator: Main orchestration class

Compatible with PyTorch 1.7.0+cu110
"""

from .progressive_training import (
    ProgressiveTrainingScheduler,
    AdaptiveLearningRateScheduler,
    DynamicLossWeightAdjuster,
    TrainingStateMonitor,
    ProgressiveTrainingCoordinator
)

__all__ = [
    'ProgressiveTrainingScheduler',
    'AdaptiveLearningRateScheduler', 
    'DynamicLossWeightAdjuster',
    'TrainingStateMonitor',
    'ProgressiveTrainingCoordinator'
]
