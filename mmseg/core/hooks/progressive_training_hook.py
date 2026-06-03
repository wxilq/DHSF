"""
Progressive Training Hook for MMSegmentation Integration

This hook integrates the progressive training strategy with MMSegmentation's
training pipeline, providing seamless integration with the existing framework.

Features:
1. Automatic stage transitions during training
2. Dynamic learning rate adjustment
3. Loss weight adaptation
4. Training monitoring and logging
5. Checkpoint management for progressive training

Compatible with PyTorch 1.7.0+cu110 and MMSegmentation
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from mmcv.runner import HOOKS, Hook
from mmcv.utils import get_logger

from ..training.progressive_training import ProgressiveTrainingCoordinator


@HOOKS.register_module()
class ProgressiveTrainingHook(Hook):
    """
    Progressive Training Hook for MMSegmentation
    
    Integrates progressive training strategy with MMSegmentation's training loop,
    automatically managing stage transitions, learning rate adjustments, and
    loss weight adaptations.
    """
    
    def __init__(self,
                 stages_config: List[Dict[str, Any]],
                 base_lr: float = 1e-4,
                 initial_loss_weights: Optional[Dict[str, float]] = None,
                 log_interval: int = 50,
                 save_state_interval: int = 5000,
                 enable_monitoring: bool = True):
        """
        Initialize Progressive Training Hook
        
        Args:
            stages_config: Configuration for training stages
            base_lr: Base learning rate
            initial_loss_weights: Initial loss weights
            log_interval: Logging interval
            save_state_interval: State saving interval
            enable_monitoring: Enable training monitoring
        """
        self.stages_config = stages_config
        self.base_lr = base_lr
        self.initial_loss_weights = initial_loss_weights
        self.log_interval = log_interval
        self.save_state_interval = save_state_interval
        self.enable_monitoring = enable_monitoring
        
        # Will be initialized in before_run
        self.coordinator = None
        self.logger = None
        
        # State tracking
        self.current_loss_components = {}
        self.current_performance_metrics = {}
        
    def before_run(self, runner):
        """Initialize progressive training coordinator before training starts"""
        self.logger = get_logger('ProgressiveTraining')
        
        # Initialize coordinator
        total_iters = runner.max_iters
        self.coordinator = ProgressiveTrainingCoordinator(
            stages_config=self.stages_config,
            total_iters=total_iters,
            base_lr=self.base_lr,
            initial_loss_weights=self.initial_loss_weights
        )
        
        self.logger.info("Progressive Training Hook initialized")
        self.logger.info(f"Total iterations: {total_iters}")
        self.logger.info(f"Training stages: {len(self.stages_config)}")
        
        # Log stage configurations
        for i, stage_config in enumerate(self.stages_config):
            stage_name = stage_config.get('name', f'stage_{i}')
            self.logger.info(f"Stage {i} ({stage_name}): {stage_config}")
    
    def before_train_iter(self, runner):
        """Update training state before each iteration"""
        current_iter = runner.iter
        current_epoch = runner.epoch

        # 🔧 设置模型的训练阶段（用于时序处理控制）
        current_stage = self._get_current_stage_by_epoch(current_epoch)
        if current_stage and hasattr(runner.model.module, 'set_training_stage'):
            runner.model.module.set_training_stage(current_stage['name'])

        # Extract loss components from runner if available
        if hasattr(runner, 'outputs') and runner.outputs:
            self._extract_loss_components(runner.outputs)

        # Extract performance metrics from runner if available
        if hasattr(runner, 'log_buffer') and runner.log_buffer:
            self._extract_performance_metrics(runner.log_buffer)

    def _get_current_stage_by_epoch(self, epoch):
        """根据epoch获取当前训练阶段"""
        for stage in self.stages:
            epoch_range = stage.get('epoch_range', (0, 50))
            if epoch_range[0] <= epoch < epoch_range[1]:
                return stage
        return None
        
        # Update progressive training coordinator
        if self.coordinator:
            training_state = self.coordinator.update(
                current_iter=current_iter,
                loss_components=self.current_loss_components,
                performance_metrics=self.current_performance_metrics
            )
            
            # Apply learning rate updates
            self._apply_learning_rate_updates(runner, training_state)
            
            # Apply loss weight updates
            self._apply_loss_weight_updates(runner, training_state)
            
            # Log training state
            if current_iter % self.log_interval == 0:
                self._log_training_state(runner, training_state)
            
            # Save training state
            if current_iter % self.save_state_interval == 0:
                self._save_training_state(runner, current_iter)
    
    def after_train_iter(self, runner):
        """Process training state after each iteration"""
        # Update performance metrics from latest results
        if hasattr(runner, 'log_buffer') and runner.log_buffer:
            self._extract_performance_metrics(runner.log_buffer)
    
    def _extract_loss_components(self, outputs: Dict[str, Any]):
        """Extract loss components from training outputs"""
        self.current_loss_components = {}
        
        if 'loss' in outputs:
            loss_dict = outputs['loss']
            
            # Extract hierarchical loss components
            hierarchical_losses = [
                'pixel_loss', 'object_loss', 'room_loss', 'scene_loss', 'temporal_loss'
            ]
            
            for loss_name in hierarchical_losses:
                if loss_name in loss_dict:
                    loss_value = loss_dict[loss_name]
                    if torch.is_tensor(loss_value):
                        self.current_loss_components[loss_name.replace('_loss', '')] = loss_value.item()
                    else:
                        self.current_loss_components[loss_name.replace('_loss', '')] = float(loss_value)
            
            # Extract total loss
            if 'total_loss' in loss_dict:
                total_loss = loss_dict['total_loss']
                if torch.is_tensor(total_loss):
                    self.current_loss_components['total'] = total_loss.item()
                else:
                    self.current_loss_components['total'] = float(total_loss)
    
    def _extract_performance_metrics(self, log_buffer):
        """Extract performance metrics from log buffer"""
        self.current_performance_metrics = {}
        
        # Common performance metrics
        metric_names = ['mIoU', 'mAcc', 'aAcc', 'boundary_IoU', 'temporal_consistency']
        
        for metric_name in metric_names:
            if metric_name in log_buffer.output:
                metric_value = log_buffer.output[metric_name]
                if torch.is_tensor(metric_value):
                    self.current_performance_metrics[metric_name] = metric_value.item()
                else:
                    self.current_performance_metrics[metric_name] = float(metric_value)
    
    def _apply_learning_rate_updates(self, runner, training_state: Dict[str, Any]):
        """Apply learning rate updates to optimizer"""
        learning_rates = training_state.get('learning_rates', {})
        
        if not learning_rates:
            return
        
        # Update optimizer learning rates
        for param_group in runner.optimizer.param_groups:
            param_name = param_group.get('name', 'default')
            
            # Map parameter group names to component names
            component_name = self._map_param_group_to_component(param_name)
            
            if component_name in learning_rates:
                new_lr = learning_rates[component_name]
                param_group['lr'] = new_lr
    
    def _apply_loss_weight_updates(self, runner, training_state: Dict[str, Any]):
        """Apply loss weight updates to model"""
        loss_weights = training_state.get('loss_weights', {})
        
        if not loss_weights:
            return
        
        # Update loss weights in model if it has hierarchical loss
        if hasattr(runner.model, 'decode_head') and hasattr(runner.model.decode_head, 'loss_decode'):
            loss_module = runner.model.decode_head.loss_decode
            
            # Update weights if it's a hierarchical loss
            if hasattr(loss_module, 'current_weights'):
                loss_module.current_weights.update(loss_weights)
            
            # Update training stage if supported
            if hasattr(loss_module, 'set_training_stage'):
                current_stage = training_state.get('stage_info', {}).get('current_stage', 0)
                loss_module.set_training_stage(current_stage)
    
    def _map_param_group_to_component(self, param_name: str) -> str:
        """Map parameter group name to component name"""
        # Default mapping - can be customized based on model structure
        if 'backbone' in param_name.lower():
            return 'backbone'
        elif 'neck' in param_name.lower():
            return 'neck'
        elif 'head' in param_name.lower():
            return 'head'
        elif 'sdsm' in param_name.lower():
            return 'sdsm'
        elif 'chsm' in param_name.lower():
            return 'chsm'
        else:
            return 'default'
    
    def _log_training_state(self, runner, training_state: Dict[str, Any]):
        """Log current training state"""
        current_iter = runner.iter
        stage_info = training_state.get('stage_info', {})
        monitoring = training_state.get('monitoring', {})
        
        # Log stage information
        stage_name = stage_info.get('stage_name', 'unknown')
        stage_progress = stage_info.get('stage_progress', 0.0)
        
        self.logger.info(f"Iter {current_iter}: Stage {stage_name} ({stage_progress:.1%} complete)")
        
        # Log learning rates
        learning_rates = training_state.get('learning_rates', {})
        if learning_rates:
            lr_str = ', '.join([f"{k}: {v:.2e}" for k, v in learning_rates.items()])
            self.logger.info(f"Learning rates: {lr_str}")
        
        # Log loss weights
        loss_weights = training_state.get('loss_weights', {})
        if loss_weights:
            weight_str = ', '.join([f"{k}: {v:.3f}" for k, v in loss_weights.items()])
            self.logger.info(f"Loss weights: {weight_str}")
        
        # Log training health
        training_health = monitoring.get('training_health', 'unknown')
        if training_health != 'healthy':
            self.logger.warning(f"Training health: {training_health}")
        
        # Log recommendations
        recommendations = training_state.get('recommendations', [])
        if recommendations:
            self.logger.info(f"Recommendations: {'; '.join(recommendations[:3])}")
        
        # Log stage transition
        if stage_info.get('stage_changed', False):
            self.logger.info(f"🔄 Stage transition detected at iteration {current_iter}")
    
    def _save_training_state(self, runner, current_iter: int):
        """Save training state to checkpoint directory"""
        if not self.coordinator:
            return
        
        try:
            # Save to work directory
            work_dir = runner.work_dir
            state_file = f"{work_dir}/progressive_training_state_iter_{current_iter}.json"
            
            self.coordinator.save_training_state(state_file)
            self.logger.info(f"Progressive training state saved to {state_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save progressive training state: {e}")
    
    def after_run(self, runner):
        """Finalize progressive training after training completes"""
        if self.coordinator:
            # Save final training state
            final_state_file = f"{runner.work_dir}/progressive_training_final_state.json"
            self.coordinator.save_training_state(final_state_file)
            
            # Log training summary
            training_summary = self.coordinator.get_training_summary()
            self.logger.info("Progressive Training Summary:")
            self.logger.info(f"Total iterations: {training_summary.get('total_iterations', 0)}")
            self.logger.info(f"Final stage: {training_summary.get('current_stage', 0)}")
            self.logger.info(f"Final training health: {training_summary.get('training_health', 'unknown')}")
            
            # Log performance trends
            performance_trends = training_summary.get('performance_trends', {})
            if 'mIoU' in performance_trends:
                miou_info = performance_trends['mIoU']
                self.logger.info(f"Final mIoU: {miou_info.get('current', 0):.4f}")
                self.logger.info(f"mIoU trend: {miou_info.get('trend', 0):.6f}")
        
        self.logger.info("Progressive Training Hook completed")
