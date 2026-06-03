"""
Progressive Training Strategy for Hierarchical Video Semantic Segmentation

This module implements progressive training strategies that adapt to different
training stages and optimize the hierarchical loss functions dynamically.

Key Features:
1. Multi-stage Training Scheduler
   - Stage 0: Spatial Feature Learning (focus on pixel/object levels)
   - Stage 1: Temporal Modeling (introduce temporal consistency)
   - Stage 2: Joint Optimization (balance all components)

2. Adaptive Learning Rate Adjustment
   - Stage-specific learning rate schedules
   - Performance-based adaptive adjustment
   - Component-specific learning rate scaling

3. Dynamic Loss Weight Adjustment
   - Automatic loss weight adaptation based on training progress
   - Performance-driven weight rebalancing
   - Convergence-aware weight scheduling

4. Training State Monitoring
   - Real-time training progress tracking
   - Automatic stage transition detection
   - Performance plateau detection and response

Compatible with PyTorch 1.7.0+cu110 and MMSegmentation framework
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict, deque
import logging


class ProgressiveTrainingScheduler:
    """
    Progressive Training Scheduler for multi-stage training coordination
    
    Manages the transition between different training stages and coordinates
    the adaptation of learning rates, loss weights, and model components.
    """
    
    def __init__(self,
                 stages_config: List[Dict[str, Any]],
                 total_iters: int,
                 warmup_iters: int = 1000,
                 transition_smoothing: int = 500,
                 auto_transition: bool = True,
                 performance_threshold: float = 0.01):
        """
        Initialize Progressive Training Scheduler
        
        Args:
            stages_config: List of stage configurations
            total_iters: Total training iterations
            warmup_iters: Warmup iterations for each stage
            transition_smoothing: Iterations for smooth stage transition
            auto_transition: Enable automatic stage transition
            performance_threshold: Performance improvement threshold for auto transition
        """
        self.stages_config = stages_config
        self.total_iters = total_iters
        self.warmup_iters = warmup_iters
        self.transition_smoothing = transition_smoothing
        self.auto_transition = auto_transition
        self.performance_threshold = performance_threshold
        
        # Initialize stage boundaries
        self.stage_boundaries = self._compute_stage_boundaries()
        self.current_stage = 0
        self.current_iter = 0
        
        # Performance tracking for auto transition
        self.performance_history = deque(maxlen=100)
        self.stage_start_iter = 0
        self.last_transition_iter = 0
        
        # Stage-specific configurations
        self.stage_lr_multipliers = self._extract_lr_multipliers()
        self.stage_loss_weights = self._extract_loss_weights()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def _compute_stage_boundaries(self) -> List[int]:
        """Compute iteration boundaries for each stage"""
        boundaries = []
        
        if len(self.stages_config) == 0:
            return [self.total_iters]
        
        # Equal division if no specific boundaries provided
        stage_length = self.total_iters // len(self.stages_config)
        
        for i, stage_config in enumerate(self.stages_config):
            if 'end_iter' in stage_config:
                boundaries.append(stage_config['end_iter'])
            elif 'duration_ratio' in stage_config:
                boundaries.append(int(self.total_iters * stage_config['duration_ratio']))
            else:
                boundaries.append((i + 1) * stage_length)
        
        # Ensure last boundary is total_iters
        boundaries[-1] = self.total_iters
        
        return boundaries
    
    def _extract_lr_multipliers(self) -> Dict[int, Dict[str, float]]:
        """Extract learning rate multipliers for each stage"""
        lr_multipliers = {}
        
        for i, stage_config in enumerate(self.stages_config):
            stage_multipliers = stage_config.get('lr_multipliers', {})
            
            # Default multipliers based on stage characteristics
            if i == 0:  # Spatial learning stage
                default_multipliers = {
                    'backbone': 0.1,
                    'neck': 1.0,
                    'head': 2.0,
                    'sdsm': 0.5,
                    'chsm': 0.3
                }
            elif i == 1:  # Temporal modeling stage
                default_multipliers = {
                    'backbone': 0.05,
                    'neck': 0.8,
                    'head': 1.5,
                    'sdsm': 1.0,
                    'chsm': 0.8
                }
            else:  # Joint optimization stage
                default_multipliers = {
                    'backbone': 0.02,
                    'neck': 0.6,
                    'head': 1.0,
                    'sdsm': 0.8,
                    'chsm': 1.0
                }
            
            # Merge with user-specified multipliers
            default_multipliers.update(stage_multipliers)
            lr_multipliers[i] = default_multipliers
        
        return lr_multipliers
    
    def _extract_loss_weights(self) -> Dict[int, Dict[str, float]]:
        """Extract loss weights for each stage"""
        loss_weights = {}
        
        for i, stage_config in enumerate(self.stages_config):
            stage_weights = stage_config.get('loss_weights', {})
            
            # Default weights based on stage focus
            if i == 0:  # Spatial learning stage
                default_weights = {
                    'pixel': 0.6,
                    'object': 0.36,
                    'room': 0.16,
                    'scene': 0.05,
                    'temporal': 0.1
                }
            elif i == 1:  # Temporal modeling stage
                default_weights = {
                    'pixel': 0.32,
                    'object': 0.3,
                    'room': 0.24,
                    'scene': 0.15,
                    'temporal': 0.5
                }
            else:  # Joint optimization stage
                default_weights = {
                    'pixel': 0.4,
                    'object': 0.3,
                    'room': 0.2,
                    'scene': 0.1,
                    'temporal': 0.3
                }
            
            # Merge with user-specified weights
            default_weights.update(stage_weights)
            loss_weights[i] = default_weights
        
        return loss_weights
    
    def update(self, current_iter: int, performance_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Update training scheduler state
        
        Args:
            current_iter: Current training iteration
            performance_metrics: Current performance metrics
        Returns:
            Dictionary with current training state and recommendations
        """
        self.current_iter = current_iter
        
        # Update performance history
        if performance_metrics:
            self.performance_history.append(performance_metrics)
        
        # Check for stage transition
        stage_changed = self._check_stage_transition()
        
        # Get current training state
        training_state = {
            'current_stage': self.current_stage,
            'stage_name': self.stages_config[self.current_stage]['name'] if self.current_stage < len(self.stages_config) else 'final',
            'stage_progress': self._get_stage_progress(),
            'stage_changed': stage_changed,
            'lr_multipliers': self.stage_lr_multipliers.get(self.current_stage, {}),
            'loss_weights': self.stage_loss_weights.get(self.current_stage, {}),
            'in_transition': self._is_in_transition(),
            'transition_factor': self._get_transition_factor()
        }
        
        return training_state
    
    def _check_stage_transition(self) -> bool:
        """Check if stage transition should occur"""
        old_stage = self.current_stage
        
        # Check iteration-based transition
        for i, boundary in enumerate(self.stage_boundaries):
            if self.current_iter <= boundary:
                self.current_stage = i
                break
        
        # Check auto transition based on performance
        if self.auto_transition and self._should_auto_transition():
            self.current_stage = min(self.current_stage + 1, len(self.stages_config) - 1)
        
        # Handle stage change
        if old_stage != self.current_stage:
            self.stage_start_iter = self.current_iter
            self.last_transition_iter = self.current_iter
            self.logger.info(f"Stage transition: {old_stage} -> {self.current_stage} at iter {self.current_iter}")
            return True
        
        return False
    
    def _should_auto_transition(self) -> bool:
        """Determine if automatic stage transition should occur"""
        if len(self.performance_history) < 20:  # Need sufficient history
            return False
        
        # Check if performance has plateaued
        recent_performance = list(self.performance_history)[-10:]
        early_performance = list(self.performance_history)[-20:-10]
        
        if len(recent_performance) == 0 or len(early_performance) == 0:
            return False
        
        # Calculate performance improvement
        recent_avg = np.mean([p.get('mIoU', 0) for p in recent_performance])
        early_avg = np.mean([p.get('mIoU', 0) for p in early_performance])
        
        improvement = recent_avg - early_avg
        
        # Transition if improvement is below threshold and minimum stage duration met
        min_stage_duration = 2000  # Minimum iterations per stage
        stage_duration = self.current_iter - self.stage_start_iter
        
        return (improvement < self.performance_threshold and 
                stage_duration > min_stage_duration and
                self.current_stage < len(self.stages_config) - 1)
    
    def _get_stage_progress(self) -> float:
        """Get current stage progress (0.0 to 1.0)"""
        if self.current_stage >= len(self.stage_boundaries):
            return 1.0
        
        stage_start = self.stage_boundaries[self.current_stage - 1] if self.current_stage > 0 else 0
        stage_end = self.stage_boundaries[self.current_stage]
        
        if stage_end <= stage_start:
            return 1.0
        
        progress = (self.current_iter - stage_start) / (stage_end - stage_start)
        return min(max(progress, 0.0), 1.0)
    
    def _is_in_transition(self) -> bool:
        """Check if currently in stage transition period"""
        return (self.current_iter - self.last_transition_iter) < self.transition_smoothing
    
    def _get_transition_factor(self) -> float:
        """Get smooth transition factor (0.0 to 1.0)"""
        if not self._is_in_transition():
            return 1.0
        
        transition_progress = (self.current_iter - self.last_transition_iter) / self.transition_smoothing
        # Smooth transition using cosine function
        return 0.5 * (1 + math.cos(math.pi * (1 - transition_progress)))
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current stage configuration"""
        if self.current_stage < len(self.stages_config):
            return self.stages_config[self.current_stage]
        return {}
    
    def force_stage_transition(self, target_stage: int):
        """Force transition to specific stage"""
        if 0 <= target_stage < len(self.stages_config):
            old_stage = self.current_stage
            self.current_stage = target_stage
            self.stage_start_iter = self.current_iter
            self.last_transition_iter = self.current_iter
            self.logger.info(f"Forced stage transition: {old_stage} -> {self.current_stage}")


class AdaptiveLearningRateScheduler:
    """
    Adaptive Learning Rate Scheduler for progressive training
    
    Adjusts learning rates based on training stage, performance metrics,
    and component-specific requirements.
    """
    
    def __init__(self,
                 base_lr: float = 1e-4,
                 warmup_iters: int = 1000,
                 min_lr_ratio: float = 0.01,
                 performance_patience: int = 10,
                 performance_factor: float = 0.5):
        """
        Initialize Adaptive Learning Rate Scheduler
        
        Args:
            base_lr: Base learning rate
            warmup_iters: Warmup iterations
            min_lr_ratio: Minimum learning rate ratio
            performance_patience: Patience for performance-based adjustment
            performance_factor: Factor for performance-based reduction
        """
        self.base_lr = base_lr
        self.warmup_iters = warmup_iters
        self.min_lr_ratio = min_lr_ratio
        self.performance_patience = performance_patience
        self.performance_factor = performance_factor
        
        # State tracking
        self.current_lr = base_lr
        self.performance_history = deque(maxlen=performance_patience * 2)
        self.last_improvement_iter = 0
        self.plateau_count = 0
        
        # Component-specific learning rates
        self.component_lrs = {}
        
    def update(self, 
               current_iter: int,
               training_state: Dict[str, Any],
               performance_metrics: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Update learning rates based on current training state
        
        Args:
            current_iter: Current training iteration
            training_state: Current training state from progressive scheduler
            performance_metrics: Performance metrics
        Returns:
            Dictionary of component-specific learning rates
        """
        # Update performance tracking
        if performance_metrics:
            self.performance_history.append(performance_metrics)
            self._check_performance_plateau(current_iter)
        
        # Get base learning rate for current stage
        stage_lr = self._get_stage_learning_rate(current_iter, training_state)
        
        # Apply performance-based adjustment
        performance_lr = self._apply_performance_adjustment(stage_lr)
        
        # Apply warmup if in transition
        if training_state.get('in_transition', False):
            transition_factor = training_state.get('transition_factor', 1.0)
            performance_lr *= transition_factor
        
        # Compute component-specific learning rates
        lr_multipliers = training_state.get('lr_multipliers', {})
        component_lrs = {}
        
        for component, multiplier in lr_multipliers.items():
            component_lrs[component] = performance_lr * multiplier
        
        # Ensure minimum learning rate
        min_lr = self.base_lr * self.min_lr_ratio
        for component in component_lrs:
            component_lrs[component] = max(component_lrs[component], min_lr)
        
        self.component_lrs = component_lrs
        self.current_lr = performance_lr
        
        return component_lrs
    
    def _get_stage_learning_rate(self, current_iter: int, training_state: Dict[str, Any]) -> float:
        """Get learning rate based on current stage"""
        stage_progress = training_state.get('stage_progress', 0.0)
        current_stage = training_state.get('current_stage', 0)
        
        # Stage-specific base learning rate adjustment
        stage_factors = [1.0, 0.8, 0.6]  # Decreasing LR for later stages
        stage_factor = stage_factors[min(current_stage, len(stage_factors) - 1)]
        
        # Apply cosine annealing within stage
        stage_lr = self.base_lr * stage_factor * (0.5 * (1 + math.cos(math.pi * stage_progress)))
        
        return stage_lr
    
    def _apply_performance_adjustment(self, base_lr: float) -> float:
        """Apply performance-based learning rate adjustment"""
        if self.plateau_count > 0:
            # Reduce learning rate if performance has plateaued
            reduction_factor = self.performance_factor ** self.plateau_count
            return base_lr * reduction_factor
        
        return base_lr
    
    def _check_performance_plateau(self, current_iter: int):
        """Check for performance plateau and update plateau count"""
        if len(self.performance_history) < self.performance_patience:
            return
        
        # Check if performance has improved recently
        recent_performance = list(self.performance_history)[-self.performance_patience:]
        best_recent = max(p.get('mIoU', 0) for p in recent_performance)
        
        # Compare with earlier performance
        if len(self.performance_history) >= self.performance_patience * 2:
            earlier_performance = list(self.performance_history)[-self.performance_patience * 2:-self.performance_patience]
            best_earlier = max(p.get('mIoU', 0) for p in earlier_performance)
            
            improvement = best_recent - best_earlier
            
            if improvement < 0.001:  # Very small improvement threshold
                self.plateau_count += 1
                self.last_improvement_iter = current_iter
            else:
                self.plateau_count = 0
    
    def get_current_lrs(self) -> Dict[str, float]:
        """Get current component learning rates"""
        return self.component_lrs.copy()
    
    def reset_plateau_count(self):
        """Reset plateau count (useful for stage transitions)"""
        self.plateau_count = 0


class DynamicLossWeightAdjuster:
    """
    Dynamic Loss Weight Adjuster for hierarchical loss functions

    Automatically adjusts loss weights based on training progress,
    performance metrics, and convergence characteristics.
    """

    def __init__(self,
                 initial_weights: Dict[str, float],
                 adaptation_rate: float = 0.1,
                 convergence_threshold: float = 0.001,
                 rebalance_patience: int = 20):
        """
        Initialize Dynamic Loss Weight Adjuster

        Args:
            initial_weights: Initial loss weights
            adaptation_rate: Rate of weight adaptation
            convergence_threshold: Threshold for convergence detection
            rebalance_patience: Patience for rebalancing
        """
        self.initial_weights = initial_weights.copy()
        self.current_weights = initial_weights.copy()
        self.adaptation_rate = adaptation_rate
        self.convergence_threshold = convergence_threshold
        self.rebalance_patience = rebalance_patience

        # Loss tracking for each component
        self.loss_history = {key: deque(maxlen=50) for key in initial_weights.keys()}
        self.convergence_status = {key: False for key in initial_weights.keys()}
        self.last_rebalance_iter = 0

    def update(self,
               current_iter: int,
               loss_components: Dict[str, float],
               training_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Update loss weights based on current loss components and training state

        Args:
            current_iter: Current training iteration
            loss_components: Current loss component values
            training_state: Training state from progressive scheduler
        Returns:
            Updated loss weights
        """
        # Update loss history
        for component, loss_value in loss_components.items():
            if component in self.loss_history:
                self.loss_history[component].append(loss_value)

        # Get stage-specific base weights
        stage_weights = training_state.get('loss_weights', self.initial_weights)

        # Apply dynamic adjustment if not in transition
        if not training_state.get('in_transition', False):
            adjusted_weights = self._apply_dynamic_adjustment(stage_weights, current_iter)
        else:
            # Smooth transition between stage weights
            transition_factor = training_state.get('transition_factor', 1.0)
            adjusted_weights = self._smooth_weight_transition(stage_weights, transition_factor)

        self.current_weights = adjusted_weights
        return adjusted_weights.copy()

    def _apply_dynamic_adjustment(self, base_weights: Dict[str, float], current_iter: int) -> Dict[str, float]:
        """Apply dynamic adjustment to base weights"""
        adjusted_weights = base_weights.copy()

        # Check convergence status for each component
        self._update_convergence_status()

        # Rebalance weights if needed
        if current_iter - self.last_rebalance_iter > self.rebalance_patience:
            adjusted_weights = self._rebalance_weights(adjusted_weights)
            self.last_rebalance_iter = current_iter

        return adjusted_weights

    def _update_convergence_status(self):
        """Update convergence status for each loss component"""
        for component, history in self.loss_history.items():
            if len(history) >= 10:
                # Check if loss has converged (small variance)
                recent_losses = list(history)[-10:]
                loss_variance = np.var(recent_losses)
                loss_mean = np.mean(recent_losses)

                # Normalized variance as convergence indicator
                if loss_mean > 0:
                    normalized_variance = loss_variance / (loss_mean ** 2)
                    self.convergence_status[component] = normalized_variance < self.convergence_threshold

    def _rebalance_weights(self, current_weights: Dict[str, float]) -> Dict[str, float]:
        """Rebalance weights based on convergence status and loss magnitudes"""
        rebalanced_weights = current_weights.copy()

        # Calculate relative loss magnitudes
        loss_magnitudes = {}
        for component, history in self.loss_history.items():
            if len(history) > 0:
                loss_magnitudes[component] = np.mean(list(history)[-5:])  # Recent average

        if not loss_magnitudes:
            return rebalanced_weights

        # Normalize loss magnitudes
        total_magnitude = sum(loss_magnitudes.values())
        if total_magnitude > 0:
            normalized_magnitudes = {k: v / total_magnitude for k, v in loss_magnitudes.items()}
        else:
            normalized_magnitudes = {k: 1.0 / len(loss_magnitudes) for k in loss_magnitudes}

        # Adjust weights based on convergence and magnitude
        for component in rebalanced_weights:
            if component in self.convergence_status and component in normalized_magnitudes:
                # Reduce weight for converged components
                if self.convergence_status[component]:
                    rebalanced_weights[component] *= 0.9
                else:
                    # Increase weight for non-converged components with high loss
                    magnitude_factor = normalized_magnitudes[component]
                    rebalanced_weights[component] *= (1.0 + self.adaptation_rate * magnitude_factor)

        # Normalize weights to maintain total weight
        total_weight = sum(rebalanced_weights.values())
        if total_weight > 0:
            for component in rebalanced_weights:
                rebalanced_weights[component] /= total_weight

        return rebalanced_weights

    def _smooth_weight_transition(self, target_weights: Dict[str, float], transition_factor: float) -> Dict[str, float]:
        """Smooth transition between current and target weights"""
        smoothed_weights = {}

        for component in target_weights:
            current_weight = self.current_weights.get(component, target_weights[component])
            target_weight = target_weights[component]

            # Linear interpolation with transition factor
            smoothed_weights[component] = (
                current_weight * (1 - transition_factor) +
                target_weight * transition_factor
            )

        return smoothed_weights

    def get_convergence_status(self) -> Dict[str, bool]:
        """Get current convergence status for all components"""
        return self.convergence_status.copy()

    def reset_for_stage_transition(self):
        """Reset state for stage transition"""
        self.convergence_status = {key: False for key in self.convergence_status}
        self.last_rebalance_iter = 0


class TrainingStateMonitor:
    """
    Training State Monitor for comprehensive training progress tracking

    Monitors training metrics, detects anomalies, and provides
    recommendations for training adjustments.
    """

    def __init__(self,
                 monitoring_window: int = 100,
                 anomaly_threshold: float = 2.0,
                 improvement_patience: int = 50):
        """
        Initialize Training State Monitor

        Args:
            monitoring_window: Window size for monitoring metrics
            anomaly_threshold: Threshold for anomaly detection (in standard deviations)
            improvement_patience: Patience for improvement detection
        """
        self.monitoring_window = monitoring_window
        self.anomaly_threshold = anomaly_threshold
        self.improvement_patience = improvement_patience

        # Metric tracking
        self.metrics_history = defaultdict(lambda: deque(maxlen=monitoring_window))
        self.loss_history = defaultdict(lambda: deque(maxlen=monitoring_window))
        self.lr_history = defaultdict(lambda: deque(maxlen=monitoring_window))

        # State tracking
        self.training_health = 'healthy'
        self.anomalies_detected = []
        self.recommendations = []

    def update(self,
               current_iter: int,
               loss_components: Dict[str, float],
               performance_metrics: Dict[str, float],
               learning_rates: Dict[str, float],
               training_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update training state monitoring

        Args:
            current_iter: Current training iteration
            loss_components: Current loss components
            performance_metrics: Performance metrics
            learning_rates: Current learning rates
            training_state: Training state
        Returns:
            Monitoring report with health status and recommendations
        """
        # Update histories
        for component, loss_value in loss_components.items():
            self.loss_history[component].append(loss_value)

        for metric, value in performance_metrics.items():
            self.metrics_history[metric].append(value)

        for component, lr in learning_rates.items():
            self.lr_history[component].append(lr)

        # Analyze training health
        self._analyze_training_health(current_iter)

        # Generate recommendations
        self._generate_recommendations(training_state)

        # Prepare monitoring report
        report = {
            'training_health': self.training_health,
            'anomalies': self.anomalies_detected.copy(),
            'recommendations': self.recommendations.copy(),
            'metrics_summary': self._get_metrics_summary(),
            'convergence_analysis': self._analyze_convergence(),
            'stage_performance': self._analyze_stage_performance(training_state)
        }

        return report

    def _analyze_training_health(self, current_iter: int):
        """Analyze overall training health"""
        self.anomalies_detected = []

        # Check for loss anomalies
        for component, history in self.loss_history.items():
            if len(history) >= 20:
                recent_losses = list(history)[-10:]
                earlier_losses = list(history)[-20:-10]

                if len(recent_losses) > 0 and len(earlier_losses) > 0:
                    recent_mean = np.mean(recent_losses)
                    earlier_mean = np.mean(earlier_losses)
                    earlier_std = np.std(earlier_losses)

                    # Detect sudden loss spikes
                    if earlier_std > 0 and recent_mean > earlier_mean + self.anomaly_threshold * earlier_std:
                        self.anomalies_detected.append(f"Loss spike detected in {component}")

                    # Detect loss explosion
                    if recent_mean > 10 * earlier_mean and recent_mean > 1.0:
                        self.anomalies_detected.append(f"Loss explosion in {component}")

        # Check for performance degradation
        for metric, history in self.metrics_history.items():
            if len(history) >= 20:
                recent_performance = list(history)[-10:]
                earlier_performance = list(history)[-20:-10]

                if len(recent_performance) > 0 and len(earlier_performance) > 0:
                    recent_mean = np.mean(recent_performance)
                    earlier_mean = np.mean(earlier_performance)

                    # Detect significant performance drop
                    if recent_mean < earlier_mean - 0.05:  # 5% drop
                        self.anomalies_detected.append(f"Performance degradation in {metric}")

        # Determine overall health
        if len(self.anomalies_detected) == 0:
            self.training_health = 'healthy'
        elif len(self.anomalies_detected) <= 2:
            self.training_health = 'warning'
        else:
            self.training_health = 'critical'

    def _generate_recommendations(self, training_state: Dict[str, Any]):
        """Generate training recommendations based on current state"""
        self.recommendations = []

        # Recommendations based on anomalies
        if 'Loss spike' in str(self.anomalies_detected):
            self.recommendations.append("Consider reducing learning rate")

        if 'Loss explosion' in str(self.anomalies_detected):
            self.recommendations.append("Reduce learning rate significantly or restart from checkpoint")

        if 'Performance degradation' in str(self.anomalies_detected):
            self.recommendations.append("Consider adjusting loss weights or learning rate schedule")

        # Stage-specific recommendations
        current_stage = training_state.get('current_stage', 0)
        stage_progress = training_state.get('stage_progress', 0.0)

        if stage_progress > 0.8 and current_stage < 2:
            self.recommendations.append("Consider early stage transition")

        if stage_progress < 0.2 and len(self.metrics_history.get('mIoU', [])) > 20:
            recent_improvement = self._calculate_recent_improvement('mIoU')
            if recent_improvement < 0.001:
                self.recommendations.append("Consider increasing learning rate or adjusting loss weights")

    def _get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics"""
        summary = {}

        for metric, history in self.metrics_history.items():
            if len(history) > 0:
                values = list(history)
                summary[metric] = {
                    'current': values[-1],
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'trend': self._calculate_trend(values)
                }

        return summary

    def _analyze_convergence(self) -> Dict[str, Dict[str, Any]]:
        """Analyze convergence status for all components"""
        convergence_analysis = {}

        for component, history in self.loss_history.items():
            if len(history) >= 20:
                values = list(history)
                recent_values = values[-10:]

                convergence_analysis[component] = {
                    'converged': np.std(recent_values) < 0.01 * np.mean(recent_values),
                    'trend': self._calculate_trend(values),
                    'stability': 1.0 / (1.0 + np.std(recent_values)),
                    'recent_mean': np.mean(recent_values)
                }

        return convergence_analysis

    def _analyze_stage_performance(self, training_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance within current stage"""
        stage_analysis = {
            'stage_name': training_state.get('stage_name', 'unknown'),
            'stage_progress': training_state.get('stage_progress', 0.0),
            'performance_trend': {},
            'efficiency_score': 0.0
        }

        # Calculate performance trends within stage
        for metric, history in self.metrics_history.items():
            if len(history) > 0:
                stage_analysis['performance_trend'][metric] = self._calculate_recent_improvement(metric)

        # Calculate efficiency score (improvement per iteration)
        if 'mIoU' in self.metrics_history and len(self.metrics_history['mIoU']) > 10:
            recent_improvement = self._calculate_recent_improvement('mIoU')
            stage_analysis['efficiency_score'] = recent_improvement * 1000  # Scale for readability

        return stage_analysis

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) of values"""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        y = np.array(values)

        # Simple linear regression
        slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x)) if np.std(x) > 0 else 0.0
        return slope

    def _calculate_recent_improvement(self, metric: str) -> float:
        """Calculate recent improvement in a metric"""
        if metric not in self.metrics_history:
            return 0.0

        history = list(self.metrics_history[metric])
        if len(history) < 20:
            return 0.0

        recent_mean = np.mean(history[-10:])
        earlier_mean = np.mean(history[-20:-10])

        return recent_mean - earlier_mean


class ProgressiveTrainingCoordinator:
    """
    Main Progressive Training Coordinator

    Orchestrates all progressive training components including stage scheduling,
    learning rate adaptation, loss weight adjustment, and training monitoring.
    """

    def __init__(self,
                 stages_config: List[Dict[str, Any]],
                 total_iters: int,
                 base_lr: float = 1e-4,
                 initial_loss_weights: Optional[Dict[str, float]] = None):
        """
        Initialize Progressive Training Coordinator

        Args:
            stages_config: Configuration for training stages
            total_iters: Total training iterations
            base_lr: Base learning rate
            initial_loss_weights: Initial loss weights
        """
        self.stages_config = stages_config
        self.total_iters = total_iters
        self.base_lr = base_lr

        # Initialize components
        self.scheduler = ProgressiveTrainingScheduler(
            stages_config=stages_config,
            total_iters=total_iters,
            auto_transition=True
        )

        self.lr_scheduler = AdaptiveLearningRateScheduler(
            base_lr=base_lr,
            warmup_iters=1000
        )

        # Default loss weights if not provided
        if initial_loss_weights is None:
            initial_loss_weights = {
                'pixel': 0.4,
                'object': 0.3,
                'room': 0.2,
                'scene': 0.1,
                'temporal': 0.3
            }

        self.loss_adjuster = DynamicLossWeightAdjuster(
            initial_weights=initial_loss_weights,
            adaptation_rate=0.1
        )

        self.monitor = TrainingStateMonitor(
            monitoring_window=100,
            anomaly_threshold=2.0
        )

        # State tracking
        self.current_iter = 0
        self.training_history = []

    def update(self,
               current_iter: int,
               loss_components: Dict[str, float],
               performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Main update method for progressive training coordination

        Args:
            current_iter: Current training iteration
            loss_components: Current loss component values
            performance_metrics: Current performance metrics
        Returns:
            Complete training state and recommendations
        """
        self.current_iter = current_iter

        # Update progressive scheduler
        training_state = self.scheduler.update(current_iter, performance_metrics)

        # Update learning rate scheduler
        learning_rates = self.lr_scheduler.update(current_iter, training_state, performance_metrics)

        # Update loss weight adjuster
        loss_weights = self.loss_adjuster.update(current_iter, loss_components, training_state)

        # Update training monitor
        monitoring_report = self.monitor.update(
            current_iter, loss_components, performance_metrics, learning_rates, training_state
        )

        # Handle stage transitions
        if training_state.get('stage_changed', False):
            self._handle_stage_transition(training_state)

        # Compile complete training state
        complete_state = {
            'iteration': current_iter,
            'stage_info': training_state,
            'learning_rates': learning_rates,
            'loss_weights': loss_weights,
            'monitoring': monitoring_report,
            'recommendations': self._compile_recommendations(monitoring_report, training_state)
        }

        # Store in history
        self.training_history.append(complete_state)

        return complete_state

    def _handle_stage_transition(self, training_state: Dict[str, Any]):
        """Handle stage transition events"""
        new_stage = training_state['current_stage']
        stage_name = training_state.get('stage_name', f'stage_{new_stage}')

        # Reset adaptive components for new stage
        self.lr_scheduler.reset_plateau_count()
        self.loss_adjuster.reset_for_stage_transition()

        # Log stage transition
        print(f"🔄 Stage Transition: Entering {stage_name} (Stage {new_stage}) at iteration {self.current_iter}")

        # Stage-specific adjustments
        if new_stage == 0:  # Spatial learning stage
            print("   Focus: Spatial feature learning (pixel/object levels)")
        elif new_stage == 1:  # Temporal modeling stage
            print("   Focus: Temporal consistency and sequence modeling")
        elif new_stage == 2:  # Joint optimization stage
            print("   Focus: Joint optimization of all components")

    def _compile_recommendations(self,
                                monitoring_report: Dict[str, Any],
                                training_state: Dict[str, Any]) -> List[str]:
        """Compile comprehensive training recommendations"""
        recommendations = []

        # Add monitoring recommendations
        recommendations.extend(monitoring_report.get('recommendations', []))

        # Add stage-specific recommendations
        current_stage = training_state.get('current_stage', 0)
        stage_progress = training_state.get('stage_progress', 0.0)

        if stage_progress > 0.9 and current_stage < len(self.stages_config) - 1:
            recommendations.append("Consider preparing for next stage transition")

        # Add health-based recommendations
        training_health = monitoring_report.get('training_health', 'healthy')
        if training_health == 'warning':
            recommendations.append("Monitor training closely - potential issues detected")
        elif training_health == 'critical':
            recommendations.append("Consider intervention - critical issues detected")

        return recommendations

    def get_current_config(self) -> Dict[str, Any]:
        """Get current training configuration"""
        return {
            'current_stage': self.scheduler.current_stage,
            'stage_config': self.scheduler.get_current_config(),
            'current_lr': self.lr_scheduler.current_lr,
            'current_loss_weights': self.loss_adjuster.current_weights,
            'training_health': self.monitor.training_health
        }

    def force_stage_transition(self, target_stage: int):
        """Force transition to specific stage"""
        self.scheduler.force_stage_transition(target_stage)
        self._handle_stage_transition({'current_stage': target_stage, 'stage_name': f'stage_{target_stage}'})

    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        if not self.training_history:
            return {}

        latest_state = self.training_history[-1]

        return {
            'total_iterations': self.current_iter,
            'current_stage': latest_state['stage_info']['current_stage'],
            'stage_progress': latest_state['stage_info']['stage_progress'],
            'training_health': latest_state['monitoring']['training_health'],
            'current_lr': latest_state['learning_rates'],
            'current_loss_weights': latest_state['loss_weights'],
            'recent_recommendations': latest_state['recommendations'][-5:] if latest_state['recommendations'] else [],
            'performance_trends': latest_state['monitoring']['metrics_summary'],
            'convergence_status': latest_state['monitoring']['convergence_analysis']
        }

    def save_training_state(self, filepath: str):
        """Save current training state to file"""
        import json

        state_to_save = {
            'current_iter': self.current_iter,
            'scheduler_state': {
                'current_stage': self.scheduler.current_stage,
                'stage_start_iter': self.scheduler.stage_start_iter,
                'last_transition_iter': self.scheduler.last_transition_iter
            },
            'lr_scheduler_state': {
                'current_lr': self.lr_scheduler.current_lr,
                'plateau_count': self.lr_scheduler.plateau_count,
                'last_improvement_iter': self.lr_scheduler.last_improvement_iter
            },
            'loss_adjuster_state': {
                'current_weights': self.loss_adjuster.current_weights,
                'last_rebalance_iter': self.loss_adjuster.last_rebalance_iter
            },
            'training_summary': self.get_training_summary()
        }

        with open(filepath, 'w') as f:
            json.dump(state_to_save, f, indent=2)

    def load_training_state(self, filepath: str):
        """Load training state from file"""
        import json

        with open(filepath, 'r') as f:
            saved_state = json.load(f)

        # Restore state
        self.current_iter = saved_state['current_iter']

        # Restore scheduler state
        scheduler_state = saved_state['scheduler_state']
        self.scheduler.current_stage = scheduler_state['current_stage']
        self.scheduler.stage_start_iter = scheduler_state['stage_start_iter']
        self.scheduler.last_transition_iter = scheduler_state['last_transition_iter']

        # Restore lr scheduler state
        lr_state = saved_state['lr_scheduler_state']
        self.lr_scheduler.current_lr = lr_state['current_lr']
        self.lr_scheduler.plateau_count = lr_state['plateau_count']
        self.lr_scheduler.last_improvement_iter = lr_state['last_improvement_iter']

        # Restore loss adjuster state
        loss_state = saved_state['loss_adjuster_state']
        self.loss_adjuster.current_weights = loss_state['current_weights']
        self.loss_adjuster.last_rebalance_iter = loss_state['last_rebalance_iter']
