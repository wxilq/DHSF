"""
Enhanced Training Monitor Hook
增强训练监控钩子 - 全面监控训练过程中的各种指标和状态

功能包括：
1. 类别训练监控
2. 模型组件监控  
3. 渐进式训练监控
4. 训练健康度监控
5. 性能指标监控
6. 数据加载和处理监控
"""

import time
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, deque
from mmcv.runner import HOOKS, Hook
from mmcv.utils import get_logger
import psutil
import gc


@HOOKS.register_module()
class EnhancedTrainingMonitorHook(Hook):
    """增强训练监控钩子"""
    
    def __init__(self,
                 log_interval=50,
                 detailed_log_interval=200,
                 memory_log_interval=100,
                 gradient_log_interval=100,
                 feature_log_interval=500,
                 save_dir='./training_logs',
                 enable_feature_monitoring=True,
                 enable_gradient_monitoring=True,
                 enable_memory_monitoring=True,
                 enable_mamba_monitoring=True,
                 enable_patch_monitoring=True,
                 gradient_clip_threshold=5.0,
                 nan_detection=True,
                 priority='NORMAL'):
        
        self.log_interval = log_interval
        self.detailed_log_interval = detailed_log_interval
        self.memory_log_interval = memory_log_interval
        self.gradient_log_interval = gradient_log_interval
        self.feature_log_interval = feature_log_interval
        self.save_dir = save_dir
        
        # 监控开关
        self.enable_feature_monitoring = enable_feature_monitoring
        self.enable_gradient_monitoring = enable_gradient_monitoring
        self.enable_memory_monitoring = enable_memory_monitoring
        self.enable_mamba_monitoring = enable_mamba_monitoring
        self.enable_patch_monitoring = enable_patch_monitoring
        
        # 异常检测配置
        self.gradient_clip_threshold = gradient_clip_threshold
        self.nan_detection = nan_detection
        
        # 统计缓存
        self.loss_history = deque(maxlen=100)
        self.lr_history = deque(maxlen=100)
        self.batch_time_history = deque(maxlen=50)
        self.data_time_history = deque(maxlen=50)
        
        # 性能统计
        self.class_predictions = defaultdict(int)
        self.class_targets = defaultdict(int)
        self.ignored_pixels = 0
        self.total_pixels = 0
        
        # 时间统计
        self.batch_start_time = None
        self.data_start_time = None
        
        # 渐进式训练状态
        self.current_stage = None
        self.stage_start_iter = None
        self.stage_performance = {}
        
        self.logger = get_logger('EnhancedTrainingMonitor')
        
    def before_run(self, runner):
        """训练开始前的初始化"""
        self.logger.info("🚀 Enhanced Training Monitor Hook 已启动")
        self.logger.info("=" * 80)
        
        # 打印模型信息
        self._log_model_info(runner)
        
        # 打印数据集信息
        self._log_dataset_info(runner)
        
        # 打印训练配置
        self._log_training_config(runner)
        
        # 创建保存目录
        import os
        os.makedirs(self.save_dir, exist_ok=True)
        
    def before_train_epoch(self, runner):
        """每个epoch开始前"""
        self.logger.info(f"📅 开始 Epoch {runner.epoch + 1}/{runner.max_epochs}")
        
        # 检查渐进式训练阶段
        self._check_progressive_stage(runner)
        
        # 重置统计
        self.class_predictions.clear()
        self.class_targets.clear()
        self.ignored_pixels = 0
        self.total_pixels = 0
        
    def before_train_iter(self, runner):
        """每个iteration开始前"""
        self.data_start_time = time.time()
        
    def after_train_iter(self, runner):
        """每个iteration结束后"""
        # 计算时间
        if self.batch_start_time is not None:
            batch_time = time.time() - self.batch_start_time
            self.batch_time_history.append(batch_time)
        
        if self.data_start_time is not None:
            data_time = time.time() - self.data_start_time
            self.data_time_history.append(data_time)
        
        self.batch_start_time = time.time()
        
        # 基础日志
        if self.every_n_iters(runner, self.log_interval):
            self._log_basic_info(runner)
        
        # 详细日志
        if self.every_n_iters(runner, self.detailed_log_interval):
            self._log_detailed_info(runner)
        
        # 内存监控
        if self.enable_memory_monitoring and self.every_n_iters(runner, self.memory_log_interval):
            self._log_memory_info(runner)
        
        # 梯度监控
        if self.enable_gradient_monitoring and self.every_n_iters(runner, self.gradient_log_interval):
            self._log_gradient_info(runner)
        
        # 特征监控
        if self.enable_feature_monitoring and self.every_n_iters(runner, self.feature_log_interval):
            self._log_feature_info(runner)
        
        # 异常检测
        if self.nan_detection:
            self._detect_anomalies(runner)
    
    def _log_model_info(self, runner):
        """记录模型信息"""
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
            
        self.logger.info("🏗️  模型架构信息:")
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"   总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        self.logger.info(f"   可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        
        # 模型组件信息
        if hasattr(model, 'backbone'):
            backbone_params = sum(p.numel() for p in model.backbone.parameters())
            self.logger.info(f"   Backbone参数: {backbone_params:,} ({backbone_params/1e6:.2f}M)")
            
        if hasattr(model, 'neck'):
            neck_params = sum(p.numel() for p in model.neck.parameters())
            self.logger.info(f"   Neck参数: {neck_params:,} ({neck_params/1e6:.2f}M)")
            
        if hasattr(model, 'decode_head'):
            head_params = sum(p.numel() for p in model.decode_head.parameters())
            self.logger.info(f"   Head参数: {head_params:,} ({head_params/1e6:.2f}M)")
        
        # 创新模块信息
        self._log_innovation_modules(model)
        
    def _log_innovation_modules(self, model):
        """记录创新模块信息"""
        self.logger.info("🔬 创新模块状态:")
        
        # 检查SDSM
        if hasattr(model, 'neck') and hasattr(model.neck, 'enable_sdsm'):
            status = "✅ 已启用" if model.neck.enable_sdsm else "❌ 已禁用"
            self.logger.info(f"   SDSM时空解耦: {status}")
        else:
            self.logger.info("   SDSM时空解耦: ❌ 未检测到")

        # 检查Enhanced CHSM
        if hasattr(model, 'neck') and hasattr(model.neck, 'cross_hierarchical_fusion'):
            self.logger.info("   Enhanced CHSM: ✅ 已启用")
        else:
            self.logger.info("   Enhanced CHSM: ❌ 未检测到")

        # Enhanced AMSSM状态
        self.logger.info("   Enhanced AMSSM: ❌ 已禁用（节省显存）")
                
        # 检查自适应Patch Embedding
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'enable_adaptive_patch'):
            status = "✅ 已启用" if model.backbone.enable_adaptive_patch else "❌ 已禁用"
            self.logger.info(f"   自适应Patch Embedding: {status}")
            if model.backbone.enable_adaptive_patch:
                patch_type = getattr(model.backbone, 'adaptive_patch_type', 'unknown')
                self.logger.info(f"   Patch类型: {patch_type}")
        
    def _log_dataset_info(self, runner):
        """记录数据集信息"""
        self.logger.info("📊 数据集信息:")
        
        # 获取数据集配置
        if hasattr(runner, 'data_loader'):
            dataset = runner.data_loader.dataset
            self.logger.info(f"   训练样本数: {len(dataset):,}")
            
            # 时序信息
            if hasattr(dataset, 'temporal_length'):
                self.logger.info(f"   时序长度: {dataset.temporal_length}")
            if hasattr(dataset, 'temporal_stride'):
                self.logger.info(f"   时序步长: {dataset.temporal_stride}")
        
        # 类别信息
        if hasattr(runner.model, 'module'):
            model = runner.model.module
        else:
            model = runner.model
            
        if hasattr(model, 'decode_head') and hasattr(model.decode_head, 'num_classes'):
            num_classes = model.decode_head.num_classes
            self.logger.info(f"   类别数量: {num_classes}")
            self.logger.info("   ⚠️  注意: 只训练占比>0.5%的25个类别，其他类别已ignore")
        
    def _log_training_config(self, runner):
        """记录训练配置"""
        self.logger.info("⚙️  训练配置:")
        
        # 优化器信息
        if hasattr(runner, 'optimizer'):
            optimizer = runner.optimizer
            self.logger.info(f"   优化器: {type(optimizer).__name__}")
            if hasattr(optimizer, 'param_groups'):
                for i, group in enumerate(optimizer.param_groups):
                    lr = group.get('lr', 'N/A')
                    weight_decay = group.get('weight_decay', 'N/A')
                    self.logger.info(f"   参数组{i}: lr={lr}, weight_decay={weight_decay}")
        
        # 学习率调度
        if hasattr(runner, 'lr_config'):
            lr_config = runner.lr_config
            self.logger.info(f"   学习率策略: {lr_config.get('policy', 'N/A')}")
        
        # 批次大小
        if hasattr(runner, 'data_loader'):
            batch_size = runner.data_loader.batch_size
            self.logger.info(f"   批次大小: {batch_size}")
        
        self.logger.info("=" * 80)
    
    def _check_progressive_stage(self, runner):
        """检查渐进式训练阶段"""
        # 这里需要根据具体的渐进式训练钩子来获取当前阶段
        # 暂时使用epoch来判断
        epoch = runner.epoch
        
        if epoch < 20:
            stage = "spatial_learning"
        elif epoch < 40:
            stage = "temporal_modeling"
        else:
            stage = "joint_optimization"
        
        if self.current_stage != stage:
            if self.current_stage is not None:
                self.logger.info(f"🔄 训练阶段切换: {self.current_stage} → {stage}")
            else:
                self.logger.info(f"🎯 当前训练阶段: {stage}")
            
            self.current_stage = stage
            self.stage_start_iter = runner.iter
    
    def _log_basic_info(self, runner):
        """记录基础信息"""
        # 获取当前损失
        if 'loss' in runner.log_buffer.output:
            current_loss = runner.log_buffer.output['loss']
            self.loss_history.append(current_loss)
        
        # 获取当前学习率
        if hasattr(runner, 'current_lr'):
            current_lr = runner.current_lr()[0] if runner.current_lr() else 0
            self.lr_history.append(current_lr)
        
        # 计算平均时间
        avg_batch_time = np.mean(self.batch_time_history) if self.batch_time_history else 0
        avg_data_time = np.mean(self.data_time_history) if self.data_time_history else 0
        
        # 基础日志
        info_str = (f"Epoch [{runner.epoch + 1}/{runner.max_epochs}] "
                   f"Iter [{runner.iter + 1}/{runner.max_iters}] "
                   f"Stage: {self.current_stage} ")
        
        if self.loss_history:
            info_str += f"Loss: {self.loss_history[-1]:.4f} "
        if self.lr_history:
            info_str += f"LR: {self.lr_history[-1]:.2e} "
        
        info_str += f"Time: {avg_batch_time:.3f}s Data: {avg_data_time:.3f}s"
        
        self.logger.info(info_str)

    def _log_detailed_info(self, runner):
        """记录详细信息"""
        self.logger.info("📈 详细训练统计:")

        # 损失趋势
        if len(self.loss_history) >= 10:
            recent_loss = np.mean(list(self.loss_history)[-10:])
            early_loss = np.mean(list(self.loss_history)[:10])
            loss_trend = "📉 下降" if recent_loss < early_loss else "📈 上升"
            self.logger.info(f"   损失趋势: {loss_trend} (最近10次: {recent_loss:.4f})")

        # 学习率变化
        if len(self.lr_history) >= 2:
            lr_change = self.lr_history[-1] - self.lr_history[-2]
            lr_trend = "📈 增加" if lr_change > 0 else "📉 减少" if lr_change < 0 else "➡️ 不变"
            self.logger.info(f"   学习率变化: {lr_trend} (当前: {self.lr_history[-1]:.2e})")

        # 类别预测统计
        if self.class_predictions:
            total_predictions = sum(self.class_predictions.values())
            ignore_ratio = self.ignored_pixels / max(self.total_pixels, 1) * 100
            self.logger.info(f"   类别预测: {len(self.class_predictions)}个类别, ignore比例: {ignore_ratio:.1f}%")

            # 显示前5个最常预测的类别
            top_classes = sorted(self.class_predictions.items(), key=lambda x: x[1], reverse=True)[:5]
            for class_id, count in top_classes:
                ratio = count / total_predictions * 100
                self.logger.info(f"     类别{class_id}: {count}次 ({ratio:.1f}%)")

        # 时间统计
        if self.batch_time_history and self.data_time_history:
            avg_batch = np.mean(self.batch_time_history)
            avg_data = np.mean(self.data_time_history)
            compute_ratio = (avg_batch - avg_data) / avg_batch * 100
            self.logger.info(f"   时间分析: 计算{compute_ratio:.1f}% 数据加载{100-compute_ratio:.1f}%")

    def _log_memory_info(self, runner):
        """记录内存信息"""
        if not torch.cuda.is_available():
            return

        self.logger.info("💾 内存使用情况:")

        # GPU内存
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3

            self.logger.info(f"   GPU {i}: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB, 峰值 {max_allocated:.2f}GB")

        # CPU内存
        cpu_percent = psutil.virtual_memory().percent
        cpu_used = psutil.virtual_memory().used / 1024**3
        self.logger.info(f"   CPU内存: {cpu_used:.2f}GB ({cpu_percent:.1f}%)")

        # 垃圾回收
        gc_counts = gc.get_count()
        self.logger.info(f"   GC计数: {gc_counts}")

    def _log_gradient_info(self, runner):
        """记录梯度信息"""
        if not self.enable_gradient_monitoring:
            return

        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        self.logger.info("📊 梯度统计:")

        # 计算梯度统计
        total_norm = 0
        param_count = 0
        grad_stats = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

                # 按模块分组统计
                module_name = name.split('.')[0]
                if module_name not in grad_stats:
                    grad_stats[module_name] = []
                grad_stats[module_name].append(param_norm.item())

        total_norm = total_norm ** (1. / 2)

        self.logger.info(f"   总梯度范数: {total_norm:.4f}")
        self.logger.info(f"   参数数量: {param_count}")

        # 各模块梯度统计
        for module_name, norms in grad_stats.items():
            avg_norm = np.mean(norms)
            max_norm = np.max(norms)
            self.logger.info(f"   {module_name}: 平均 {avg_norm:.4f}, 最大 {max_norm:.4f}")

        # 梯度异常检测
        if total_norm > self.gradient_clip_threshold:
            self.logger.warning(f"⚠️  梯度范数过大: {total_norm:.4f} > {self.gradient_clip_threshold}")

    def _log_feature_info(self, runner):
        """记录特征信息"""
        if not self.enable_feature_monitoring:
            return

        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        self.logger.info("🔍 特征统计:")

        # 监控backbone特征
        self._monitor_backbone_features(model)

        # 监控Mamba状态
        if self.enable_mamba_monitoring:
            self._monitor_mamba_states(model)

        # 监控自适应Patch
        if self.enable_patch_monitoring:
            self._monitor_adaptive_patch(model)

    def _monitor_backbone_features(self, model):
        """监控backbone特征"""
        if not hasattr(model, 'backbone'):
            return

        # 这里需要hook来获取中间特征，暂时跳过具体实现
        self.logger.info("   Backbone特征: 需要添加feature hook")

    def _monitor_mamba_states(self, model):
        """监控Mamba状态"""
        # AMSSM已禁用，检查其他Mamba相关模块
        if hasattr(model, 'neck'):
            self.logger.info("   Mamba状态: 轻量级时序处理已启用")
        else:
            self.logger.info("   Mamba状态: 未检测到时序处理模块")

    def _monitor_adaptive_patch(self, model):
        """监控自适应Patch"""
        if not (hasattr(model, 'backbone') and hasattr(model.backbone, 'enable_adaptive_patch')):
            return

        if not model.backbone.enable_adaptive_patch:
            return

        self.logger.info("   自适应Patch: 需要添加patch监控")

    def _detect_anomalies(self, runner):
        """异常检测"""
        if not self.nan_detection:
            return

        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        # 检测NaN值
        nan_params = []
        inf_params = []

        for name, param in model.named_parameters():
            if param.data.isnan().any():
                nan_params.append(name)
            if param.data.isinf().any():
                inf_params.append(name)

        if nan_params:
            self.logger.error(f"🚨 检测到NaN参数: {nan_params}")
        if inf_params:
            self.logger.error(f"🚨 检测到Inf参数: {inf_params}")

        # 检测损失异常
        if 'loss' in runner.log_buffer.output:
            current_loss = runner.log_buffer.output['loss']
            if np.isnan(current_loss) or np.isinf(current_loss):
                self.logger.error(f"🚨 异常损失值: {current_loss}")

    def after_train_epoch(self, runner):
        """每个epoch结束后"""
        self.logger.info(f"✅ Epoch {runner.epoch + 1} 完成")

        # 保存epoch统计
        self._save_epoch_stats(runner)

    def _save_epoch_stats(self, runner):
        """保存epoch统计信息"""
        import json
        import os

        stats = {
            'epoch': runner.epoch + 1,
            'stage': self.current_stage,
            'loss_history': list(self.loss_history),
            'lr_history': list(self.lr_history),
            'class_predictions': dict(self.class_predictions),
            'ignored_pixels': self.ignored_pixels,
            'total_pixels': self.total_pixels,
            'avg_batch_time': np.mean(self.batch_time_history) if self.batch_time_history else 0,
            'avg_data_time': np.mean(self.data_time_history) if self.data_time_history else 0
        }

        stats_file = os.path.join(self.save_dir, f'epoch_{runner.epoch + 1}_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

    def after_run(self, runner):
        """训练结束后"""
        self.logger.info("🎉 训练完成！")
        self.logger.info("=" * 80)

        # 保存最终统计
        self._save_final_stats(runner)

    def _save_final_stats(self, runner):
        """保存最终统计信息"""
        import json
        import os

        final_stats = {
            'total_epochs': runner.epoch + 1,
            'total_iters': runner.iter + 1,
            'final_stage': self.current_stage,
            'training_time': time.time() - runner._epoch_start_time if hasattr(runner, '_epoch_start_time') else 0,
            'final_loss': self.loss_history[-1] if self.loss_history else None,
            'final_lr': self.lr_history[-1] if self.lr_history else None
        }

        stats_file = os.path.join(self.save_dir, 'final_training_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(final_stats, f, indent=2)
