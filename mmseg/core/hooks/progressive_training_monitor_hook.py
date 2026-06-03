"""
Progressive Training Monitor Hook
渐进式训练监控钩子 - 监控渐进式训练的阶段切换和性能变化

功能包括：
1. 监控当前训练阶段
2. 阶段切换时的性能变化分析
3. 各阶段的损失权重监控
4. 学习率调度监控
5. 阶段性能对比分析
"""

import torch
import numpy as np
from collections import defaultdict, deque
from mmcv.runner import HOOKS, Hook
from mmcv.utils import get_logger
import json
import os

# 兼容性导入 - 处理可视化库
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found, visualization features will be disabled")


@HOOKS.register_module()
class ProgressiveTrainingMonitorHook(Hook):
    """渐进式训练监控钩子"""
    
    def __init__(self,
                 log_interval=50,
                 stage_analysis_interval=200,
                 save_dir='./progressive_training_logs',
                 enable_stage_comparison=True,
                 enable_loss_weight_plot=True,
                 enable_lr_schedule_plot=True,
                 priority='NORMAL'):
        
        self.log_interval = log_interval
        self.stage_analysis_interval = stage_analysis_interval
        self.save_dir = save_dir
        self.enable_stage_comparison = enable_stage_comparison
        self.enable_loss_weight_plot = enable_loss_weight_plot
        self.enable_lr_schedule_plot = enable_lr_schedule_plot
        
        # 阶段定义
        self.stage_definitions = {
            'spatial_learning': {'epoch_range': (0, 20), 'description': '空间学习阶段'},
            'temporal_modeling': {'epoch_range': (20, 40), 'description': '时序引入阶段'},
            'joint_optimization': {'epoch_range': (40, 50), 'description': '联合优化阶段'}
        }
        
        # 当前状态
        self.current_stage = None
        self.stage_start_epoch = None
        self.stage_start_iter = None
        
        # 阶段性能统计
        self.stage_performance = {}
        self.stage_loss_history = defaultdict(list)
        self.stage_lr_history = defaultdict(list)
        self.stage_metrics_history = defaultdict(lambda: defaultdict(list))
        
        # 损失权重历史
        self.loss_weights_history = []
        
        # 阶段切换记录
        self.stage_switches = []
        
        self.logger = get_logger('ProgressiveTrainingMonitor')
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
    
    def before_run(self, runner):
        """训练开始前初始化"""
        self.logger.info("🎯 渐进式训练监控钩子已启动")
        self.logger.info("📋 训练阶段配置:")
        
        for stage_name, config in self.stage_definitions.items():
            epoch_start, epoch_end = config['epoch_range']
            description = config['description']
            self.logger.info(f"   {stage_name:20s}: Epoch {epoch_start:2d}-{epoch_end:2d} ({description})")
        
        # 初始化当前阶段
        self._update_current_stage(runner)
    
    def before_train_epoch(self, runner):
        """每个epoch开始前检查阶段"""
        old_stage = self.current_stage
        self._update_current_stage(runner)
        
        # 检查是否发生阶段切换
        if old_stage != self.current_stage and old_stage is not None:
            self._handle_stage_switch(runner, old_stage, self.current_stage)
    
    def after_train_iter(self, runner):
        """每个iteration后更新统计"""
        # 记录当前阶段的性能
        self._record_stage_performance(runner)
        
        # 基础日志
        if self.every_n_iters(runner, self.log_interval):
            self._log_basic_stage_info(runner)
        
        # 阶段分析
        if self.every_n_iters(runner, self.stage_analysis_interval):
            self._log_stage_analysis(runner)
    
    def _update_current_stage(self, runner):
        """更新当前训练阶段"""
        current_epoch = runner.epoch
        
        for stage_name, config in self.stage_definitions.items():
            epoch_start, epoch_end = config['epoch_range']
            if epoch_start <= current_epoch < epoch_end:
                if self.current_stage != stage_name:
                    self.current_stage = stage_name
                    self.stage_start_epoch = current_epoch
                    self.stage_start_iter = runner.iter
                break
    
    def _handle_stage_switch(self, runner, old_stage, new_stage):
        """处理阶段切换"""
        self.logger.info("=" * 80)
        self.logger.info(f"🔄 训练阶段切换: {old_stage} → {new_stage}")
        self.logger.info(f"   切换时间: Epoch {runner.epoch}, Iter {runner.iter}")
        
        # 记录切换信息
        switch_info = {
            'epoch': runner.epoch,
            'iter': runner.iter,
            'old_stage': old_stage,
            'new_stage': new_stage,
            'timestamp': runner._epoch_start_time if hasattr(runner, '_epoch_start_time') else 0
        }
        self.stage_switches.append(switch_info)
        
        # 分析上一阶段的性能
        if old_stage in self.stage_performance:
            self._analyze_stage_performance(old_stage)
        
        # 打印新阶段信息
        new_stage_config = self.stage_definitions[new_stage]
        self.logger.info(f"   新阶段: {new_stage_config['description']}")
        
        # 获取当前损失权重和学习率
        self._log_stage_configuration(runner)
        
        self.logger.info("=" * 80)
    
    def _record_stage_performance(self, runner):
        """记录当前阶段的性能"""
        if self.current_stage is None:
            return
        
        # 记录损失
        if 'loss' in runner.log_buffer.output:
            loss = runner.log_buffer.output['loss']
            self.stage_loss_history[self.current_stage].append(loss)
        
        # 记录学习率
        if hasattr(runner, 'current_lr') and runner.current_lr():
            lr = runner.current_lr()[0]
            self.stage_lr_history[self.current_stage].append(lr)
        
        # 记录其他指标
        for key, value in runner.log_buffer.output.items():
            if key.startswith(('acc', 'iou', 'dice')):
                self.stage_metrics_history[self.current_stage][key].append(value)
        
        # 记录损失权重（如果可用）
        self._record_loss_weights(runner)
    
    def _record_loss_weights(self, runner):
        """记录损失权重"""
        # 尝试从模型或损失函数中获取权重
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        
        loss_weights = {}
        
        # 检查是否有层次化损失
        if hasattr(model, 'decode_head') and hasattr(model.decode_head, 'loss_decode'):
            loss_decode = model.decode_head.loss_decode
            if hasattr(loss_decode, 'loss_weights'):
                loss_weights = loss_decode.loss_weights
        
        # 如果找到权重，记录下来
        if loss_weights:
            weight_record = {
                'epoch': runner.epoch,
                'iter': runner.iter,
                'stage': self.current_stage,
                'weights': dict(loss_weights)
            }
            self.loss_weights_history.append(weight_record)
    
    def _log_basic_stage_info(self, runner):
        """记录基础阶段信息"""
        if self.current_stage is None:
            return
        
        # 计算阶段进度
        stage_config = self.stage_definitions[self.current_stage]
        epoch_start, epoch_end = stage_config['epoch_range']
        stage_progress = (runner.epoch - epoch_start) / (epoch_end - epoch_start) * 100
        
        # 阶段内的iteration数
        stage_iters = runner.iter - self.stage_start_iter if self.stage_start_iter is not None else 0
        
        info_str = (f"🎯 阶段: {self.current_stage} "
                   f"进度: {stage_progress:.1f}% "
                   f"阶段内Iter: {stage_iters}")
        
        # 添加阶段性能
        if self.current_stage in self.stage_loss_history:
            recent_losses = self.stage_loss_history[self.current_stage][-10:]
            if recent_losses:
                avg_loss = np.mean(recent_losses)
                info_str += f" 近期损失: {avg_loss:.4f}"
        
        self.logger.info(info_str)
    
    def _log_stage_analysis(self, runner):
        """记录阶段分析"""
        if self.current_stage is None:
            return
        
        self.logger.info(f"📊 {self.current_stage} 阶段分析:")
        
        # 损失趋势分析
        if self.current_stage in self.stage_loss_history:
            losses = self.stage_loss_history[self.current_stage]
            if len(losses) >= 20:
                recent_avg = np.mean(losses[-10:])
                early_avg = np.mean(losses[:10])
                trend = "📉 下降" if recent_avg < early_avg else "📈 上升"
                change_rate = (recent_avg - early_avg) / early_avg * 100
                self.logger.info(f"   损失趋势: {trend} ({change_rate:+.1f}%)")
        
        # 学习率变化
        if self.current_stage in self.stage_lr_history:
            lrs = self.stage_lr_history[self.current_stage]
            if len(lrs) >= 2:
                lr_change = (lrs[-1] - lrs[0]) / lrs[0] * 100
                lr_trend = "📈 增加" if lr_change > 0 else "📉 减少" if lr_change < 0 else "➡️ 稳定"
                self.logger.info(f"   学习率变化: {lr_trend} ({lr_change:+.1f}%)")
        
        # 其他指标分析
        for metric_name, values in self.stage_metrics_history[self.current_stage].items():
            if len(values) >= 10:
                recent_avg = np.mean(values[-5:])
                self.logger.info(f"   {metric_name}: {recent_avg:.4f}")
    
    def _log_stage_configuration(self, runner):
        """记录阶段配置"""
        self.logger.info("⚙️  当前阶段配置:")
        
        # 学习率配置
        if hasattr(runner, 'current_lr') and runner.current_lr():
            lrs = runner.current_lr()
            for i, lr in enumerate(lrs):
                self.logger.info(f"   参数组{i} 学习率: {lr:.2e}")
        
        # 损失权重配置
        if self.loss_weights_history:
            latest_weights = self.loss_weights_history[-1]['weights']
            self.logger.info("   损失权重:")
            for loss_name, weight in latest_weights.items():
                self.logger.info(f"     {loss_name}: {weight:.3f}")
    
    def _analyze_stage_performance(self, stage_name):
        """分析阶段性能"""
        self.logger.info(f"📈 {stage_name} 阶段性能总结:")
        
        # 损失分析
        if stage_name in self.stage_loss_history:
            losses = self.stage_loss_history[stage_name]
            if losses:
                min_loss = min(losses)
                max_loss = max(losses)
                avg_loss = np.mean(losses)
                final_loss = losses[-1]
                
                self.logger.info(f"   损失统计: 最小 {min_loss:.4f}, 最大 {max_loss:.4f}, "
                                f"平均 {avg_loss:.4f}, 最终 {final_loss:.4f}")
        
        # 保存阶段性能
        self.stage_performance[stage_name] = {
            'loss_history': self.stage_loss_history[stage_name].copy(),
            'lr_history': self.stage_lr_history[stage_name].copy(),
            'metrics_history': dict(self.stage_metrics_history[stage_name])
        }
    
    def after_train_epoch(self, runner):
        """每个epoch结束后保存统计"""
        # 保存阶段统计
        self._save_stage_stats(runner.epoch)
        
        # 生成可视化图表
        if self.enable_loss_weight_plot:
            self._save_loss_weight_plot(runner.epoch)
        
        if self.enable_lr_schedule_plot:
            self._save_lr_schedule_plot(runner.epoch)
    
    def _save_stage_stats(self, epoch):
        """保存阶段统计"""
        stats = {
            'epoch': epoch + 1,
            'current_stage': self.current_stage,
            'stage_switches': self.stage_switches,
            'stage_performance': self.stage_performance,
            'loss_weights_history': self.loss_weights_history
        }
        
        stats_file = os.path.join(self.save_dir, f'progressive_stats_epoch_{epoch + 1}.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
    
    def _save_loss_weight_plot(self, epoch):
        """保存损失权重变化图"""
        if not self.loss_weights_history or not HAS_MATPLOTLIB:
            if not HAS_MATPLOTLIB:
                self.logger.warning("matplotlib not available, skipping loss weight plot")
            return

        plt.figure(figsize=(12, 8))
        
        # 提取数据
        epochs = [record['epoch'] for record in self.loss_weights_history]
        weight_names = set()
        for record in self.loss_weights_history:
            weight_names.update(record['weights'].keys())
        
        # 绘制每个权重的变化
        for weight_name in weight_names:
            weights = []
            for record in self.loss_weights_history:
                weights.append(record['weights'].get(weight_name, 0))
            plt.plot(epochs, weights, label=weight_name, marker='o', markersize=3)
        
        # 添加阶段分割线
        for switch in self.stage_switches:
            plt.axvline(x=switch['epoch'], color='red', linestyle='--', alpha=0.7)
            plt.text(switch['epoch'], plt.ylim()[1] * 0.9, 
                    f"{switch['new_stage']}", rotation=90, ha='right')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss Weight')
        plt.title(f'Loss Weights Evolution (Epoch {epoch + 1})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'loss_weights_epoch_{epoch + 1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_lr_schedule_plot(self, epoch):
        """保存学习率调度图"""
        if not any(self.stage_lr_history.values()) or not HAS_MATPLOTLIB:
            if not HAS_MATPLOTLIB:
                self.logger.warning("matplotlib not available, skipping lr schedule plot")
            return

        plt.figure(figsize=(12, 6))
        
        # 绘制各阶段的学习率
        colors = ['blue', 'green', 'red']
        for i, (stage_name, lr_history) in enumerate(self.stage_lr_history.items()):
            if lr_history:
                x = range(len(lr_history))
                plt.plot(x, lr_history, label=f'{stage_name}', 
                        color=colors[i % len(colors)], linewidth=2)
        
        plt.xlabel('Iteration (within stage)')
        plt.ylabel('Learning Rate')
        plt.title(f'Learning Rate Schedule by Stage (Epoch {epoch + 1})')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'lr_schedule_epoch_{epoch + 1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def after_run(self, runner):
        """训练结束后的总结"""
        self.logger.info("🎉 渐进式训练完成！")
        self.logger.info("=" * 80)
        
        # 最终阶段性能分析
        if self.current_stage and self.current_stage not in self.stage_performance:
            self._analyze_stage_performance(self.current_stage)
        
        # 生成最终对比分析
        if self.enable_stage_comparison:
            self._generate_final_comparison()
        
        # 保存最终统计
        self._save_final_progressive_stats()
    
    def _generate_final_comparison(self):
        """生成最终的阶段对比分析"""
        self.logger.info("📊 各阶段性能对比:")
        
        for stage_name, performance in self.stage_performance.items():
            loss_history = performance.get('loss_history', [])
            if loss_history:
                avg_loss = np.mean(loss_history)
                min_loss = min(loss_history)
                final_loss = loss_history[-1]
                
                stage_desc = self.stage_definitions[stage_name]['description']
                self.logger.info(f"   {stage_desc}:")
                self.logger.info(f"     平均损失: {avg_loss:.4f}")
                self.logger.info(f"     最小损失: {min_loss:.4f}")
                self.logger.info(f"     最终损失: {final_loss:.4f}")
    
    def _save_final_progressive_stats(self):
        """保存最终的渐进式训练统计"""
        final_stats = {
            'total_stages': len(self.stage_definitions),
            'stage_switches': self.stage_switches,
            'stage_performance': self.stage_performance,
            'loss_weights_history': self.loss_weights_history,
            'final_stage': self.current_stage
        }
        
        stats_file = os.path.join(self.save_dir, 'final_progressive_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(final_stats, f, indent=2, default=str)
