"""
Performance and Health Monitor Hook
性能和健康度监控钩子 - 监控训练过程中的性能指标和健康状态

功能包括：
1. 性能指标监控（损失、准确率、IoU等）
2. 训练健康度监控（梯度、内存、异常检测）
3. 数据加载和处理时间监控
4. 模型组件特征统计监控
5. 异常检测和预警
"""

import time
import torch
import torch.nn as nn
import numpy as np
import gc
from collections import defaultdict, deque
from mmcv.runner import HOOKS, Hook
from mmcv.utils import get_logger
import json
import os
import warnings

# 兼容性导入 - 处理psutil
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not found, system monitoring features will be disabled")

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
class PerformanceHealthMonitorHook(Hook):
    """性能和健康度监控钩子"""
    
    def __init__(self,
                 log_interval=50,
                 health_check_interval=100,
                 performance_log_interval=200,
                 feature_monitor_interval=500,
                 save_dir='./performance_health_logs',
                 enable_gradient_monitoring=True,
                 enable_memory_monitoring=True,
                 enable_feature_monitoring=True,
                 enable_anomaly_detection=True,
                 gradient_clip_threshold=5.0,
                 memory_warning_threshold=0.9,
                 loss_spike_threshold=2.0,
                 priority='NORMAL'):
        
        self.log_interval = log_interval
        self.health_check_interval = health_check_interval
        self.performance_log_interval = performance_log_interval
        self.feature_monitor_interval = feature_monitor_interval
        self.save_dir = save_dir
        
        # 监控开关
        self.enable_gradient_monitoring = enable_gradient_monitoring
        self.enable_memory_monitoring = enable_memory_monitoring
        self.enable_feature_monitoring = enable_feature_monitoring
        self.enable_anomaly_detection = enable_anomaly_detection
        
        # 阈值设置
        self.gradient_clip_threshold = gradient_clip_threshold
        self.memory_warning_threshold = memory_warning_threshold
        self.loss_spike_threshold = loss_spike_threshold
        
        # 性能统计
        self.performance_metrics = defaultdict(deque)
        self.timing_stats = {
            'data_loading_times': deque(maxlen=100),
            'forward_times': deque(maxlen=100),
            'backward_times': deque(maxlen=100),
            'batch_times': deque(maxlen=100)
        }
        
        # 健康度统计
        self.health_stats = {
            'gradient_norms': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'gpu_utilization': deque(maxlen=100),
            'nan_detections': [],
            'inf_detections': [],
            'loss_spikes': []
        }
        
        # 特征统计
        self.feature_stats = defaultdict(lambda: defaultdict(deque))
        
        # 时间记录
        self.iter_start_time = None
        self.data_start_time = None
        self.forward_start_time = None
        self.backward_start_time = None
        
        # 异常计数
        self.anomaly_counts = defaultdict(int)
        
        self.logger = get_logger('PerformanceHealthMonitor')
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
    
    def before_run(self, runner):
        """训练开始前初始化"""
        self.logger.info("🔍 性能和健康度监控钩子已启动")
        self.logger.info("📋 监控配置:")
        self.logger.info(f"   梯度监控: {'✅' if self.enable_gradient_monitoring else '❌'}")
        self.logger.info(f"   内存监控: {'✅' if self.enable_memory_monitoring else '❌'}")
        self.logger.info(f"   特征监控: {'✅' if self.enable_feature_monitoring else '❌'}")
        self.logger.info(f"   异常检测: {'✅' if self.enable_anomaly_detection else '❌'}")
        self.logger.info(f"   梯度阈值: {self.gradient_clip_threshold}")
        self.logger.info(f"   内存警告阈值: {self.memory_warning_threshold * 100:.0f}%")
        
        # 记录初始系统状态
        self._log_system_info()
    
    def before_train_iter(self, runner):
        """每个iteration开始前"""
        self.iter_start_time = time.time()
        self.data_start_time = time.time()
    
    def after_train_iter(self, runner):
        """每个iteration结束后"""
        # 记录时间统计
        self._record_timing_stats(runner)
        
        # 记录性能指标
        self._record_performance_metrics(runner)
        
        # 基础日志
        if self.every_n_iters(runner, self.log_interval):
            self._log_basic_performance(runner)
        
        # 健康检查
        if self.every_n_iters(runner, self.health_check_interval):
            self._perform_health_check(runner)
        
        # 详细性能日志
        if self.every_n_iters(runner, self.performance_log_interval):
            self._log_detailed_performance(runner)
        
        # 特征监控
        if self.enable_feature_monitoring and self.every_n_iters(runner, self.feature_monitor_interval):
            self._monitor_model_features(runner)
        
        # 异常检测
        if self.enable_anomaly_detection:
            self._detect_anomalies(runner)
    
    def _log_system_info(self):
        """记录系统信息"""
        self.logger.info("💻 系统信息:")

        if HAS_PSUTIL:
            # CPU信息
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            self.logger.info(f"   CPU: {cpu_count}核心, {cpu_freq.current:.0f}MHz")

            # 内存信息
            memory = psutil.virtual_memory()
            self.logger.info(f"   内存: {memory.total / 1024**3:.1f}GB 总计")
        else:
            self.logger.info("   系统信息监控不可用 (psutil未安装)")
        
        # GPU信息
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                self.logger.info(f"   GPU {i}: {gpu_name}, {gpu_memory:.1f}GB")
        else:
            self.logger.info("   GPU: 未检测到CUDA设备")
    
    def _record_timing_stats(self, runner):
        """记录时间统计"""
        current_time = time.time()
        
        if self.iter_start_time is not None:
            batch_time = current_time - self.iter_start_time
            self.timing_stats['batch_times'].append(batch_time)
        
        # 这里需要在适当的位置记录其他时间
        # 由于无法直接hook到forward/backward，暂时使用估算
        if hasattr(runner, '_forward_time'):
            self.timing_stats['forward_times'].append(runner._forward_time)
        if hasattr(runner, '_backward_time'):
            self.timing_stats['backward_times'].append(runner._backward_time)
    
    def _record_performance_metrics(self, runner):
        """记录性能指标"""
        # 记录损失
        for key, value in runner.log_buffer.output.items():
            if isinstance(value, (int, float)):
                self.performance_metrics[key].append(value)
        
        # 记录学习率
        if hasattr(runner, 'current_lr') and runner.current_lr():
            lr = runner.current_lr()[0]
            self.performance_metrics['learning_rate'].append(lr)
    
    def _log_basic_performance(self, runner):
        """记录基础性能信息"""
        # 计算平均时间
        avg_batch_time = np.mean(self.timing_stats['batch_times']) if self.timing_stats['batch_times'] else 0
        
        # 获取最新指标
        latest_loss = self.performance_metrics['loss'][-1] if self.performance_metrics['loss'] else 0
        latest_lr = self.performance_metrics['learning_rate'][-1] if self.performance_metrics['learning_rate'] else 0
        
        # 计算吞吐量
        throughput = 1.0 / avg_batch_time if avg_batch_time > 0 else 0
        
        info_str = (f"⚡ 性能: 损失 {latest_loss:.4f}, "
                   f"学习率 {latest_lr:.2e}, "
                   f"批次时间 {avg_batch_time:.3f}s, "
                   f"吞吐量 {throughput:.1f} batch/s")
        
        self.logger.info(info_str)
    
    def _perform_health_check(self, runner):
        """执行健康检查"""
        self.logger.info("🏥 健康检查:")
        
        # 梯度健康检查
        if self.enable_gradient_monitoring:
            self._check_gradient_health(runner)
        
        # 内存健康检查
        if self.enable_memory_monitoring:
            self._check_memory_health(runner)
        
        # 模型参数健康检查
        self._check_parameter_health(runner)
    
    def _check_gradient_health(self, runner):
        """检查梯度健康状态"""
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        
        total_norm = 0
        param_count = 0
        zero_grad_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                if param_norm.item() < 1e-8:
                    zero_grad_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            self.health_stats['gradient_norms'].append(total_norm)
            
            zero_grad_ratio = zero_grad_count / param_count * 100
            
            # 梯度健康状态
            if total_norm > self.gradient_clip_threshold:
                status = "🔴 异常 (梯度过大)"
                self.anomaly_counts['large_gradient'] += 1
            elif total_norm < 1e-6:
                status = "🟡 警告 (梯度过小)"
                self.anomaly_counts['small_gradient'] += 1
            elif zero_grad_ratio > 50:
                status = "🟡 警告 (零梯度过多)"
                self.anomaly_counts['zero_gradient'] += 1
            else:
                status = "🟢 正常"
            
            self.logger.info(f"   梯度状态: {status}")
            self.logger.info(f"   梯度范数: {total_norm:.4f}")
            self.logger.info(f"   零梯度比例: {zero_grad_ratio:.1f}%")
    
    def _check_memory_health(self, runner):
        """检查内存健康状态"""
        # GPU内存检查
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i)
                reserved = torch.cuda.memory_reserved(i)
                total = torch.cuda.get_device_properties(i).total_memory
                
                allocated_ratio = allocated / total
                reserved_ratio = reserved / total
                
                self.health_stats['memory_usage'].append(allocated_ratio)
                
                if allocated_ratio > self.memory_warning_threshold:
                    status = "🔴 警告 (内存使用过高)"
                    self.anomaly_counts['high_memory'] += 1
                elif allocated_ratio > 0.7:
                    status = "🟡 注意 (内存使用较高)"
                else:
                    status = "🟢 正常"
                
                self.logger.info(f"   GPU {i} 内存: {status}")
                self.logger.info(f"   已分配: {allocated / 1024**3:.2f}GB ({allocated_ratio * 100:.1f}%)")
                self.logger.info(f"   已保留: {reserved / 1024**3:.2f}GB ({reserved_ratio * 100:.1f}%)")
        
        # CPU内存检查
        if HAS_PSUTIL:
            cpu_memory = psutil.virtual_memory()
            cpu_usage_ratio = cpu_memory.percent / 100

            if cpu_usage_ratio > 0.9:
                cpu_status = "🔴 警告 (CPU内存使用过高)"
            elif cpu_usage_ratio > 0.7:
                cpu_status = "🟡 注意 (CPU内存使用较高)"
            else:
                cpu_status = "🟢 正常"

            self.logger.info(f"   CPU内存: {cpu_status} ({cpu_memory.percent:.1f}%)")
        else:
            self.logger.info("   CPU内存监控不可用 (psutil未安装)")
    
    def _check_parameter_health(self, runner):
        """检查模型参数健康状态"""
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        
        nan_params = 0
        inf_params = 0
        total_params = 0
        
        for name, param in model.named_parameters():
            total_params += 1
            if torch.isnan(param.data).any():
                nan_params += 1
            if torch.isinf(param.data).any():
                inf_params += 1
        
        if nan_params > 0 or inf_params > 0:
            status = "🔴 异常 (检测到NaN/Inf参数)"
            self.anomaly_counts['nan_inf_params'] += 1
        else:
            status = "🟢 正常"
        
        self.logger.info(f"   参数状态: {status}")
        if nan_params > 0:
            self.logger.info(f"   NaN参数: {nan_params}/{total_params}")
        if inf_params > 0:
            self.logger.info(f"   Inf参数: {inf_params}/{total_params}")
    
    def _log_detailed_performance(self, runner):
        """记录详细性能信息"""
        self.logger.info("📊 详细性能分析:")
        
        # 时间分析
        if self.timing_stats['batch_times']:
            avg_batch = np.mean(self.timing_stats['batch_times'])
            std_batch = np.std(self.timing_stats['batch_times'])
            min_batch = min(self.timing_stats['batch_times'])
            max_batch = max(self.timing_stats['batch_times'])
            
            self.logger.info(f"   批次时间: 平均 {avg_batch:.3f}s ± {std_batch:.3f}s")
            self.logger.info(f"   时间范围: {min_batch:.3f}s - {max_batch:.3f}s")
        
        # 性能指标趋势
        for metric_name, values in self.performance_metrics.items():
            if len(values) >= 10:
                recent_avg = np.mean(list(values)[-10:])
                early_avg = np.mean(list(values)[:10])
                
                if early_avg != 0:
                    change_rate = (recent_avg - early_avg) / abs(early_avg) * 100
                    trend = "📈 上升" if change_rate > 5 else "📉 下降" if change_rate < -5 else "➡️ 稳定"
                    self.logger.info(f"   {metric_name}: {recent_avg:.4f} ({trend}, {change_rate:+.1f}%)")
        
        # 异常统计
        if any(self.anomaly_counts.values()):
            self.logger.info("   异常统计:")
            for anomaly_type, count in self.anomaly_counts.items():
                if count > 0:
                    self.logger.info(f"     {anomaly_type}: {count}次")
    
    def _monitor_model_features(self, runner):
        """监控模型特征"""
        self.logger.info("🔬 模型特征监控:")
        
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        
        # 监控各模块的参数统计
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                self._monitor_module_stats(name, module)
    
    def _monitor_module_stats(self, module_name, module):
        """监控单个模块的统计信息"""
        if hasattr(module, 'weight') and module.weight is not None:
            weight = module.weight.data
            
            # 计算统计量
            mean_val = weight.mean().item()
            std_val = weight.std().item()
            norm_val = weight.norm().item()
            
            # 记录统计量
            self.feature_stats[module_name]['mean'].append(mean_val)
            self.feature_stats[module_name]['std'].append(std_val)
            self.feature_stats[module_name]['norm'].append(norm_val)
            
            # 只显示主要模块的统计
            if any(key in module_name.lower() for key in ['backbone', 'neck', 'head']):
                self.logger.info(f"   {module_name}: 均值 {mean_val:.4f}, "
                                f"标准差 {std_val:.4f}, 范数 {norm_val:.4f}")
    
    def _detect_anomalies(self, runner):
        """异常检测"""
        # 损失异常检测
        if 'loss' in self.performance_metrics and len(self.performance_metrics['loss']) >= 2:
            current_loss = self.performance_metrics['loss'][-1]
            prev_loss = self.performance_metrics['loss'][-2]
            
            # 检测损失突增
            if prev_loss > 0 and current_loss / prev_loss > self.loss_spike_threshold:
                self.logger.warning(f"⚠️  检测到损失突增: {prev_loss:.4f} → {current_loss:.4f}")
                self.health_stats['loss_spikes'].append({
                    'iter': runner.iter,
                    'prev_loss': prev_loss,
                    'current_loss': current_loss,
                    'ratio': current_loss / prev_loss
                })
                self.anomaly_counts['loss_spike'] += 1
            
            # 检测NaN损失
            if np.isnan(current_loss) or np.isinf(current_loss):
                self.logger.error(f"🚨 检测到异常损失值: {current_loss}")
                self.health_stats['nan_detections'].append({
                    'iter': runner.iter,
                    'loss': current_loss,
                    'type': 'nan' if np.isnan(current_loss) else 'inf'
                })
                self.anomaly_counts['nan_loss'] += 1
    
    def after_train_epoch(self, runner):
        """每个epoch结束后保存统计"""
        # 保存性能统计
        self._save_performance_stats(runner.epoch)
        
        # 生成性能图表
        self._save_performance_plots(runner.epoch)
    
    def _save_performance_stats(self, epoch):
        """保存性能统计"""
        stats = {
            'epoch': epoch + 1,
            'performance_metrics': {k: list(v) for k, v in self.performance_metrics.items()},
            'timing_stats': {k: list(v) for k, v in self.timing_stats.items()},
            'health_stats': {
                'gradient_norms': list(self.health_stats['gradient_norms']),
                'memory_usage': list(self.health_stats['memory_usage']),
                'anomaly_counts': dict(self.anomaly_counts),
                'loss_spikes': self.health_stats['loss_spikes'],
                'nan_detections': self.health_stats['nan_detections']
            }
        }
        
        stats_file = os.path.join(self.save_dir, f'performance_health_epoch_{epoch + 1}.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
    
    def _save_performance_plots(self, epoch):
        """保存性能图表"""
        if not HAS_MATPLOTLIB:
            self.logger.warning("matplotlib not available, skipping performance plots")
            return

        # 损失曲线
        if 'loss' in self.performance_metrics:
            plt.figure(figsize=(10, 6))
            losses = list(self.performance_metrics['loss'])
            plt.plot(losses, label='Training Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title(f'Training Loss Curve (Epoch {epoch + 1})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            save_path = os.path.join(self.save_dir, f'loss_curve_epoch_{epoch + 1}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        # 梯度范数曲线
        if self.health_stats['gradient_norms']:
            plt.figure(figsize=(10, 6))
            grad_norms = list(self.health_stats['gradient_norms'])
            plt.plot(grad_norms, label='Gradient Norm')
            plt.axhline(y=self.gradient_clip_threshold, color='r', linestyle='--', 
                       label=f'Clip Threshold ({self.gradient_clip_threshold})')
            plt.xlabel('Iteration')
            plt.ylabel('Gradient Norm')
            plt.title(f'Gradient Norm Evolution (Epoch {epoch + 1})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            save_path = os.path.join(self.save_dir, f'gradient_norm_epoch_{epoch + 1}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    def after_run(self, runner):
        """训练结束后的总结"""
        self.logger.info("🎉 性能和健康度监控完成！")
        
        # 生成最终报告
        self._generate_final_report()
    
    def _generate_final_report(self):
        """生成最终报告"""
        self.logger.info("📋 最终性能和健康度报告:")
        
        # 异常总结
        if any(self.anomaly_counts.values()):
            self.logger.info("   异常事件总结:")
            for anomaly_type, count in self.anomaly_counts.items():
                if count > 0:
                    self.logger.info(f"     {anomaly_type}: {count}次")
        else:
            self.logger.info("   ✅ 未检测到异常事件")
        
        # 性能总结
        if 'loss' in self.performance_metrics:
            final_loss = self.performance_metrics['loss'][-1]
            min_loss = min(self.performance_metrics['loss'])
            self.logger.info(f"   最终损失: {final_loss:.4f}")
            self.logger.info(f"   最小损失: {min_loss:.4f}")
        
        # 时间总结
        if self.timing_stats['batch_times']:
            avg_batch_time = np.mean(self.timing_stats['batch_times'])
            self.logger.info(f"   平均批次时间: {avg_batch_time:.3f}s")
            self.logger.info(f"   平均吞吐量: {1.0/avg_batch_time:.1f} batch/s")
