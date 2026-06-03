"""
Feature Monitoring Hook
特征监控钩子 - 专门监控模型各组件的特征统计

功能包括：
1. Backbone特征统计（均值、方差、范数等）
2. Mamba状态监控（隐状态统计、注意力权重熵值）
3. 自适应Patch监控（patch大小分布、自适应权重统计）
4. 特征分布可视化
5. 异常特征检测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, deque
from mmcv.runner import HOOKS, Hook
from mmcv.utils import get_logger
import os
import json

# 兼容性导入 - 处理可视化库
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found, visualization features will be disabled")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not found, some advanced plots will be disabled")


@HOOKS.register_module()
class FeatureMonitoringHook(Hook):
    """特征监控钩子"""
    
    def __init__(self,
                 log_interval=500,
                 detailed_log_interval=1000,
                 visualization_interval=2000,
                 save_dir='./feature_monitoring',
                 enable_backbone_monitoring=True,
                 enable_mamba_monitoring=True,
                 enable_patch_monitoring=True,
                 enable_visualization=True,
                 max_samples_per_batch=4,
                 priority='LOW'):
        
        self.log_interval = log_interval
        self.detailed_log_interval = detailed_log_interval
        self.visualization_interval = visualization_interval
        self.save_dir = save_dir
        
        # 监控开关
        self.enable_backbone_monitoring = enable_backbone_monitoring
        self.enable_mamba_monitoring = enable_mamba_monitoring
        self.enable_patch_monitoring = enable_patch_monitoring
        self.enable_visualization = enable_visualization
        
        self.max_samples_per_batch = max_samples_per_batch
        
        # 特征统计存储
        self.backbone_stats = defaultdict(lambda: defaultdict(deque))
        self.mamba_stats = defaultdict(lambda: defaultdict(deque))
        self.patch_stats = defaultdict(deque)
        
        # 特征钩子存储
        self.feature_hooks = []
        self.feature_cache = {}
        
        self.logger = get_logger('FeatureMonitoring')
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
    
    def before_run(self, runner):
        """训练开始前注册特征钩子"""
        self.logger.info("🔬 特征监控钩子已启动")
        self.logger.info(f"   Backbone监控: {'✅' if self.enable_backbone_monitoring else '❌'}")
        self.logger.info(f"   Mamba监控: {'✅' if self.enable_mamba_monitoring else '❌'}")
        self.logger.info(f"   Patch监控: {'✅' if self.enable_patch_monitoring else '❌'}")
        
        # 注册特征钩子
        self._register_feature_hooks(runner)
    
    def _register_feature_hooks(self, runner):
        """注册特征提取钩子"""
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        
        # 注册Backbone特征钩子
        if self.enable_backbone_monitoring and hasattr(model, 'backbone'):
            self._register_backbone_hooks(model.backbone)
        
        # 注册Mamba特征钩子
        if self.enable_mamba_monitoring and hasattr(model, 'neck'):
            self._register_mamba_hooks(model.neck)
    
    def _register_backbone_hooks(self, backbone):
        """注册Backbone特征钩子"""
        def create_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.feature_cache[f'backbone_{name}'] = output.detach()
                elif isinstance(output, (list, tuple)):
                    for i, feat in enumerate(output):
                        if isinstance(feat, torch.Tensor):
                            self.feature_cache[f'backbone_{name}_stage{i}'] = feat.detach()
            return hook_fn
        
        # 注册主要stage的钩子
        for name, module in backbone.named_modules():
            if any(stage in name for stage in ['block1', 'block2', 'block3', 'block4']):
                if len(name.split('.')) == 1:  # 只注册顶层block
                    hook = module.register_forward_hook(create_hook(name))
                    self.feature_hooks.append(hook)
    
    def _register_mamba_hooks(self, neck):
        """注册Mamba特征钩子"""
        def create_mamba_hook(name):
            def hook_fn(module, input, output):
                # 存储Mamba模块的输出
                if isinstance(output, torch.Tensor):
                    self.feature_cache[f'mamba_{name}'] = output.detach()
                elif isinstance(output, (list, tuple)):
                    for i, feat in enumerate(output):
                        if isinstance(feat, torch.Tensor):
                            self.feature_cache[f'mamba_{name}_level{i}'] = feat.detach()
                
                # 如果模块有状态信息，也记录下来
                if hasattr(module, 'last_hidden_state'):
                    self.feature_cache[f'mamba_{name}_hidden'] = module.last_hidden_state.detach()
            return hook_fn
        
        # 注册Mamba相关模块的钩子（排除AMSSM）
        for name, module in neck.named_modules():
            if any(keyword in name.lower() for keyword in ['mamba', 'ssm', 'sdsm']) and 'amssm' not in name.lower():
                hook = module.register_forward_hook(create_mamba_hook(name))
                self.feature_hooks.append(hook)
    
    def after_train_iter(self, runner):
        """每个iteration后分析特征"""
        # 基础特征监控
        if self.every_n_iters(runner, self.log_interval):
            self._analyze_features(runner)
        
        # 详细特征分析
        if self.every_n_iters(runner, self.detailed_log_interval):
            self._detailed_feature_analysis(runner)
        
        # 特征可视化
        if self.enable_visualization and self.every_n_iters(runner, self.visualization_interval):
            self._visualize_features(runner)
        
        # 清空特征缓存，避免内存积累
        self.feature_cache.clear()
    
    def _analyze_features(self, runner):
        """分析特征统计"""
        if not self.feature_cache:
            return
        
        self.logger.info("🔍 特征统计分析:")
        
        # 分析Backbone特征
        if self.enable_backbone_monitoring:
            self._analyze_backbone_features()
        
        # 分析Mamba特征
        if self.enable_mamba_monitoring:
            self._analyze_mamba_features()
        
        # 分析自适应Patch特征
        if self.enable_patch_monitoring:
            self._analyze_patch_features(runner)
    
    def _analyze_backbone_features(self):
        """分析Backbone特征"""
        backbone_features = {k: v for k, v in self.feature_cache.items() if k.startswith('backbone_')}
        
        if not backbone_features:
            return
        
        self.logger.info("   📊 Backbone特征统计:")
        
        for feat_name, feat_tensor in backbone_features.items():
            if feat_tensor.numel() == 0:
                continue
            
            # 计算基础统计量
            mean_val = feat_tensor.mean().item()
            std_val = feat_tensor.std().item()
            min_val = feat_tensor.min().item()
            max_val = feat_tensor.max().item()
            norm_val = feat_tensor.norm().item()
            
            # 存储历史统计
            stage_name = feat_name.replace('backbone_', '')
            self.backbone_stats[stage_name]['mean'].append(mean_val)
            self.backbone_stats[stage_name]['std'].append(std_val)
            self.backbone_stats[stage_name]['norm'].append(norm_val)
            
            # 检测异常值
            anomaly_flags = []
            if abs(mean_val) > 10:
                anomaly_flags.append("均值过大")
            if std_val > 10:
                anomaly_flags.append("方差过大")
            if norm_val > 100:
                anomaly_flags.append("范数过大")
            
            anomaly_str = f" ⚠️ {', '.join(anomaly_flags)}" if anomaly_flags else ""
            
            self.logger.info(f"     {stage_name:15s}: 均值 {mean_val:7.4f}, 标准差 {std_val:7.4f}, "
                           f"范围 [{min_val:7.4f}, {max_val:7.4f}], 范数 {norm_val:7.4f}{anomaly_str}")
    
    def _analyze_mamba_features(self):
        """分析Mamba特征"""
        mamba_features = {k: v for k, v in self.feature_cache.items() if k.startswith('mamba_')}
        
        if not mamba_features:
            return
        
        self.logger.info("   🌊 Mamba状态统计:")
        
        for feat_name, feat_tensor in mamba_features.items():
            if feat_tensor.numel() == 0:
                continue
            
            # 计算Mamba特有的统计量
            mean_val = feat_tensor.mean().item()
            std_val = feat_tensor.std().item()
            
            # 计算激活稀疏性
            activation_ratio = (feat_tensor > 0).float().mean().item()
            
            # 计算注意力熵值（如果是注意力权重）
            entropy_val = 0
            if feat_tensor.dim() >= 3:  # 可能是注意力权重
                # 对最后一个维度计算softmax和熵值
                softmax_weights = F.softmax(feat_tensor.view(-1, feat_tensor.size(-1)), dim=-1)
                entropy_val = -(softmax_weights * torch.log(softmax_weights + 1e-8)).sum(dim=-1).mean().item()
            
            # 存储历史统计
            module_name = feat_name.replace('mamba_', '')
            self.mamba_stats[module_name]['mean'].append(mean_val)
            self.mamba_stats[module_name]['std'].append(std_val)
            self.mamba_stats[module_name]['activation_ratio'].append(activation_ratio)
            if entropy_val > 0:
                self.mamba_stats[module_name]['entropy'].append(entropy_val)
            
            # 状态健康检查
            health_flags = []
            if activation_ratio < 0.1:
                health_flags.append("激活稀疏")
            elif activation_ratio > 0.9:
                health_flags.append("激活饱和")
            
            if entropy_val > 0:
                if entropy_val < 0.5:
                    health_flags.append("注意力集中")
                elif entropy_val > 3.0:
                    health_flags.append("注意力分散")
            
            health_str = f" 🔍 {', '.join(health_flags)}" if health_flags else ""
            
            info_str = (f"     {module_name:15s}: 均值 {mean_val:7.4f}, 标准差 {std_val:7.4f}, "
                       f"激活率 {activation_ratio:5.3f}")
            
            if entropy_val > 0:
                info_str += f", 熵值 {entropy_val:5.3f}"
            
            info_str += health_str
            self.logger.info(info_str)
    
    def _analyze_patch_features(self, runner):
        """分析自适应Patch特征"""
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        
        if not (hasattr(model, 'backbone') and hasattr(model.backbone, 'enable_adaptive_patch')):
            return
        
        if not model.backbone.enable_adaptive_patch:
            return
        
        self.logger.info("   🎯 自适应Patch统计:")
        
        # 尝试获取patch大小信息
        if hasattr(model.backbone, 'patch_embed') and hasattr(model.backbone.patch_embed, 'last_patch_sizes'):
            patch_sizes = model.backbone.patch_embed.last_patch_sizes
            if patch_sizes is not None:
                # 统计patch大小分布
                unique_sizes, counts = torch.unique(patch_sizes, return_counts=True)
                
                self.logger.info("     Patch大小分布:")
                for size, count in zip(unique_sizes, counts):
                    ratio = count.item() / patch_sizes.numel() * 100
                    self.logger.info(f"       大小 {size.item()}: {count.item()}个 ({ratio:.1f}%)")
                
                # 存储统计
                avg_patch_size = patch_sizes.float().mean().item()
                std_patch_size = patch_sizes.float().std().item()
                
                self.patch_stats['avg_size'].append(avg_patch_size)
                self.patch_stats['std_size'].append(std_patch_size)
                
                self.logger.info(f"     平均Patch大小: {avg_patch_size:.2f} ± {std_patch_size:.2f}")
        
        # 分析自适应权重（如果有的话）
        if hasattr(model.backbone, 'patch_embed') and hasattr(model.backbone.patch_embed, 'last_adaptive_weights'):
            adaptive_weights = model.backbone.patch_embed.last_adaptive_weights
            if adaptive_weights is not None:
                weight_mean = adaptive_weights.mean().item()
                weight_std = adaptive_weights.std().item()
                weight_entropy = -(adaptive_weights * torch.log(adaptive_weights + 1e-8)).sum(dim=-1).mean().item()
                
                self.patch_stats['weight_mean'].append(weight_mean)
                self.patch_stats['weight_std'].append(weight_std)
                self.patch_stats['weight_entropy'].append(weight_entropy)
                
                self.logger.info(f"     自适应权重: 均值 {weight_mean:.4f}, 标准差 {weight_std:.4f}, 熵值 {weight_entropy:.4f}")
    
    def _detailed_feature_analysis(self, runner):
        """详细特征分析"""
        self.logger.info("📈 详细特征趋势分析:")
        
        # Backbone特征趋势
        if self.backbone_stats:
            self.logger.info("   Backbone特征趋势:")
            for stage_name, stats in self.backbone_stats.items():
                if len(stats['mean']) >= 10:
                    recent_mean = np.mean(list(stats['mean'])[-5:])
                    early_mean = np.mean(list(stats['mean'])[:5])
                    trend = "📈 上升" if recent_mean > early_mean else "📉 下降"
                    change_rate = (recent_mean - early_mean) / abs(early_mean) * 100 if early_mean != 0 else 0
                    
                    self.logger.info(f"     {stage_name}: 均值趋势 {trend} ({change_rate:+.1f}%)")
        
        # Mamba状态趋势
        if self.mamba_stats:
            self.logger.info("   Mamba状态趋势:")
            for module_name, stats in self.mamba_stats.items():
                if len(stats['activation_ratio']) >= 10:
                    recent_activation = np.mean(list(stats['activation_ratio'])[-5:])
                    early_activation = np.mean(list(stats['activation_ratio'])[:5])
                    
                    if abs(recent_activation - early_activation) > 0.1:
                        trend = "📈 增加" if recent_activation > early_activation else "📉 减少"
                        self.logger.info(f"     {module_name}: 激活率 {trend} "
                                       f"({early_activation:.3f} → {recent_activation:.3f})")
        
        # Patch特征趋势
        if self.patch_stats and len(self.patch_stats['avg_size']) >= 10:
            recent_size = np.mean(list(self.patch_stats['avg_size'])[-5:])
            early_size = np.mean(list(self.patch_stats['avg_size'])[:5])
            
            if abs(recent_size - early_size) > 0.1:
                trend = "📈 增大" if recent_size > early_size else "📉 减小"
                self.logger.info(f"   自适应Patch: 平均大小 {trend} "
                               f"({early_size:.2f} → {recent_size:.2f})")
    
    def _visualize_features(self, runner):
        """可视化特征分布"""
        if not self.enable_visualization or not HAS_MATPLOTLIB:
            if not HAS_MATPLOTLIB:
                self.logger.warning("matplotlib not available, skipping feature visualization")
            return

        self.logger.info("📊 生成特征可视化图表...")
        
        # 可视化Backbone特征分布
        self._plot_backbone_features(runner.epoch)
        
        # 可视化Mamba状态分布
        self._plot_mamba_features(runner.epoch)
        
        # 可视化Patch分布
        self._plot_patch_features(runner.epoch)
    
    def _plot_backbone_features(self, epoch):
        """绘制Backbone特征图"""
        if not self.backbone_stats:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Backbone Features Analysis (Epoch {epoch + 1})', fontsize=16)
        
        # 均值趋势
        ax = axes[0, 0]
        for stage_name, stats in self.backbone_stats.items():
            if stats['mean']:
                ax.plot(list(stats['mean']), label=stage_name, alpha=0.7)
        ax.set_title('Feature Mean Trends')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 标准差趋势
        ax = axes[0, 1]
        for stage_name, stats in self.backbone_stats.items():
            if stats['std']:
                ax.plot(list(stats['std']), label=stage_name, alpha=0.7)
        ax.set_title('Feature Std Trends')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Std Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 范数趋势
        ax = axes[1, 0]
        for stage_name, stats in self.backbone_stats.items():
            if stats['norm']:
                ax.plot(list(stats['norm']), label=stage_name, alpha=0.7)
        ax.set_title('Feature Norm Trends')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Norm Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 特征分布热力图
        ax = axes[1, 1]
        if self.feature_cache:
            # 选择一个backbone特征进行分布可视化
            backbone_features = {k: v for k, v in self.feature_cache.items() if k.startswith('backbone_')}
            if backbone_features:
                feat_name, feat_tensor = next(iter(backbone_features.items()))
                if feat_tensor.dim() >= 4:  # [B, C, H, W]
                    # 取第一个batch的第一个channel
                    feat_2d = feat_tensor[0, 0].cpu().numpy()
                    im = ax.imshow(feat_2d, cmap='viridis', aspect='auto')
                    ax.set_title(f'Feature Map: {feat_name}')
                    plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'backbone_features_epoch_{epoch + 1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_mamba_features(self, epoch):
        """绘制Mamba特征图"""
        if not self.mamba_stats:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Mamba States Analysis (Epoch {epoch + 1})', fontsize=16)
        
        # 激活率趋势
        ax = axes[0, 0]
        for module_name, stats in self.mamba_stats.items():
            if stats['activation_ratio']:
                ax.plot(list(stats['activation_ratio']), label=module_name, alpha=0.7)
        ax.set_title('Activation Ratio Trends')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Activation Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 熵值趋势
        ax = axes[0, 1]
        for module_name, stats in self.mamba_stats.items():
            if stats['entropy']:
                ax.plot(list(stats['entropy']), label=module_name, alpha=0.7)
        ax.set_title('Attention Entropy Trends')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Entropy Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 均值和标准差
        ax = axes[1, 0]
        for module_name, stats in self.mamba_stats.items():
            if stats['mean']:
                ax.plot(list(stats['mean']), label=f'{module_name}_mean', alpha=0.7)
        ax.set_title('Mamba Feature Mean')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        for module_name, stats in self.mamba_stats.items():
            if stats['std']:
                ax.plot(list(stats['std']), label=f'{module_name}_std', alpha=0.7)
        ax.set_title('Mamba Feature Std')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Std Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'mamba_features_epoch_{epoch + 1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_patch_features(self, epoch):
        """绘制Patch特征图"""
        if not self.patch_stats:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Adaptive Patch Analysis (Epoch {epoch + 1})', fontsize=16)
        
        # Patch大小趋势
        if self.patch_stats['avg_size']:
            ax = axes[0]
            ax.plot(list(self.patch_stats['avg_size']), label='Average Size', color='blue')
            if self.patch_stats['std_size']:
                sizes = np.array(list(self.patch_stats['avg_size']))
                stds = np.array(list(self.patch_stats['std_size']))
                ax.fill_between(range(len(sizes)), sizes - stds, sizes + stds, alpha=0.3, color='blue')
            ax.set_title('Patch Size Evolution')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Patch Size')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 自适应权重熵值
        if self.patch_stats['weight_entropy']:
            ax = axes[1]
            ax.plot(list(self.patch_stats['weight_entropy']), label='Weight Entropy', color='red')
            ax.set_title('Adaptive Weight Entropy')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Entropy Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'patch_features_epoch_{epoch + 1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def after_train_epoch(self, runner):
        """每个epoch结束后保存统计"""
        self._save_feature_stats(runner.epoch)
    
    def _save_feature_stats(self, epoch):
        """保存特征统计"""
        stats = {
            'epoch': epoch + 1,
            'backbone_stats': {k: {kk: list(vv) for kk, vv in v.items()} 
                             for k, v in self.backbone_stats.items()},
            'mamba_stats': {k: {kk: list(vv) for kk, vv in v.items()} 
                          for k, v in self.mamba_stats.items()},
            'patch_stats': {k: list(v) for k, v in self.patch_stats.items()}
        }
        
        stats_file = os.path.join(self.save_dir, f'feature_stats_epoch_{epoch + 1}.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
    
    def after_run(self, runner):
        """训练结束后清理钩子"""
        self.logger.info("🎉 特征监控完成，清理钩子...")
        
        # 移除所有注册的钩子
        for hook in self.feature_hooks:
            hook.remove()
        
        self.feature_hooks.clear()
        self.feature_cache.clear()
