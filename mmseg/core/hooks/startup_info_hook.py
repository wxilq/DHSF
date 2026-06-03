"""
Startup Info Hook
启动信息钩子 - 在训练开始时显示详细的配置和环境信息

功能包括：
1. 显示SegFormer基线风格的启动日志
2. 环境信息检查和显示
3. 模型配置详细信息
4. 数据集配置信息
5. 训练策略配置信息
6. 创新模块状态检查
"""

import torch
import torch.nn as nn
import numpy as np
import platform
import sys
import os
from mmcv.runner import HOOKS, Hook
from mmcv.utils import get_logger
import mmcv
import mmseg
from collections import OrderedDict


@HOOKS.register_module()
class StartupInfoHook(Hook):
    """启动信息钩子"""
    
    def __init__(self, priority='HIGHEST'):
        self.logger = get_logger('StartupInfo')
    
    def before_run(self, runner):
        """训练开始前显示启动信息"""
        self._print_banner()
        self._print_environment_info()
        self._print_model_info(runner)
        self._print_dataset_info(runner)
        self._print_training_config(runner)
        self._print_innovation_modules(runner)
        self._print_final_summary(runner)
    
    def _print_banner(self):
        """打印横幅"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🚀 Enhanced Video SegFormer Training                      ║
║                                                                              ║
║    基于SegFormer的视频语义分割增强模型 - 集成6大创新模块                      ║
║                                                                              ║
║    1. SDSM时空解耦状态建模                                                    ║
║    2. Enhanced AMSSM增强多尺度状态空间模型                                   ║
║    3. Enhanced CHSM增强跨层次状态建模                                        ║
║    4. Hierarchical Semantic Loss层次化语义损失                               ║
║    5. Progressive Training Strategy渐进式训练策略                            ║
║    6. Adaptive Patch Embedding自适应Patch Embedding                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
        self.logger.info("🎉 Enhanced Video SegFormer 训练启动")
    
    def _print_environment_info(self):
        """打印环境信息"""
        self.logger.info("=" * 80)
        self.logger.info("🌍 环境信息检查")
        self.logger.info("=" * 80)
        
        # Python环境
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self.logger.info(f"Python版本: {python_version}")
        
        # 操作系统
        os_info = f"{platform.system()} {platform.release()}"
        self.logger.info(f"操作系统: {os_info}")
        
        # PyTorch信息
        self.logger.info(f"PyTorch版本: {torch.__version__}")
        self.logger.info(f"CUDA版本: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
        self.logger.info(f"cuDNN版本: {torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A'}")
        
        # MMSegmentation信息
        try:
            self.logger.info(f"MMSegmentation版本: {mmseg.__version__}")
        except:
            self.logger.info("MMSegmentation版本: 未知")
        
        try:
            self.logger.info(f"MMCV版本: {mmcv.__version__}")
        except:
            self.logger.info("MMCV版本: 未知")
        
        # GPU信息
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            self.logger.info(f"GPU数量: {gpu_count}")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                self.logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            self.logger.info("GPU: 未检测到CUDA设备")
        
        # 内存信息
        try:
            import psutil
            memory = psutil.virtual_memory()
            self.logger.info(f"系统内存: {memory.total / 1024**3:.1f}GB 总计, {memory.available / 1024**3:.1f}GB 可用")
        except ImportError:
            self.logger.info("系统内存: 无法获取信息 (需要psutil)")
    
    def _print_model_info(self, runner):
        """打印模型信息"""
        self.logger.info("=" * 80)
        self.logger.info("🏗️  模型架构信息")
        self.logger.info("=" * 80)
        
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        
        # 模型类型
        model_type = type(model).__name__
        self.logger.info(f"模型类型: {model_type}")
        
        # 参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        self.logger.info(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        self.logger.info(f"可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        if frozen_params > 0:
            self.logger.info(f"冻结参数: {frozen_params:,} ({frozen_params/1e6:.2f}M)")
        
        # 各组件参数统计
        component_params = OrderedDict()
        
        if hasattr(model, 'backbone'):
            backbone_params = sum(p.numel() for p in model.backbone.parameters())
            component_params['Backbone'] = backbone_params
            
            # Backbone详细信息
            backbone_type = type(model.backbone).__name__
            self.logger.info(f"Backbone: {backbone_type} ({backbone_params/1e6:.2f}M 参数)")
            
            # 检查自适应Patch Embedding
            if hasattr(model.backbone, 'enable_adaptive_patch'):
                if model.backbone.enable_adaptive_patch:
                    patch_type = getattr(model.backbone, 'adaptive_patch_type', 'unknown')
                    self.logger.info(f"  ✅ 自适应Patch Embedding: {patch_type}")
                else:
                    self.logger.info("  ❌ 自适应Patch Embedding: 已禁用")
        
        if hasattr(model, 'neck') and model.neck is not None:
            neck_params = sum(p.numel() for p in model.neck.parameters())
            component_params['Neck'] = neck_params
            
            neck_type = type(model.neck).__name__
            self.logger.info(f"Neck: {neck_type} ({neck_params/1e6:.2f}M 参数)")
        
        if hasattr(model, 'decode_head'):
            head_params = sum(p.numel() for p in model.decode_head.parameters())
            component_params['Decode Head'] = head_params
            
            head_type = type(model.decode_head).__name__
            self.logger.info(f"Decode Head: {head_type} ({head_params/1e6:.2f}M 参数)")
            
            # 类别数量
            if hasattr(model.decode_head, 'num_classes'):
                num_classes = model.decode_head.num_classes
                self.logger.info(f"  输出类别数: {num_classes}")
        
        # 参数分布
        self.logger.info("参数分布:")
        for component, params in component_params.items():
            percentage = params / total_params * 100
            self.logger.info(f"  {component:15s}: {params/1e6:6.2f}M ({percentage:5.1f}%)")
    
    def _print_dataset_info(self, runner):
        """打印数据集信息"""
        self.logger.info("=" * 80)
        self.logger.info("📊 数据集配置信息")
        self.logger.info("=" * 80)
        
        # 训练数据集
        if hasattr(runner, 'data_loader') and runner.data_loader is not None:
            dataset = runner.data_loader.dataset
            dataset_type = type(dataset).__name__
            
            self.logger.info(f"数据集类型: {dataset_type}")
            self.logger.info(f"训练样本数: {len(dataset):,}")
            self.logger.info(f"批次大小: {runner.data_loader.batch_size}")
            self.logger.info(f"工作进程数: {runner.data_loader.num_workers}")
            
            # 时序信息
            if hasattr(dataset, 'temporal_length'):
                self.logger.info(f"时序长度: {dataset.temporal_length}")
            if hasattr(dataset, 'temporal_stride'):
                self.logger.info(f"时序步长: {dataset.temporal_stride}")
            
            # 数据路径
            if hasattr(dataset, 'data_root'):
                self.logger.info(f"数据根目录: {dataset.data_root}")
            if hasattr(dataset, 'img_dir'):
                self.logger.info(f"图像目录: {dataset.img_dir}")
            if hasattr(dataset, 'ann_dir'):
                self.logger.info(f"标注目录: {dataset.ann_dir}")
        
        # 验证数据集
        if hasattr(runner, 'val_dataloader') and runner.val_dataloader is not None:
            val_dataset = runner.val_dataloader.dataset
            self.logger.info(f"验证样本数: {len(val_dataset):,}")
        
        # 类别信息
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        
        if hasattr(model, 'decode_head') and hasattr(model.decode_head, 'num_classes'):
            num_classes = model.decode_head.num_classes
            self.logger.info(f"分割类别数: {num_classes}")
            self.logger.info("⚠️  注意: 只训练占比>0.5%的25个类别，其他类别已设为ignore")
    
    def _print_training_config(self, runner):
        """打印训练配置"""
        self.logger.info("=" * 80)
        self.logger.info("⚙️  训练配置信息")
        self.logger.info("=" * 80)
        
        # 训练器信息
        runner_type = type(runner).__name__
        self.logger.info(f"训练器类型: {runner_type}")
        
        if hasattr(runner, 'max_epochs'):
            self.logger.info(f"最大训练轮数: {runner.max_epochs}")
        elif hasattr(runner, 'max_iters'):
            self.logger.info(f"最大训练迭代数: {runner.max_iters:,}")
        
        # 优化器信息
        if hasattr(runner, 'optimizer'):
            optimizer = runner.optimizer
            optimizer_type = type(optimizer).__name__
            self.logger.info(f"优化器: {optimizer_type}")
            
            # 学习率信息
            if hasattr(optimizer, 'param_groups'):
                self.logger.info("学习率配置:")
                for i, group in enumerate(optimizer.param_groups):
                    lr = group.get('lr', 'N/A')
                    weight_decay = group.get('weight_decay', 'N/A')
                    self.logger.info(f"  参数组 {i}: lr={lr:.2e}, weight_decay={weight_decay}")
        
        # 学习率调度
        if hasattr(runner, 'lr_config'):
            lr_config = runner.lr_config
            policy = lr_config.get('policy', 'N/A')
            self.logger.info(f"学习率策略: {policy}")
            
            if 'warmup' in lr_config:
                warmup = lr_config['warmup']
                warmup_iters = lr_config.get('warmup_iters', 'N/A')
                self.logger.info(f"预热策略: {warmup}, 预热迭代数: {warmup_iters}")
        
        # 评估配置
        if hasattr(runner, 'eval_config'):
            eval_config = runner.eval_config
            interval = eval_config.get('interval', 'N/A')
            metric = eval_config.get('metric', 'N/A')
            self.logger.info(f"评估间隔: {interval}, 评估指标: {metric}")
    
    def _print_innovation_modules(self, runner):
        """打印创新模块信息"""
        self.logger.info("=" * 80)
        self.logger.info("🔬 创新模块状态检查")
        self.logger.info("=" * 80)
        
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        
        innovation_status = []
        
        # 1. SDSM时空解耦状态建模
        sdsm_status = self._check_sdsm_status(model)
        innovation_status.append(("SDSM时空解耦状态建模", sdsm_status))
        
        # 2. Enhanced AMSSM (已禁用)
        amssm_status = {'enabled': False, 'description': '已禁用（节省显存）', 'details': []}
        innovation_status.append(("Enhanced AMSSM增强多尺度状态空间", amssm_status))
        
        # 3. Enhanced CHSM
        chsm_status = self._check_enhanced_chsm_status(model)
        innovation_status.append(("Enhanced CHSM增强跨层次状态建模", chsm_status))
        
        # 4. Hierarchical Semantic Loss
        hierarchical_loss_status = self._check_hierarchical_loss_status(model)
        innovation_status.append(("Hierarchical Semantic Loss层次化语义损失", hierarchical_loss_status))
        
        # 5. Progressive Training Strategy
        progressive_training_status = self._check_progressive_training_status(runner)
        innovation_status.append(("Progressive Training Strategy渐进式训练", progressive_training_status))
        
        # 6. Adaptive Patch Embedding
        adaptive_patch_status = self._check_adaptive_patch_status(model)
        innovation_status.append(("Adaptive Patch Embedding自适应Patch嵌入", adaptive_patch_status))
        
        # 显示状态
        for module_name, status in innovation_status:
            status_icon = "✅" if status['enabled'] else "❌"
            self.logger.info(f"{status_icon} {module_name}: {status['description']}")
            if status['details']:
                for detail in status['details']:
                    self.logger.info(f"    {detail}")
    
    def _check_sdsm_status(self, model):
        """检查SDSM状态"""
        # 检查neck中的SDSM配置
        if hasattr(model, 'neck') and hasattr(model.neck, 'enable_sdsm'):
            if model.neck.enable_sdsm:
                return {
                    'enabled': True,
                    'description': '已启用',
                    'details': ['时空解耦建模已激活', '支持独立的空间和时序状态建模']
                }

        return {
            'enabled': False,
            'description': '未检测到或已禁用',
            'details': []
        }
    
    def _check_enhanced_amssm_status(self, model):
        """检查Enhanced AMSSM状态 - 已禁用"""
        # AMSSM已被禁用以节省显存
        return {
            'enabled': False,
            'description': '已禁用（节省显存）',
            'details': ['AMSSM模块已移除', '使用轻量级时序处理替代']
        }
    
    def _check_enhanced_chsm_status(self, model):
        """检查Enhanced CHSM状态"""
        # 检查neck中的CHSM配置
        if hasattr(model, 'neck') and hasattr(model.neck, 'cross_hierarchical_fusion'):
            return {
                'enabled': True,
                'description': '已启用',
                'details': ['跨层次状态传递已激活', '支持像素→物体→房间→场景的层次建模']
            }

        return {
            'enabled': False,
            'description': '未检测到',
            'details': []
        }
    
    def _check_hierarchical_loss_status(self, model):
        """检查层次化语义损失状态"""
        if hasattr(model, 'decode_head'):
            head_type = type(model.decode_head).__name__
            if 'Hierarchical' in head_type:
                return {
                    'enabled': True,
                    'description': f'已启用 ({head_type})',
                    'details': ['多层次损失函数已激活', '支持像素、物体、房间、场景级损失']
                }
        
        return {
            'enabled': False,
            'description': '使用标准损失函数',
            'details': []
        }
    
    def _check_progressive_training_status(self, runner):
        """检查渐进式训练状态"""
        # 检查是否有渐进式训练钩子
        if hasattr(runner, 'hooks'):
            for hook in runner.hooks:
                if 'Progressive' in type(hook).__name__:
                    return {
                        'enabled': True,
                        'description': '已启用',
                        'details': ['3阶段渐进式训练已配置', 'Stage1: 空间学习 → Stage2: 时序引入 → Stage3: 联合优化']
                    }
        
        return {
            'enabled': False,
            'description': '未启用',
            'details': []
        }
    
    def _check_adaptive_patch_status(self, model):
        """检查自适应Patch Embedding状态"""
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'enable_adaptive_patch'):
            if model.backbone.enable_adaptive_patch:
                patch_type = getattr(model.backbone, 'adaptive_patch_type', 'unknown')
                return {
                    'enabled': True,
                    'description': f'已启用 ({patch_type})',
                    'details': ['自适应patch大小选择已激活', '支持动态特征提取优化']
                }
        
        return {
            'enabled': False,
            'description': '使用标准Patch Embedding',
            'details': []
        }
    
    def _print_final_summary(self, runner):
        """打印最终总结"""
        self.logger.info("=" * 80)
        self.logger.info("🎯 训练准备就绪")
        self.logger.info("=" * 80)
        
        # 计算预计训练时间
        if hasattr(runner, 'data_loader'):
            total_samples = len(runner.data_loader.dataset)
            batch_size = runner.data_loader.batch_size
            iters_per_epoch = total_samples // batch_size
            
            if hasattr(runner, 'max_epochs'):
                total_iters = iters_per_epoch * runner.max_epochs
                self.logger.info(f"预计总迭代数: {total_iters:,}")
                self.logger.info(f"每轮迭代数: {iters_per_epoch:,}")
        
        self.logger.info("🚀 开始训练...")
        self.logger.info("=" * 80)
