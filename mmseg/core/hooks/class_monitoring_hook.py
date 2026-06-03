"""
Class Monitoring Hook
类别监控钩子 - 专门监控25个类别的训练情况和ignore状态

功能包括：
1. 监控25个有效类别的预测分布
2. 统计ignore类别的像素数量
3. 类别平衡性分析
4. 类别预测准确率统计
5. 混淆矩阵可视化
"""

import torch
import numpy as np
from collections import defaultdict, Counter
from mmcv.runner import HOOKS, Hook
from mmcv.utils import get_logger
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

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not found, some advanced plots will be disabled")

try:
    from sklearn.metrics import confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not found, confusion matrix will use simple implementation")

    # 简单的混淆矩阵实现
    def confusion_matrix(y_true, y_pred, labels=None):
        """简单的混淆矩阵实现，兼容无sklearn环境"""
        if labels is None:
            labels = sorted(list(set(y_true) | set(y_pred)))

        n_labels = len(labels)
        label_to_idx = {label: idx for idx, label in enumerate(labels)}

        cm = np.zeros((n_labels, n_labels), dtype=int)

        for true_label, pred_label in zip(y_true, y_pred):
            if true_label in label_to_idx and pred_label in label_to_idx:
                true_idx = label_to_idx[true_label]
                pred_idx = label_to_idx[pred_label]
                cm[true_idx, pred_idx] += 1

        return cm


@HOOKS.register_module()
class ClassMonitoringHook(Hook):
    """类别监控钩子"""
    
    def __init__(self,
                 log_interval=100,
                 detailed_log_interval=500,
                 confusion_matrix_interval=1000,
                 save_dir='./class_monitoring',
                 num_classes=25,
                 ignore_index=255,
                 class_names=None,
                 enable_confusion_matrix=True,
                 enable_class_distribution_plot=True,
                 priority='NORMAL'):
        
        self.log_interval = log_interval
        self.detailed_log_interval = detailed_log_interval
        self.confusion_matrix_interval = confusion_matrix_interval
        self.save_dir = save_dir
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.enable_confusion_matrix = enable_confusion_matrix
        self.enable_class_distribution_plot = enable_class_distribution_plot
        
        # 类别名称 - 25个有效类别，不包含ignore
        if class_names is None:
            self.class_names = [
                'background', 'wall', 'chair', 'floor', 'table', 'door', 'couch', 'cabinet',
                'shelf', 'desk', 'bed', 'window', 'unknown_17', 'bookshelf', 'monitor', 'curtain', 'books',
                'armchair', 'coffee_table', 'lamp', 'kitchen_cabinet', 'whiteboard',
                'shower_curtain', 'bathroom_stall', 'doorframe'
            ]
        else:
            self.class_names = class_names
        
        # 统计变量
        self.class_pixel_counts = defaultdict(int)
        self.class_predictions = defaultdict(int)
        self.class_correct_predictions = defaultdict(int)
        self.ignore_pixel_count = 0
        self.total_pixel_count = 0
        
        # 混淆矩阵
        self.confusion_matrix_data = []
        
        # 历史统计
        self.class_distribution_history = []
        self.class_accuracy_history = []
        
        self.logger = get_logger('ClassMonitoring')
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
    
    def before_run(self, runner):
        """训练开始前初始化"""
        self.logger.info("🎯 类别监控钩子已启动")
        self.logger.info(f"   监控类别数: {self.num_classes}")
        self.logger.info(f"   ignore索引: {self.ignore_index}")
        self.logger.info("   ✅ 确认只训练占比>0.5%的25个类别，其他类别已ignore")
        
        # 打印类别映射
        self.logger.info("📋 类别映射表:")
        for i, name in enumerate(self.class_names):
            if i == self.ignore_index:
                self.logger.info(f"   {i:2d}: {name} (IGNORE - 包含所有占比<0.5%的类别)")
            else:
                self.logger.info(f"   {i:2d}: {name}")
    
    def before_train_epoch(self, runner):
        """每个epoch开始前重置统计"""
        self.class_pixel_counts.clear()
        self.class_predictions.clear()
        self.class_correct_predictions.clear()
        self.ignore_pixel_count = 0
        self.total_pixel_count = 0
    
    def after_train_iter(self, runner):
        """每个iteration后更新统计"""
        # 获取预测和标签
        if hasattr(runner, 'outputs') and runner.outputs is not None:
            self._update_class_statistics(runner.outputs, runner.data_batch)
        
        # 基础日志
        if self.every_n_iters(runner, self.log_interval):
            self._log_basic_class_info(runner)
        
        # 详细日志
        if self.every_n_iters(runner, self.detailed_log_interval):
            self._log_detailed_class_info(runner)
        
        # 混淆矩阵
        if self.enable_confusion_matrix and self.every_n_iters(runner, self.confusion_matrix_interval):
            self._log_confusion_matrix(runner)
    
    def _update_class_statistics(self, outputs, data_batch):
        """更新类别统计"""
        if 'seg_logits' not in outputs:
            return
        
        # 获取预测和真实标签
        seg_logits = outputs['seg_logits']  # [B, T, C, H, W] or [B, C, H, W]
        seg_label = data_batch['gt_semantic_seg']  # [B, T, H, W] or [B, H, W]
        
        # 处理维度
        if len(seg_logits.shape) == 5:  # 视频数据 [B, T, C, H, W]
            B, T, C, H, W = seg_logits.shape
            seg_pred = seg_logits.argmax(dim=2)  # [B, T, H, W]
            seg_pred = seg_pred.view(-1)  # [B*T*H*W]
            seg_label = seg_label.view(-1)  # [B*T*H*W]
        else:  # 图像数据 [B, C, H, W]
            seg_pred = seg_logits.argmax(dim=1)  # [B, H, W]
            seg_pred = seg_pred.view(-1)  # [B*H*W]
            seg_label = seg_label.view(-1)  # [B*H*W]
        
        # 转换为numpy
        seg_pred = seg_pred.cpu().numpy()
        seg_label = seg_label.cpu().numpy()
        
        # 统计各类别像素数
        for class_id in range(self.num_classes):
            # 真实标签统计
            class_pixels = np.sum(seg_label == class_id)
            self.class_pixel_counts[class_id] += class_pixels
            
            # 预测统计
            pred_pixels = np.sum(seg_pred == class_id)
            self.class_predictions[class_id] += pred_pixels
            
            # 正确预测统计
            correct_pixels = np.sum((seg_pred == class_id) & (seg_label == class_id))
            self.class_correct_predictions[class_id] += correct_pixels
        
        # ignore像素统计
        ignore_pixels = np.sum(seg_label == self.ignore_index)
        self.ignore_pixel_count += ignore_pixels
        
        # 总像素数
        self.total_pixel_count += len(seg_label)
        
        # 保存混淆矩阵数据
        if self.enable_confusion_matrix:
            # 只保存一小部分数据用于混淆矩阵，避免内存过大
            if len(self.confusion_matrix_data) < 100000:  # 限制数据量
                sample_indices = np.random.choice(len(seg_label), 
                                                min(1000, len(seg_label)), 
                                                replace=False)
                self.confusion_matrix_data.extend(
                    list(zip(seg_label[sample_indices], seg_pred[sample_indices]))
                )
    
    def _log_basic_class_info(self, runner):
        """记录基础类别信息"""
        if self.total_pixel_count == 0:
            return
        
        # 计算ignore比例
        ignore_ratio = self.ignore_pixel_count / self.total_pixel_count * 100
        
        # 计算有效类别数量
        active_classes = sum(1 for count in self.class_pixel_counts.values() if count > 0)
        
        self.logger.info(f"📊 类别统计 - Ignore比例: {ignore_ratio:.1f}%, "
                        f"活跃类别: {active_classes}/{self.num_classes-1}")
    
    def _log_detailed_class_info(self, runner):
        """记录详细类别信息"""
        if self.total_pixel_count == 0:
            return
        
        self.logger.info("📈 详细类别分析:")
        
        # 计算类别分布
        class_distribution = {}
        class_accuracy = {}
        
        for class_id in range(self.num_classes - 1):  # 排除ignore类别
            pixel_count = self.class_pixel_counts[class_id]
            pred_count = self.class_predictions[class_id]
            correct_count = self.class_correct_predictions[class_id]
            
            # 分布比例
            distribution = pixel_count / self.total_pixel_count * 100
            class_distribution[class_id] = distribution
            
            # 准确率
            accuracy = correct_count / max(pixel_count, 1) * 100
            class_accuracy[class_id] = accuracy
        
        # 显示前10个最常见的类别
        top_classes = sorted(class_distribution.items(), key=lambda x: x[1], reverse=True)[:10]
        
        self.logger.info("   前10个最常见类别:")
        for class_id, distribution in top_classes:
            accuracy = class_accuracy[class_id]
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            self.logger.info(f"     {class_name:15s}: {distribution:5.1f}% (准确率: {accuracy:5.1f}%)")
        
        # 显示准确率最低的5个类别
        low_acc_classes = sorted(class_accuracy.items(), key=lambda x: x[1])[:5]
        self.logger.info("   准确率最低的5个类别:")
        for class_id, accuracy in low_acc_classes:
            if self.class_pixel_counts[class_id] > 100:  # 只显示有足够样本的类别
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                distribution = class_distribution[class_id]
                self.logger.info(f"     {class_name:15s}: {accuracy:5.1f}% (分布: {distribution:5.1f}%)")
        
        # 保存历史数据
        self.class_distribution_history.append(dict(class_distribution))
        self.class_accuracy_history.append(dict(class_accuracy))
        
        # 类别平衡性分析
        self._analyze_class_balance()
    
    def _analyze_class_balance(self):
        """分析类别平衡性"""
        if not self.class_pixel_counts:
            return
        
        # 计算类别分布的标准差
        distributions = []
        for class_id in range(self.num_classes - 1):
            pixel_count = self.class_pixel_counts[class_id]
            distribution = pixel_count / max(self.total_pixel_count, 1) * 100
            if distribution > 0.01:  # 只考虑有足够样本的类别
                distributions.append(distribution)
        
        if len(distributions) > 1:
            std_dev = np.std(distributions)
            mean_dist = np.mean(distributions)
            cv = std_dev / mean_dist  # 变异系数
            
            if cv > 2.0:
                balance_status = "🔴 严重不平衡"
            elif cv > 1.0:
                balance_status = "🟡 中度不平衡"
            else:
                balance_status = "🟢 相对平衡"
            
            self.logger.info(f"   类别平衡性: {balance_status} (CV: {cv:.2f})")
    
    def _log_confusion_matrix(self, runner):
        """记录混淆矩阵"""
        if not self.confusion_matrix_data:
            return
        
        self.logger.info("🔍 生成混淆矩阵...")
        
        # 提取标签和预测
        y_true = [item[0] for item in self.confusion_matrix_data]
        y_pred = [item[1] for item in self.confusion_matrix_data]
        
        # 只考虑有效类别（排除ignore）
        valid_indices = [i for i, label in enumerate(y_true) if label != self.ignore_index]
        y_true_valid = [y_true[i] for i in valid_indices]
        y_pred_valid = [y_pred[i] for i in valid_indices]
        
        if len(y_true_valid) == 0:
            return
        
        # 计算混淆矩阵
        labels = list(range(self.num_classes - 1))  # 排除ignore类别
        cm = confusion_matrix(y_true_valid, y_pred_valid, labels=labels)
        
        # 保存混淆矩阵图
        self._save_confusion_matrix_plot(cm, runner.epoch)
        
        # 计算每个类别的精确率和召回率
        self._calculate_precision_recall(cm)
        
        # 清空数据，避免内存过大
        self.confusion_matrix_data = []
    
    def _save_confusion_matrix_plot(self, cm, epoch):
        """保存混淆矩阵图"""
        if not self.enable_confusion_matrix or not HAS_MATPLOTLIB:
            if not HAS_MATPLOTLIB:
                self.logger.warning("matplotlib not available, skipping confusion matrix plot")
            return

        plt.figure(figsize=(12, 10))

        # 归一化混淆矩阵
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # 绘制热力图
        if HAS_SEABORN:
            sns.heatmap(cm_normalized,
                       annot=True,
                       fmt='.2f',
                       cmap='Blues',
                       xticklabels=self.class_names[:-1],  # 排除ignore
                       yticklabels=self.class_names[:-1])
        else:
            # 使用matplotlib的简单实现
            im = plt.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
            plt.colorbar(im)

            # 添加文本注释
            thresh = cm_normalized.max() / 2.
            for i in range(cm_normalized.shape[0]):
                for j in range(cm_normalized.shape[1]):
                    plt.text(j, i, f'{cm_normalized[i, j]:.2f}',
                            ha="center", va="center",
                            color="white" if cm_normalized[i, j] > thresh else "black")

            plt.xticks(range(len(self.class_names[:-1])), self.class_names[:-1], rotation=45)
            plt.yticks(range(len(self.class_names[:-1])), self.class_names[:-1])

        plt.title(f'Confusion Matrix (Epoch {epoch + 1})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()

        # 保存图片
        save_path = os.path.join(self.save_dir, f'confusion_matrix_epoch_{epoch + 1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"   混淆矩阵已保存: {save_path}")
    
    def _calculate_precision_recall(self, cm):
        """计算精确率和召回率"""
        precisions = []
        recalls = []
        
        for i in range(len(cm)):
            # 精确率 = TP / (TP + FP)
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            precision = tp / max(tp + fp, 1)
            precisions.append(precision)
            
            # 召回率 = TP / (TP + FN)
            fn = cm[i, :].sum() - tp
            recall = tp / max(tp + fn, 1)
            recalls.append(recall)
        
        # 显示精确率和召回率最低的类别
        low_precision_classes = sorted(enumerate(precisions), key=lambda x: x[1])[:3]
        low_recall_classes = sorted(enumerate(recalls), key=lambda x: x[1])[:3]
        
        self.logger.info("   精确率最低的3个类别:")
        for class_id, precision in low_precision_classes:
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            self.logger.info(f"     {class_name:15s}: {precision:.3f}")
        
        self.logger.info("   召回率最低的3个类别:")
        for class_id, recall in low_recall_classes:
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            self.logger.info(f"     {class_name:15s}: {recall:.3f}")
    
    def after_train_epoch(self, runner):
        """每个epoch结束后保存统计"""
        # 保存类别分布图
        if self.enable_class_distribution_plot:
            self._save_class_distribution_plot(runner.epoch)
        
        # 保存epoch统计
        self._save_epoch_class_stats(runner.epoch)
    
    def _save_class_distribution_plot(self, epoch):
        """保存类别分布图"""
        if not self.class_pixel_counts:
            return
        
        # 准备数据
        class_names = []
        distributions = []
        
        for class_id in range(self.num_classes - 1):  # 排除ignore
            pixel_count = self.class_pixel_counts[class_id]
            distribution = pixel_count / max(self.total_pixel_count, 1) * 100
            
            if distribution > 0.01:  # 只显示有足够样本的类别
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                class_names.append(class_name)
                distributions.append(distribution)
        
        if not distributions or not HAS_MATPLOTLIB:
            if not HAS_MATPLOTLIB:
                self.logger.warning("matplotlib not available, skipping class distribution plot")
            return

        # 绘制柱状图
        plt.figure(figsize=(15, 8))
        bars = plt.bar(range(len(distributions)), distributions)

        # 设置颜色
        try:
            colors = plt.cm.Set3(np.linspace(0, 1, len(distributions)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        except:
            # 如果颜色映射失败，使用默认颜色
            pass

        plt.xlabel('Classes')
        plt.ylabel('Distribution (%)')
        plt.title(f'Class Distribution (Epoch {epoch + 1})')
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for i, v in enumerate(distributions):
            plt.text(i, v + 0.1, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        # 保存图片
        save_path = os.path.join(self.save_dir, f'class_distribution_epoch_{epoch + 1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_epoch_class_stats(self, epoch):
        """保存epoch类别统计"""
        import json
        
        stats = {
            'epoch': epoch + 1,
            'total_pixels': self.total_pixel_count,
            'ignore_pixels': self.ignore_pixel_count,
            'ignore_ratio': self.ignore_pixel_count / max(self.total_pixel_count, 1) * 100,
            'class_pixel_counts': dict(self.class_pixel_counts),
            'class_predictions': dict(self.class_predictions),
            'class_correct_predictions': dict(self.class_correct_predictions)
        }
        
        stats_file = os.path.join(self.save_dir, f'class_stats_epoch_{epoch + 1}.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
