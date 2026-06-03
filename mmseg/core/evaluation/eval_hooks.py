'''
import os.path as osp
import torch
from mmcv.runner import Hook
from torch.utils.data import DataLoader


class EvalHook(Hook):
    """增强的评估钩子，支持保存最佳模型。

    属性:
        dataloader (DataLoader): PyTorch数据加载器
        interval (int): 评估间隔（按轮次）。默认: 1
        save_best (str): 用于保存最佳检查点的指标名称。例如: 'mIoU'
        rule (str): 比较规则。'greater' 或 'less'
    """

    def __init__(self, dataloader, interval=1, by_epoch=False,
                 save_best=None, rule='greater', **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader必须是pytorch DataLoader类型，但获得了'
                            f'{type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.by_epoch = by_epoch
        self.eval_kwargs = eval_kwargs

        # 🔥 添加 save_best 功能
        self.save_best = save_best
        self.rule = rule
        self.best_score = None
        self.best_epoch = None

        # 验证rule参数
        if rule not in ['greater', 'less']:
            raise ValueError(f"rule必须是'greater'或'less'，得到了{rule}")

        # 打印配置
        if self.save_best:
            print(f"🎯 EvalHook: 将根据 '{save_best}' 保存最佳模型 (rule={rule})")

    def after_train_iter(self, runner):
        """训练迭代后钩子。"""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        """训练轮次后钩子。"""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

        # 🔧 关键修复：验证完成后深度清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()

    def evaluate(self, runner, results):
        """调用数据集的评估函数并保存最佳检查点。"""
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)

        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True

        # 🔥 添加 save_best 逻辑
        if self.save_best is not None:
            self._save_best_checkpoint(runner, eval_res)

    def _save_best_checkpoint(self, runner, eval_res):
        """基于评估结果保存最佳检查点。

        Args:
            runner: 训练运行器
            eval_res: 评估结果字典，例如 {'mIoU': 0.5, 'mAcc': 0.6}
        """
        # 检查指标是否存在
        if self.save_best not in eval_res:
            # 尝试不同的大小写组合
            possible_keys = [self.save_best, self.save_best.lower(),
                             self.save_best.upper(), 'mIoU', 'miou', 'mIOU']
            found_key = None
            for key in possible_keys:
                if key in eval_res:
                    found_key = key
                    break

            if found_key is None:
                runner.logger.warning(
                    f"❌ save_best指标'{self.save_best}'在评估结果中未找到！")
                runner.logger.warning(f"   可用指标: {list(eval_res.keys())}")
                return
            else:
                # 找到了匹配的key，使用它
                runner.logger.info(
                    f"ℹ️  save_best: '{self.save_best}'未找到，改用'{found_key}'")
                metric_key = found_key
        else:
            metric_key = self.save_best

        current_score = eval_res[metric_key]

        # 初始化或更新best_score
        if self.best_score is None:
            is_best = True
        else:
            if self.rule == 'greater':
                is_best = current_score > self.best_score
            else:
                is_best = current_score < self.best_score

        if is_best:
            self.best_score = current_score
            self.best_epoch = runner.epoch

            # 🔄 修改：使用固定的文件名
            best_ckpt_path = osp.join(
                runner.work_dir,
                f'best_{metric_key}.pth'
            )

            # 如果已存在，先删除旧的最佳模型文件
            if osp.exists(best_ckpt_path):
                import os
                os.remove(best_ckpt_path)

            # 使用固定的文件名保存检查点
            runner.save_checkpoint(
                runner.work_dir,
                filename_tmpl=f'best_{metric_key}',
                create_symlink=False
            )

            # 保存一个包含当前轮次信息的文件
            epoch_info_path = osp.join(
                runner.work_dir,
                f'best_{metric_key}_epoch_info.txt'
            )
            with open(epoch_info_path, 'w') as f:
                f.write(f"最佳{metric_key}={current_score:.4f}，在轮次{runner.epoch}\n")

            # 记录日志
            runner.logger.info(
                f"✅ 新的最佳检查点已保存！"
                f"{metric_key}={current_score:.4f}，在轮次{runner.epoch}")
            runner.logger.info(f"   保存到: {best_ckpt_path}")
            runner.logger.info(f"   轮次信息保存到: {epoch_info_path}")

        else:
            runner.logger.info(
                f"📊 当前{metric_key}={current_score:.4f}，"
                f"最佳{metric_key}={self.best_score:.4f}，在轮次{self.best_epoch}")


class DistEvalHook(EvalHook):
    """分布式评估钩子，支持保存最佳模型。"""

    def __init__(self,
                 dataloader,
                 interval=1,
                 gpu_collect=False,
                 by_epoch=False,
                 save_best=None,
                 rule='greater',
                 **eval_kwargs):
        super().__init__(dataloader, interval, by_epoch, save_best, rule, **eval_kwargs)
        self.gpu_collect = gpu_collect

    def after_train_iter(self, runner):
        """训练迭代后钩子。"""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmseg.apis import multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        """训练轮次后钩子。"""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmseg.apis import multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)
'''

import os.path as osp
import torch
from mmcv.runner import Hook
from torch.utils.data import DataLoader


class EvalHook(Hook):
    """增强的评估钩子，支持保存最佳模型。

    属性:
        dataloader (DataLoader): PyTorch数据加载器
        interval (int): 评估间隔（按轮次）。默认: 1
        save_best (str): 用于保存最佳检查点的指标名称。例如: 'mIoU'
        rule (str): 比较规则。'greater' 或 'less'
    """

    def __init__(self, dataloader, interval=1, by_epoch=False,
                 save_best=None, rule='greater', **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader必须是pytorch DataLoader类型，但获得了'
                            f'{type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.by_epoch = by_epoch
        self.eval_kwargs = eval_kwargs

        # 🔥 添加 save_best 功能
        self.save_best = save_best
        self.rule = rule
        self.best_score = None
        self.best_epoch = None

        # 验证rule参数
        if rule not in ['greater', 'less']:
            raise ValueError(f"rule必须是'greater'或'less'，得到了{rule}")

    def after_train_iter(self, runner):
        """训练迭代后钩子。"""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        """训练轮次后钩子。"""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

        # 🔧 关键修复：验证完成后深度清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()

    def evaluate(self, runner, results):
        """调用数据集的评估函数并保存最佳检查点。"""
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)

        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True

        # 🔥 添加 save_best 逻辑
        if self.save_best is not None:
            self._save_best_checkpoint(runner, eval_res)

    def _save_best_checkpoint(self, runner, eval_res):
        """基于评估结果保存最佳检查点。

        Args:
            runner: 训练运行器
            eval_res: 评估结果字典，例如 {'mIoU': 0.5, 'mAcc': 0.6}
        """
        # 检查指标是否存在
        if self.save_best not in eval_res:
            # 尝试不同的大小写组合
            possible_keys = [self.save_best, self.save_best.lower(),
                             self.save_best.upper(), 'mIoU', 'miou', 'mIOU']
            found_key = None
            for key in possible_keys:
                if key in eval_res:
                    found_key = key
                    break

            if found_key is None:
                runner.logger.warning(
                    f"❌ save_best指标'{self.save_best}'在评估结果中未找到！")
                runner.logger.warning(f"   可用指标: {list(eval_res.keys())}")
                return
            else:
                # 找到了匹配的key，使用它
                runner.logger.info(
                    f"ℹ️  save_best: '{self.save_best}'未找到，改用'{found_key}'")
                metric_key = found_key
        else:
            metric_key = self.save_best

        current_score = eval_res[metric_key]

        # 初始化或更新best_score
        if self.best_score is None:
            is_best = True
        else:
            if self.rule == 'greater':
                is_best = current_score > self.best_score
            else:
                is_best = current_score < self.best_score

        if is_best:
            self.best_score = current_score
            self.best_epoch = runner.epoch

            # 🔄 修改：使用固定的文件名
            best_ckpt_path = osp.join(
                runner.work_dir,
                f'best_{metric_key}.pth'
            )

            # 如果已存在，先删除旧的最佳模型文件
            if osp.exists(best_ckpt_path):
                import os
                os.remove(best_ckpt_path)

            # 使用固定的文件名保存检查点
            runner.save_checkpoint(
                runner.work_dir,
                filename_tmpl=f'best_{metric_key}',
                create_symlink=False
            )

            # 保存一个包含当前轮次信息的文件
            epoch_info_path = osp.join(
                runner.work_dir,
                f'best_{metric_key}_epoch_info.txt'
            )
            with open(epoch_info_path, 'w') as f:
                f.write(f"最佳{metric_key}={current_score:.4f}，在轮次{runner.epoch}\n")

            # 记录日志
            runner.logger.info(
                f"✅ 新的最佳检查点已保存！"
                f"{metric_key}={current_score:.4f}，在轮次{runner.epoch}")
            runner.logger.info(f"   保存到: {best_ckpt_path}")
            runner.logger.info(f"   轮次信息保存到: {epoch_info_path}")

        else:
            runner.logger.info(
                f"📊 当前{metric_key}={current_score:.4f}，"
                f"最佳{metric_key}={self.best_score:.4f}，在轮次{self.best_epoch}")


class DistEvalHook(EvalHook):
    """分布式评估钩子，支持保存最佳模型。"""

    def __init__(self,
                 dataloader,
                 interval=1,
                 gpu_collect=False,
                 by_epoch=False,
                 save_best=None,
                 rule='greater',
                 **eval_kwargs):
        super().__init__(dataloader, interval, by_epoch, save_best, rule, **eval_kwargs)
        self.gpu_collect = gpu_collect

    def after_train_iter(self, runner):
        """训练迭代后钩子。"""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmseg.apis import multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        """训练轮次后钩子。"""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmseg.apis import multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)