import os
import torch
import threading

from neural_angelo.Util.misc import to_cpu


class Checkpointer(object):
    """检查点管理器，用于保存和加载模型状态。"""

    def __init__(self, model, optim=None, sched=None):
        self.model = model
        self.optim = optim
        self.sched = sched
        self.resume = False
        self.resume_epoch = self.resume_iteration = None

    def save(self, checkpoint_path, current_epoch, current_iteration):
        """将网络权重、优化器参数、调度器参数保存到指定的检查点文件。

        Args:
            checkpoint_path (str): 检查点文件的完整路径。
            current_epoch (int): 当前 epoch。
            current_iteration (int): 当前迭代次数。

        Returns:
            checkpoint_path: 检查点文件路径。
        """
        save_dict = to_cpu(self._collect_state_dicts())
        save_dict.update(
            epoch=current_epoch,
            iteration=current_iteration,
        )
        # 在单独的线程中运行检查点保存
        threading.Thread(
            target=self._save_worker, daemon=False, args=(save_dict, checkpoint_path)).start()
        return checkpoint_path

    def _save_worker(self, save_dict, checkpoint_path):
        """在后台线程中保存检查点。"""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(save_dict, checkpoint_path)
        print('Saved checkpoint to {}'.format(checkpoint_path))

    def _collect_state_dicts(self):
        """收集所有要保存的网络模块的状态字典。"""
        return dict(
            model=self.model.state_dict(),
            optim=self.optim.state_dict() if self.optim is not None else None,
            sched=self.sched.state_dict() if self.sched is not None else None,
        )

    def load(self, checkpoint_path, load_opt=True, load_sch=True, iteration_mode=True, strict_resume=True):
        """从指定的检查点文件加载网络权重、优化器参数、调度器参数。

        Args:
            checkpoint_path (str): 检查点文件的完整路径。
            load_opt (bool): 是否加载优化器状态字典。
            load_sch (bool): 是否加载调度器状态字典。
            iteration_mode (bool): 是否使用迭代模式（用于设置 scheduler 的 last_epoch）。
            strict_resume (bool): 是否严格加载模型权重。
        """
        if checkpoint_path is None:
            print('No checkpoint path provided. Training from scratch.')
            torch.cuda.empty_cache()
            return

        # 检查文件是否存在
        if not os.path.exists(checkpoint_path):
            print(f'警告: 检查点文件不存在: {checkpoint_path}')
            print('从头开始训练...')
            torch.cuda.empty_cache()
            return

        # 尝试加载检查点
        try:
            print(f"Loading checkpoint: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

            # 加载模型状态字典
            print('- Loading the model...')
            self.model.load_state_dict(state_dict['model'], strict=strict_resume)

            # 恢复训练状态
            self.resume = True
            self.resume_epoch = state_dict['epoch']
            self.resume_iteration = state_dict['iteration']

            if self.sched is not None:
                self.sched.last_epoch = self.resume_iteration if iteration_mode else self.resume_epoch

            if load_opt and self.optim is not None and state_dict.get('optim') is not None:
                print('- Loading the optimizer...')
                self.optim.load_state_dict(state_dict['optim'])
            if load_sch and self.sched is not None and state_dict.get('sched') is not None:
                print('- Loading the scheduler...')
                self.sched.load_state_dict(state_dict['sched'])

            print(f"Done with loading the checkpoint (epoch {self.resume_epoch}, iter {self.resume_iteration}).")

        except Exception as e:
            print(f'警告: 加载检查点失败: {checkpoint_path}')
            print(f'错误信息: {e}')
            print('从头开始训练...')
            self.resume = False
            self.resume_epoch = self.resume_iteration = None

        torch.cuda.empty_cache()
