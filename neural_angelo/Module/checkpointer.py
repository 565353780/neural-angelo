import os
import torch
import threading

from neural_angelo.Util.misc import to_cpu


class Checkpointer(object):
    """检查点管理器，用于保存和加载模型状态。"""

    def __init__(self, cfg, model, optim=None, sched=None):
        self.model = model
        self.optim = optim
        self.sched = sched
        self.logdir = cfg.logdir
        self.save_period = cfg.checkpoint.save_period
        self.strict_resume = cfg.checkpoint.strict_resume
        self.iteration_mode = cfg.optim.sched.iteration_mode
        self.resume = False
        self.resume_epoch = self.resume_iteration = None

    def save(self, current_epoch, current_iteration, latest=False):
        """将网络权重、优化器参数、调度器参数保存到检查点。

        Args:
            current_epoch (int): 当前 epoch。
            current_iteration (int): 当前迭代次数。
            latest (bool): 如果为 True，则使用 'latest_checkpoint.pt' 名称保存。

        Returns:
            checkpoint_path: 检查点文件路径。
        """
        checkpoint_file = 'latest_checkpoint.pt' if latest else \
                          f'epoch_{current_epoch:05}_iteration_{current_iteration:09}_checkpoint.pt'
        save_dict = to_cpu(self._collect_state_dicts())
        save_dict.update(
            epoch=current_epoch,
            iteration=current_iteration,
        )
        # 在单独的线程中运行检查点保存
        threading.Thread(
            target=self._save_worker, daemon=False, args=(save_dict, checkpoint_file)).start()
        checkpoint_path = self._get_full_path(checkpoint_file)
        return checkpoint_path

    def _save_worker(self, save_dict, checkpoint_file):
        """在后台线程中保存检查点。"""
        checkpoint_path = self._get_full_path(checkpoint_file)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(save_dict, checkpoint_path)
        self.write_latest_checkpoint_file(checkpoint_file)
        print('Saved checkpoint to {}'.format(checkpoint_path))

    def _collect_state_dicts(self):
        """收集所有要保存的网络模块的状态字典。"""
        return dict(
            model=self.model.state_dict(),
            optim=self.optim.state_dict(),
            sched=self.sched.state_dict(),
        )

    def load(self, checkpoint_path=None, resume=False, load_opt=True, load_sch=True, **kwargs):
        """从检查点加载网络权重、优化器参数、调度器参数。

        Args:
            checkpoint_path (str): 检查点路径（本地文件）。
            resume (bool): 如果为 False，只加载模型权重。如果为 True，也会加载元数据（epoch/iteration）和优化器/调度器（可选）。
            load_opt (bool): 是否加载优化器状态字典（resume 应为 True）。
            load_sch (bool): 是否加载调度器状态字典（resume 应为 True）。
        """
        self.resume = resume
        # 如果未指定检查点路径，尝试从同一运行中加载最新的检查点
        if resume and checkpoint_path is None:
            latest_checkpoint_file = self.read_latest_checkpoint_file()
            if latest_checkpoint_file is not None:
                checkpoint_path = self._get_full_path(latest_checkpoint_file)
        # 加载检查点
        if checkpoint_path is not None:
            self._check_checkpoint_exists(checkpoint_path)
            self.checkpoint_path = checkpoint_path
            state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            print(f"Loading checkpoint (local): {checkpoint_path}")
            # 加载状态字典
            print('- Loading the model...')
            self.model.load_state_dict(state_dict['model'], strict=self.strict_resume)
            if resume:
                self.resume_epoch = state_dict['epoch']
                self.resume_iteration = state_dict['iteration']
                self.sched.last_epoch = self.resume_iteration if self.iteration_mode else self.resume_epoch
                if load_opt:
                    print('- Loading the optimizer...')
                    self.optim.load_state_dict(state_dict['optim'])
                if load_sch:
                    print('- Loading the scheduler...')
                    self.sched.load_state_dict(state_dict['sched'])
                print(f"Done with loading the checkpoint (epoch {self.resume_epoch}, iter {self.resume_iteration}).")
            else:
                print('Done with loading the checkpoint.')
            self.eval_epoch = state_dict['epoch']
            self.eval_iteration = state_dict['iteration']
        else:
            print('Training from scratch.')
        torch.cuda.empty_cache()

    def _get_full_path(self, file):
        return os.path.join(self.logdir, file)

    def _get_latest_pointer_path(self):
        return self._get_full_path('latest_checkpoint.txt')

    def read_latest_checkpoint_file(self):
        """读取最新检查点文件名。"""
        checkpoint_file = None
        latest_path = self._get_latest_pointer_path()
        if os.path.exists(latest_path):
            checkpoint_file = open(latest_path).read().strip()
            if checkpoint_file.startswith("latest_checkpoint:"):  # 向后兼容
                checkpoint_file = checkpoint_file.split(' ')[-1]
        return checkpoint_file

    def write_latest_checkpoint_file(self, checkpoint_file):
        """写入最新检查点文件指针。"""
        latest_path = self._get_latest_pointer_path()
        content = f"{checkpoint_file}\n"
        with open(latest_path, "w") as file:
            file.write(content)

    def _check_checkpoint_exists(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'File not found (local): {checkpoint_path}')

    def reached_checkpointing_period(self, timer):
        """检查是否达到检查点保存周期。"""
        if timer.checkpoint_toc() > self.save_period:
            print('checkpointing period!')
            return True
        return False
