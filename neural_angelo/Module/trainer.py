import os
import torch
import inspect
import torch.nn.functional as torch_F

from tqdm import tqdm
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from neural_angelo.Util.init_weight import weights_init, weights_rescale
from neural_angelo.Util.model_average import ModelAverage
from neural_angelo.Util.misc import to_cuda, requires_grad
from neural_angelo.Util.visualization import tensorboard_image

from neural_angelo.Dataset.dataloader import get_train_dataloader, get_val_dataloader
from neural_angelo.Model.model import Model
from neural_angelo.Loss.eikonal import eikonal_loss
from neural_angelo.Loss.curvature import curvature_loss
from neural_angelo.Method.time import getCurrentTime
from neural_angelo.Method.cudnn import init_cudnn
from neural_angelo.Module.checkpointer import Checkpointer


def cycle_dataloader(data_loader):
    """创建一个无限循环的数据迭代器，确保数据按顺序循环获取。

    每次 data_loader 耗尽时，重新从头开始迭代，保证所有图片都被均匀使用。
    """
    while True:
        for data in data_loader:
            yield data


def _calculate_model_size(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def collate_test_data_batches(data_batches):
    """将多个测试数据批次合并为一个字典。"""
    if not data_batches:
        return {}
    data_all = {}
    for key in data_batches[0].keys():
        if isinstance(data_batches[0][key], torch.Tensor):
            data_all[key] = torch.cat([batch[key] for batch in data_batches], dim=0)
        elif isinstance(data_batches[0][key], (list, tuple)):
            data_all[key] = [item for batch in data_batches for item in batch[key]]
        else:
            data_all[key] = [batch[key] for batch in data_batches]
    return data_all


def trim_test_samples(data, max_samples=None):
    """将测试样本数量限制在 max_samples 以内。"""
    if max_samples is None or max_samples <= 0:
        return
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key][:max_samples]
        elif isinstance(data[key], (list, tuple)):
            data[key] = data[key][:max_samples]


class Trainer(object):
    r"""Trainer class for Neuralangelo training.

    训练循环基于 epoch，每个 epoch 包含 cfg.iters_per_epoch 次迭代。
    所有保存、验证、日志记录操作都在 epoch 结束时统一进行。

    Args:
        cfg (obj): Global configuration.
    """

    def __init__(self, cfg):
        print('Setup trainer.')
        # 初始化 cuDNN
        init_cudnn(deterministic=False, benchmark=True)

        self.cfg = cfg

        # Create objects for the networks, optimizers, and schedulers.
        self.model = self.setup_model()
        self.optim = self.setup_optimizer(self.model)
        self.sched = self.setup_scheduler(self.optim)
        self.model = self.wrap_model(self.model)
        # Initialize automatic mixed precision training.
        self.init_amp()
        # Initialize loss functions.
        self.init_losses()

        self.checkpointer = Checkpointer(self.model, self.optim, self.sched)

        # 检查点路径设置
        self.checkpoint_path_last = os.path.join(self.cfg.logdir, 'model_last.pt')
        self.checkpoint_path_best = os.path.join(self.cfg.logdir, 'model_best.pt')
        self.best_psnr = float('-inf')  # 追踪最佳PSNR

        # Initialize logging attributes.
        self.init_logging_attributes()
        if 'TORCH_HOME' not in os.environ:
            os.environ['TORCH_HOME'] = os.path.join(os.environ['HOME'], ".cache")

        # Trainer-specific initialization
        self.metrics = dict()

        # 每个 epoch 的迭代次数
        self.iters_per_epoch = self.cfg.iters_per_epoch

        self.warm_up_end = self.cfg.optim.sched.warm_up_end
        self.cfg_gradient = self.cfg.model.object.sdf.gradient

        self.c2f_step = self.cfg.model.object.sdf.encoding.coarse2fine.step
        self.model_module.neural_sdf.warm_up_end = self.warm_up_end

        self.criteria["render"] = torch.nn.L1Loss()

        self.mode = 'train'

        self.train_data_loader = get_train_dataloader(self.cfg, shuffle=True)
        self.eval_data_loader = get_val_dataloader(self.cfg)

        self.init_tensorboard(self.cfg.logdir)
        return

    def setup_model(self):
        r"""Return the networks.
        will wrap the network with a moving average model if applicable.

        The following objects are constructed as class members:
          - model (obj): Model object (historically: generator network object).
        """
        # Construct networks
        model = Model(self.cfg.model, self.cfg.data)
        print('model parameter count: {:,}'.format(_calculate_model_size(model)))
        print(f'Initialize model weights using type: {self.cfg.trainer.init.type}, gain: {self.cfg.trainer.init.gain}')
        init_bias = getattr(self.cfg.trainer.init, 'bias', None)
        init_gain = self.cfg.trainer.init.gain or 1.
        model.apply(weights_init(self.cfg.trainer.init.type, init_gain, init_bias))
        model.apply(weights_rescale())
        model = model.to('cuda')
        return model

    def setup_optimizer(self, model):
        optim = AdamW(
            params=model.parameters(),
            lr=self.cfg.optim.params.lr,
            weight_decay=self.cfg.optim.params.weight_decay,
        )

        self.optim_zero_grad_kwargs = {}
        if 'set_to_none' in inspect.signature(optim.zero_grad).parameters:
            self.optim_zero_grad_kwargs['set_to_none'] = True
        return optim

    def setup_scheduler(self, optim):
        warm_up_end = self.cfg.optim.sched.warm_up_end
        two_steps = self.cfg.optim.sched.two_steps
        gamma = self.cfg.optim.sched.gamma

        def sch(x):
            if x < warm_up_end:
                return x / warm_up_end
            else:
                if x > two_steps[1]:
                    return 1.0 / gamma ** 2
                elif x > two_steps[0]:
                    return 1.0 / gamma
                else:
                    return 1.0

        return LambdaLR(optim, lambda x: sch(x))

    def wrap_model(self, model):
        # 使用指数移动平均（EMA）包装模型
        if self.cfg.trainer.ema_config.enabled:
            model = ModelAverage(model,
                                self.cfg.trainer.ema_config.beta,
                                self.cfg.trainer.ema_config.start_iteration)
            self.model_module = model.module
        else:
            self.model_module = model
        return model

    def init_amp(self):
        r"""Initialize automatic mixed precision training."""

        if getattr(self.cfg.trainer, 'allow_tf32', True):
            print("Allow TensorFloat32 operations on supported devices")
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
        else:
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False

        if self.cfg.trainer.amp_config.enabled:
            print("Using automatic mixed precision training.")

        # amp scaler can be used without mixed precision training
        if hasattr(self.cfg.trainer, 'scaler_config'):
            scaler_kwargs = vars(self.cfg.trainer.scaler_config)
            scaler_kwargs['enabled'] = self.cfg.trainer.amp_config.enabled and \
                getattr(self.cfg.trainer.scaler_config, 'enabled', True)
        else:
            scaler_kwargs = vars(self.cfg.trainer.amp_config)   # backward compatibility
            scaler_kwargs.pop('dtype', None)
            scaler_kwargs.pop('cache_enabled', None)

        self.scaler = GradScaler('cuda', **scaler_kwargs)

    def init_logging_attributes(self):
        r"""Initialize logging attributes."""
        self.current_iteration = 0
        self.current_epoch = 0

    def load_checkpoint(self, checkpoint_path=None, load_opt=True, load_sch=True):
        """加载检查点以恢复训练或进行推理。

        Args:
            checkpoint_path (str): 检查点文件路径。如果为 None，则尝试加载 model_last.pt。
            load_opt (bool): 是否加载优化器状态。
            load_sch (bool): 是否加载调度器状态。
        """
        if checkpoint_path is None:
            # 默认尝试加载 model_last.pt
            if os.path.exists(self.checkpoint_path_last):
                checkpoint_path = self.checkpoint_path_last
        self.checkpointer.load(
            checkpoint_path,
            load_opt=load_opt,
            load_sch=load_sch,
            iteration_mode=self.cfg.optim.sched.iteration_mode,
            strict_resume=self.cfg.checkpoint.strict_resume
        )

    def init_tensorboard(self, logdir: str) -> bool:
        print('Initialize TensorBoard')
        tensorboard_dir = os.path.join(logdir, getCurrentTime())
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.tensorboard_writer = SummaryWriter(log_dir=tensorboard_dir)
        return True

    def start_of_epoch(self, current_epoch):
        r"""Things to do before an epoch.

        Args:
            current_epoch (int): Current number of epoch.
        """
        self.current_epoch = current_epoch

    def start_of_iteration(self, data, current_iteration):
        r"""Things to do before an iteration."""
        self.current_iteration = current_iteration
        self._update_progress(current_iteration)
        data = to_cuda(data)
        self.model.train()
        return data

    def end_of_iteration(self, current_iteration):
        r"""Things to do after each iteration (minimal operations).

        Args:
            current_iteration (int): Current number of iteration.
        """
        self.current_iteration = current_iteration
        # 每次迭代后更新 scheduler（iteration mode）
        if self.cfg.optim.sched.iteration_mode:
            self.sched.step()

    def end_of_epoch(self, data, current_epoch, current_iteration):
        r"""Things to do after an epoch.

        所有的验证、保存、日志记录操作都在这里统一进行。

        Args:
            data (dict): Data used for the last iteration in the epoch.
            current_epoch (int): Current number of epoch.
            current_iteration (int): Current total iteration count.
        """
        self.current_epoch = current_epoch
        self.current_iteration = current_iteration

        # 更新 scheduler（epoch mode）
        if not self.cfg.optim.sched.iteration_mode:
            self.sched.step()

        # 日志打印
        print(f'Epoch: {current_epoch}, iter: {current_iteration}')

        # TensorBoard scalar 记录
        self.log_tensorboard_scalars(data, mode="train")
        # Exit if the training loss has gone to NaN/inf.
        if self.losses["total"].isnan() or self.losses["total"].isinf():
            self.finalize()
            raise ValueError("Training loss has gone to NaN or infinity!!!")

        # 验证和 TensorBoard 图像记录
        data_all = self.test(self.eval_data_loader, mode="val")
        self.log_tensorboard_scalars(data_all, mode="val")
        self.log_tensorboard_images(data_all, mode="val", max_samples=self.cfg.data.val.max_viz_samples)
        # 判断是否为最佳模型并保存
        self._save_best_model_if_needed(current_epoch, current_iteration)

        # 保存 model_last.pt
        self.checkpointer.save(self.checkpoint_path_last, current_epoch, current_iteration)

    def _update_progress(self, current_iteration):
        r"""Update training progress and model parameters."""
        model = self.model_module
        max_iter = self.cfg.max_epoch * self.iters_per_epoch
        self.progress = model.progress = current_iteration / max_iter

        model.neural_sdf.set_active_levels(current_iteration)

        model.neural_sdf.set_normal_epsilon()
        self.get_curvature_weight(current_iteration, self.cfg.trainer.loss_weight.curvature)

    def _save_best_model_if_needed(self, current_epoch, current_iteration):
        """根据PSNR判断是否保存最佳模型。"""
        current_psnr = self.metrics["psnr"].detach().item()
        if current_psnr > self.best_psnr:
            self.best_psnr = current_psnr
            self.checkpointer.save(self.checkpoint_path_best, current_epoch, current_iteration)
            print(f'New best model saved! PSNR: {current_psnr:.4f}')

    def train_step(self, data, last_iter_in_epoch=False):
        r"""One training step.

        Args:
            data (dict): Data used for the current iteration.
        """
        # Set requires_grad flags.
        requires_grad(self.model_module, True)

        autocast_dtype = getattr(self.cfg.trainer.amp_config, 'dtype', 'float16')
        autocast_dtype = torch.bfloat16 if autocast_dtype == 'bfloat16' else torch.float16
        amp_kwargs = {
            'enabled': self.cfg.trainer.amp_config.enabled,
            'dtype': autocast_dtype
        }
        with autocast('cuda', **amp_kwargs):
            total_loss = self.model_forward(data)
            # Scale down the loss w.r.t. gradient accumulation iterations.
            total_loss = total_loss / float(self.cfg.trainer.grad_accum_iter)

        # Backpropagate the loss.
        self.scaler.scale(total_loss).backward()

        # Perform an optimizer step. This enables gradient accumulation when grad_accum_iter is not 1.
        if (self.current_iteration + 1) % self.cfg.trainer.grad_accum_iter == 0 or last_iter_in_epoch:
            self.scaler.step(self.optim)
            self.scaler.update()
            # Zero out the gradients.
            self.optim.zero_grad(**self.optim_zero_grad_kwargs)

        # Update model average.
        if self.cfg.trainer.ema_config.enabled:
            self.model.update_average()

        self._detach_losses()

    def model_forward(self, data):
        # Model forward.
        output = self.model(data)
        data.update(output)
        # Compute loss.
        self._compute_loss(data, mode="train")
        total_loss = self._get_total_loss()
        return total_loss

    def train(self):
        """训练主循环。

        使用基于 epoch 的训练方式，每个 epoch 包含 iters_per_epoch 次迭代。
        数据通过循环迭代器按顺序获取，确保所有图片都被均匀使用。
        """
        # Resume from checkpoint if available
        start_epoch = self.checkpointer.resume_epoch or self.current_epoch
        current_iteration = self.checkpointer.resume_iteration or self.current_iteration
        self.current_epoch = start_epoch
        self.current_iteration = current_iteration
        max_iter = self.cfg.max_epoch * self.iters_per_epoch
        self.progress = self.model_module.progress = current_iteration / max_iter

        # Initial validation
        data_all = self.test(self.eval_data_loader, mode="val")
        self.log_tensorboard_scalars(data_all, mode="val")
        self.log_tensorboard_images(data_all, mode="val", max_samples=self.cfg.data.val.max_viz_samples)
        self._save_best_model_if_needed(start_epoch, current_iteration)

        # 创建循环数据迭代器，确保数据按顺序循环获取
        data_iter = cycle_dataloader(self.train_data_loader)

        for current_epoch in range(start_epoch, self.cfg.max_epoch):
            self.start_of_epoch(current_epoch)

            # 使用 tqdm 显示 epoch 内的迭代进度
            epoch_pbar = tqdm(range(self.iters_per_epoch), 
                            desc=f"Epoch {current_epoch + 1}/{self.cfg.max_epoch}", 
                            leave=False)

            for it in epoch_pbar:
                # 从循环迭代器获取下一批数据
                data = next(data_iter)
                data = self.start_of_iteration(data, current_iteration)

                # 判断是否是 epoch 最后一次迭代（用于梯度累积）
                last_iter_in_epoch = (it == self.iters_per_epoch - 1)
                self.train_step(data, last_iter_in_epoch=last_iter_in_epoch)

                current_iteration += 1
                epoch_pbar.set_postfix(iter=current_iteration, loss=f"{self.losses['total'].item():.4f}")

                self.end_of_iteration(current_iteration)

            # Epoch 结束时执行所有操作（验证、保存、日志）
            self.end_of_epoch(data, current_epoch + 1, current_iteration)

        # 训练结束，保存最终模型
        self.checkpointer.save(self.checkpoint_path_last, self.cfg.max_epoch, current_iteration)
        print('Done with training!!!')

    @torch.no_grad()
    def test(self, data_loader, mode="test"):
        """The evaluation/inference engine."""
        if self.cfg.trainer.ema_config.enabled:
            model = self.model.averaged_model
        else:
            model = self.model_module
        model.eval()
        data_loader = tqdm(data_loader, desc="Evaluating", leave=False)
        data_batches = []
        for it, data in enumerate(data_loader):
            data = self.start_of_iteration(data, current_iteration=self.current_iteration)
            output = model.inference(data)
            data.update(output)
            data_batches.append(data)
        # 合并所有批次的数据
        data_all = collate_test_data_batches(data_batches)
        tqdm.write(f"Evaluating with {len(data_all['idx'])} samples.")
        # Validate/test.
        if mode == "val":
            self._compute_loss(data_all, mode=mode)
            _ = self._get_total_loss()
        return data_all

    def _get_total_loss(self):
        r"""Return the total loss to be backpropagated.
        """
        total_loss = torch.tensor(0., device=torch.device('cuda'))
        # Iterates over all possible losses.
        for loss_name in self.weights:
            if loss_name in self.losses:
                # Multiply it with the corresponding weight and add it to the total loss.
                total_loss += self.losses[loss_name] * self.weights[loss_name]
        self.losses['total'] = total_loss  # logging purpose
        return total_loss

    def _detach_losses(self):
        r"""Detach all logging variables to prevent potential memory leak."""
        for loss_name in self.losses:
            self.losses[loss_name] = self.losses[loss_name].detach()

    def finalize(self):
        # Finish the TensorBoard logger.
        if hasattr(self, 'tensorboard_writer') and self.tensorboard_writer is not None:
            self.tensorboard_writer.close()

    def init_losses(self):
        r"""Initialize loss functions."""
        self.losses = dict()
        self.criteria = torch.nn.ModuleDict()
        self.weights = dict()

        # Extract loss weights from config (supports both dict and class formats)
        loss_weight_obj = self.cfg.trainer.loss_weight
        if hasattr(loss_weight_obj, 'items'):
            self.weights = {k: v for k, v in loss_weight_obj.items() if v}
        else:
            self.weights = {k: getattr(loss_weight_obj, k) for k in dir(loss_weight_obj)
                           if not k.startswith('_') and not callable(getattr(loss_weight_obj, k))
                           and getattr(loss_weight_obj, k)}

        for loss_name, loss_weight in self.weights.items():
            print("Loss {:<20} Weight {}".format(loss_name, loss_weight))

    def _compute_loss(self, data, mode=None):
        if mode == "train":
            # Compute loss only on randomly sampled rays.
            self.losses["render"] = self.criteria["render"](data["rgb"], data["image_sampled"]) * 3
            self.metrics["psnr"] = -10 * torch_F.mse_loss(data["rgb"], data["image_sampled"]).log10()
            if "eikonal" in self.weights.keys():
                self.losses["eikonal"] = eikonal_loss(data["gradients"], outside=data["outside"])
            if "curvature" in self.weights:
                self.losses["curvature"] = curvature_loss(data["hessians"], outside=data["outside"])
        else:
            # Compute loss on the entire image.
            self.losses["render"] = self.criteria["render"](data["rgb_map"], data["image"])
            self.metrics["psnr"] = -10 * torch_F.mse_loss(data["rgb_map"], data["image"]).log10()

    def get_curvature_weight(self, current_iteration, init_weight):
        if "curvature" in self.weights:
            if current_iteration <= self.warm_up_end:
                self.weights["curvature"] = current_iteration / self.warm_up_end * init_weight
            else:
                model = self.model_module
                decay_factor = model.neural_sdf.growth_rate ** (model.neural_sdf.anneal_levels - 1)
                self.weights["curvature"] = init_weight / decay_factor

    def log_tensorboard_scalars(self, data, mode=None):
        if not hasattr(self, 'tensorboard_writer') or self.tensorboard_writer is None:
            return
        # Log scalars (basic info & losses).
        if mode == "train":
            self.tensorboard_writer.add_scalar("optim/lr", self.sched.get_last_lr()[0], self.current_iteration)
        for key, value in self.losses.items():
            if isinstance(value, torch.Tensor):
                self.tensorboard_writer.add_scalar(f"{mode}/loss/{key}", value.item() if value.numel() == 1 else value.mean().item(), self.current_iteration)
            else:
                self.tensorboard_writer.add_scalar(f"{mode}/loss/{key}", value, self.current_iteration)
        self.tensorboard_writer.add_scalar("iteration", self.current_iteration, self.current_iteration)
        self.tensorboard_writer.add_scalar("epoch", self.current_epoch, self.current_iteration)
        self.tensorboard_writer.add_scalar(f"{mode}/PSNR", self.metrics["psnr"].detach().item(), self.current_iteration)
        self.tensorboard_writer.add_scalar(f"{mode}/s-var", self.model_module.s_var.item(), self.current_iteration)
        if "curvature" in self.weights:
            self.tensorboard_writer.add_scalar(f"{mode}/curvature_weight", self.weights["curvature"], self.current_iteration)
        if "eikonal" in self.weights:
            self.tensorboard_writer.add_scalar(f"{mode}/eikonal_weight", self.weights["eikonal"], self.current_iteration)
        if mode == "train":
            self.tensorboard_writer.add_scalar(f"{mode}/epsilon", self.model_module.neural_sdf.normal_eps, self.current_iteration)
        self.tensorboard_writer.add_scalar(f"{mode}/active_levels", self.model_module.neural_sdf.active_levels, self.current_iteration)

    def log_tensorboard_images(self, data, mode=None, max_samples=None):
        trim_test_samples(data, max_samples=max_samples)

        if not hasattr(self, 'tensorboard_writer') or self.tensorboard_writer is None:
            return
        if mode == "val":
            images_error = (data["rgb_map"] - data["image"]).abs()
            self.tensorboard_writer.add_image(f"{mode}/vis/rgb_target", tensorboard_image(data["image"]), self.current_iteration)
            self.tensorboard_writer.add_image(f"{mode}/vis/rgb_render", tensorboard_image(data["rgb_map"]), self.current_iteration)
            self.tensorboard_writer.add_image(f"{mode}/vis/rgb_error", tensorboard_image(images_error), self.current_iteration)
            self.tensorboard_writer.add_image(f"{mode}/vis/normal", tensorboard_image(data["normal_map"], from_range=(-1, 1)), self.current_iteration)
            self.tensorboard_writer.add_image(f"{mode}/vis/inv_depth", tensorboard_image(1 / (data["depth_map"] + 1e-8) * self.cfg.trainer.depth_vis_scale), self.current_iteration)
            self.tensorboard_writer.add_image(f"{mode}/vis/opacity", tensorboard_image(data["opacity_map"]), self.current_iteration)
