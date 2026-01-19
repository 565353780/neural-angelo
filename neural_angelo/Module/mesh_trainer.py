import os
import torch
import trimesh
import inspect
import warp as wp
import numpy as np
import torch.nn.functional as torch_F

from tqdm import tqdm
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from neural_angelo.Util.init_weight import weights_init, weights_rescale
from neural_angelo.Util.misc import to_cuda, requires_grad
from neural_angelo.Util.visualization import tensorboard_image
from neural_angelo.Util import nerf_util, render

from neural_angelo.Dataset.dataloader import get_train_dataloader, get_val_dataloader
from neural_angelo.Model.model import Model
from neural_angelo.Loss.eikonal import eikonal_loss
from neural_angelo.Loss.curvature import curvature_loss
from neural_angelo.Method.io import loadMeshFile
from neural_angelo.Method.time import getCurrentTime
from neural_angelo.Method.cudnn import init_cudnn
from neural_angelo.Module.checkpointer import Checkpointer
from neural_angelo.Module.model_average import ModelAverage


# 1. 初始化 Warp
wp.init()

@wp.kernel
def compute_sdf_kernel(
    mesh: wp.uint64,                 # Mesh 句柄
    query_points: wp.array(dtype=wp.vec3),
    out_sdf: wp.array(dtype=float),
    out_gradients: wp.array(dtype=wp.vec3)  # 可选：SDF 梯度（即方向）
):
    tid = wp.tid()
    p = query_points[tid]

    # max_dist 设置为一个足够大的数
    # MeshQueryPoint 包含: result, face, u, v, sign
    query_res = wp.mesh_query_point(mesh, p, 1.0e6)

    if query_res.result:
        # 使用重心坐标计算最近点位置
        face_idx = query_res.face
        u = query_res.u
        v = query_res.v
        closest_p = wp.mesh_eval_position(mesh, face_idx, u, v)

        # 计算距离 (Unsigned)
        dist = wp.length(p - closest_p)

        # 使用 query_res.sign 直接获取符号（正数表示外部，负数表示内部）
        sdf_val = query_res.sign * dist

        out_sdf[tid] = sdf_val

        # 计算方向向量
        diff = p - closest_p

        # 计算梯度（SDF 的导数就是指向表面的单位向量）
        if dist > 1e-6:
            out_gradients[tid] = wp.normalize(diff)
        else:
            # 在表面上直接使用法线
            normal = wp.mesh_eval_face_normal(mesh, face_idx)
            out_gradients[tid] = normal

    else:
        # 如果超出 max_dist
        out_sdf[tid] = 1.0e6
        out_gradients[tid] = wp.vec3(0.0, 0.0, 0.0)


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

    def __init__(self, cfg, device: str='cuda:0'):
        print('Setup trainer.')
        # 初始化 cuDNN
        init_cudnn(deterministic=False, benchmark=True)

        self.cfg = cfg
        self.device = device

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
        model = model.to(self.device)
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

    def _get_trainable_params(self, freeze_sdf: bool = False):
        """根据 freeze_sdf 参数获取可训练的参数列表。

        Args:
            freeze_sdf: 是否冻结 SDF 网络参数。
                - False: 返回所有模型参数（正常训练模式）
                - True: 冻结 SDF 参数，只返回 RGB 相关网络参数

        Returns:
            list: 可训练参数列表
        """
        if not freeze_sdf:
            # 训练所有参数
            return list(self.model.parameters())

        # 冻结 SDF，只训练 RGB 相关网络
        neural_sdf = self.model_module.neural_sdf
        neural_rgb = self.model_module.neural_rgb
        background_nerf = self.model_module.background_nerf

        # 冻结 SDF 参数
        for param in neural_sdf.parameters():
            param.requires_grad = False
        neural_sdf.eval()

        # 确保 RGB 相关网络可训练
        neural_rgb.train()
        rgb_params = list(neural_rgb.parameters())

        if background_nerf is not None:
            background_nerf.train()
            rgb_params.extend(background_nerf.parameters())

        # 也训练 s_var（SDF 转 alpha 的参数）
        rgb_params.append(self.model_module.s_var)

        # 如果有 appearance embedding，也加入训练
        if self.model_module.appear_embed is not None:
            rgb_params.extend(self.model_module.appear_embed.parameters())
        if self.model_module.appear_embed_outside is not None:
            rgb_params.extend(self.model_module.appear_embed_outside.parameters())

        frozen_count = sum(p.numel() for p in neural_sdf.parameters())
        trainable_count = sum(p.numel() for p in rgb_params if p.requires_grad)
        print(f'[INFO] freeze_sdf=True: Trainable params: {trainable_count:,}, Frozen SDF params: {frozen_count:,}')

        return rgb_params

    def _setup_optimizer_for_train(self, params, freeze_sdf: bool = False):
        """为 train() 创建优化器。

        Args:
            params: 要优化的参数列表
            freeze_sdf: 是否冻结 SDF（影响学习率等配置）

        Returns:
            AdamW: 优化器实例
        """
        if freeze_sdf:
            # 冻结 SDF 时使用独立配置
            optim = AdamW(
                params=params,
                lr=1e-4,
                weight_decay=1e-6,
            )
        else:
            # 正常训练使用配置文件中的参数
            optim = AdamW(
                params=params,
                lr=self.cfg.optim.params.lr,
                weight_decay=self.cfg.optim.params.weight_decay,
            )

        self.optim_zero_grad_kwargs = {}
        if 'set_to_none' in inspect.signature(optim.zero_grad).parameters:
            self.optim_zero_grad_kwargs['set_to_none'] = True

        return optim

    def _setup_scheduler_for_train(self, optim, freeze_sdf: bool = False):
        """为 train() 创建学习率调度器。

        Args:
            optim: 优化器实例
            freeze_sdf: 是否冻结 SDF（影响调度策略）

        Returns:
            LambdaLR: 学习率调度器实例
        """
        if freeze_sdf:
            # 冻结 SDF 时使用余弦退火调度
            max_epochs = self.cfg.max_epoch

            def lr_schedule(epoch):
                if epoch < 100:
                    return epoch / 100  # warm up
                else:
                    progress = (epoch - 100) / max(max_epochs - 100, 1)
                    return 0.5 * (1 + np.cos(np.pi * progress))

            return LambdaLR(optim, lr_lambda=lr_schedule)
        else:
            # 正常训练使用配置文件中的调度策略
            return self.setup_scheduler(optim)

    def _restore_sdf_trainable(self):
        """恢复 SDF 网络参数为可训练状态。"""
        neural_sdf = self.model_module.neural_sdf
        for param in neural_sdf.parameters():
            param.requires_grad = True

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

        # Use the device from trainer, extract device index if needed
        if isinstance(self.device, str):
            # Extract base device name (e.g., 'cuda:0' -> 'cuda')
            scaler_device = self.device.split(':')[0] if ':' in self.device else self.device
        else:
            scaler_device = str(self.device).split(':')[0] if ':' in str(self.device) else str(self.device)
        self.scaler = GradScaler(scaler_device, **scaler_kwargs)

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

    def fitMeshSDF(
        self,
        mesh: trimesh.Trimesh,
        sample_point_num: int = 2048,
        lr: float = 1e-4,
        log_interval: int = 100,
        patience: int = 50,
        min_delta: float = 1e-6,
        max_epochs_per_level: int = 500,
    ) -> bool:
        """使用自适应coarse-to-fine策略训练NeuralSDF网络拟合三角网格的SDF场。

        自适应策略：从初始level开始拟合，当loss连续patience个epoch不再下降时，
        自动增加level，直到最深层级拟合完毕。

        Args:
            mesh (trimesh.Trimesh): 输入的三角网格对象。
            sample_point_num (int): 每轮采样的点数，默认8192。
            lr (float): 学习率，默认1e-4。
            log_interval (int): 日志打印间隔，默认100。
            patience (int): 连续多少个epoch loss不下降则增加层级，默认50。
            min_delta (float): 判断loss下降的最小阈值，默认1e-6。
            max_epochs_per_level (int): 每个层级最多训练的epoch数，防止卡住，默认500。

        Returns:
            bool: 训练是否成功。
        """
        # 将 PyTorch 设备转换为 Warp 设备格式（字符串）
        warp_device = str(self.device) if not isinstance(self.device, str) else self.device

        # 创建 Warp Mesh 对象 (注意：points 需要 dtype=wp.vec3，indices 需要扁平化)
        wp_mesh = wp.Mesh(
            points=wp.array(mesh.vertices.astype(np.float32), dtype=wp.vec3, device=warp_device),
            indices=wp.array(mesh.faces.flatten().astype(np.int32), device=warp_device)
        )

        # 获取 mesh 的边界盒，用于采样点的范围
        bbox_min = mesh.bounds[0]
        bbox_max = mesh.bounds[1]
        bbox_center = (bbox_min + bbox_max) / 2.0
        bbox_extent = (bbox_max - bbox_min).max() / 2.0 * 1.2  # 扩展 20% 边界

        def sample_sdf_with_warp(num_points: int) -> tuple:
            """使用 Warp 查询采样点的 SDF 值。

            采样策略：
            - 50% 点在表面附近采样（使用 mesh 表面采样 + 扰动）
            - 50% 点在边界盒内均匀采样

            Args:
                num_points: 采样点数量

            Returns:
                points_np: 采样点坐标 [N, 3]
                sdf_np: SDF 值 [N]
            """
            # 表面附近采样
            near_surface_num = num_points // 2
            surface_points = mesh.sample(near_surface_num)
            # 添加高斯噪声，标准差为边界盒大小的 1-5%
            noise_std = np.random.uniform(0.01, 0.05) * bbox_extent
            near_surface_points = surface_points + np.random.randn(near_surface_num, 3) * noise_std

            # 边界盒内均匀采样
            uniform_num = num_points - near_surface_num
            uniform_points = (np.random.rand(uniform_num, 3) - 0.5) * 2.0 * bbox_extent + bbox_center

            # 合并采样点
            query_points_np = np.vstack([near_surface_points, uniform_points]).astype(np.float32)

            # 使用 Warp 查询 SDF
            query_points_wp = wp.array(query_points_np, dtype=wp.vec3, device=warp_device)
            out_sdf_wp = wp.zeros(num_points, dtype=float, device=warp_device)
            out_grad_wp = wp.zeros(num_points, dtype=wp.vec3, device=warp_device)

            wp.launch(
                kernel=compute_sdf_kernel,
                dim=num_points,
                inputs=[wp_mesh.id, query_points_wp, out_sdf_wp, out_grad_wp],
                device=warp_device
            )

            wp.synchronize_device(warp_device)

            sdf_np = out_sdf_wp.numpy()
            return query_points_np, sdf_np

        # 获取NeuralSDF模块和coarse2fine配置
        neural_sdf = self.model_module.neural_sdf
        init_active_level = self.cfg.model.object.sdf.encoding.coarse2fine.init_active_level
        #total_levels = self.cfg.model.object.sdf.encoding.levels
        total_levels = 6

        print('[INFO][Trainer::fitMeshSDF]')
        print(f'\t Adaptive Coarse-to-fine SDF fitting:')
        print(f'\t   - init_active_level: {init_active_level}')
        print(f'\t   - total_levels: {total_levels}')
        print(f'\t   - patience: {patience}')
        print(f'\t   - min_delta: {min_delta}')
        print(f'\t   - max_epochs_per_level: {max_epochs_per_level}')
        print(f'\t   - sample_point_num: {sample_point_num}')

        # 设置初始active levels（从低分辨率开始）
        neural_sdf.active_levels = init_active_level
        neural_sdf.anneal_levels = init_active_level
        neural_sdf.set_normal_epsilon()

        # 创建专用的优化器，只优化NeuralSDF的参数
        sdf_optimizer = AdamW(
            params=neural_sdf.parameters(),
            lr=lr,
            weight_decay=1e-6,
        )

        # 定义损失函数
        sdf_loss_fn = torch.nn.L1Loss()

        # 训练状态变量
        neural_sdf.train()
        current_level = init_active_level
        total_epoch = 0

        # 逐层训练
        while current_level <= total_levels:
            print(f'\n[INFO] Training level {current_level}/{total_levels}')

            # 设置当前层级
            neural_sdf.active_levels = current_level
            neural_sdf.anneal_levels = current_level
            neural_sdf.set_normal_epsilon()

            # 当前层级的训练状态
            best_loss = float('inf')
            epochs_without_improvement = 0
            level_epoch = 0

            pbar = tqdm(desc=f"Level {current_level}/{total_levels}", leave=False)

            while epochs_without_improvement < patience and level_epoch < max_epochs_per_level:
                # 每轮使用 Warp 重新采样点和 SDF 值
                points_np, sdf_np = sample_sdf_with_warp(sample_point_num)

                # 转换为tensor并移动到指定设备
                points = torch.from_numpy(points_np).float().to(self.device)  # [N, 3]
                sdf_gt = torch.from_numpy(sdf_np).float().to(self.device)  # [N]

                # 前向传播：预测SDF值
                sdf_pred, _ = neural_sdf.forward(points, with_sdf=True, with_feat=False)  # [N, 1]
                sdf_pred = sdf_pred.squeeze(-1)  # [N]

                # 计算SDF损失
                loss_sdf = sdf_loss_fn(sdf_pred, sdf_gt)

                # 计算Eikonal损失（约束梯度长度为1）
                gradients, _ = neural_sdf.compute_gradients(points, training=False)  # [N, 3]
                grad_norm = gradients.norm(dim=-1)  # [N]
                loss_eikonal = ((grad_norm - 1.0) ** 2).mean()

                # 总损失
                total_loss = loss_sdf + 0.1 * loss_eikonal
                current_loss = total_loss.item()

                # 反向传播
                sdf_optimizer.zero_grad()
                total_loss.backward()
                sdf_optimizer.step()

                # 检查是否有改进
                if current_loss < best_loss - min_delta:
                    best_loss = current_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                # 更新进度条
                pbar.update(1)
                pbar.set_postfix(
                    epoch=level_epoch,
                    loss=f"{current_loss:.6f}",
                    best=f"{best_loss:.6f}",
                    no_improve=epochs_without_improvement
                )

                # 日志打印
                if (level_epoch + 1) % log_interval == 0:
                    print(f'\t level {current_level}, epoch {level_epoch + 1}: '
                          f'loss={current_loss:.6f}, best={best_loss:.6f}, '
                          f'no_improve={epochs_without_improvement}/{patience}')

                    # TensorBoard记录
                    if hasattr(self, 'tensorboard_writer') and self.tensorboard_writer is not None:
                        self.tensorboard_writer.add_scalar("sdf_fit/loss_sdf", loss_sdf.item(), total_epoch)
                        self.tensorboard_writer.add_scalar("sdf_fit/loss_eikonal", loss_eikonal.item(), total_epoch)
                        self.tensorboard_writer.add_scalar("sdf_fit/total_loss", total_loss.item(), total_epoch)
                        self.tensorboard_writer.add_scalar("sdf_fit/active_levels", current_level, total_epoch)
                        self.tensorboard_writer.add_scalar("sdf_fit/best_loss", best_loss, total_epoch)

                level_epoch += 1
                total_epoch += 1

            pbar.close()

            # 打印当前层级的训练结果
            if epochs_without_improvement >= patience:
                print(f'\t [C2F] Level {current_level} converged after {level_epoch} epochs (patience reached)')
            else:
                print(f'\t [C2F] Level {current_level} reached max epochs ({max_epochs_per_level})')
            print(f'\t       Best loss: {best_loss:.6f}')

            # 增加层级
            current_level += 1

        print('\n[INFO][Trainer::fitMeshSDF]')
        print(f'\t Adaptive coarse-to-fine SDF fitting completed!')
        print(f'\t Total epochs: {total_epoch}, Final level: {neural_sdf.active_levels}')
        return True

    def fitAll(
        self,
        mesh: trimesh.Trimesh,
        # SDF fitting 参数
        sdf_sample_point_num: int = 2048,
        sdf_lr: float = 1e-4,
        sdf_patience: int = 50,
        sdf_min_delta: float = 1e-6,
        sdf_max_epochs_per_level: int = 500,
        # 联合训练参数
        run_joint_training: bool = True,
        log_interval: int = 100,
    ) -> bool:
        """完整的三阶段训练流程：SDF -> RGB -> 联合优化。

        这个函数按顺序执行：
        1. fitMeshSDF: 使用 mesh 的 SDF 值训练 NeuralSDF 网络
        2. train(freeze_sdf=True): 冻结 SDF，训练 NeuralRGB 和 BackgroundNeRF
        3. train(freeze_sdf=False): 联合优化所有网络（可选）

        Args:
            mesh (trimesh.Trimesh): 输入的三角网格对象。

            SDF fitting 参数:
                sdf_sample_point_num (int): 每轮采样的点数，默认2048。
                sdf_lr (float): SDF 学习率，默认1e-4。
                sdf_patience (int): SDF 早停耐心值，默认50。
                sdf_min_delta (float): SDF loss 下降最小阈值，默认1e-6。
                sdf_max_epochs_per_level (int): 每层最大 epoch 数，默认500。

            联合训练参数:
                run_joint_training (bool): 是否运行联合训练阶段，默认True。
                log_interval (int): 日志打印间隔，默认100。

        Returns:
            bool: 训练是否成功。
        """
        print('=' * 60)
        print('[INFO][Trainer::fitAll] Starting complete training pipeline')
        print('=' * 60)

        # ==================== 阶段 1: SDF 训练 ====================
        print('\n' + '=' * 60)
        print('[STAGE 1/3] Training SDF from mesh')
        print('=' * 60)

        sdf_checkpoint_path = os.path.join(self.cfg.logdir, 'model_after_sdf.pt')

        # 检查 SDF 检查点是否存在
        if os.path.exists(sdf_checkpoint_path):
            print(f'[INFO] Found existing SDF checkpoint: {sdf_checkpoint_path}')
            print('[INFO] Loading checkpoint and skipping SDF training...')
            self.load_checkpoint(sdf_checkpoint_path, load_opt=False, load_sch=False)
            print('[INFO] SDF checkpoint loaded successfully, skipping SDF training stage.')
        else:
            print(f'[INFO] SDF checkpoint not found, starting SDF training...')
            success = self.fitMeshSDF(
                mesh,
                sample_point_num=sdf_sample_point_num,
                lr=sdf_lr,
                log_interval=log_interval,
                patience=sdf_patience,
                min_delta=sdf_min_delta,
                max_epochs_per_level=sdf_max_epochs_per_level,
            )

            if not success:
                print('[ERROR][Trainer::fitAll] SDF training failed!')
                return False

            # 保存 SDF 阶段的检查点
            self.checkpointer.save(sdf_checkpoint_path, current_epoch=0, current_iteration=0)
            print(f'[INFO] SDF checkpoint saved to: {sdf_checkpoint_path}')

        # ==================== 阶段 2: RGB 训练（冻结 SDF）====================
        print('\n' + '=' * 60)
        print('[STAGE 2/3] Training RGB and Background with frozen SDF')
        print('=' * 60)

        rgb_checkpoint_path = os.path.join(self.cfg.logdir, 'model_after_rgb.pt')

        # 检查 RGB 检查点是否存在
        if os.path.exists(rgb_checkpoint_path):
            print(f'[INFO] Found existing RGB checkpoint: {rgb_checkpoint_path}')
            print('[INFO] Loading checkpoint and skipping RGB training...')
            self.load_checkpoint(rgb_checkpoint_path, load_opt=False, load_sch=False)
            print('[INFO] RGB checkpoint loaded successfully, skipping RGB training stage.')
        else:
            print(f'[INFO] RGB checkpoint not found, starting RGB training with freeze_sdf=True...')
            # 使用 train(freeze_sdf=True) 冻结 SDF，只训练 RGB 相关网络
            self.train(freeze_sdf=True)

            # 保存 RGB 阶段的检查点
            self.checkpointer.save(rgb_checkpoint_path, current_epoch=0, current_iteration=0)
            print(f'[INFO] RGB checkpoint saved to: {rgb_checkpoint_path}')

        # ==================== 阶段 3: 联合优化（可选）====================
        if run_joint_training:
            print('\n' + '=' * 60)
            print('[STAGE 3/3] Joint optimization of SDF, RGB, and Background')
            print('=' * 60)

            joint_checkpoint_path = os.path.join(self.cfg.logdir, 'model_after_joint.pt')

            # 检查联合训练检查点是否存在
            if os.path.exists(joint_checkpoint_path):
                print(f'[INFO] Found existing joint training checkpoint: {joint_checkpoint_path}')
                print('[INFO] Loading checkpoint and skipping joint training...')
                self.load_checkpoint(joint_checkpoint_path, load_opt=True, load_sch=True)
                print('[INFO] Joint training checkpoint loaded successfully, skipping joint training stage.')
            else:
                print(f'[INFO] Joint training checkpoint not found, starting joint training...')
                # 运行联合训练（train() 内部会创建优化器和调度器，freeze_sdf=False 训练所有参数）
                self.train(freeze_sdf=False)

                # 保存联合训练阶段的检查点
                self.checkpointer.save(joint_checkpoint_path, current_epoch=0, current_iteration=0)
                print(f'[INFO] Joint training checkpoint saved to: {joint_checkpoint_path}')

        print('\n' + '=' * 60)
        print('[INFO][Trainer::fitAll] Complete training pipeline finished!')
        print('=' * 60)

        return True

    def fitMeshFileAll(
        self,
        mesh_file_path: str,
        # SDF fitting 参数
        sdf_sample_point_num: int = 2048,
        sdf_lr: float = 1e-4,
        sdf_patience: int = 50,
        sdf_min_delta: float = 1e-6,
        sdf_max_epochs_per_level: int = 500,
        # 联合训练参数
        run_joint_training: bool = True,
        log_interval: int = 100,
    ) -> bool:
        """从文件加载 mesh 并执行完整的三阶段训练。

        Args:
            mesh_file_path (str): mesh 文件路径。
            其他参数见 fitAll 方法。

        Returns:
            bool: 训练是否成功。
        """
        if not os.path.exists(mesh_file_path):
            print('[ERROR][Trainer::fitMeshFileAll]')
            print('\t mesh file not exist!')
            print('\t mesh_file_path:', mesh_file_path)
            return False

        mesh = loadMeshFile(mesh_file_path)

        return self.fitAll(
            mesh,
            sdf_sample_point_num=sdf_sample_point_num,
            sdf_lr=sdf_lr,
            sdf_patience=sdf_patience,
            sdf_min_delta=sdf_min_delta,
            sdf_max_epochs_per_level=sdf_max_epochs_per_level,
            run_joint_training=run_joint_training,
            log_interval=log_interval,
        )

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
        data = to_cuda(data, device=self.device)
        self.model.train()
        # 如果 freeze_sdf 模式，确保 SDF 保持 eval 模式
        if getattr(self, '_freeze_sdf', False):
            self.model_module.neural_sdf.eval()
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

    def train(self, freeze_sdf: bool = False):
        """训练主循环。

        使用基于 epoch 的训练方式，每个 epoch 包含 iters_per_epoch 次迭代。
        数据通过循环迭代器按顺序获取，确保所有图片都被均匀使用。

        Args:
            freeze_sdf: 是否冻结 SDF 网络参数，只训练 RGB 相关网络。
                - False（默认）: 训练所有参数（完整训练模式）
                - True: 冻结 SDF 参数，只训练 NeuralRGB、BackgroundNeRF、s_var 和 appearance embedding
        """
        # ==================== 根据 freeze_sdf 设置可训练参数和优化器 ====================
        trainable_params = self._get_trainable_params(freeze_sdf)
        self.optim = self._setup_optimizer_for_train(trainable_params, freeze_sdf)
        self.sched = self._setup_scheduler_for_train(self.optim, freeze_sdf)

        # 更新 checkpointer 的引用（因为 optim 和 sched 已更新）
        self.checkpointer = Checkpointer(self.model, self.optim, self.sched)

        # 保存 freeze_sdf 状态，供 train_step 使用
        self._freeze_sdf = freeze_sdf

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

        # ==================== 训练结束，恢复 SDF 参数为可训练状态 ====================
        if freeze_sdf:
            self._restore_sdf_trainable()

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
        total_loss = torch.tensor(0., device=self.device)
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
