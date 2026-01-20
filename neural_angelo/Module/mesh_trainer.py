import os
import torch
import trimesh
import inspect
import warp as wp
import numpy as np

from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from neural_angelo.Method.io import loadMeshFile
from neural_angelo.Module.checkpointer import Checkpointer
from neural_angelo.Module.trainer import Trainer


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


class MeshTrainer(Trainer):
    r"""Trainer class for Neuralangelo training with mesh SDF fitting support.

    继承自 Trainer，增加了以下功能：
    - 支持自定义 device 参数
    - 支持根据 mesh 的 bounding box 自动设置 hashgrid 范围
    - 支持 SDF 拟合和分阶段训练

    Args:
        cfg (obj): Global configuration.
        mesh_file_path (str, optional): 三角网格文件路径。如果提供，会自动根据 mesh 的 
            bounding box 计算 hashgrid 的空间范围（默认为 bbox 的 1.1 倍）。
        device (str): 训练设备，默认为 'cuda:0'。
    """

    def __init__(self, cfg, mesh_file_path: str, device: str = 'cuda:0'):
        # 先保存 mesh_file_path，在父类初始化之前设置 hashgrid 范围
        self.mesh_file_path = mesh_file_path

        # 根据 mesh 的 bbox 自动设置 hashgrid 范围
        self._setup_hashgrid_range_from_mesh(cfg, mesh_file_path)

        # 调用父类初始化（传入 device 参数）
        super().__init__(cfg, device=device)

    def _setup_hashgrid_range_from_mesh(self, cfg, mesh_file_path: str, scale_factor: float = 1.1) -> bool:
        """根据 mesh 的 bounding box 自动设置 hashgrid 的空间范围。

        会将 mesh 的 bbox 扩展 scale_factor 倍（默认 1.1 倍），确保物体表面完全在 
        hashgrid 内部。

        Args:
            cfg: 配置对象。
            mesh_file_path (str): mesh 文件路径。
            scale_factor (float): bbox 扩展因子，默认 1.1（扩展 10%）。

        Returns:
            bool: 设置是否成功。
        """
        if not os.path.exists(mesh_file_path):
            print('[WARN][MeshTrainer::_setup_hashgrid_range_from_mesh]')
            print(f'\t mesh file not exist: {mesh_file_path}')
            print('\t Using default hashgrid range.')
            return False

        mesh = loadMeshFile(mesh_file_path)
        if mesh is None:
            print('[WARN][MeshTrainer::_setup_hashgrid_range_from_mesh]')
            print('\t Failed to load mesh file.')
            print('\t Using default hashgrid range.')
            return False

        # 计算 mesh 的 bounding box
        bbox_min = mesh.bounds[0]  # [3,]
        bbox_max = mesh.bounds[1]  # [3,]
        bbox_center = (bbox_min + bbox_max) / 2.0
        bbox_extent = (bbox_max - bbox_min) / 2.0

        # 扩展 bbox（默认 1.1 倍）
        scaled_extent = bbox_extent * scale_factor
        scaled_min = bbox_center - scaled_extent
        scaled_max = bbox_center + scaled_extent

        # 为了简化，使用各轴的最小/最大值构建一个统一的立方体范围
        # hashgrid.range 期望的是 [min_val, max_val]，所有轴使用相同范围
        # 取所有轴中最大的范围
        range_min = float(scaled_min.min())
        range_max = float(scaled_max.max())

        # 更新配置中的 hashgrid.range
        old_range = cfg.model.object.sdf.encoding.hashgrid.range
        cfg.model.object.sdf.encoding.hashgrid.range = [range_min, range_max]

        print('[INFO][MeshTrainer::_setup_hashgrid_range_from_mesh]')
        print(f'\t Mesh file: {mesh_file_path}')
        print(f'\t Original bbox: min={bbox_min}, max={bbox_max}')
        print(f'\t Bbox center: {bbox_center}')
        print(f'\t Bbox extent: {bbox_extent}')
        print(f'\t Scale factor: {scale_factor}')
        print(f'\t Old hashgrid range: {old_range}')
        print(f'\t New hashgrid range: [{range_min:.4f}, {range_max:.4f}]')

        return True

    def start_of_iteration(self, data, current_iteration):
        r"""Things to do before an iteration. 增加 freeze_sdf 模式支持。"""
        # 调用父类方法处理基本逻辑
        data = super().start_of_iteration(data, current_iteration)
        # 如果 freeze_sdf 模式，确保 SDF 保持 eval 模式
        if getattr(self, '_freeze_sdf', False):
            self.model_module.neural_sdf.eval()
        return data

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

    def train(self, freeze_sdf: bool = False):
        """训练主循环。

        使用基于 epoch 的训练方式，每个 epoch 包含 iters_per_epoch 次迭代。
        数据通过循环迭代器按顺序获取，确保所有图片都被均匀使用。

        Args:
            freeze_sdf: 是否冻结 SDF 网络参数，只训练 RGB 相关网络。
                - False（默认）: 训练所有参数（完整训练模式）
                - True: 冻结 SDF 参数，只训练 NeuralRGB、BackgroundNeRF、s_var 和 appearance embedding
        """
        from neural_angelo.Module.trainer import cycle_dataloader

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
