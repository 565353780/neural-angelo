import os
import torch
import trimesh
import warp as wp
import numpy as np

from neural_angelo.Method.io import loadMeshFile
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
    - 支持基于 Mesh 的 OccupancyGrid 初始化和冻结
    - 支持 SDF 拟合和分阶段训练

    核心功能：基于 Mesh 固定 OccupancyGrid
    =========================================
    当提供了 mesh 文件时，本类会：
    1. **Voxelization (体素化)**：将 Mesh 转换成与 NerfAcc Grid 分辨率一致的 3D Boolean Tensor
    2. **Inject (注入)**：将这个 Tensor 强行赋值给 NerfAcc 的 OccGridEstimator
    3. **Freeze (冻结)**：在训练循环中，不再调用 estimator.update()

    重要说明：
    - Mesh 需要预处理到与配置中的空间范围一致（例如 [-1, 1]）
    - HashGrid 和 NerfAcc OccupancyGrid 会使用配置中的统一空间范围
    - 这样可以保证体素化和渲染的空间对齐

    优点：
    - 彻底根除"破碎"/"空洞"问题
    - 渲染速度极快
    - 防止"云雾"伪影

    Args:
        cfg (obj): Global configuration.
        mesh_file_path (str): 三角网格文件路径。Mesh 应该已经预处理到配置的空间范围内。
        device (str): 训练设备，默认为 'cuda:0'。
        freeze_occ_grid (bool): 是否冻结 OccupancyGrid，默认为 True。
        occ_margin_voxels (float): OccupancyGrid 的 margin（以体素为单位），默认为 2.0。
            这会在 Mesh 表面附近保留一个"缓冲带"，给 NeRF 模拟毛发、布料等的空间。
    """

    def __init__(
        self,
        cfg,
        mesh_file_path: str,
        device: str = 'cuda:0',
        freeze_occ_grid: bool = True,
        occ_margin_voxels: float = 2.0,
        sdf_loss_weight: float = 1.0,
    ):
        # 保存参数
        self.mesh_file_path = mesh_file_path
        self.freeze_occ_grid = freeze_occ_grid
        self.occ_margin_voxels = occ_margin_voxels
        self._occ_grid_initialized = False
        self._mesh = None

        # SDF 损失参数
        self.sdf_loss_weight_init = sdf_loss_weight
        self.sdf_loss_weight = sdf_loss_weight

        # 预加载 mesh（不修改任何配置）
        self._load_mesh(mesh_file_path)

        # 初始化 Warp mesh（用于 SDF 计算）
        self._wp_mesh = None
        self._init_warp_mesh()

        # 调用父类初始化（传入 device 参数）
        super().__init__(cfg, device=device)

        # 父类初始化后，设置基于 Mesh 的 OccupancyGrid
        if self.freeze_occ_grid:
            self._setup_frozen_occupancy_grid_from_mesh()

    def _load_mesh(self, mesh_file_path: str) -> bool:
        """加载 Mesh 文件（不修改任何空间范围配置）。

        Mesh 应该已经预处理到配置的空间范围内（例如 [-1, 1]）。
        这样可以保证 HashGrid 和 NerfAcc OccupancyGrid 使用相同的空间范围。

        Args:
            mesh_file_path (str): mesh 文件路径。

        Returns:
            bool: 是否成功加载。
        """
        if not os.path.exists(mesh_file_path):
            print('[WARN][MeshTrainer::_load_mesh]')
            print(f'\t Mesh file not exist: {mesh_file_path}')
            return False

        mesh = loadMeshFile(mesh_file_path)
        if mesh is None:
            print('[WARN][MeshTrainer::_load_mesh]')
            print('\t Failed to load mesh file.')
            return False

        # 保存 mesh 供后续使用
        self._mesh = mesh

        # 打印 mesh 的 bounding box 信息（仅供参考，不修改配置）
        bbox_min = mesh.bounds[0]
        bbox_max = mesh.bounds[1]
        bbox_center = (bbox_min + bbox_max) / 2.0

        print('[INFO][MeshTrainer::_load_mesh]')
        print(f'\t Mesh file: {mesh_file_path}')
        print(f'\t Mesh bbox: min={bbox_min}, max={bbox_max}')
        print(f'\t Mesh center: {bbox_center}')
        print('\t Note: Mesh should be pre-transformed to match the config space range.')

        return True

    def _init_warp_mesh(self):
        """初始化 Warp mesh 用于快速 SDF 计算。"""
        if self._mesh is None:
            return

        vertices = self._mesh.vertices.astype(np.float32)
        faces = self._mesh.faces.astype(np.int32).flatten()

        self._wp_mesh = wp.Mesh(
            points=wp.array(vertices, dtype=wp.vec3),
            indices=wp.array(faces, dtype=int)
        )

    def _get_scene_aabb_from_config(self) -> torch.Tensor:
        """从 config 中获取场景 AABB。

        统一使用 hashgrid.range 来设置 AABB，确保 NerfAcc OccupancyGrid 
        与 HashGrid 使用相同的空间范围。

        Returns:
            aabb: [6] tensor [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        # 统一使用 hashgrid.range（与 model.py 中 _build_nerfacc 保持一致）
        hashgrid_range = self.cfg.model.object.sdf.encoding.hashgrid.range
        aabb = torch.tensor([
            hashgrid_range[0], hashgrid_range[0], hashgrid_range[0],  # x_min, y_min, z_min
            hashgrid_range[1], hashgrid_range[1], hashgrid_range[1]   # x_max, y_max, z_max
        ], dtype=torch.float32, device=self.device)

        return aabb

    def _setup_frozen_occupancy_grid_from_mesh(self) -> bool:
        """基于 Mesh 初始化并冻结 OccupancyGrid。

        核心步骤：
        1. 从 config 获取 hashgrid.range 作为场景 AABB（保证与 HashGrid 对齐）
        2. 重新创建 OccGridEstimator 使用正确的 AABB
        3. 使用 Warp 计算每个体素中心到 Mesh 的 SDF
        4. 根据 SDF 生成 binary occupancy mask（使用 margin 扩张）
        5. 将 mask 注入到 NerfAcc 的 OccGridEstimator

        Returns:
            bool: 是否成功初始化。
        """
        from neural_angelo.Util.nerfacc_util import NerfAccEstimator

        # 检查模型是否启用了 nerfacc
        if not hasattr(self.model_module, 'use_nerfacc') or not self.model_module.use_nerfacc:
            print('[WARN][MeshTrainer::_setup_frozen_occupancy_grid_from_mesh]')
            print('\t NerfAcc is not enabled in the model.')
            print('\t Skipping frozen OccupancyGrid setup.')
            return False

        if not hasattr(self.model_module, 'estimator') or self.model_module.estimator is None:
            print('[WARN][MeshTrainer::_setup_frozen_occupancy_grid_from_mesh]')
            print('\t No estimator found in the model.')
            return False

        # 检查是否有加载的 mesh
        if not hasattr(self, '_mesh') or self._mesh is None:
            mesh = loadMeshFile(self.mesh_file_path)
            if mesh is None:
                print('[WARN][MeshTrainer::_setup_frozen_occupancy_grid_from_mesh]')
                print('\t Failed to load mesh file.')
                return False
            self._mesh = mesh

        print('[INFO][MeshTrainer::_setup_frozen_occupancy_grid_from_mesh]')
        print('\t Setting up frozen OccupancyGrid from mesh...')

        # ========== 获取场景 AABB (优先使用 radius) ==========
        scene_aabb = self._get_scene_aabb_from_config()
        
        # 获取 Grid 分辨率
        resolution = self.cfg.model.nerfacc.occ_grid.resolution
        
        hashgrid_range = self.cfg.model.object.sdf.encoding.hashgrid.range
        print(f'\t Using hashgrid.range: {hashgrid_range}')
        print(f'\t Scene AABB: {scene_aabb.tolist()}')
        print(f'\t Grid resolution: {resolution}')

        # ========== 重新创建 estimator，使用正确的 AABB ==========
        # 创建新的 OccGridEstimator，AABB 与 hashgrid.range 对齐
        new_estimator = NerfAccEstimator(
            aabb=scene_aabb,
            resolution=resolution,
            device=self.device
        )
        
        # 替换 model 中的 estimator
        old_aabb = self.model_module.estimator.aabb.tolist() if hasattr(self.model_module.estimator, 'aabb') else 'N/A'
        self.model_module.estimator = new_estimator
        
        # 同时更新 model 的 scene_aabb buffer
        if hasattr(self.model_module, 'scene_aabb'):
            self.model_module.scene_aabb = scene_aabb
        
        print(f'\t Old estimator AABB: {old_aabb}')
        print(f'\t New estimator AABB: {scene_aabb.tolist()}')

        # 获取内部的 OccGridEstimator
        estimator = new_estimator.estimator

        # ========== 生成基于 Mesh 的 occupancy grid ==========
        aabb_np = scene_aabb.cpu().numpy()
        binary_grid = self._compute_mesh_occupancy_grid(
            mesh=self._mesh,
            aabb=aabb_np,
            resolution=resolution,
            margin_voxels=self.occ_margin_voxels,
            estimator=estimator
        )

        # ========== 注入到 NerfAcc estimator ==========
        self._inject_occupancy_grid(estimator, binary_grid)

        self._occ_grid_initialized = True

        print('[INFO][MeshTrainer::_setup_frozen_occupancy_grid_from_mesh]')
        print(f'\t Grid resolution: {resolution}')
        print(f'\t Margin (voxels): {self.occ_margin_voxels}')
        print(f'\t Occupied voxels: {binary_grid.sum().item()} / {binary_grid.numel()}')
        print(f'\t Occupancy rate: {binary_grid.float().mean().item():.4f}')
        print(f'\t OccupancyGrid has been frozen (using hashgrid.range={hashgrid_range}).')

        return True

    def _compute_mesh_occupancy_grid(
        self,
        mesh: trimesh.Trimesh,
        aabb: np.ndarray,
        resolution: int,
        margin_voxels: float = 2.0,
        estimator=None,
    ) -> torch.Tensor:
        """计算基于 Mesh 的 occupancy grid（与 NerfAcc 对齐）。

        Args:
            mesh: trimesh 网格对象。
            aabb: [6] 场景边界 [x_min, y_min, z_min, x_max, y_max, z_max]。
            resolution: Grid 分辨率。
            margin_voxels: 扩张的体素数量。
            estimator: NerfAcc OccGridEstimator 实例（可选，用于直接获取对齐的坐标）。

        Returns:
            binary_grid: [resolution^3] 的 bool tensor。
        """
        device = self.device

        # 计算体素大小
        aabb_min = aabb[:3]
        aabb_max = aabb[3:]
        voxel_size = (aabb_max - aabb_min).max() / resolution

        # 计算 margin（以实际距离为单位）
        margin = voxel_size * margin_voxels

        # 生成体素中心坐标（与 NerfAcc 对齐）
        if estimator is not None:
            # 直接从 estimator 获取坐标（最准确），传递已知的 resolution
            grid_coords = self._make_grid_coords_from_estimator(estimator, resolution=resolution)
        else:
            # 使用手动生成的坐标（确保与 NerfAcc 对齐）
            grid_coords = self._make_grid_coords(aabb, resolution)

        # 使用 Warp 计算 SDF
        sdf_values = self._compute_sdf_with_warp(mesh, grid_coords.cpu())  # [N]

        # 生成 binary mask：如果 abs(SDF) < margin，则认为该体素被占用
        # 注意：SDF 内部是负数，外部是正数
        binary_grid = (sdf_values.abs() < margin)

        print(f'[DEBUG] Grid coords shape: {grid_coords.shape}')
        print(f'[DEBUG] Grid coords range: x=[{grid_coords[:, 0].min():.4f}, {grid_coords[:, 0].max():.4f}]')
        print(f'[DEBUG]                    y=[{grid_coords[:, 1].min():.4f}, {grid_coords[:, 1].max():.4f}]')
        print(f'[DEBUG]                    z=[{grid_coords[:, 2].min():.4f}, {grid_coords[:, 2].max():.4f}]')
        print(f'[DEBUG] Voxel size: {voxel_size:.6f}')
        print(f'[DEBUG] Margin distance: {margin:.6f}')
        print(f'[DEBUG] SDF range: [{sdf_values.min().item():.4f}, {sdf_values.max().item():.4f}]')

        return binary_grid.to(device)

    def _make_grid_coords_from_estimator(self, estimator, resolution: int = None) -> torch.Tensor:
        """从 NerfAcc estimator 获取所有体素中心坐标。

        这是最准确的方法，直接使用 NerfAcc 内部的坐标生成逻辑。

        Args:
            estimator: NerfAcc OccGridEstimator 实例。
            resolution: Grid 分辨率（可选）。如果提供，则使用此值；否则从 estimator 获取。

        Returns:
            coords: [N, 3] 的坐标 tensor。
        """
        # 获取 AABB
        aabb = estimator.aabbs[0]  # [6] - 第一个 level 的 AABB
        
        # 获取分辨率：优先使用传入的参数，否则从 estimator 获取
        if resolution is None:
            resolution = estimator.resolution
            # 确保 resolution 是 int 类型
            if isinstance(resolution, torch.Tensor):
                # 如果是 tensor，尝试获取标量值
                if resolution.numel() == 1:
                    resolution = int(resolution.item())
                else:
                    # 如果是多元素 tensor，取第一个元素
                    resolution = int(resolution[0].item())
            else:
                resolution = int(resolution)
        else:
            # 确保传入的 resolution 是 int
            resolution = int(resolution)

        # NerfAcc 内部使用的坐标生成方式：
        # 1. 生成整数索引 [0, resolution-1]
        # 2. 归一化到 [0, 1]: (idx + 0.5) / resolution
        # 3. 映射到 AABB: aabb_min + normalized * (aabb_max - aabb_min)

        aabb_min = aabb[:3]
        aabb_max = aabb[3:]

        # 生成整数索引（与 NerfAcc 一致的顺序：x, y, z）
        # NerfAcc 使用 C-order flatten，所以 z 变化最快
        indices = torch.arange(resolution, dtype=torch.float32, device=aabb.device)

        # 创建 3D 网格 - 使用 indexing='ij' 确保与 C-order 一致
        # xx: [res, res, res], x 变化最慢
        # zz: [res, res, res], z 变化最快
        xx, yy, zz = torch.meshgrid(indices, indices, indices, indexing='ij')

        # 归一化到体素中心 [0.5/res, (res-0.5)/res]
        resolution_float = float(resolution)
        xx = (xx + 0.5) / resolution_float
        yy = (yy + 0.5) / resolution_float
        zz = (zz + 0.5) / resolution_float

        # 映射到实际坐标
        coords_x = aabb_min[0] + xx * (aabb_max[0] - aabb_min[0])
        coords_y = aabb_min[1] + yy * (aabb_max[1] - aabb_min[1])
        coords_z = aabb_min[2] + zz * (aabb_max[2] - aabb_min[2])

        # 展平并堆叠（按 C-order：x 最慢，z 最快）
        coords = torch.stack([
            coords_x.flatten(),
            coords_y.flatten(),
            coords_z.flatten()
        ], dim=-1)  # [resolution^3, 3]

        return coords

    def _make_grid_coords(self, aabb: np.ndarray, resolution: int) -> torch.Tensor:
        """生成 Grid 体素中心坐标（与 NerfAcc 对齐）。

        Args:
            aabb: [6] 场景边界 [x_min, y_min, z_min, x_max, y_max, z_max]。
            resolution: Grid 分辨率。

        Returns:
            coords: [resolution^3, 3] 的坐标 tensor。
        """
        aabb_min = torch.tensor(aabb[:3], dtype=torch.float32)
        aabb_max = torch.tensor(aabb[3:], dtype=torch.float32)

        # 生成整数索引 [0, resolution-1]
        indices = torch.arange(resolution, dtype=torch.float32)

        # 创建 3D 网格 - 使用 indexing='ij' 确保与 C-order 一致
        # NerfAcc 内部 flatten 顺序：x 变化最慢，z 变化最快
        xx, yy, zz = torch.meshgrid(indices, indices, indices, indexing='ij')

        # 归一化到体素中心：(idx + 0.5) / resolution
        xx = (xx + 0.5) / resolution
        yy = (yy + 0.5) / resolution
        zz = (zz + 0.5) / resolution

        # 映射到实际坐标
        coords_x = aabb_min[0] + xx * (aabb_max[0] - aabb_min[0])
        coords_y = aabb_min[1] + yy * (aabb_max[1] - aabb_min[1])
        coords_z = aabb_min[2] + zz * (aabb_max[2] - aabb_min[2])

        # 展平并堆叠（按 C-order：x 最慢，z 最快）
        coords = torch.stack([
            coords_x.flatten(),
            coords_y.flatten(),
            coords_z.flatten()
        ], dim=-1)  # [resolution^3, 3]

        return coords

    def _compute_sdf_with_warp(self, mesh: trimesh.Trimesh, query_points: torch.Tensor) -> torch.Tensor:
        """使用 Warp 计算 query points 到 Mesh 的 SDF。

        Args:
            mesh: trimesh 网格对象。
            query_points: [N, 3] 查询点坐标。

        Returns:
            sdf: [N] SDF 值。
        """
        # 准备 Warp 数据
        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.int32).flatten()

        # 创建 Warp mesh
        wp_mesh = wp.Mesh(
            points=wp.array(vertices, dtype=wp.vec3),
            indices=wp.array(faces, dtype=int)
        )

        # 准备查询点
        points_np = query_points.cpu().numpy().astype(np.float32)
        num_points = points_np.shape[0]

        # 创建 Warp 数组
        wp_points = wp.array(points_np, dtype=wp.vec3)
        wp_sdf = wp.zeros(num_points, dtype=float)
        wp_gradients = wp.zeros(num_points, dtype=wp.vec3)

        # 运行 kernel
        wp.launch(
            kernel=compute_sdf_kernel,
            dim=num_points,
            inputs=[wp_mesh.id, wp_points, wp_sdf, wp_gradients]
        )

        # 同步并转换回 PyTorch
        wp.synchronize()
        sdf_np = wp_sdf.numpy()
        sdf = torch.from_numpy(sdf_np).float()

        return sdf

    def _inject_occupancy_grid(self, estimator, binary_grid: torch.Tensor):
        """将 binary grid 注入到 NerfAcc estimator。

        Args:
            estimator: NerfAcc OccGridEstimator 实例。
            binary_grid: [N] 或 [resolution^3] 的 bool tensor。
        """
        # NerfAcc 的 OccGridEstimator 内部有以下属性：
        # - binaries: bool tensor 表示每个体素是否被占用
        # - occs: float tensor 表示占据概率
        # 确保 binary_grid 是正确的形状和类型
        binary_grid = binary_grid.bool()

        # 确保形状匹配
        binary_grid_reshaped = binary_grid.view_as(estimator.binaries)
        estimator.binaries.data.copy_(binary_grid_reshaped)

        print(f'[INFO] Injected occupancy grid to estimator')
        if hasattr(estimator, 'binaries'):
            print(f'  binaries shape: {estimator.binaries.shape}, sum: {estimator.binaries.sum().item()}')
        if hasattr(estimator, 'occs'):
            print(f'  occs shape: {estimator.occs.shape}, mean: {estimator.occs.mean().item():.4f}')

    def _update_progress(self, current_iteration):
        """Update training progress and model parameters.

        Override 父类方法以实现 OccupancyGrid 冻结功能。
        当 freeze_occ_grid=True 时，不会调用 model.update_occupancy_grid()。
        """
        model = self.model_module
        max_iter = self.cfg.max_epoch * self.iters_per_epoch
        self.progress = model.progress = current_iteration / max_iter

        model.neural_sdf.set_active_levels(current_iteration)
        model.neural_sdf.set_normal_epsilon()
        self.get_curvature_weight(current_iteration, self.cfg.trainer.loss_weight.curvature)

        # 更新 SDF 损失权重（线性衰减）
        self._update_sdf_loss_weight(current_iteration)

        # 关键修改：如果启用了 freeze_occ_grid，则不更新 OccupancyGrid
        if self.freeze_occ_grid and self._occ_grid_initialized:
            # 不调用 update_occupancy_grid，保持 Grid 固定
            pass
        else:
            # 原有逻辑：更新 nerfacc OccupancyGrid (仅在训练模式下)
            if hasattr(model, 'update_occupancy_grid') and model.training:
                model.update_occupancy_grid(current_iteration)

    def _update_sdf_loss_weight(self, current_iteration):
        """更新 SDF 损失权重（线性衰减到 0）。"""
        max_iter = self.cfg.max_epoch * self.iters_per_epoch
        # 线性衰减：从 init_weight 衰减到 0
        decay_ratio = 1.0 - (current_iteration / max_iter)
        self.sdf_loss_weight = self.sdf_loss_weight_init * decay_ratio

    def _compute_sdf_loss(self, points: torch.Tensor) -> torch.Tensor:
        """计算 SDF L1 损失。

        Args:
            points: [N, 3] 采样点坐标。

        Returns:
            loss: SDF L1 损失标量。
        """
        if self._wp_mesh is None or points.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        # 获取预测的 SDF
        pred_sdf = self.model_module.neural_sdf.sdf(points)  # [N, 1]

        # 计算 GT SDF（使用 Warp）
        with torch.no_grad():
            gt_sdf = self._compute_gt_sdf_batch(points)  # [N]

        # L1 损失
        loss = (pred_sdf.squeeze(-1) - gt_sdf).abs().mean()
        return loss

    def _compute_gt_sdf_batch(self, points: torch.Tensor) -> torch.Tensor:
        """使用 Warp 计算 GT SDF。

        Args:
            points: [N, 3] 或 [..., 3] 采样点坐标。

        Returns:
            sdf: [N] 或 [...] GT SDF 值。
        """
        original_shape = points.shape[:-1]
        points_flat = points.view(-1, 3)
        num_points = points_flat.shape[0]

        # 准备查询点
        points_np = points_flat.detach().cpu().numpy().astype(np.float32)

        # 创建 Warp 数组
        wp_points = wp.array(points_np, dtype=wp.vec3)
        wp_sdf = wp.zeros(num_points, dtype=float)
        wp_gradients = wp.zeros(num_points, dtype=wp.vec3)

        # 运行 kernel
        wp.launch(
            kernel=compute_sdf_kernel,
            dim=num_points,
            inputs=[self._wp_mesh.id, wp_points, wp_sdf, wp_gradients]
        )

        # 同步并转换回 PyTorch
        wp.synchronize()
        sdf_np = wp_sdf.numpy()
        sdf = torch.from_numpy(sdf_np).float().to(self.device)

        # 恢复原始形状
        sdf = sdf.view(*original_shape)
        return sdf

    def _compute_loss(self, data, mode=None):
        """Override 父类的 _compute_loss 以添加 SDF 损失。"""
        # 调用父类的 loss 计算
        super()._compute_loss(data, mode)

        # 添加 SDF 损失（仅在训练模式下）
        if mode == "train" and self.sdf_loss_weight > 0 and self._wp_mesh is not None:
            # 从渲染数据中获取采样点
            # 使用 gradients 对应的点（这些点已经在场景内）
            if "gradients" in data and data["gradients"] is not None:
                gradients = data["gradients"]
                # 获取对应的 3D 点
                # gradients 形状: [B, R, N, 3] 或 [M, 3]
                if gradients.dim() == 4:
                    # 从 data 中获取采样点
                    # 需要重新计算点的位置
                    if "dists" in data and "outside" in data:
                        # 使用渲染时的采样点
                        # 这里我们采样一些随机点来计算 SDF 损失
                        sdf_loss = self._compute_sdf_loss_from_random_samples()
                    else:
                        sdf_loss = torch.tensor(0.0, device=self.device)
                else:
                    # [M, 3] 格式（nerfacc）
                    sdf_loss = self._compute_sdf_loss_from_random_samples()
            else:
                sdf_loss = self._compute_sdf_loss_from_random_samples()

            self.losses["sdf"] = sdf_loss
            self.weights["sdf"] = self.sdf_loss_weight

    def _compute_sdf_loss_from_random_samples(self, num_samples: int = 4096) -> torch.Tensor:
        """从场景中随机采样点计算 SDF 损失。

        Args:
            num_samples: 采样点数量。

        Returns:
            loss: SDF L1 损失。
        """
        # 获取场景边界
        hashgrid_range = self.cfg.model.object.sdf.encoding.hashgrid.range
        vol_min, vol_max = hashgrid_range

        # 随机采样点
        points = torch.rand(num_samples, 3, device=self.device) * (vol_max - vol_min) + vol_min

        return self._compute_sdf_loss(points)

    def reinitialize_occupancy_grid(self, margin_voxels: float = None):
        """重新初始化 OccupancyGrid。

        允许在训练过程中动态调整 margin 并重新生成 Grid。

        Args:
            margin_voxels: 新的 margin 值（以体素为单位）。如果为 None，使用当前值。
        """
        if margin_voxels is not None:
            self.occ_margin_voxels = margin_voxels

        self._occ_grid_initialized = False
        self._setup_frozen_occupancy_grid_from_mesh()

    def unfreeze_occupancy_grid(self):
        """解冻 OccupancyGrid，允许其在训练过程中更新。"""
        self.freeze_occ_grid = False
        print('[INFO] OccupancyGrid unfrozen. It will now be updated during training.')

    def get_occupancy_stats(self) -> dict:
        """获取当前 OccupancyGrid 的统计信息。

        Returns:
            dict: 包含占据统计信息的字典。
        """
        stats = {
            'frozen': self.freeze_occ_grid,
            'initialized': self._occ_grid_initialized,
        }

        if hasattr(self.model_module, 'estimator') and self.model_module.estimator is not None:
            estimator = self.model_module.estimator.estimator
            
            if hasattr(estimator, 'binaries'):
                stats['total_voxels'] = estimator.binaries.numel()
                stats['occupied_voxels'] = estimator.binaries.sum().item()
                stats['occupancy_rate'] = estimator.binaries.float().mean().item()

            if hasattr(estimator, 'occs'):
                stats['occs_mean'] = estimator.occs.mean().item()
                stats['occs_min'] = estimator.occs.min().item()
                stats['occs_max'] = estimator.occs.max().item()

        return stats

    def visualize_occupancy_grid(self, save_path: str = None) -> trimesh.Trimesh:
        """可视化 OccupancyGrid 为点云或体素网格。

        Args:
            save_path: 保存路径（可选）。如果提供，会保存为 PLY 文件。

        Returns:
            点云的 trimesh 对象。
        """
        if not hasattr(self.model_module, 'estimator') or self.model_module.estimator is None:
            print('[WARN] No estimator found.')
            return None

        estimator = self.model_module.estimator.estimator
        
        if not hasattr(estimator, 'binaries'):
            print('[WARN] No binaries found in estimator.')
            return None

        # 生成所有体素中心坐标（直接从 estimator 获取，确保对齐）
        all_coords = self._make_grid_coords_from_estimator(estimator)

        # 获取被占据的体素
        occupied_mask = estimator.binaries.cpu().view(-1).bool()
        occupied_coords = all_coords.cpu()[occupied_mask].numpy()

        if len(occupied_coords) == 0:
            print('[WARN] No occupied voxels.')
            return None

        # 创建点云
        pcd = trimesh.PointCloud(occupied_coords)

        if save_path:
            pcd.export(save_path)
            print(f'[INFO] Saved occupancy visualization to {save_path}')

        return pcd


# ==================== 辅助函数 ====================

def create_mesh_trainer(
    cfg,
    mesh_file_path: str,
    device: str = 'cuda:0',
    freeze_occ_grid: bool = True,
    occ_margin_voxels: float = 2.0,
    sdf_loss_weight: float = 1.0
) -> MeshTrainer:
    """创建 MeshTrainer 的便捷函数。

    Args:
        cfg: 配置对象。
        mesh_file_path: Mesh 文件路径。
        device: 训练设备。
        freeze_occ_grid: 是否冻结 OccupancyGrid。
        occ_margin_voxels: OccupancyGrid 的 margin（以体素为单位）。
        sdf_loss_weight: SDF 损失的初始权重（会线性衰减到 0）。

    Returns:
        MeshTrainer 实例。
    """
    return MeshTrainer(
        cfg=cfg,
        mesh_file_path=mesh_file_path,
        device=device,
        freeze_occ_grid=freeze_occ_grid,
        occ_margin_voxels=occ_margin_voxels,
        sdf_loss_weight=sdf_loss_weight
    )
