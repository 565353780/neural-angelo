"""NerfAcc 加速工具模块

适配最新版 nerfacc (>=0.5.3) 的工具函数和类
用于加速 NeuS 风格的 SDF 渲染
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 尝试导入 nerfacc
try:
    import nerfacc
    from nerfacc import OccGridEstimator, render_weight_from_alpha, accumulate_along_rays
    NERFACC_AVAILABLE = True
    NERFACC_VERSION = getattr(nerfacc, '__version__', '0.5.0')
except ImportError:
    NERFACC_AVAILABLE = False
    NERFACC_VERSION = None
    print("Warning: nerfacc not available, using vanilla ray marching")


def check_nerfacc_available():
    """检查 nerfacc 是否可用"""
    return NERFACC_AVAILABLE


def get_nerfacc_version():
    """获取 nerfacc 版本"""
    return NERFACC_VERSION


class NerfAccEstimator(nn.Module):
    """NerfAcc OccupancyGrid 估计器包装类
    
    封装 OccGridEstimator，提供统一的接口用于：
    - 射线采样
    - 占据网格更新
    - 体积渲染
    """
    
    def __init__(self, aabb, resolution=128, device='cuda'):
        """
        Args:
            aabb: [6] tensor 或 list，表示场景边界 [x_min, y_min, z_min, x_max, y_max, z_max]
            resolution: OccupancyGrid 分辨率
            device: 计算设备
        """
        super().__init__()
        
        if not NERFACC_AVAILABLE:
            raise ImportError("nerfacc is not available. Please install it via: pip install nerfacc")
        
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        
        self.register_buffer('aabb', aabb)
        self.resolution = resolution
        
        # 创建 OccGridEstimator
        self.estimator = OccGridEstimator(
            roi_aabb=aabb,
            resolution=resolution,
        )
        
        # 移动到指定设备
        self.to(device)
        
    def sampling(self, rays_o, rays_d, sigma_fn=None, alpha_fn=None,
                 near_plane=0.0, far_plane=1e10, render_step_size=1e-3,
                 stratified=True, alpha_thre=0.0):
        """使用 OccupancyGrid 进行高效射线采样
        
        Args:
            rays_o: [N, 3] 射线起点
            rays_d: [N, 3] 射线方向
            sigma_fn: 密度函数 (用于 NeRF)
            alpha_fn: alpha 函数 (用于 NeuS)
            near_plane: 近平面
            far_plane: 远平面
            render_step_size: 渲染步长
            stratified: 是否使用分层采样
            alpha_thre: alpha 阈值
            
        Returns:
            ray_indices: [M] 采样点对应的射线索引
            t_starts: [M] 采样区间起点
            t_ends: [M] 采样区间终点
        """
        ray_indices, t_starts, t_ends = self.estimator.sampling(
            rays_o, rays_d,
            sigma_fn=sigma_fn,
            alpha_fn=alpha_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=stratified,
            alpha_thre=alpha_thre,
        )
        return ray_indices, t_starts, t_ends
    
    def update(self, step, occ_eval_fn, occ_thre=0.01, ema_decay=0.95):
        """更新 OccupancyGrid
        
        Args:
            step: 当前训练步数
            occ_eval_fn: 占据评估函数，输入 [N, 3] 坐标，输出 [N] 占据值
            occ_thre: 占据阈值
            ema_decay: EMA 衰减系数
        """
        self.estimator.update_every_n_steps(
            step=step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=occ_thre,
            ema_decay=ema_decay,
        )


def compute_neus_alpha_nerfacc(sdf, normals, dirs, dists, s_var, progress=1.0, anneal_end=0.1):
    """计算 NeuS 风格的 alpha 值（nerfacc 兼容格式）
    
    Args:
        sdf: [M, 1] SDF 值
        normals: [M, 3] 法向量
        dirs: [M, 3] 射线方向
        dists: [M] 采样区间长度
        s_var: s-variance 参数
        progress: 训练进度 [0, 1]
        anneal_end: 退火结束进度
        
    Returns:
        alpha: [M] alpha 值
    """
    sdf = sdf.squeeze(-1) if sdf.dim() > 1 else sdf
    
    inv_s = s_var.exp().clamp(1e-6, 1e6)
    
    # 计算 cosine
    true_cos = (dirs * normals).sum(dim=-1)
    
    # NeuS 退火策略
    anneal_ratio = min(progress / anneal_end, 1.0)
    iter_cos = -((-true_cos * 0.5 + 0.5).relu() * (1.0 - anneal_ratio) +
                 (-true_cos).relu() * anneal_ratio)
    
    # 估计前后 SDF
    estimated_next_sdf = sdf + iter_cos * dists * 0.5
    estimated_prev_sdf = sdf - iter_cos * dists * 0.5
    
    # 计算 CDF 和 alpha
    prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
    next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
    
    alpha = ((prev_cdf - next_cdf) / (prev_cdf + 1e-5)).clamp(0.0, 1.0)
    
    return alpha


def render_with_nerfacc(ray_indices, t_starts, t_ends, n_rays,
                        rgb, alpha, normal=None, depth_values=None):
    """使用 nerfacc 进行体积渲染
    
    Args:
        ray_indices: [M] 采样点对应的射线索引
        t_starts: [M] 采样区间起点
        t_ends: [M] 采样区间终点
        n_rays: 射线总数
        rgb: [M, 3] RGB 值
        alpha: [M] alpha 值
        normal: [M, 3] 法向量（可选）
        depth_values: [M] 深度值（可选，如果为 None，使用 (t_starts + t_ends) / 2）
        
    Returns:
        output: 包含渲染结果的字典
    """
    device = rgb.device
    
    if len(ray_indices) == 0:
        # 无有效采样点
        return {
            'rgb': torch.zeros(n_rays, 3, device=device),
            'depth': torch.zeros(n_rays, 1, device=device),
            'opacity': torch.zeros(n_rays, 1, device=device),
            'normal': torch.zeros(n_rays, 3, device=device) if normal is not None else None,
            'weights': None,
            'ray_indices': ray_indices,
            'num_samples': torch.tensor([0], device=device),
        }
    
    # 计算渲染权重
    # nerfacc >= 0.5.0 返回 (weights, trans)，需要解包
    weights_result = render_weight_from_alpha(
        alpha,
        ray_indices=ray_indices,
        n_rays=n_rays
    )
    # 兼容不同版本的 nerfacc 返回值
    if isinstance(weights_result, tuple):
        weights = weights_result[0]
    else:
        weights = weights_result
    
    # 累积 RGB
    comp_rgb = accumulate_along_rays(
        weights, values=rgb,
        ray_indices=ray_indices, n_rays=n_rays
    )
    
    # 累积深度
    if depth_values is None:
        depth_values = (t_starts + t_ends) / 2.0
    comp_depth = accumulate_along_rays(
        weights, values=depth_values[:, None],
        ray_indices=ray_indices, n_rays=n_rays
    )
    
    # 累积不透明度
    comp_opacity = accumulate_along_rays(
        weights, values=None,
        ray_indices=ray_indices, n_rays=n_rays
    )
    
    # 累积法向量
    comp_normal = None
    if normal is not None:
        comp_normal = accumulate_along_rays(
            weights, values=normal,
            ray_indices=ray_indices, n_rays=n_rays
        )
        comp_normal = F.normalize(comp_normal, dim=-1)
    
    return {
        'rgb': comp_rgb,
        'depth': comp_depth,
        'opacity': comp_opacity,
        'normal': comp_normal,
        'weights': weights,
        'ray_indices': ray_indices,
        'num_samples': torch.tensor([len(t_starts)], device=device),
    }


def get_aabb_from_radius(radius, center=None):
    """从半径生成 AABB
    
    Args:
        radius: 场景半径
        center: 场景中心，默认为原点
        
    Returns:
        aabb: [6] tensor
    """
    if center is None:
        center = [0.0, 0.0, 0.0]
    
    aabb = torch.tensor([
        center[0] - radius, center[1] - radius, center[2] - radius,
        center[0] + radius, center[1] + radius, center[2] + radius
    ], dtype=torch.float32)
    
    return aabb


def estimate_render_step_size(radius, num_samples_per_ray=512):
    """估计渲染步长
    
    Args:
        radius: 场景半径
        num_samples_per_ray: 每条射线的采样点数
        
    Returns:
        step_size: 渲染步长
    """
    # 场景对角线长度 = 2 * radius * sqrt(3)
    diagonal = 2 * radius * 1.732
    step_size = diagonal / num_samples_per_ray
    return step_size
