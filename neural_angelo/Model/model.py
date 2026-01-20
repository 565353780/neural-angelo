import os
import torch
import pickle
import torch.nn.functional as torch_F
from functools import partial
from collections import defaultdict

from neural_angelo.Util import nerf_util, camera, render
from neural_angelo.Util import nerf_misc as misc
from neural_angelo.Util.nerfacc_util import (
    NerfAccEstimator,
    compute_neus_alpha_nerfacc, render_with_nerfacc,
    get_aabb_from_radius, get_aabb_from_hashgrid_range, estimate_render_step_size
)

from neural_angelo.Model.neural_sdf import NeuralSDF
from neural_angelo.Model.neural_rgb import NeuralRGB
from neural_angelo.Model.background_nerf import BackgroundNeRF


class Model(torch.nn.Module):
    def __init__(self, cfg_model, cfg_data):
        super().__init__()
        self.cfg_model = cfg_model
        self.cfg_render = cfg_model.render
        self.white_background = cfg_model.background.white
        self.with_background = cfg_model.background.enabled
        self.with_appear_embed = cfg_model.appear_embed.enabled
        self.anneal_end = cfg_model.object.s_var.anneal_end
        self.outside_val = 1000. * (-1 if cfg_model.object.sdf.mlp.inside_out else 1)

        # 从 camera.pkl 加载相机和图像数据
        camera_pkl_file_path = cfg_data.root + '../camera_cpu.pkl'
        assert os.path.exists(camera_pkl_file_path), f"camera.pkl not found at {camera_pkl_file_path}"
        with open(camera_pkl_file_path, 'rb') as f:
            self.camera_list = pickle.load(f)
            self.image_size_train = [self.camera_list[0].height, self.camera_list[0].width]

        self.image_size_val = cfg_data.val.image_size
        # 初始化训练进度（用于 NeuS 退火策略，会在训练器中更新）
        self.progress = 0.
        # Define models.
        self.build_model(cfg_model, cfg_data)
        # Define functions.
        self.ray_generator = partial(nerf_util.ray_generator,
                                     camera_ndc=False,
                                     num_rays=cfg_model.render.rand_rays)
        self.sample_dists_from_pdf = partial(nerf_util.sample_dists_from_pdf,
                                             intvs_fine=cfg_model.render.num_samples.fine)
        self.to_full_val_image = partial(misc.to_full_image, image_size=cfg_data.val.image_size)

        # NerfAcc 加速配置
        self.use_nerfacc = False
        self.estimator = None
        self.estimator_bg = None
        if hasattr(cfg_model, 'nerfacc') and cfg_model.nerfacc.enabled:
            self._build_nerfacc(cfg_model)

    def get_param_groups(self):
        return self.parameters()

    def device(self):
        return next(self.parameters()).device

    def _build_nerfacc(self, cfg_model):
        """构建 nerfacc 加速结构"""
        print("Initializing NerfAcc acceleration...")
        
        # 使用 hashgrid.range 计算 AABB（确保与 HashGrid 空间范围一致）
        hashgrid_range = cfg_model.object.sdf.encoding.hashgrid.range
        aabb = get_aabb_from_hashgrid_range(hashgrid_range)
        self.register_buffer('scene_aabb', aabb)
        
        # 计算等效半径（用于步长估计等）
        self.radius = hashgrid_range[1]  # 使用 max 值作为半径
        
        print(f"  Using hashgrid.range {hashgrid_range} for AABB: {aabb.tolist()}")
        
        # 计算渲染步长
        num_samples = cfg_model.render.num_samples.coarse + \
                      cfg_model.render.num_samples.fine * cfg_model.render.num_sample_hierarchy
        if cfg_model.nerfacc.render_step_size is not None:
            self.render_step_size = cfg_model.nerfacc.render_step_size
        else:
            self.render_step_size = estimate_render_step_size(self.radius, num_samples)

        # 创建 OccupancyGrid 估计器
        if cfg_model.nerfacc.grid_prune:
            self.estimator = NerfAccEstimator(
                aabb=aabb,
                resolution=cfg_model.nerfacc.occ_grid.resolution,
            )

            # 背景 OccupancyGrid
            if cfg_model.background.enabled:
                self.estimator_bg = NerfAccEstimator(
                    aabb=aabb,
                    resolution=cfg_model.nerfacc.occ_grid.resolution_bg,
                )

        self.use_nerfacc = True
        self.nerfacc_cfg = cfg_model.nerfacc
        print(f"NerfAcc initialized with step_size={self.render_step_size:.6f}, "
              f"grid_resolution={cfg_model.nerfacc.occ_grid.resolution}")

    def build_model(self, cfg_model, cfg_data):
        # appearance encoding
        if cfg_model.appear_embed.enabled:
            assert cfg_data.num_images is not None
            self.appear_embed = torch.nn.Embedding(cfg_data.num_images, cfg_model.appear_embed.dim)
            if cfg_model.background.enabled:
                self.appear_embed_outside = torch.nn.Embedding(cfg_data.num_images, cfg_model.appear_embed.dim)
            else:
                self.appear_embed_outside = None
        else:
            self.appear_embed = self.appear_embed_outside = None
        self.neural_sdf = NeuralSDF(cfg_model.object.sdf)
        self.neural_rgb = NeuralRGB(cfg_model.object.rgb, feat_dim=cfg_model.object.sdf.mlp.hidden_dim,
                                    appear_embed=cfg_model.appear_embed)
        if cfg_model.background.enabled:
            self.background_nerf = BackgroundNeRF(cfg_model.background, appear_embed=cfg_model.appear_embed)
        else:
            self.background_nerf = None
        self.s_var = torch.nn.Parameter(torch.tensor(cfg_model.object.s_var.init_val, dtype=torch.float32))

    def forward(self, data):
        # Randomly sample and render the pixels.
        output = self.render_pixels(data["pose"], data["intr"], image_size=self.image_size_train,
                                    stratified=self.cfg_render.stratified, sample_idx=data["idx"],
                                    ray_idx=data["ray_idx"])
        return output

    @torch.no_grad()
    def inference(self, data):
        self.eval()
        # Render the full images.
        output = self.render_image(data["pose"], data["intr"], image_size=self.image_size_val,
                                   stratified=False, sample_idx=data["idx"])  # [B,N,C]
        # Get full rendered RGB and depth images.
        rot = data["pose"][..., :3, :3]  # [B,3,3]
        normal_cam = -output["gradient"] @ rot.transpose(-1, -2)  # [B,HW,3]
        output.update(
            rgb_map=self.to_full_val_image(output["rgb"]),  # [B,3,H,W]
            opacity_map=self.to_full_val_image(output["opacity"]),  # [B,1,H,W]
            depth_map=self.to_full_val_image(output["depth"]),  # [B,1,H,W]
            normal_map=self.to_full_val_image(normal_cam),  # [B,3,H,W]
        )
        return output

    def render_image(self, pose, intr, image_size, stratified=False, sample_idx=None):
        """ Render the rays given the camera intrinsics and poses.
        Args:
            pose (tensor [batch,3,4]): Camera poses ([R,t]).
            intr (tensor [batch,3,3]): Camera intrinsics.
            stratified (bool): Whether to stratify the depth sampling.
            sample_idx (tensor [batch]): Data sample index.
        Returns:
            output: A dictionary containing the outputs.
        """
        output = defaultdict(list)
        # 推理时需要保留的字段（可正确 cat 的字段）
        inference_keys = {'rgb', 'depth', 'opacity', 'gradient', 'normal', 'outside'}
        
        for center, ray, _ in self.ray_generator(pose, intr, image_size, full_image=True):
            ray_unit = torch_F.normalize(ray, dim=-1)  # [B,R,3]
            output_batch = self.render_rays(center, ray_unit, sample_idx=sample_idx, stratified=stratified)
            if not self.training:
                # nerfacc 已经计算了 depth，只有 vanilla 渲染需要从 dists/weights 计算
                if "depth" not in output_batch or output_batch["depth"] is None:
                    dist = render.composite(output_batch["dists"], output_batch["weights"])  # [B,R,1]
                    depth = dist / ray.norm(dim=-1, keepdim=True)
                    output_batch.update(depth=depth)
            for key, value in output_batch.items():
                if value is not None:
                    # 推理时只收集可以正确 cat 的字段
                    if not self.training and key not in inference_keys:
                        continue
                    # 确保 tensor 至少有 2 维才能在 dim=1 上 cat
                    if isinstance(value, torch.Tensor) and value.dim() >= 2:
                        output[key].append(value.detach())
        # Concat each item (list) in output into one tensor. Concatenate along the ray dimension (1)
        for key, value in output.items():
            if len(value) > 0:
                output[key] = torch.cat(value, dim=1)
        return output

    def render_pixels(self, pose, intr, image_size, stratified=False, sample_idx=None, ray_idx=None):
        center, ray = camera.get_center_and_ray(pose, intr, image_size)  # [B,HW,3]
        center = nerf_util.slice_by_ray_idx(center, ray_idx)  # [B,R,3]
        ray = nerf_util.slice_by_ray_idx(ray, ray_idx)  # [B,R,3]
        ray_unit = torch_F.normalize(ray, dim=-1)  # [B,R,3]
        output = self.render_rays(center, ray_unit, sample_idx=sample_idx, stratified=stratified)
        return output

    def render_rays(self, center, ray_unit, sample_idx=None, stratified=False):
        """渲染光线，根据配置选择 nerfacc 加速或 vanilla 渲染"""
        # 使用 nerfacc 加速渲染
        if self.use_nerfacc and self.estimator is not None:
            return self.render_rays_nerfacc(center, ray_unit, sample_idx=sample_idx, stratified=stratified)
        
        # Vanilla 渲染
        return self.render_rays_vanilla(center, ray_unit, sample_idx=sample_idx, stratified=stratified)
    
    def render_rays_vanilla(self, center, ray_unit, sample_idx=None, stratified=False):
        """Vanilla 渲染方法（不使用 nerfacc）"""
        with torch.no_grad():
            near, far, outside = self.get_dist_bounds(center, ray_unit)
        app, app_outside = self.get_appearance_embedding(sample_idx, ray_unit.shape[1])
        output_object = self.render_rays_object(center, ray_unit, near, far, outside, app, stratified=stratified)
        if self.with_background:
            output_background = self.render_rays_background(center, ray_unit, far, app_outside, stratified=stratified)
            # Concatenate object and background samples.
            rgbs = torch.cat([output_object["rgbs"], output_background["rgbs"]], dim=2)  # [B,R,No+Nb,3]
            dists = torch.cat([output_object["dists"], output_background["dists"]], dim=2)  # [B,R,No+Nb,1]
            alphas = torch.cat([output_object["alphas"], output_background["alphas"]], dim=2)  # [B,R,No+Nb]
        else:
            rgbs = output_object["rgbs"]  # [B,R,No,3]
            dists = output_object["dists"]  # [B,R,No,1]
            alphas = output_object["alphas"]  # [B,R,No]
        weights = render.alpha_compositing_weights(alphas)  # [B,R,No+Nb,1]
        # Compute weights and composite samples.
        rgb = render.composite(rgbs, weights)  # [B,R,3]
        if self.white_background:
            opacity_all = render.composite(1., weights)  # [B,R,1]
            rgb = rgb + (1 - opacity_all)
        # Collect output.
        output = dict(
            rgb=rgb,  # [B,R,3]
            opacity=output_object["opacity"],  # [B,R,1]/None
            outside=outside,  # [B,R,1]
            dists=dists,  # [B,R,No+Nb,1]
            weights=weights,  # [B,R,No+Nb,1]
            gradient=output_object["gradient"],  # [B,R,3]/None
            gradients=output_object["gradients"],  # [B,R,No,3]
            hessians=output_object["hessians"],  # [B,R,No,3]/None
        )
        return output
    
    def render_rays_nerfacc(self, center, ray_unit, sample_idx=None, stratified=False):
        """使用 nerfacc 加速的光线渲染
        
        Args:
            center: [B, R, 3] 光线起点
            ray_unit: [B, R, 3] 归一化光线方向
            sample_idx: [B] 图像索引
            stratified: 是否使用分层采样
            
        Returns:
            output: 渲染结果字典
        """
        batch_size, n_rays = center.shape[:2]
        device = center.device
        
        # 获取边界
        with torch.no_grad():
            near, far, outside = self.get_dist_bounds(center, ray_unit)
        
        # 获取外观嵌入
        app, app_outside = self.get_appearance_embedding(sample_idx, n_rays)
        
        # 展平 batch 维度用于 nerfacc
        rays_o = center.view(-1, 3)  # [B*R, 3]
        rays_d = ray_unit.view(-1, 3)  # [B*R, 3]
        total_rays = rays_o.shape[0]
        
        # 定义 alpha 函数用于采样
        def alpha_fn(t_starts, t_ends, ray_indices):
            """计算 NeuS alpha 用于 OccupancyGrid 采样"""
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            
            sdf = self.neural_sdf.sdf(positions)
            
            inv_s = self.s_var.exp().clamp(1e-6, 1e6)
            dists = t_ends - t_starts
            
            # 简化的 alpha 估计
            estimated_next_sdf = sdf.squeeze(-1) - dists * 0.5
            estimated_prev_sdf = sdf.squeeze(-1) + dists * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
            alpha = ((prev_cdf - next_cdf) / (prev_cdf + 1e-5)).clamp(0.0, 1.0)
            return alpha
        
        # 使用 OccupancyGrid 采样
        ray_indices, t_starts, t_ends = self.estimator.sampling(
            rays_o, rays_d,
            alpha_fn=alpha_fn,
            near_plane=self.nerfacc_cfg.near_plane,
            far_plane=self.nerfacc_cfg.far_plane,
            render_step_size=self.render_step_size,
            stratified=stratified,
            alpha_thre=self.nerfacc_cfg.alpha_thre,
        )
        
        # 渲染前景采样点
        output = self._render_samples_nerfacc(
            rays_o, rays_d, ray_indices, t_starts, t_ends, 
            app, total_rays, outside.view(-1, 1)
        )
        
        # 处理背景
        if self.white_background:
            opacity = output['opacity']
            output['rgb'] = output['rgb'] + (1 - opacity)
        
        # 将输出 reshape 回 [B, R, ...] 格式
        output = self._reshape_nerfacc_output(output, batch_size, n_rays)
        output['outside'] = outside
        
        return output
    
    def _render_samples_nerfacc(self, rays_o, rays_d, ray_indices, t_starts, t_ends, 
                                 app, n_rays, outside):
        """渲染 nerfacc 采样点
        
        Args:
            rays_o: [N, 3] 光线起点
            rays_d: [N, 3] 光线方向
            ray_indices: [M] 采样点的光线索引
            t_starts: [M] 采样区间起点
            t_ends: [M] 采样区间终点
            app: 外观嵌入
            n_rays: 总光线数
            outside: [N, 1] 是否在场景外
            
        Returns:
            output: 渲染结果字典
        """
        device = rays_o.device
        
        if len(ray_indices) == 0:
            # 无有效采样点 - 创建需要梯度的零张量以保持计算图
            # 通过与 s_var 相乘来保持梯度连接
            zero_with_grad = self.s_var * 0.0
            return {
                'rgb': torch.zeros(n_rays, 3, device=device) + zero_with_grad,
                'depth': torch.zeros(n_rays, 1, device=device),
                'opacity': torch.zeros(n_rays, 1, device=device),
                'gradient': torch.zeros(n_rays, 3, device=device),
                'gradients': torch.zeros(n_rays, 1, 3, device=device) + zero_with_grad,
                'hessians': None,
                'dists': torch.zeros(n_rays, 1, 1, device=device),
                'weights': torch.zeros(n_rays, 1, 1, device=device),
                'num_samples': torch.tensor([0], device=device),
            }
        
        # 计算采样点位置
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.0
        positions = t_origins + t_dirs * midpoints[:, None]
        dists = t_ends - t_starts
        
        # SDF 和特征
        sdfs, feats = self.neural_sdf.forward(positions)  # [M, 1], [M, K]
        
        # 设置场景外的 SDF 值（使用 torch.where 避免 in-place 操作断开计算图）
        sample_outside = outside[ray_indices].squeeze(-1)
        if sample_outside.any():
            outside_mask = sample_outside.unsqueeze(-1)  # [M, 1]
            sdfs = torch.where(outside_mask, torch.full_like(sdfs, self.outside_val), sdfs)
        
        # 计算梯度和法向量
        gradients, hessians = self.neural_sdf.compute_gradients(positions, training=self.training, sdf=sdfs)
        normals = torch_F.normalize(gradients, dim=-1)  # [M, 3]
        
        # 获取外观嵌入（如果有）
        if app is not None:
            # app 是 [B, 1, N_samples, C] 格式，需要展平并索引
            app_flat = app.view(-1, app.shape[-1])  # [B*R*N, C]
            # 对于 nerfacc，我们需要为每个采样点获取对应的外观嵌入
            # 简化处理：使用射线索引获取外观嵌入
            n_rays_per_batch = app.shape[1] * app.shape[2]  # R * N
            batch_indices = ray_indices // (rays_o.shape[0] // app.shape[0]) if app.shape[0] > 1 else torch.zeros_like(ray_indices)
            app_samples = app[batch_indices.long(), 0, 0, :]  # [M, C]
        else:
            app_samples = None
        
        # 计算 RGB
        rgbs = self.neural_rgb.forward(positions, normals, t_dirs, feats, app=app_samples)  # [M, 3]
        
        # 计算 NeuS alpha
        alphas = compute_neus_alpha_nerfacc(
            sdfs, normals, t_dirs, dists,
            self.s_var, self.progress, self.anneal_end
        )
        
        # 使用 nerfacc 进行体积渲染
        output = render_with_nerfacc(
            ray_indices, t_starts, t_ends, n_rays,
            rgbs, alphas, normal=normals, depth_values=midpoints
        )
        
        # 添加训练所需的额外输出
        output['gradients'] = self._aggregate_gradients(gradients, ray_indices, n_rays)
        output['hessians'] = self._aggregate_gradients(hessians, ray_indices, n_rays) if hessians is not None else None
        output['gradient'] = output['normal'] if not self.training else None
        
        return output
    
    def _aggregate_gradients(self, values, ray_indices, n_rays):
        """将采样点的值聚合到射线级别（取加权平均或第一个值）"""
        device = values.device
        # 简化处理：返回所有采样点的梯度（用于 loss 计算）
        # 为了与原始格式兼容，我们需要重组
        # 这里返回一个 [N, max_samples, 3] 的张量可能太大
        # 简化为返回 [M, 3] 格式，在 loss 计算时处理
        return values
    
    def _reshape_nerfacc_output(self, output, batch_size, n_rays):
        """将 nerfacc 输出 reshape 回 [B, R, ...] 格式"""
        reshaped = {}
        total_rays = batch_size * n_rays
        for key, value in output.items():
            if value is None:
                reshaped[key] = None
            elif isinstance(value, torch.Tensor):
                # 跳过不需要 reshape 的特殊字段
                if key in ('num_samples', 'ray_indices', 'weights'):
                    reshaped[key] = value
                elif value.dim() == 2 and value.shape[0] == total_rays:
                    # [total_rays, C] -> [B, R, C]
                    reshaped[key] = value.view(batch_size, n_rays, -1)
                elif value.dim() == 1 and value.shape[0] == total_rays:
                    # [total_rays] -> [B, R]
                    reshaped[key] = value.view(batch_size, n_rays)
                else:
                    # 形状不匹配或其他维度，保持原样
                    reshaped[key] = value
            else:
                reshaped[key] = value
        return reshaped
    
    def update_occupancy_grid(self, step):
        """更新 OccupancyGrid（在训练循环中调用）"""
        if not self.use_nerfacc or self.estimator is None:
            return
        
        warmup = self.nerfacc_cfg.occ_grid.warmup_steps
        update_interval = self.nerfacc_cfg.occ_grid.update_interval
        
        if step < warmup or step % update_interval != 0:
            return
        
        def occ_eval_fn(x):
            """评估占据值"""
            sdf = self.neural_sdf.sdf(x)
            inv_s = self.s_var.exp().clamp(1e-6, 1e6)
            # 使用 sigmoid 将 SDF 转换为占据概率
            occ = torch.sigmoid(-sdf.abs() * inv_s)
            return occ.squeeze(-1)
        
        self.estimator.update(
            step=step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=self.nerfacc_cfg.occ_grid.occ_thre,
        )

    def render_rays_object(self, center, ray_unit, near, far, outside, app, stratified=False):
        with torch.no_grad():
            dists = self.sample_dists_all(center, ray_unit, near, far, stratified=stratified)  # [B,R,N,3]
        points = camera.get_3D_points_from_dist(center, ray_unit, dists)  # [B,R,N,3]
        sdfs, feats = self.neural_sdf.forward(points)  # [B,R,N,1],[B,R,N,K]
        sdfs[outside[..., None].expand_as(sdfs)] = self.outside_val
        # Compute 1st- and 2nd-order gradients.
        rays_unit = ray_unit[..., None, :].expand_as(points).contiguous()  # [B,R,N,3]
        gradients, hessians = self.neural_sdf.compute_gradients(points, training=self.training, sdf=sdfs)
        normals = torch_F.normalize(gradients, dim=-1)  # [B,R,N,3]
        rgbs = self.neural_rgb.forward(points, normals, rays_unit, feats, app=app)  # [B,R,N,3]
        # SDF volume rendering.
        alphas = self.compute_neus_alphas(ray_unit, sdfs, gradients, dists, dist_far=far[..., None],
                                          progress=self.progress)  # [B,R,N]
        if not self.training:
            weights = render.alpha_compositing_weights(alphas)  # [B,R,N,1]
            opacity = render.composite(1., weights)  # [B,R,1]
            gradient = render.composite(gradients, weights)  # [B,R,3]
        else:
            opacity = None
            gradient = None
        # Collect output.
        output = dict(
            rgbs=rgbs,  # [B,R,N,3]
            sdfs=sdfs[..., 0],  # [B,R,N]
            dists=dists,  # [B,R,N,1]
            alphas=alphas,  # [B,R,N]
            opacity=opacity,  # [B,R,3]/None
            gradient=gradient,  # [B,R,3]/None
            gradients=gradients,  # [B,R,N,3]
            hessians=hessians,  # [B,R,N,3]/None
        )
        return output

    def render_rays_background(self, center, ray_unit, far, app_outside, stratified=False):
        with torch.no_grad():
            dists = self.sample_dists_background(ray_unit, far, stratified=stratified)
        points = camera.get_3D_points_from_dist(center, ray_unit, dists)  # [B,R,N,3]
        rays_unit = ray_unit[..., None, :].expand_as(points)  # [B,R,N,3]
        rgbs, densities = self.background_nerf.forward(points, rays_unit, app_outside)  # [B,R,N,3]
        alphas = render.volume_rendering_alphas_dist(densities, dists)  # [B,R,N]
        # Collect output.
        output = dict(
            rgbs=rgbs,  # [B,R,3]
            dists=dists,  # [B,R,N,1]
            alphas=alphas,  # [B,R,N]
        )
        return output

    @torch.no_grad()
    def get_dist_bounds(self, center, ray_unit):
        dist_near, dist_far = nerf_util.intersect_with_sphere(center, ray_unit, radius=1.)
        dist_near.relu_()  # Distance (and thus depth) should be non-negative.
        outside = dist_near.isnan()
        dist_near[outside], dist_far[outside] = 1, 1.2  # Dummy distances. Density will be set to 0.
        return dist_near, dist_far, outside

    def get_appearance_embedding(self, sample_idx, num_rays):
        if self.with_appear_embed:
            # Object appearance embedding.
            num_samples_all = self.cfg_render.num_samples.coarse + \
                self.cfg_render.num_samples.fine * self.cfg_render.num_sample_hierarchy
            app = self.appear_embed(sample_idx)[:, None, None]  # [B,1,1,C]
            app = app.expand(-1, num_rays, num_samples_all, -1)  # [B,R,N,C]
            # Background appearance embedding.
            if self.with_background:
                app_outside = self.appear_embed_outside(sample_idx)[:, None, None]  # [B,1,1,C]
                app_outside = app_outside.expand(-1, num_rays, self.cfg_render.num_samples.background, -1)  # [B,R,N,C]
            else:
                app_outside = None
        else:
            app = app_outside = None
        return app, app_outside

    @torch.no_grad()
    def sample_dists_all(self, center, ray_unit, near, far, stratified=False):
        dists = nerf_util.sample_dists(ray_unit.shape[:2], dist_range=(near[..., None], far[..., None]),
                                       intvs=self.cfg_render.num_samples.coarse, stratified=stratified,
                                       device=ray_unit.device)
        if self.cfg_render.num_sample_hierarchy > 0:
            points = camera.get_3D_points_from_dist(center, ray_unit, dists)  # [B,R,N,3]
            sdfs = self.neural_sdf.sdf(points)  # [B,R,N]
        for h in range(self.cfg_render.num_sample_hierarchy):
            dists_fine = self.sample_dists_hierarchical(dists, sdfs, inv_s=(64 * 2 ** h))  # [B,R,Nf,1]
            dists = torch.cat([dists, dists_fine], dim=2)  # [B,R,N+Nf,1]
            dists, sort_idx = dists.sort(dim=2)
            if h != self.cfg_render.num_sample_hierarchy - 1:
                points_fine = camera.get_3D_points_from_dist(center, ray_unit, dists_fine)  # [B,R,Nf,3]
                sdfs_fine = self.neural_sdf.sdf(points_fine)  # [B,R,Nf]
                sdfs = torch.cat([sdfs, sdfs_fine], dim=2)  # [B,R,N+Nf]
                sdfs = sdfs.gather(dim=2, index=sort_idx.expand_as(sdfs))  # [B,R,N+Nf,1]
        return dists

    def sample_dists_hierarchical(self, dists, sdfs, inv_s, robust=True, eps=1e-5):
        sdfs = sdfs[..., 0]  # [B,R,N]
        prev_sdfs, next_sdfs = sdfs[..., :-1], sdfs[..., 1:]  # [B,R,N-1]
        prev_dists, next_dists = dists[..., :-1, 0], dists[..., 1:, 0]  # [B,R,N-1]
        mid_sdfs = (prev_sdfs + next_sdfs) * 0.5  # [B,R,N-1]
        cos_val = (next_sdfs - prev_sdfs) / (next_dists - prev_dists + 1e-5)  # [B,R,N-1]
        if robust:
            prev_cos_val = torch.cat([torch.zeros_like(cos_val)[..., :1], cos_val[..., :-1]], dim=-1)  # [B,R,N-1]
            cos_val = torch.stack([prev_cos_val, cos_val], dim=-1).min(dim=-1).values  # [B,R,N-1]
        dist_intvs = dists[..., 1:, 0] - dists[..., :-1, 0]  # [B,R,N-1]
        est_prev_sdf = mid_sdfs - cos_val * dist_intvs * 0.5  # [B,R,N-1]
        est_next_sdf = mid_sdfs + cos_val * dist_intvs * 0.5  # [B,R,N-1]
        prev_cdf = (est_prev_sdf * inv_s).sigmoid()  # [B,R,N-1]
        next_cdf = (est_next_sdf * inv_s).sigmoid()  # [B,R,N-1]
        alphas = ((prev_cdf - next_cdf) / (prev_cdf + eps)).clip_(0.0, 1.0)  # [B,R,N-1]
        weights = render.alpha_compositing_weights(alphas)  # [B,R,N-1,1]
        dists_fine = self.sample_dists_from_pdf(dists, weights=weights[..., 0])  # [B,R,Nf,1]
        return dists_fine

    def sample_dists_background(self, ray_unit, far, stratified=False, eps=1e-5):
        inv_dists = nerf_util.sample_dists(ray_unit.shape[:2], dist_range=(1, 0),
                                           intvs=self.cfg_render.num_samples.background, stratified=stratified,
                                           device=ray_unit.device)
        dists = far[..., None] / (inv_dists + eps)  # [B,R,N,1]
        return dists

    def compute_neus_alphas(self, ray_unit, sdfs, gradients, dists, dist_far=None, progress=1., eps=1e-5):
        sdfs = sdfs[..., 0]  # [B,R,N]
        # SDF volume rendering in NeuS.
        inv_s = self.s_var.exp()
        true_cos = (ray_unit[..., None, :] * gradients).sum(dim=-1, keepdim=False)  # [B,R,N]
        iter_cos = self._get_iter_cos(true_cos, progress=progress)  # [B,R,N]
        # Estimate signed distances at section points
        if dist_far is None:
            dist_far = torch.empty_like(dists[..., :1, :]).fill_(1e10)  # [B,R,1,1]
        dists = torch.cat([dists, dist_far], dim=2)  # [B,R,N+1,1]
        dist_intvs = dists[..., 1:, 0] - dists[..., :-1, 0]  # [B,R,N]
        est_prev_sdf = sdfs - iter_cos * dist_intvs * 0.5  # [B,R,N]
        est_next_sdf = sdfs + iter_cos * dist_intvs * 0.5  # [B,R,N]
        prev_cdf = (est_prev_sdf * inv_s).sigmoid()  # [B,R,N]
        next_cdf = (est_next_sdf * inv_s).sigmoid()  # [B,R,N]
        alphas = ((prev_cdf - next_cdf) / (prev_cdf + eps)).clip_(0.0, 1.0)  # [B,R,N]
        # weights = render.alpha_compositing_weights(alphas)  # [B,R,N,1]
        return alphas

    def _get_iter_cos(self, true_cos, progress=1.):
        anneal_ratio = min(progress / self.anneal_end, 1.)
        # The anneal strategy below keeps the cos value alive at the beginning of training iterations.
        return -((-true_cos * 0.5 + 0.5).relu() * (1.0 - anneal_ratio) +
                 (-true_cos).relu() * anneal_ratio)  # always non-positive
