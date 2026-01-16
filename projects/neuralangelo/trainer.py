'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import torch
import torch.nn.functional as torch_F

from imaginaire.utils.distributed import master_only
from imaginaire.utils.visualization import tensorboard_image
from projects.nerf.trainers.base import BaseTrainer
from projects.neuralangelo.utils.misc import get_scheduler, eikonal_loss, curvature_loss


class Trainer(BaseTrainer):

    def __init__(self, cfg, is_inference=True, seed=0):
        super().__init__(cfg, is_inference=is_inference, seed=seed)
        self.metrics = dict()
        self.warm_up_end = cfg.optim.sched.warm_up_end
        self.cfg_gradient = cfg.model.object.sdf.gradient
        if cfg.model.object.sdf.encoding.type == "hashgrid" and cfg.model.object.sdf.encoding.coarse2fine.enabled:
            self.c2f_step = cfg.model.object.sdf.encoding.coarse2fine.step
            self.model.module.neural_sdf.warm_up_end = self.warm_up_end

    def _init_loss(self, cfg):
        self.criteria["render"] = torch.nn.L1Loss()

    def setup_scheduler(self, cfg, optim):
        return get_scheduler(cfg.optim, optim)

    def _compute_loss(self, data, mode=None):
        if mode == "train":
            # Compute loss only on randomly sampled rays.
            self.losses["render"] = self.criteria["render"](data["rgb"], data["image_sampled"]) * 3  # FIXME:sumRGB?!
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

    def _start_of_iteration(self, data, current_iteration):
        model = self.model_module
        self.progress = model.progress = current_iteration / self.cfg.max_iter
        if self.cfg.model.object.sdf.encoding.coarse2fine.enabled:
            model.neural_sdf.set_active_levels(current_iteration)
            if self.cfg_gradient.mode == "numerical":
                model.neural_sdf.set_normal_epsilon()
                self.get_curvature_weight(current_iteration, self.cfg.trainer.loss_weight.curvature)
        elif self.cfg_gradient.mode == "numerical":
            model.neural_sdf.set_normal_epsilon()

        return super()._start_of_iteration(data, current_iteration)

    @master_only
    def log_tensorboard_scalars(self, data, mode=None):
        super().log_tensorboard_scalars(data, mode=mode)
        if not hasattr(self, 'tensorboard_writer') or self.tensorboard_writer is None:
            return
        self.tensorboard_writer.add_scalar(f"{mode}/PSNR", self.metrics["psnr"].detach().item(), self.current_iteration)
        self.tensorboard_writer.add_scalar(f"{mode}/s-var", self.model_module.s_var.item(), self.current_iteration)
        if "curvature" in self.weights:
            self.tensorboard_writer.add_scalar(f"{mode}/curvature_weight", self.weights["curvature"], self.current_iteration)
        if "eikonal" in self.weights:
            self.tensorboard_writer.add_scalar(f"{mode}/eikonal_weight", self.weights["eikonal"], self.current_iteration)
        if mode == "train" and self.cfg_gradient.mode == "numerical":
            self.tensorboard_writer.add_scalar(f"{mode}/epsilon", self.model.module.neural_sdf.normal_eps, self.current_iteration)
        if self.cfg.model.object.sdf.encoding.coarse2fine.enabled:
            self.tensorboard_writer.add_scalar(f"{mode}/active_levels", self.model.module.neural_sdf.active_levels, self.current_iteration)

    @master_only
    def log_tensorboard_images(self, data, mode=None, max_samples=None):
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

    def train(self, cfg, data_loader, single_gpu=False, profile=False, show_pbar=False):
        self.progress = self.model_module.progress = self.current_iteration / self.cfg.max_iter
        super().train(cfg, data_loader, single_gpu, profile, show_pbar)
