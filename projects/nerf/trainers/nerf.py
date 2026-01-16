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
import skvideo.io

from imaginaire.utils.distributed import master_only
from projects.nerf.trainers.base import BaseTrainer
from imaginaire.utils.visualization import tensorboard_image, preprocess_image


class Trainer(BaseTrainer):

    def __init__(self, cfg, is_inference=True, seed=0):
        super().__init__(cfg, is_inference=is_inference, seed=seed)
        self.batch_idx, _ = torch.meshgrid(torch.arange(cfg.data.train.batch_size),
                                           torch.arange(cfg.model.rand_rays), indexing="ij")  # [B,R]
        self.batch_idx = self.batch_idx.cuda()

    def _init_loss(self, cfg):
        self.criteria["render"] = self.criteria["render_fine"] = torch.nn.MSELoss()

    def _compute_loss(self, data, mode=None):
        if mode == "train":
            # Extract the corresponding sampled rays.
            batch_size = len(data["image"])
            image_vec = data["image"].permute(0, 2, 3, 1).view(batch_size, -1, 3)  # [B,HW,3]
            image_sampled = image_vec[self.batch_idx, data["ray_idx"]]  # [B,R,3]
            # Compute loss only on randomly sampled rays.
            self.losses["render"] = self.criteria["render"](data["rgb"], image_sampled)
            self.metrics["psnr"] = -10 * torch_F.mse_loss(data["rgb"], image_sampled).log10()
            if self.cfg.model.fine_sampling:
                self.losses["render_fine"] = self.criteria["render_fine"](data["rgb_fine"], image_sampled)
                self.metrics["psnr_fine"] = -10 * torch_F.mse_loss(data["rgb_fine"], image_sampled).log10()
        else:
            # Compute loss on the entire image.
            self.losses["render"] = self.criteria["render"](data["rgb_map"], data["image"])
            self.metrics["psnr"] = -10 * torch_F.mse_loss(data["rgb_map"], data["image"]).log10()
            if self.cfg.model.fine_sampling:
                self.losses["render_fine"] = self.criteria["render_fine"](data["rgb_map_fine"], data["image"])
                self.metrics["psnr_fine"] = -10 * torch_F.mse_loss(data["rgb_map_fine"], data["image"]).log10()

    @master_only
    def log_tensorboard_scalars(self, data, mode=None):
        super().log_tensorboard_scalars(data, mode=mode)
        if not hasattr(self, 'tensorboard_writer') or self.tensorboard_writer is None:
            return
        self.tensorboard_writer.add_scalar(f"{mode}/PSNR/nerf", self.metrics["psnr"].detach().item(), self.current_iteration)
        if "render_fine" in self.losses:
            self.tensorboard_writer.add_scalar(f"{mode}/PSNR/nerf_fine", self.metrics["psnr_fine"].detach().item(), self.current_iteration)

    @master_only
    def log_tensorboard_images(self, data, mode=None, max_samples=None):
        super().log_tensorboard_images(data, mode=mode, max_samples=max_samples)
        if not hasattr(self, 'tensorboard_writer') or self.tensorboard_writer is None:
            return
        image_target = tensorboard_image(data["image"])
        self.tensorboard_writer.add_image(f"{mode}/image_target", image_target, self.current_iteration)
        if mode == "val":
            images_error = (data["rgb_map"] - data["image"]).abs()
            self.tensorboard_writer.add_image(f"{mode}/images", tensorboard_image(data["rgb_map"]), self.current_iteration)
            self.tensorboard_writer.add_image(f"{mode}/images_error", tensorboard_image(images_error), self.current_iteration)
            self.tensorboard_writer.add_image(f"{mode}/inv_depth", tensorboard_image(data["inv_depth_map"]), self.current_iteration)
            if self.cfg.model.fine_sampling:
                images_error_fine = (data["rgb_map_fine"] - data["image"]).abs()
                self.tensorboard_writer.add_image(f"{mode}/images_fine", tensorboard_image(data["rgb_map_fine"]), self.current_iteration)
                self.tensorboard_writer.add_image(f"{mode}/images_error_fine", tensorboard_image(images_error_fine), self.current_iteration)
                self.tensorboard_writer.add_image(f"{mode}/inv_depth_fine", tensorboard_image(data["inv_depth_map_fine"]), self.current_iteration)

    def dump_test_results(self, data_all, output_dir):
        results = dict(
            images_target=preprocess_image(data_all["images_target"]),
            image=preprocess_image(data_all["rgb_map"]),
            inv_depth=preprocess_image(data_all["inv_depth_map"]),
        )
        if self.cfg.model.fine_sampling:
            results.update(
                image_fine=preprocess_image(data_all["rgb_map_fine"]),
                inv_depth_fine=preprocess_image(data_all["inv_depth_map_fine"]),
            )
        # Write results as videos.
        inputdict, outputdict = self._get_ffmpeg_dicts()
        for key, image_list in results.items():
            print(f"writing video ({key})...")
            video_fname = f"{output_dir}/{key}.mp4"
            video_writer = skvideo.io.FFmpegWriter(video_fname, inputdict=inputdict, outputdict=outputdict)
            for image in image_list:
                image = (image * 255).byte().permute(1, 2, 0).numpy()
                video_writer.writeFrame(image)
            video_writer.close()

    def _get_ffmpeg_dicts(self):
        inputdict = {"-r": str(30)}
        outputdict = {"-crf": str(10), "-pix_fmt": "yuv420p"}
        return inputdict, outputdict
