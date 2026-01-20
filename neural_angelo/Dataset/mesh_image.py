import os
import json
import torch
import pickle
import trimesh
import numpy as np
import torchvision.transforms.functional as torchvision_F
from PIL import Image, ImageFile

from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer

from neural_angelo.Dataset.base import BaseDataset
from neural_angelo.Method.io import loadMeshFile
from neural_angelo.Util import camera

ImageFile.LOAD_TRUNCATED_IMAGES = True


class MeshImageDataset(BaseDataset):
    def __init__(self, cfg, is_inference=False):
        super().__init__(is_inference)
        cfg_data = cfg.data
        self.root = cfg_data.root
        self.preload = cfg_data.preload

        # 从 camera.pkl 加载相机和图像数据
        camera_pkl_file_path = self.root + '../camera.pkl'
        assert os.path.exists(camera_pkl_file_path), f"camera.pkl not found at {camera_pkl_file_path}"
        with open(camera_pkl_file_path, 'rb') as f:
            self.camera_list = pickle.load(f)

        if is_inference:
            self.H, self.W = cfg_data.val.image_size
        else:
            self.W = self.camera_list[0].width
            self.H = self.camera_list[0].height

        # 将所有相机数据转换为 float32 并移到 CPU
        for i in range(len(self.camera_list)):
            self.camera_list[i].to(dtype=torch.float32, device='cpu')

        # 从 transforms.json 读取场景归一化参数
        meta_fname = f"{cfg_data.root}/transforms.json"
        if os.path.exists(meta_fname):
            with open(meta_fname) as file:
                meta = json.load(file)
            self.sphere_center = np.array(meta.get("sphere_center", [0.0, 0.0, 0.0]))
            self.sphere_radius = meta.get("sphere_radius", 1.0)
        else:
            # 如果没有 transforms.json，根据相机位置自动计算
            print("[WARNING] transforms.json not found, computing sphere_center and sphere_radius from cameras")
            camera_positions = []
            for cam in self.camera_list:
                c2w = cam.camera2world
                if isinstance(c2w, torch.Tensor):
                    pos = c2w[:3, 3].numpy()
                else:
                    pos = c2w[:3, 3]
                camera_positions.append(pos)
            camera_positions = np.array(camera_positions)
            self.sphere_center = camera_positions.mean(axis=0)
            self.sphere_radius = np.linalg.norm(camera_positions - self.sphere_center, axis=1).max()

        self.num_rays = cfg.model.render.rand_rays
        self.readjust = getattr(cfg_data, "readjust", None)

        # Preload dataset if possible.
        if cfg_data.preload:
            self.images = self.preload_threading(self.get_image, cfg_data.num_workers)
            self.cameras = self.preload_threading(self.get_camera, cfg_data.num_workers, data_str="cameras")

        self.mesh: trimesh.Trimesh = None

    def __len__(self):
        return len(self.camera_list)

    def loadMeshFile(self, mesh_file_path: str) -> bool:
        if not os.path.exists(mesh_file_path):
            print('[ERROR][MeshImageDataset::loadMeshFile]')
            print('\t mesh file not exist!')
            print('\t mesh_file_path:', mesh_file_path)
            return False

        self.mesh = loadMeshFile(mesh_file_path)
        return True

    def __getitem__(self, idx):
        """Process raw data and return processed data in a dictionary.

        Args:
            idx: The index of the sample of the dataset.
        Returns: A dictionary containing the data.
                 idx (scalar): The index of the sample of the dataset.
                 image (R tensor): Image idx for per-image embedding.
                 image (Rx3 tensor): Image with pixel values in [0,1] for supervision.
                 intr (3x3 tensor): The camera intrinsics of `image`.
                 pose (3x4 tensor): The camera extrinsics [R,t] of `image`.
        """
        # Keep track of sample index for convenience.
        sample = dict(idx=idx)
        # Get the images.
        image, image_size_raw = self.images[idx] if self.preload else self.get_image(idx)
        image = self.preprocess_image(image)
        # Get the cameras (intrinsics and pose).
        intr, pose = self.cameras[idx] if self.preload else self.get_camera(idx)
        intr, pose = self.preprocess_camera(intr, pose, image_size_raw)
        # Pre-sample ray indices.
        if self.split == "train":
            ray_idx = torch.randperm(self.H * self.W)[:self.num_rays]  # [R]
            image_sampled = image.flatten(1, 2)[:, ray_idx].t()  # [R,3]
            sample.update(
                ray_idx=ray_idx,
                image_sampled=image_sampled,
                intr=intr,
                pose=pose,
            )
        else:  # keep image during inference
            sample.update(
                image=image,
                intr=intr,
                pose=pose,
            )
        return sample

    def get_image(self, idx):
        """从 camera.pkl 中获取图像"""
        cam = self.camera_list[idx]
        # image_cv 是 OpenCV 格式 (BGR, HWC, numpy array 或 tensor)
        image_cv = cam.image_cv

        # 从 BGR 转换为 RGB，然后创建 PIL Image
        image_rgb = image_cv[..., ::-1].copy()  # BGR to RGB
        image = Image.fromarray(image_rgb)
        image_size_raw = image.size  # (W, H)
        return image, image_size_raw

    def preprocess_image(self, image):
        # Resize the image.
        image = image.resize((self.W, self.H))
        image = torchvision_F.to_tensor(image)
        rgb = image[:3]
        return rgb

    def get_camera(self, idx):
        """从 camera.pkl 中获取相机内参和位姿"""
        cam = self.camera_list[idx]

        # 转换 OpenGL 坐标系到 CV 坐标系
        c2w = self._gl_to_cv(cam.camera2world)

        '''
        # 场景归一化：中心化
        center = self.sphere_center.copy()
        center += np.array(getattr(self.readjust, "center", [0])) if self.readjust else 0.
        c2w[:3, -1] -= center

        # 场景归一化：缩放
        scale = self.sphere_radius
        scale *= getattr(self.readjust, "scale", 1.) if self.readjust else 1.
        c2w[:3, -1] /= scale
        '''

        # 计算 w2c (world to camera)
        w2c = camera.Pose().invert(c2w[:3])
        return cam.intrinsic, w2c

    def preprocess_camera(self, intr, pose, image_size_raw):
        # Adjust the intrinsics according to the resized image.
        intr = intr.clone()
        raw_W, raw_H = image_size_raw
        intr[0] *= self.W / raw_W
        intr[1] *= self.H / raw_H
        return intr, pose

    def _gl_to_cv(self, gl):
        # convert to CV convention used in Imaginaire
        cv = gl * torch.tensor([1, -1, -1, 1])
        return cv
