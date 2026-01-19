import torch
import numpy as np


class LatticeGrid(torch.utils.data.Dataset):
    """用于分块提取 SDF 的网格数据集"""

    def __init__(self, bounds, intv, block_res=64):
        super().__init__()
        self.block_res = block_res
        ((x_min, x_max), (y_min, y_max), (z_min, z_max)) = bounds
        self.x_grid = torch.arange(x_min, x_max, intv)
        self.y_grid = torch.arange(y_min, y_max, intv)
        self.z_grid = torch.arange(z_min, z_max, intv)
        res_x, res_y, res_z = len(self.x_grid), len(self.y_grid), len(self.z_grid)
        print(f"提取表面分辨率: {res_x} x {res_y} x {res_z}")
        self.num_blocks_x = int(np.ceil(res_x / block_res))
        self.num_blocks_y = int(np.ceil(res_y / block_res))
        self.num_blocks_z = int(np.ceil(res_z / block_res))

    def __getitem__(self, idx):
        sample = dict(idx=idx)
        block_idx_x = idx // (self.num_blocks_y * self.num_blocks_z)
        block_idx_y = (idx // self.num_blocks_z) % self.num_blocks_y
        block_idx_z = idx % self.num_blocks_z
        xi = block_idx_x * self.block_res
        yi = block_idx_y * self.block_res
        zi = block_idx_z * self.block_res
        x, y, z = torch.meshgrid(
            self.x_grid[xi:xi + self.block_res + 1],
            self.y_grid[yi:yi + self.block_res + 1],
            self.z_grid[zi:zi + self.block_res + 1],
            indexing="ij"
        )
        xyz = torch.stack([x, y, z], dim=-1)
        sample.update(xyz=xyz)
        return sample

    def __len__(self):
        return self.num_blocks_x * self.num_blocks_y * self.num_blocks_z
