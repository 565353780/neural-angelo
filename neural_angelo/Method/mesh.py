import torch
import mcubes
import trimesh
import numpy as np
import torch.nn.functional as torch_F
from tqdm import tqdm

from neural_angelo.Module.lattice_grid import LatticeGrid


def get_lattice_grid_loader(dataset, num_workers=8):
    """创建网格数据加载器"""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False
    )


def filter_points_outside_bounding_sphere(old_mesh):
    """过滤单位球外的顶点"""
    mask = np.linalg.norm(old_mesh.vertices, axis=-1) < 1.0
    if np.any(mask):
        indices = np.ones(len(old_mesh.vertices), dtype=int) * -1
        indices[mask] = np.arange(mask.sum())
        faces_mask = mask[old_mesh.faces[:, 0]] & mask[old_mesh.faces[:, 1]] & mask[old_mesh.faces[:, 2]]
        new_faces = indices[old_mesh.faces[faces_mask]]
        new_vertices = old_mesh.vertices[mask]
        new_colors = old_mesh.visual.vertex_colors[mask] if hasattr(old_mesh.visual, 'vertex_colors') else None
        new_mesh = trimesh.Trimesh(new_vertices, new_faces, vertex_colors=new_colors)
    else:
        new_mesh = trimesh.Trimesh()
    return new_mesh


def filter_largest_cc(mesh):
    """只保留最大连通分量"""
    components = mesh.split(only_watertight=False)
    areas = np.array([c.area for c in components], dtype=float)
    if len(areas) > 0 and mesh.vertices.shape[0] > 0:
        new_mesh = components[areas.argmax()]
    else:
        new_mesh = trimesh.Trimesh()
    return new_mesh


def marching_cubes(sdf, xyz, intv, texture_func=None, filter_lcc=False):
    """执行 Marching Cubes 算法"""
    V, F = mcubes.marching_cubes(sdf, 0.)
    if V.shape[0] > 0:
        V = V * intv + xyz[0, 0, 0]
        if texture_func is not None:
            C = texture_func(V)
            mesh = trimesh.Trimesh(V, F, vertex_colors=C)
        else:
            mesh = trimesh.Trimesh(V, F)
        mesh = filter_points_outside_bounding_sphere(mesh)
        mesh = filter_largest_cc(mesh) if filter_lcc else mesh
    else:
        mesh = trimesh.Trimesh()
    return mesh


@torch.no_grad()
def extract_mesh_from_sdf(sdf_func, bounds, intv, block_res=64, texture_func=None, filter_lcc=False):
    """
    提取三角网格

    Args:
        sdf_func: SDF 函数，输入点坐标，输出 SDF 值
        bounds: 边界范围 [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        intv: 采样间隔
        block_res: 分块分辨率
        texture_func: 纹理函数（可选）
        filter_lcc: 是否只保留最大连通分量

    Returns:
        trimesh.Trimesh: 提取的三角网格
    """
    lattice_grid = LatticeGrid(bounds, intv=intv, block_res=block_res)
    data_loader = get_lattice_grid_loader(lattice_grid)
    mesh_blocks = []

    for it, data in enumerate(tqdm(data_loader, leave=False, desc="提取网格")):
        xyz = data["xyz"][0]
        xyz_cuda = xyz.cuda()
        sdf_cuda = sdf_func(xyz_cuda)[..., 0]
        sdf = sdf_cuda.cpu()
        mesh = marching_cubes(sdf.numpy(), xyz.numpy(), intv, texture_func, filter_lcc)
        if mesh.vertices.shape[0] > 0:
            mesh_blocks.append(mesh)

    if len(mesh_blocks) > 0:
        mesh = trimesh.util.concatenate(mesh_blocks)
    else:
        mesh = trimesh.Trimesh()

    return mesh


@torch.no_grad()
def extract_texture(xyz, neural_rgb, neural_sdf, appear_embed):
    """
    提取顶点颜色

    Args:
        xyz: 顶点坐标 (numpy array)
        neural_rgb: RGB 网络
        neural_sdf: SDF 网络
        appear_embed: 外观嵌入（可为 None）

    Returns:
        numpy array: 顶点颜色 (RGB, uint8)
    """
    num_samples, _ = xyz.shape
    # Get model device to ensure tensors are on the correct device
    model_device = next(neural_sdf.mlp.parameters()).device
    xyz_cuda = torch.from_numpy(xyz).float().to(model_device)[None, None]  # [N,3] -> [1,1,N,3]
    sdfs, feats = neural_sdf(xyz_cuda)
    gradients, _ = neural_sdf.compute_gradients(xyz_cuda, training=False, sdf=sdfs)
    normals = torch_F.normalize(gradients, dim=-1)

    if appear_embed is not None:
        feat_dim = appear_embed.embedding_dim
        app = torch.zeros([1, 1, num_samples, feat_dim], device=model_device)
    else:
        app = None

    rgbs = neural_rgb.forward(xyz_cuda, normals, -normals, feats, app=app)  # [1,1,N,3]
    return (rgbs.squeeze().cpu().numpy() * 255).astype(np.uint8)
