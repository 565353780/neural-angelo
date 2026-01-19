import torch


def curvature_loss(hessian, outside=None):
    """计算曲率 loss
    
    Args:
        hessian: [..., 3] Hessian 对角元素，支持 [B, R, N, 3] 或 [M, 3] 格式
        outside: 场景外标志（可选）
    
    Returns:
        loss: curvature loss 标量 (tensor)
    """
    if hessian is None:
        return torch.tensor(0.0)
    
    laplacian = hessian.sum(dim=-1).abs()  # [...]
    laplacian = laplacian.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    
    if outside is not None:
        # 处理不同维度的 outside 标志
        if outside.dim() < laplacian.dim():
            # [B, R, 1] -> [B, R, N] 通过 expand
            outside = outside.expand_as(laplacian)
        elif outside.dim() > laplacian.dim():
            # [B, R, 1] 与 [M] 格式不兼容，忽略 outside
            return laplacian.mean()
        return (laplacian * (~outside).float()).mean()
    else:
        return laplacian.mean()
