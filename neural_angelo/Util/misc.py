import torch

string_classes = (str, bytes)


def requires_grad(model, require=True):
    """设置模型是否需要梯度。

    Args:
        model (nn.Module): 神经网络模型。
        require (bool): 网络是否需要梯度。
    """
    for p in model.parameters():
        p.requires_grad = require


def to_device(data, device):
    """将数据中的所有张量移动到指定设备。

    Args:
        data (dict, list, or tensor): 输入数据。
        device (str): 'cpu' 或 'cuda'。
    """
    if isinstance(device, str):
        device = torch.device(device)
    assert isinstance(device, torch.device)

    if isinstance(data, torch.Tensor):
        data = data.to(device)
        return data
    elif isinstance(data, dict):
        return {key: to_device(data[key], device) for key in data}
    elif isinstance(data, (list, tuple)) and not isinstance(data, string_classes):
        return type(data)([to_device(d, device) for d in data])
    else:
        return data


def to_cuda(data):
    """将数据中的所有张量移动到 GPU。

    Args:
        data (dict, list, or tensor): 输入数据。
    """
    return to_device(data, 'cuda')


def to_cpu(data):
    """将数据中的所有张量移动到 CPU。

    Args:
        data (dict, list, or tensor): 输入数据。
    """
    return to_device(data, 'cpu')



