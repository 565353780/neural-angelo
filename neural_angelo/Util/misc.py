import torch
import collections

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
    elif isinstance(data, collections.abc.Mapping):
        return type(data)({key: to_device(data[key], device) for key in data})
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
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


def to_half(data):
    """将所有浮点数转换为半精度。

    Args:
        data (dict, list or tensor): 输入数据。
    """
    if isinstance(data, torch.Tensor) and torch.is_floating_point(data):
        data = data.half()
        return data
    elif isinstance(data, collections.abc.Mapping):
        return type(data)({key: to_half(data[key]) for key in data})
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
        return type(data)([to_half(d) for d in data])
    else:
        return data


def to_float(data):
    """将所有半精度转换为浮点数。

    Args:
        data (dict, list or tensor): 输入数据。
    """
    if isinstance(data, torch.Tensor) and torch.is_floating_point(data):
        data = data.float()
        return data
    elif isinstance(data, collections.abc.Mapping):
        return type(data)({key: to_float(data[key]) for key in data})
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
        return type(data)([to_float(d) for d in data])
    else:
        return data


def slice_tensor(data, start, end):
    """对所有张量进行切片。

    Args:
        data (dict, list or tensor): 输入数据。
    """
    if isinstance(data, torch.Tensor):
        data = data[start:end]
        return data
    elif isinstance(data, collections.abc.Mapping):
        return type(data)({key: slice_tensor(data[key], start, end) for key in data})
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
        return type(data)([slice_tensor(d, start, end) for d in data])
    else:
        return data
