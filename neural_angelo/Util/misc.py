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

import collections
import functools
import os
import signal
import time
from collections import OrderedDict

import torch
import torch.nn.functional as F

from neural_angelo.Util.termcolor import alert, PP  # noqa

string_classes = (str, bytes)


def santize_args(name, locals_fn):
    args = {k: v for k, v in locals_fn.items()}
    if 'kwargs' in args and args['kwargs']:
        unused = PP(args['kwargs'])
        alert(f'{name}: Unused kwargs\n{unused}')

    keys_to_remove = ['self', 'kwargs']
    for k in keys_to_remove:
        args.pop(k, None)
    alert(f'{name}: Used args\n{PP(args)}', 'green')
    return args


def split_labels(labels, label_lengths):
    """将连接的标签分割成各个部分。

    Args:
        labels (torch.Tensor): 通过连接获得的标签。
        label_lengths (OrderedDict): 包含标签顺序及其长度。

    Returns:
        分割后的标签字典。
    """
    assert isinstance(label_lengths, OrderedDict)
    start = 0
    outputs = {}
    for data_type, length in label_lengths.items():
        end = start + length
        if labels.dim() == 5:
            outputs[data_type] = labels[:, :, start:end]
        elif labels.dim() == 4:
            outputs[data_type] = labels[:, start:end]
        elif labels.dim() == 3:
            outputs[data_type] = labels[start:end]
        start = end
    return outputs


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


def get_and_setattr(cfg, name, default):
    """获取带有默认选择的属性。如果属性不存在，则使用默认值设置它。

    Args:
        cfg (obj): 配置选项。
        name (str): 属性名称。
        default (obj): 默认属性。

    Returns:
        (obj): 所需的属性。
    """
    if not hasattr(cfg, name) or name not in cfg.__dict__:
        setattr(cfg, name, default)
    return getattr(cfg, name)


def get_nested_attr(cfg, attr_name, default):
    """迭代地尝试从 cfg 获取属性。如果未找到，返回默认值。

    Args:
        cfg (obj): 配置文件。
        attr_name (str): 属性名称（例如 XXX.YYY.ZZZ）。
        default (obj): 属性的默认返回值。

    Returns:
        (obj): 属性值。
    """
    names = attr_name.split('.')
    atr = cfg
    for name in names:
        if not hasattr(atr, name):
            return default
        atr = getattr(atr, name)
    return atr


def gradient_norm(model):
    """返回模型的梯度范数。

    Args:
        model (PyTorch module): 你的网络。
    """
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** (1. / 2)


def random_shift(x, offset=0.05, mode='bilinear', padding_mode='reflection'):
    """随机平移输入张量。

    Args:
        x (4D tensor): 输入的图像批次。
        offset (int): 最大偏移比率，范围在 [0, 1] 之间。
            每个方向的最大平移量为 offset * image_size。
        mode (str): 'F.grid_sample' 的重采样模式。
        padding_mode (str): 'F.grid_sample' 的填充模式。

    Returns:
        x (4D tensor): 随机平移后的图像。
    """
    assert x.dim() == 4, "Input must be a 4D tensor."
    batch_size = x.size(0)
    theta = torch.eye(2, 3, device=x.device).unsqueeze(0).repeat(
        batch_size, 1, 1)
    theta[:, :, 2] = 2 * offset * torch.rand(batch_size, 2) - offset
    grid = F.affine_grid(theta, x.size())
    x = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode, align_corners=False)
    return x


def apply_imagenet_normalization(input):
    """使用 ImageNet 均值和标准差进行归一化。

    Args:
        input (4D tensor NxCxHxW): 输入图像，假设范围为 [-1, 1]。

    Returns:
        使用 ImageNet 归一化的输入。
    """
    # 将输入归一化回 [0, 1]
    normalized_input = (input + 1) / 2
    # 使用 ImageNet 均值和标准差归一化输入
    mean = normalized_input.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = normalized_input.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    output = (normalized_input - mean) / std
    return output


def alarm_handler(timeout_period, signum, frame):
    """当进程卡住时的处理函数。目前，我们只是结束进程。"""
    error_message = f"Timeout error! More than {timeout_period} seconds have passed since the last iteration. Most " \
                    f"likely the process has been stuck due to node failure or PBSS error."
    ngc_job_id = os.environ.get('NGC_JOB_ID', None)
    if ngc_job_id is not None:
        error_message += f" Failed NGC job ID: {ngc_job_id}."
    alert(error_message)
    exit()


class Timer(object):
    """计时器类，用于跟踪训练时间。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.time_iteration = 0
        self.time_epoch = 0
        # 设置超时信号处理器
        signal.signal(signal.SIGALRM, functools.partial(alarm_handler, self.cfg.timeout_period))

    def reset(self):
        self.accu_forw_iter_time = 0
        self.accu_loss_iter_time = 0
        self.accu_back_iter_time = 0
        self.accu_step_iter_time = 0
        self.accu_avg_iter_time = 0

    def _time_before_forward(self):
        """记录前向传播之前的时间。"""
        if self.cfg.speed_benchmark:
            torch.cuda.synchronize()
            self.forw_time = time.time()

    def _time_before_loss(self):
        """记录计算损失之前的时间。"""
        if self.cfg.speed_benchmark:
            torch.cuda.synchronize()
            self.loss_time = time.time()

    def _time_before_backward(self):
        """记录反向传播之前的时间。"""
        if self.cfg.speed_benchmark:
            torch.cuda.synchronize()
            self.back_time = time.time()

    def _time_before_step(self):
        """记录更新权重之前的时间。"""
        if self.cfg.speed_benchmark:
            torch.cuda.synchronize()
            self.step_time = time.time()

    def _time_before_model_avg(self):
        """记录应用模型平均之前的时间。"""
        if self.cfg.speed_benchmark:
            torch.cuda.synchronize()
            self.avg_time = time.time()

    def _time_before_leave_gen(self):
        """记录网络更新的前向、反向、损失和模型平均时间。"""
        if self.cfg.speed_benchmark:
            torch.cuda.synchronize()
            end_time = time.time()
            self.accu_forw_iter_time += self.loss_time - self.forw_time
            self.accu_loss_iter_time += self.back_time - self.loss_time
            self.accu_back_iter_time += self.step_time - self.back_time
            self.accu_step_iter_time += self.avg_time - self.step_time
            self.accu_avg_iter_time += end_time - self.avg_time

    def _print_speed_benchmark(self, avg_time):
        """打印性能分析结果并重置计时器。"""
        print('{:6f}'.format(avg_time))
        print('\tModel FWD time {:6f}'.format(self.accu_forw_iter_time / self.cfg.logging_iter))
        print('\tModel LOS time {:6f}'.format(self.accu_loss_iter_time / self.cfg.logging_iter))
        print('\tModel BCK time {:6f}'.format(self.accu_back_iter_time / self.cfg.logging_iter))
        print('\tModel STP time {:6f}'.format(self.accu_step_iter_time / self.cfg.logging_iter))
        print('\tModel AVG time {:6f}'.format(self.accu_avg_iter_time / self.cfg.logging_iter))
        self.accu_forw_iter_time = 0
        self.accu_loss_iter_time = 0
        self.accu_back_iter_time = 0
        self.accu_step_iter_time = 0
        self.accu_avg_iter_time = 0

    def checkpoint_tic(self):
        """重置检查点计时器。"""
        self.checkpoint_start_time = time.time()

    def checkpoint_toc(self):
        """返回自上次重置以来的时间（分钟）。"""
        return (time.time() - self.checkpoint_start_time) / 60

    def reset_timeout_counter(self):
        """重置超时计数器。"""
        signal.alarm(self.cfg.timeout_period)
