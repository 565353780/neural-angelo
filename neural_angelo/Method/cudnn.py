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

import torch.backends.cudnn as cudnn


def init_cudnn(deterministic, benchmark):
    """初始化 cudnn 模块。需要考虑的两个方面是是否使用 cudnn benchmark 和是否使用 cudnn deterministic。
    如果设置了 cudnn benchmark，则 cudnn deterministic 自动为 False。

    Args:
        deterministic (bool): 是否使用 cudnn deterministic。
        benchmark (bool): 是否使用 cudnn benchmark。
    """
    cudnn.deterministic = deterministic
    cudnn.benchmark = benchmark
    print('cudnn benchmark: {}'.format(benchmark))
    print('cudnn deterministic: {}'.format(deterministic))
