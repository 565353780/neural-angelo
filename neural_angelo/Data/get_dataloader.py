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

from neural_angelo.Dataset.data import Dataset
from neural_angelo.Data.dataloader import MultiEpochsDataLoader


def _get_train_dataset_objects(cfg, subset_indices=None):
    """返回训练集的数据集对象。

    Args:
        cfg (obj): 全局配置。
        subset_indices (sequence): 要使用的子集索引。

    Returns:
        train_dataset (obj): PyTorch 训练数据集对象。
    """
    train_dataset = Dataset(cfg, is_inference=False)
    if subset_indices is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, subset_indices)
    print('Train dataset length:', len(train_dataset))
    return train_dataset


def _get_val_dataset_objects(cfg, subset_indices=None):
    """返回验证集的数据集对象。

    Args:
        cfg (obj): 全局配置。
        subset_indices (sequence): 要使用的子集索引。

    Returns:
        val_dataset (obj): PyTorch 验证数据集对象。
    """
    if hasattr(cfg.data.val, 'type'):
        for key in ['type', 'input_types', 'input_image']:
            setattr(cfg.data, key, getattr(cfg.data.val, key))
    val_dataset = Dataset(cfg, is_inference=True)

    if subset_indices is not None:
        val_dataset = torch.utils.data.Subset(val_dataset, subset_indices)
    print('Val dataset length:', len(val_dataset))
    return val_dataset


def _get_test_dataset_object(cfg, subset_indices=None):
    """返回测试集的数据集对象。

    Args:
        cfg (obj): 全局配置。
        subset_indices (sequence): 要使用的子集索引。

    Returns:
        (obj): PyTorch 数据集对象。
    """
    test_dataset = Dataset(cfg, is_inference=True)
    if subset_indices is not None:
        test_dataset = torch.utils.data.Subset(test_dataset, subset_indices)
    return test_dataset


def _get_data_loader(cfg, dataset, batch_size, shuffle=True, drop_last=True,
                     use_multi_epoch_loader=False):
    """返回数据加载器。

    Args:
        cfg (obj): 全局配置。
        dataset (obj): PyTorch 数据集对象。
        batch_size (int): 批量大小。
        shuffle (bool): 是否打乱数据。
        drop_last (bool): 如果样本数量小于批量大小，是否丢弃最后一个批次。
        use_multi_epoch_loader (bool): 是否使用多 epoch 数据加载器。

    Returns:
        (obj): 数据加载器。
    """
    num_workers = getattr(cfg.data, 'num_workers', 8)
    persistent_workers = getattr(cfg.data, 'persistent_workers', False)
    data_loader = (MultiEpochsDataLoader if use_multi_epoch_loader else torch.utils.data.DataLoader)(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=drop_last,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )
    return data_loader


def get_train_dataloader(cfg, shuffle=True, drop_last=True, subset_indices=None, seed=0, preemptable=False):
    """返回训练数据加载器。

    Args:
        cfg (obj): 全局配置。
        shuffle (bool): 是否打乱数据。
        drop_last (bool): 如果样本数量小于批量大小，是否丢弃最后一个批次。
        subset_indices (sequence): 要使用的子集索引。
        seed (int): 随机种子（保留参数以保持接口兼容性）。
        preemptable (bool): 是否处理抢占（保留参数以保持接口兼容性）。

    Returns:
        train_data_loader (obj): 训练数据加载器。
    """
    train_dataset = _get_train_dataset_objects(cfg, subset_indices=subset_indices)
    train_data_loader = _get_data_loader(
        cfg, train_dataset, cfg.data.train.batch_size,
        shuffle=shuffle, drop_last=drop_last,
        use_multi_epoch_loader=cfg.data.use_multi_epoch_loader
    )
    return train_data_loader


def get_val_dataloader(cfg, subset_indices=None, seed=0):
    """返回验证数据加载器。

    Args:
        cfg (obj): 全局配置。
        subset_indices (sequence): 要使用的子集索引。
        seed (int): 随机种子（保留参数以保持接口兼容性）。

    Returns:
        val_data_loader (obj): 验证数据加载器。
    """
    val_dataset = _get_val_dataset_objects(cfg, subset_indices=subset_indices)
    drop_last = getattr(cfg.data.val, 'drop_last', False)
    val_data_loader = _get_data_loader(
        cfg, val_dataset, cfg.data.val.batch_size,
        shuffle=False, drop_last=drop_last,
        use_multi_epoch_loader=False
    )
    return val_data_loader


def get_test_dataloader(cfg, subset_indices=None):
    """返回测试数据加载器。

    Args:
        cfg (obj): 全局配置。
        subset_indices (sequence): 要使用的子集索引。

    Returns:
        (obj): 测试数据加载器（可能不包含真实标签）。
    """
    test_dataset = _get_test_dataset_object(cfg, subset_indices=subset_indices)
    test_data_loader = _get_data_loader(
        cfg, test_dataset, cfg.data.val.batch_size,
        shuffle=False, drop_last=False,
        use_multi_epoch_loader=False
    )
    return test_data_loader
