import torch
import random

from neural_angelo.Dataset.data import Dataset
from neural_angelo.Dataset.mesh_image import MeshImageDataset
from neural_angelo.Module.multi_epochs_dataloader import MultiEpochsDataLoader


def _create_dataloader(cfg, is_inference, batch_size, shuffle, use_multi_epoch_loader,
                       subset_indices=None, subset_size=None):
    """创建数据加载器的通用函数。"""
    dataset = MeshImageDataset(cfg, is_inference=is_inference)
    split_name = "Val" if is_inference else "Train"

    # 处理 subset：优先使用传入的 indices，否则根据 subset_size 随机选取
    if subset_indices is None and subset_size is not None:
        dataset_len = len(dataset)
        subset_size = min(subset_size, dataset_len)
        subset_indices = sorted(random.sample(range(dataset_len), subset_size))
        print(f'{split_name} subset: randomly selected {subset_size} from {dataset_len}')

    if subset_indices is not None:
        dataset = torch.utils.data.Subset(dataset, subset_indices)

    print(f'{split_name} dataset length: {len(dataset)}')

    num_workers = getattr(cfg.data, 'num_workers', 8)
    persistent_workers = num_workers > 0 and getattr(cfg.data, 'persistent_workers', False)
    LoaderClass = MultiEpochsDataLoader if use_multi_epoch_loader else torch.utils.data.DataLoader

    return LoaderClass(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=persistent_workers
    )


def get_train_dataloader(cfg, shuffle=True, subset_indices=None, **_):
    return _create_dataloader(
        cfg,
        is_inference=False,
        batch_size=cfg.data.train.batch_size,
        shuffle=shuffle,
        use_multi_epoch_loader=cfg.data.use_multi_epoch_loader,
        subset_indices=subset_indices,
    )


def get_val_dataloader(cfg, subset_indices=None):
    return _create_dataloader(
        cfg,
        is_inference=True,
        batch_size=cfg.data.val.batch_size,
        shuffle=False,
        use_multi_epoch_loader=False,
        subset_indices=subset_indices,
        subset_size=getattr(cfg.data.val, 'subset', None),
    )
