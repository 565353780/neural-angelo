import torch
import random
import numpy as np


def set_random_seed(seed):
    """设置所有随机种子，包括 random, numpy, torch.manual_seed, torch.cuda_manual_seed。

    Args:
        seed (int): 随机种子。
    """
    print(f"Using random seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
