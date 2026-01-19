import torch
import torchvision
from matplotlib import pyplot as plt


def tensorboard_image(images, from_range=(0, 1)):
    """Convert images to TensorBoard format (torch.Tensor).
    
    Args:
        images: Input images tensor
        from_range: Range of input values (min, max)

    Returns:
        torch.Tensor: Image grid in format suitable for TensorBoard (C, H, W)
    """
    images = preprocess_image(images, from_range=from_range)
    image_grid = torchvision.utils.make_grid(images, nrow=1, pad_value=1)
    return image_grid


def preprocess_image(images, from_range=(0, 1), cmap="gray"):
    min, max = from_range
    images = (images - min) / (max - min)
    images = images.detach().cpu().float().clamp_(min=0, max=1)
    if images.shape[1] == 1:
        images = get_heatmap(images[:, 0], cmap=cmap)
    return images


def get_heatmap(gray, cmap):  # [N,H,W]
    color = plt.get_cmap(cmap)(gray.numpy())
    color = torch.from_numpy(color[..., :3]).permute(0, 3, 1, 2).float()  # [N,3,H,W]
    return color
