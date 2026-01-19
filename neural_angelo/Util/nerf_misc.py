import numpy as np

flip_mat = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])


def to_full_image(image, image_size=None, from_vec=True):
    # if from_vec is True: [B,HW,...,K] --> [B,K,H,W,...]
    # if from_vec is False: [B,H,W,...,K] --> [B,K,H,W,...]
    if from_vec:
        assert image_size is not None
        image = image.unflatten(dim=1, sizes=image_size)
    image = image.moveaxis(-1, 1)
    return image
