import torch
import numpy as np


def positional_encoding(input, num_freq_bases):
    """Encode input into position codes.
    Args:
        input (tensor [bs, ..., N]): A batch of data with N dimension.
        num_freq_bases: (int): The number of frequency base of the code.
    Returns:
        input_enc (tensor [bs, ..., 2*N*num_freq_bases]): Positional codes for input.
    """
    freq = 2 ** torch.arange(num_freq_bases, dtype=torch.float32, device=input.device) * np.pi  # [L].
    spectrum = input[..., None] * freq  # [B,...,N,L].
    sin, cos = spectrum.sin(), spectrum.cos()  # [B,...,N,L].
    input_enc = torch.stack([sin, cos], dim=-2)  # [B,...,N,2,L].
    input_enc = input_enc.view(*input.shape[:-1], -1)  # [B,...,2NL].
    return input_enc
