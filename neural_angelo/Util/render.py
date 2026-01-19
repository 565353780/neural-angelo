import torch


def volume_rendering_alphas_dist(densities, dists, dist_far=None):
    """The volume rendering function. Details can be found in the NeRF paper.
    Args:
        densities (tensor [batch,ray,samples]): The predicted volume density samples.
        dists (tensor [batch,ray,samples,1]): The corresponding distance samples.
        dist_far (tensor [batch,ray,1,1]): The farthest distance for computing the last interval.
    Returns:
        alphas (tensor [batch,ray,samples,1]): The occupancy of each sampled point along the ray (in [0,1]).
    """
    if dist_far is None:
        dist_far = torch.empty_like(dists[..., :1, :]).fill_(1e10)  # [B,R,1,1]
    dists = torch.cat([dists, dist_far], dim=2)  # [B,R,N+1,1]
    # Volume rendering: compute rendering weights (using quadrature).
    dist_intvs = dists[..., 1:, 0] - dists[..., :-1, 0]  # [B,R,N]
    sigma_delta = densities * dist_intvs  # [B,R,N]
    alphas = 1 - (-sigma_delta).exp_()  # [B,R,N]
    return alphas


def alpha_compositing_weights(alphas):
    """Alpha compositing of (sampled) MPIs given their RGBs and alphas.
    Args:
        alphas (tensor [batch,ray,samples]): The predicted opacity values.
    Returns:
        weights (tensor [batch,ray,samples,1]): The predicted weight of each MPI (in [0,1]).
    """
    alphas_front = torch.cat([torch.zeros_like(alphas[..., :1]),
                              alphas[..., :-1]], dim=2)  # [B,R,N]
    #with autocast('cuda'):  # TODO: may be unstable in some cases.
    visibility = (1 - alphas_front).cumprod(dim=2)  # [B,R,N]
    weights = (alphas * visibility)[..., None]  # [B,R,N,1]
    return weights


def composite(quantities, weights):
    """Composite the samples to render the RGB/depth/opacity of the corresponding pixels.
    Args:
        quantities (tensor [batch,ray,samples,k]): The quantity to be weighted summed.
        weights (tensor [batch,ray,samples,1]): The predicted weight of each sampled point along the ray.
    Returns:
        quantity (tensor [batch,ray,k]): The expected (rendered) quantity.
    """
    # Integrate RGB and depth weighted by probability.
    quantity = (quantities * weights).sum(dim=2)  # [B,R,K]
    return quantity
