import torch
import torch.nn.functional as F
from functools import partial

from neural_angelo.Model.Layer.positional_encoding import positional_encoding
from neural_angelo.Model.Layer.mlp_with_skip_connection import MLPwithSkipConnection
from neural_angelo.Method.spherical_harmonics import get_spherical_harmonics


class BackgroundNeRF(torch.nn.Module):

    def __init__(self, cfg_background, appear_embed):
        super().__init__()
        self.cfg_background = cfg_background
        self.cfg_appear_embed = appear_embed

        # Positional encoding.
        encoding_dim = 8 * cfg_background.encoding.levels

        # View encoding.
        self.spherical_harmonic_encoding = partial(
            get_spherical_harmonics,
            levels=cfg_background.encoding_view.levels,
        )
        encoding_view_dim = (cfg_background.encoding_view.levels + 1) ** 2

        input_dim = 4 + encoding_dim
        input_view_dim = cfg_background.mlp.hidden_dim + encoding_view_dim + \
            (appear_embed.dim if appear_embed.enabled else 0)

        # Point-wise feature.
        layer_dims = [input_dim] + [cfg_background.mlp.hidden_dim] * (cfg_background.mlp.num_layers - 1) + [cfg_background.mlp.hidden_dim + 1]
        self.mlp_feat = MLPwithSkipConnection(
            layer_dims,
            skip_connection=cfg_background.mlp.skip,
            activ=F.relu,
        )

        # RGB prediction.
        layer_dims_rgb = [input_view_dim] + [cfg_background.mlp.hidden_dim_rgb] * (cfg_background.mlp.num_layers_rgb - 1) + [3]
        self.mlp_rgb = MLPwithSkipConnection(
            layer_dims_rgb,
            skip_connection=cfg_background.mlp.skip_rgb,
            activ=F.relu,
        )

        self.activ_density = torch.nn.Softplus()
        return

    def forward(self, points_3D, rays_unit, app_outside):
        points_enc = self.encode(points_3D)  # [...,4+LD]
        # Volume density prediction.
        out = self.mlp_feat(points_enc)
        density, feat = self.activ_density(out[..., 0]), self.mlp_feat.activ(out[..., 1:])  # [...],[...,K]
        # RGB color prediction.
        if self.cfg_background.view_dep:
            view_enc = self.spherical_harmonic_encoding(rays_unit)  # [...,LD]

            input_list = [feat, view_enc]
            if app_outside is not None:
                input_list.append(app_outside)
            input_vec = torch.cat(input_list, dim=-1)
            rgb = self.mlp_rgb(input_vec).sigmoid_()  # [...,3]
        else:
            raise NotImplementedError
        return rgb, density

    def encode(self, points_3D):
        # Reparametrize the 3D points.
        # FIXME: review this
        points_3D_norm = points_3D.norm(dim=-1, keepdim=True)  # [B,R,N,1]
        points = torch.cat([points_3D / points_3D_norm, 1.0 / points_3D_norm], dim=-1)  # [B,R,N,4]

        # Positional encoding.
        points_enc = positional_encoding(points, num_freq_bases=self.cfg_background.encoding.levels)

        # TODO: 1/x?
        points_enc = torch.cat([points, points_enc], dim=-1)  # [B,R,N,4+LD]
        return points_enc
