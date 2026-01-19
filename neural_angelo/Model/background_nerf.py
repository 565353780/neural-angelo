import torch
from functools import partial

from projects.nerf.utils import nerf_util
from projects.neuralangelo.utils.misc import get_activation
from projects.neuralangelo.utils.spherical_harmonics import get_spherical_harmonics


class BackgroundNeRF(torch.nn.Module):

    def __init__(self, cfg_background, appear_embed):
        super().__init__()
        self.cfg_background = cfg_background
        self.cfg_appear_embed = appear_embed
        encoding_dim, encoding_view_dim = self.build_encoding(cfg_background.encoding, cfg_background.encoding_view)
        input_dim = 4 + encoding_dim
        input_view_dim = cfg_background.mlp.hidden_dim + encoding_view_dim + \
            (appear_embed.dim if appear_embed.enabled else 0)
        self.build_mlp(cfg_background.mlp, input_dim=input_dim, input_view_dim=input_view_dim)

    def build_encoding(self, cfg_encoding, cfg_encoding_view):
        # Positional encoding.
        if cfg_encoding.type == "fourier":
            encoding_dim = 8 * cfg_encoding.levels
        else:
            raise NotImplementedError("Unknown encoding type")
        # View encoding.
        if cfg_encoding_view.type == "fourier":
            encoding_view_dim = 6 * cfg_encoding_view.levels
        elif cfg_encoding_view.type == "spherical":
            self.spherical_harmonic_encoding = partial(get_spherical_harmonics, levels=cfg_encoding_view.levels)
            encoding_view_dim = (cfg_encoding_view.levels + 1) ** 2
        else:
            raise NotImplementedError("Unknown encoding type")
        return encoding_dim, encoding_view_dim

    def build_mlp(self, cfg_mlp, input_dim=3, input_view_dim=3):
        activ = get_activation(cfg_mlp.activ, **cfg_mlp.activ_params)
        # Point-wise feature.
        layer_dims = [input_dim] + [cfg_mlp.hidden_dim] * (cfg_mlp.num_layers - 1) + [cfg_mlp.hidden_dim + 1]
        self.mlp_feat = nerf_util.MLPwithSkipConnection(layer_dims, skip_connection=cfg_mlp.skip, activ=activ)
        self.activ_density = get_activation(cfg_mlp.activ_density, **cfg_mlp.activ_density_params)
        # RGB prediction.
        layer_dims_rgb = [input_view_dim] + [cfg_mlp.hidden_dim_rgb] * (cfg_mlp.num_layers_rgb - 1) + [3]
        self.mlp_rgb = nerf_util.MLPwithSkipConnection(layer_dims_rgb, skip_connection=cfg_mlp.skip_rgb, activ=activ)

    def forward(self, points_3D, rays_unit, app_outside):
        points_enc = self.encode(points_3D)  # [...,4+LD]
        # Volume density prediction.
        out = self.mlp_feat(points_enc)
        density, feat = self.activ_density(out[..., 0]), self.mlp_feat.activ(out[..., 1:])  # [...],[...,K]
        # RGB color prediction.
        if self.cfg_background.view_dep:
            view_enc = self.encode_view(rays_unit)  # [...,LD]
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
        if self.cfg_background.encoding.type == "fourier":
            points_enc = nerf_util.positional_encoding(points, num_freq_bases=self.cfg_background.encoding.levels)
        else:
            raise NotImplementedError("Unknown encoding type")
        # TODO: 1/x?
        points_enc = torch.cat([points, points_enc], dim=-1)  # [B,R,N,4+LD]
        return points_enc

    def encode_view(self, rays_unit):
        if self.cfg_background.encoding_view.type == "fourier":
            view_enc = nerf_util.positional_encoding(rays_unit, num_freq_bases=self.cfg_background.encoding_view.levels)
        elif self.cfg_background.encoding_view.type == "spherical":
            view_enc = self.spherical_harmonic_encoding(rays_unit)
        else:
            raise NotImplementedError("Unknown encoding type")
        return view_enc
