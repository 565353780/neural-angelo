import torch
import numpy as np
import tinycudann as tcnn

from neural_angelo.Model.Layer.mlp import MLPforNeuralSDF


class NeuralSDF(torch.nn.Module):

    def __init__(self, cfg_sdf):
        super().__init__()
        self.cfg_sdf = cfg_sdf
        encoding_dim = self.build_encoding(cfg_sdf.encoding)
        input_dim = 3 + encoding_dim
        self.build_mlp(cfg_sdf.mlp, input_dim=input_dim)

    def build_encoding(self, cfg_encoding):
        # Build the multi-resolution hash grid.
        l_min, l_max = cfg_encoding.hashgrid.min_logres, cfg_encoding.hashgrid.max_logres
        r_min, r_max = 2 ** l_min, 2 ** l_max
        num_levels = cfg_encoding.levels
        self.growth_rate = np.exp((np.log(r_max) - np.log(r_min)) / (num_levels - 1))
        config = dict(
            otype="HashGrid",
            n_levels=cfg_encoding.levels,
            n_features_per_level=cfg_encoding.hashgrid.dim,
            log2_hashmap_size=cfg_encoding.hashgrid.dict_size,
            base_resolution=2 ** cfg_encoding.hashgrid.min_logres,
            per_level_scale=self.growth_rate,
        )
        self.tcnn_encoding = tcnn.Encoding(3, config)
        self.resolutions = []
        for lv in range(0, num_levels):
            size = np.floor(r_min * self.growth_rate ** lv).astype(int) + 1
            self.resolutions.append(size)
        encoding_dim = cfg_encoding.hashgrid.dim * cfg_encoding.levels
        return encoding_dim

    def build_mlp(self, cfg_mlp, input_dim=3):
        # SDF + point-wise feature
        layer_dims = [input_dim] + [cfg_mlp.hidden_dim] * cfg_mlp.num_layers + [cfg_mlp.hidden_dim]
        self.mlp = MLPforNeuralSDF(
            layer_dims,
            skip_connection=cfg_mlp.skip,
            use_weightnorm=cfg_mlp.weight_norm,
            geometric_init=cfg_mlp.geometric_init,
            out_bias=cfg_mlp.out_bias,
            invert=cfg_mlp.inside_out,
            activ_beta=cfg_mlp.activ_params.beta,
        )

    def forward(self, points_3D, with_sdf=True, with_feat=True):
        points_enc = self.encode(points_3D)  # [...,3+LD]
        sdf, feat = self.mlp(points_enc, with_sdf=with_sdf, with_feat=with_feat)
        return sdf, feat  # [...,1],[...,K]

    def sdf(self, points_3D):
        return self.forward(points_3D, with_sdf=True, with_feat=False)[0]

    def encode(self, points_3D):
        # Tri-linear interpolate the corresponding embeddings from the dictionary.
        vol_min, vol_max = self.cfg_sdf.encoding.hashgrid.range
        points_3D_normalized = (points_3D - vol_min) / (vol_max - vol_min)  # Normalize to [0,1].
        tcnn_input = points_3D_normalized.view(-1, 3)
        tcnn_output = self.tcnn_encoding(tcnn_input)
        points_enc = tcnn_output.view(*points_3D_normalized.shape[:-1], tcnn_output.shape[-1])
        feat_dim = self.cfg_sdf.encoding.hashgrid.dim

        # Coarse-to-fine.
        mask = self._get_coarse2fine_mask(points_enc, feat_dim=feat_dim)
        points_enc = points_enc * mask

        points_enc = torch.cat([points_3D, points_enc], dim=-1)  # [B,R,N,3+LD]
        return points_enc

    def set_active_levels(self, current_iter=None):
        anneal_levels = max((current_iter - self.warm_up_end) // self.cfg_sdf.encoding.coarse2fine.step, 1)
        self.anneal_levels = min(self.cfg_sdf.encoding.levels, anneal_levels)
        self.active_levels = max(self.cfg_sdf.encoding.coarse2fine.init_active_level, self.anneal_levels)

    def set_normal_epsilon(self):
        epsilon_res = self.resolutions[self.anneal_levels - 1]
        self.normal_eps = 1. / epsilon_res

    @torch.no_grad()
    def _get_coarse2fine_mask(self, points_enc, feat_dim):
        mask = torch.zeros_like(points_enc)
        mask[..., :(self.active_levels * feat_dim)] = 1
        return mask

    def compute_gradients(self, x, training=False, sdf=None):
        # Note: hessian is not fully hessian but diagonal elements
        if self.cfg_sdf.gradient.taps == 6:
            eps = self.normal_eps
            # 1st-order gradient
            eps_x = torch.tensor([eps, 0., 0.], dtype=x.dtype, device=x.device)  # [3]
            eps_y = torch.tensor([0., eps, 0.], dtype=x.dtype, device=x.device)  # [3]
            eps_z = torch.tensor([0., 0., eps], dtype=x.dtype, device=x.device)  # [3]
            sdf_x_pos = self.sdf(x + eps_x)  # [...,1]
            sdf_x_neg = self.sdf(x - eps_x)  # [...,1]
            sdf_y_pos = self.sdf(x + eps_y)  # [...,1]
            sdf_y_neg = self.sdf(x - eps_y)  # [...,1]
            sdf_z_pos = self.sdf(x + eps_z)  # [...,1]
            sdf_z_neg = self.sdf(x - eps_z)  # [...,1]
            gradient_x = (sdf_x_pos - sdf_x_neg) / (2 * eps)
            gradient_y = (sdf_y_pos - sdf_y_neg) / (2 * eps)
            gradient_z = (sdf_z_pos - sdf_z_neg) / (2 * eps)
            gradient = torch.cat([gradient_x, gradient_y, gradient_z], dim=-1)  # [...,3]
            # 2nd-order gradient (hessian)
            if training:
                assert sdf is not None  # computed when feed-forwarding through the network
                hessian_xx = (sdf_x_pos + sdf_x_neg - 2 * sdf) / (eps ** 2)  # [...,1]
                hessian_yy = (sdf_y_pos + sdf_y_neg - 2 * sdf) / (eps ** 2)  # [...,1]
                hessian_zz = (sdf_z_pos + sdf_z_neg - 2 * sdf) / (eps ** 2)  # [...,1]
                hessian = torch.cat([hessian_xx, hessian_yy, hessian_zz], dim=-1)  # [...,3]
            else:
                hessian = None
        elif self.cfg_sdf.gradient.taps == 4:
            eps = self.normal_eps / np.sqrt(3)
            k1 = torch.tensor([1, -1, -1], dtype=x.dtype, device=x.device)  # [3]
            k2 = torch.tensor([-1, -1, 1], dtype=x.dtype, device=x.device)  # [3]
            k3 = torch.tensor([-1, 1, -1], dtype=x.dtype, device=x.device)  # [3]
            k4 = torch.tensor([1, 1, 1], dtype=x.dtype, device=x.device)  # [3]
            sdf1 = self.sdf(x + k1 * eps)  # [...,1]
            sdf2 = self.sdf(x + k2 * eps)  # [...,1]
            sdf3 = self.sdf(x + k3 * eps)  # [...,1]
            sdf4 = self.sdf(x + k4 * eps)  # [...,1]
            gradient = (k1*sdf1 + k2*sdf2 + k3*sdf3 + k4*sdf4) / (4.0 * eps)
            if training:
                assert sdf is not None  # computed when feed-forwarding through the network
                # the result of 4 taps is directly trace, but we assume they are individual components
                # so we use the same signature as 6 taps
                hessian_xx = ((sdf1 + sdf2 + sdf3 + sdf4) / 2.0 - 2 * sdf) / eps ** 2   # [N,1]
                hessian = torch.cat([hessian_xx, hessian_xx, hessian_xx], dim=-1) / 3.0
            else:
                hessian = None
        else:
            raise ValueError("Only support 4 or 6 taps.")
        return gradient, hessian
