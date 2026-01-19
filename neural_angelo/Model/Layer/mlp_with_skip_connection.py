import torch
import torch.nn.functional as F

class MLPwithSkipConnection(torch.nn.Module):

    def __init__(self, layer_dims, skip_connection=[], activ=None, use_layernorm=False, use_weightnorm=False):
        """Initialize a multi-layer perceptron with skip connection.
        Args:
            layer_dims: A list of integers representing the number of channels in each layer.
            skip_connection: A list of integers representing the index of layers to add skip connection.
        """
        super().__init__()
        self.skip_connection = skip_connection
        self.use_layernorm = use_layernorm
        self.linears = torch.nn.ModuleList()
        if use_layernorm:
            self.layer_norm = torch.nn.ModuleList()
        layer_dim_pairs = list(zip(layer_dims[:-1], layer_dims[1:]))
        for li, (k_in, k_out) in enumerate(layer_dim_pairs):
            if li in self.skip_connection:
                k_in += layer_dims[0]
            linear = torch.nn.Linear(k_in, k_out)
            if use_weightnorm:
                linear = torch.nn.utils.parametrizations.weight_norm(linear)
            self.linears.append(linear)
            if use_layernorm and li != len(layer_dim_pairs) - 1:
                self.layer_norm.append(torch.nn.LayerNorm(k_out))
            if li == len(layer_dim_pairs) - 1:
                self.linears[-1].bias.data.fill_(0.0)
        self.activ = activ or F.relu_

    def forward(self, input):
        feat = input
        for li, linear in enumerate(self.linears):
            if li in self.skip_connection:
                feat = torch.cat([feat, input], dim=-1)
            feat = linear(feat)
            if li != len(self.linears) - 1:
                if self.use_layernorm:
                    feat = self.layer_norm[li](feat)
                feat = self.activ(feat)
        return feat
