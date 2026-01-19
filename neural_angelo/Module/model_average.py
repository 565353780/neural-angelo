import copy
import torch
from torch import nn

from neural_angelo.Util.misc import requires_grad


class ModelAverage(nn.Module):
    r"""In this model average implementation, the spectral layers are
    absorbed in the model parameter by default. If such options are
    turned on, be careful with how you do the training. Remember to
    re-estimate the batch norm parameters before using the model.

    Args:
        module (torch nn module): Torch network.
        beta (float): Moving average weights. How much we weight the past.
        start_iteration (int): From which iteration, we start the update.
    """
    def __init__(self, module, beta=0.9999, start_iteration=0):
        super(ModelAverage, self).__init__()

        self.module = module
        # Get the device from the module
        module_device = next(module.parameters()).device
        # A shallow copy creates a new object which stores the reference of
        # the original elements.
        # A deep copy creates a new object and recursively adds the copies of
        # nested objects present in the original elements.
        self._averaged_model = copy.deepcopy(self.module).to(module_device)
        self.stream = torch.cuda.Stream(device=module_device)

        self.beta = beta

        self.start_iteration = start_iteration
        # This buffer is to track how many iterations has the model been
        # trained for. We will ignore the first $(start_iterations) and start
        # the averaging after.
        self.register_buffer('num_updates_tracked',
                             torch.tensor(0, dtype=torch.long))
        self.num_updates_tracked = self.num_updates_tracked.to(module_device)
        self.averaged_model.eval()

        # Averaged model does not require grad.
        requires_grad(self.averaged_model, False)

    @property
    def averaged_model(self):
        self.stream.synchronize()
        return self._averaged_model

    def forward(self, *inputs, **kwargs):
        r"""PyTorch module forward function overload."""
        return self.module(*inputs, **kwargs)

    @torch.no_grad()
    def update_average(self):
        r"""Update the moving average."""
        module_device = next(self.module.parameters()).device
        current_stream = torch.cuda.current_stream(device=module_device)
        self.stream.wait_stream(current_stream)
        with torch.cuda.stream(self.stream):
            self.num_updates_tracked += 1
            if self.num_updates_tracked <= self.start_iteration:
                beta = 0.
            else:
                beta = self.beta
            source_dict = self.module.state_dict()
            target_dict = self._averaged_model.state_dict()
            source_list = []
            target_list = []
            for key in target_dict:
                if 'num_batches_tracked' in key:
                    continue
                source_list.append(source_dict[key].data)
                target_list.append(target_dict[key].data.float())

            torch._foreach_mul_(target_list, beta)
            torch._foreach_add_(target_list, source_list, alpha=1 - beta)

    def __repr__(self):
        r"""Returns a string that holds a printable representation of an
        object"""
        return self.module.__repr__()
