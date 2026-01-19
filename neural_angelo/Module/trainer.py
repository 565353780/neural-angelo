import os
import time
import json
import torch
import inspect
import torch.nn.functional as torch_F

from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from neural_angelo.Data.get_dataloader import get_train_dataloader, get_val_dataloader, get_test_dataloader
from neural_angelo.Util.init_weight import weights_init, weights_rescale
from neural_angelo.Util.model_average import ModelAverage
from neural_angelo.Util.misc import to_cuda, requires_grad, Timer
from neural_angelo.Util.set_random_seed import set_random_seed
from neural_angelo.Util.visualization import tensorboard_image
from neural_angelo.Util.nerf_misc import eikonal_loss, curvature_loss

from neural_angelo.Model.model import Model
from neural_angelo.Module.checkpointer import Checkpointer


def _calculate_model_size(model):
    r"""Calculate number of parameters in a PyTorch network.

    Args:
        model (obj): PyTorch network.

    Returns:
        (int): Number of parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def collate_test_data_batches(data_batches):
    """Collate test data batches from a single GPU (simplified for single GPU training)."""
    if not data_batches:
        return {}
    # Concatenate all batches
    data_all = {}
    for key in data_batches[0].keys():
        if isinstance(data_batches[0][key], torch.Tensor):
            data_all[key] = torch.cat([batch[key] for batch in data_batches], dim=0)
        elif isinstance(data_batches[0][key], (list, tuple)):
            data_all[key] = [item for batch in data_batches for item in batch[key]]
        else:
            data_all[key] = [batch[key] for batch in data_batches]
    return data_all


def get_unique_test_data(data_gather, idx):
    """Get unique test data (simplified for single GPU training)."""
    return data_gather


def trim_test_samples(data, max_samples=None):
    """Trim test samples to max_samples (simplified for single GPU training)."""
    if max_samples is None or max_samples <= 0:
        return
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key][:max_samples]
        elif isinstance(data[key], (list, tuple)):
            data[key] = data[key][:max_samples]


class Trainer(object):
    r"""Trainer class for Neuralangelo training.

    Args:
        cfg (obj): Global configuration.
        is_inference (bool): if True, load the test dataloader and run in inference mode.
        seed (int): Random seed.
    """

    def __init__(self, cfg, is_inference=True, seed=0):
        print('Setup trainer.')
        self.cfg = cfg
        # Create objects for the networks, optimizers, and schedulers.
        self.model = self.setup_model(cfg, seed=seed)
        if not is_inference:
            self.optim = self.setup_optimizer(cfg, self.model)
            self.sched = self.setup_scheduler(cfg, self.optim)
        else:
            self.optim = None
            self.sched = None
        self.model = self.wrap_model(cfg, self.model)
        # Data loaders & inference mode.
        self.is_inference = is_inference
        # Initialize automatic mixed precision training.
        self.init_amp()
        # Initialize loss functions.
        self.init_losses(cfg)

        self.checkpointer = Checkpointer(cfg, self.model, self.optim, self.sched)
        self.timer = Timer(cfg)

        # -------- The initialization steps below can be skipped during inference. --------
        if self.is_inference:
            return

        # Initialize logging attributes.
        self.init_logging_attributes()
        # Initialize validation parameters.
        self.init_val_parameters()
        # AWS credentials.
        if hasattr(cfg, 'aws_credentials_file'):
            with open(cfg.aws_credentials_file) as fin:
                self.credentials = json.load(fin)
        else:
            self.credentials = None
        if 'TORCH_HOME' not in os.environ:
            os.environ['TORCH_HOME'] = os.path.join(os.environ['HOME'], ".cache")

        # Trainer-specific initialization
        self.metrics = dict()
        # The below configs should be properly overridden.
        cfg.setdefault("tensorboard_scalar_iter", 9999999999999)
        cfg.setdefault("tensorboard_image_iter", 9999999999999)
        cfg.setdefault("validation_epoch", 9999999999999)
        cfg.setdefault("validation_iter", 9999999999999)

        self.warm_up_end = cfg.optim.sched.warm_up_end
        self.cfg_gradient = cfg.model.object.sdf.gradient
        if cfg.model.object.sdf.encoding.type == "hashgrid" and cfg.model.object.sdf.encoding.coarse2fine.enabled:
            self.c2f_step = cfg.model.object.sdf.encoding.coarse2fine.step
            self.model_module.neural_sdf.warm_up_end = self.warm_up_end

        self.criteria["render"] = torch.nn.L1Loss()

    def set_data_loader(self, cfg, split, shuffle=True, drop_last=True, seed=0):
        """Set the data loader corresponding to the indicated split.
        Args:
            split (str): Must be either 'train', 'val', or 'test'.
            shuffle (bool): Whether to shuffle the data (only applies to the training set).
            drop_last (bool): Whether to drop the last batch if it is not full (only applies to the training set).
            seed (int): Random seed.
        """
        assert (split in ["train", "val", "test"])
        if split == "train":
            self.train_data_loader = get_train_dataloader(cfg, shuffle=shuffle, drop_last=drop_last, seed=seed)
        elif split == "val":
            self.eval_data_loader = get_val_dataloader(cfg, seed=seed)
        elif split == "test":
            self.eval_data_loader = get_test_dataloader(cfg)

    def setup_model(self, cfg, seed=0):
        r"""Return the networks. We will first set the random seed to a fixed value so that the network will be
        initialized with consistent weights. After this we will wrap the network with a moving average model if applicable.

        The following objects are constructed as class members:
          - model (obj): Model object (historically: generator network object).

        Args:
            cfg (obj): Global configuration.
            seed (int): Random seed.
        """
        # Set the random seed for consistent initialization.
        set_random_seed(seed, by_rank=False)
        # Construct networks
        model = Model(cfg.model, cfg.data)
        print('model parameter count: {:,}'.format(_calculate_model_size(model)))
        print(f'Initialize model weights using type: {cfg.trainer.init.type}, gain: {cfg.trainer.init.gain}')
        init_bias = getattr(cfg.trainer.init, 'bias', None)
        init_gain = cfg.trainer.init.gain or 1.
        model.apply(weights_init(cfg.trainer.init.type, init_gain, init_bias))
        model.apply(weights_rescale())
        model = model.to('cuda')
        return model

    def setup_optimizer(self, cfg, model):
        optim = AdamW(
            params=model.parameters(),
            lr=cfg.optim.params.lr,
            weight_decay=cfg.optim.params.weight_decay,
        )

        self.optim_zero_grad_kwargs = {}
        if 'set_to_none' in inspect.signature(optim.zero_grad).parameters:
            self.optim_zero_grad_kwargs['set_to_none'] = True
        return optim

    def setup_scheduler(self, cfg, optim):
        warm_up_end = cfg.optim.sched.warm_up_end
        two_steps = cfg.optim.sched.two_steps
        gamma = cfg.optim.sched.gamma

        def sch(x):
            if x < warm_up_end:
                return x / warm_up_end
            else:
                if x > two_steps[1]:
                    return 1.0 / gamma ** 2
                elif x > two_steps[0]:
                    return 1.0 / gamma
                else:
                    return 1.0

        return LambdaLR(optim, lambda x: sch(x))

    def wrap_model(self, cfg, model):
        # Moving average model wrapping (no DDP for single GPU training).
        if cfg.trainer.ema_config.enabled:
            model = ModelAverage(model,
                                cfg.trainer.ema_config.beta,
                                cfg.trainer.ema_config.start_iteration)
            self.model_module = model.module
        else:
            self.model_module = model
        return model

    def init_amp(self):
        r"""Initialize automatic mixed precision training."""

        if getattr(self.cfg.trainer, 'allow_tf32', True):
            print("Allow TensorFloat32 operations on supported devices")
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
        else:
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False

        if self.cfg.trainer.amp_config.enabled:
            print("Using automatic mixed precision training.")

        # amp scaler can be used without mixed precision training
        if hasattr(self.cfg.trainer, 'scaler_config'):
            scaler_kwargs = vars(self.cfg.trainer.scaler_config)
            scaler_kwargs['enabled'] = self.cfg.trainer.amp_config.enabled and \
                getattr(self.cfg.trainer.scaler_config, 'enabled', True)
        else:
            scaler_kwargs = vars(self.cfg.trainer.amp_config)   # backward compatibility
            scaler_kwargs.pop('dtype', None)
            scaler_kwargs.pop('cache_enabled', None)

        self.scaler = GradScaler(**scaler_kwargs)

    def init_logging_attributes(self):
        r"""Initialize logging attributes."""
        self.current_iteration = 0
        self.current_epoch = 0
        self.start_iteration_time = None
        self.start_epoch_time = None
        self.elapsed_iteration_time = 0
        if self.cfg.speed_benchmark:
            self.timer.reset()

    def init_val_parameters(self):
        r"""Initialize validation parameters."""
        if self.cfg.metrics_iter is None:
            self.cfg.metrics_iter = self.cfg.checkpoint.save_iter
        if self.cfg.metrics_epoch is None:
            self.cfg.metrics_epoch = self.cfg.checkpoint.save_epoch

    def init_tensorboard(self, cfg, enabled=True):
        r"""Initialize TensorBoard logger.

        Args:
            cfg (obj): Global configuration.
            enabled (bool): Whether to enable TensorBoard logging.
        """
        if enabled:
            print('Initialize TensorBoard')
            tensorboard_dir = os.path.join(cfg.logdir, "tensorboard")
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir=tensorboard_dir)
            # Log dataset name as a text summary
            self.tensorboard_writer.add_text('config/dataset', cfg.data.name, 0)
        else:
            self.tensorboard_writer = None

    def start_of_epoch(self, current_epoch):
        r"""Things to do before an epoch.

        Args:
            current_epoch (int): Current number of epoch.
        """
        self._start_of_epoch(current_epoch)
        self.current_epoch = current_epoch
        self.start_epoch_time = time.time()

    def start_of_iteration(self, data, current_iteration):
        r"""Things to do before an iteration.

        Args:
            data (dict): Data used for the current iteration.
            current_iteration (int): Current number of iteration.
        """
        data = self._start_of_iteration(data, current_iteration)
        data = to_cuda(data)
        self.current_iteration = current_iteration
        self.model.train()
        self.start_iteration_time = time.time()
        return data

    def end_of_iteration(self, data, current_epoch, current_iteration):
        r"""Things to do after an iteration.

        Args:
            data (dict): Data used for the current iteration.
            current_epoch (int): Current number of epoch.
            current_iteration (int): Current number of iteration.
        """
        self.current_iteration = current_iteration
        self.current_epoch = current_epoch

        # Accumulate time
        self.elapsed_iteration_time += time.time() - self.start_iteration_time
        # Logging.
        if current_iteration % self.cfg.logging_iter == 0:
            avg_time = self.elapsed_iteration_time / self.cfg.logging_iter
            self.timer.time_iteration = avg_time
            print('Iteration: {}, average iter time: {:6f}.'.format(current_iteration, avg_time))
            self.elapsed_iteration_time = 0

            if self.cfg.speed_benchmark:
                # only needed when analyzing computation bottleneck.
                self.timer._print_speed_benchmark(avg_time)

        self._end_of_iteration(data, current_epoch, current_iteration)

        # Save everything to the checkpoint by time period.
        if self.checkpointer.reached_checkpointing_period(self.timer):
            self.checkpointer.save(current_epoch, current_iteration)
            self.timer.checkpoint_tic()  # reset timer

        # Save everything to the checkpoint.
        if current_iteration % self.cfg.checkpoint.save_iter == 0 or \
                current_iteration == self.cfg.max_iter:
            self.checkpointer.save(current_epoch, current_iteration)

        # Save everything to the checkpoint using the name 'latest_checkpoint.pt'.
        if current_iteration % self.cfg.checkpoint.save_latest_iter == 0:
            if current_iteration >= self.cfg.checkpoint.save_latest_iter:
                self.checkpointer.save(current_epoch, current_iteration, True)

        # Update the learning rate policy for the generator if operating in the iteration mode.
        if self.cfg.optim.sched.iteration_mode:
            self.sched.step()

        # This iteration was successfully finished. Reset timeout counter.
        self.timer.reset_timeout_counter()

    def end_of_epoch(self, data, current_epoch, current_iteration):
        r"""Things to do after an epoch.

        Args:
            data (dict): Data used for the current iteration.

            current_epoch (int): Current number of epoch.
            current_iteration (int): Current number of iteration.
        """
        # Update the learning rate policy for the generator if operating in the epoch mode.
        self.current_iteration = current_iteration
        self.current_epoch = current_epoch
        if not self.cfg.optim.sched.iteration_mode:
            self.sched.step()
        elapsed_epoch_time = time.time() - self.start_epoch_time
        # Logging.
        print('Epoch: {}, total time: {:6f}.'.format(current_epoch, elapsed_epoch_time))
        self.timer.time_epoch = elapsed_epoch_time
        self._end_of_epoch(data, current_epoch, current_iteration)

        # Save everything to the checkpoint.
        if current_epoch % self.cfg.checkpoint.save_epoch == 0:
            self.checkpointer.save(current_epoch, current_iteration)

    def _extra_step(self, data):
        pass

    def _start_of_epoch(self, current_epoch):
        r"""Operations to do before starting an epoch.

        Args:
            current_epoch (int): Current number of epoch.
        """
        pass

    def _start_of_iteration(self, data, current_iteration):
        r"""Operations to do before starting an iteration.

        Args:
            data (dict): Data used for the current iteration.
            current_iteration (int): Current epoch number.
        Returns:
            (dict): Data used for the current iteration. They might be
                processed by the custom _start_of_iteration function.
        """
        model = self.model_module
        self.progress = model.progress = current_iteration / self.cfg.max_iter
        if self.cfg.model.object.sdf.encoding.coarse2fine.enabled:
            model.neural_sdf.set_active_levels(current_iteration)
            if self.cfg_gradient.mode == "numerical":
                model.neural_sdf.set_normal_epsilon()
                self.get_curvature_weight(current_iteration, self.cfg.trainer.loss_weight.curvature)
        elif self.cfg_gradient.mode == "numerical":
            model.neural_sdf.set_normal_epsilon()
        return data

    def _end_of_iteration(self, data, current_epoch, current_iteration):
        r"""Operations to do after an iteration.

        Args:
            data (dict): Data used for the current iteration.
            current_epoch (int): Current number of epoch.
            current_iteration (int): Current epoch number.
        """
        # Log to TensorBoard.
        if current_iteration % self.cfg.tensorboard_scalar_iter == 0:
            # Compute the elapsed time (as in the original base trainer).
            self.timer.time_iteration = self.elapsed_iteration_time / self.cfg.tensorboard_scalar_iter
            self.elapsed_iteration_time = 0
            # Log scalars.
            self.log_tensorboard_scalars(data, mode="train")
            # Exit if the training loss has gone to NaN/inf.
            if self.losses["total"].isnan():
                self.finalize(self.cfg)
                raise ValueError("Training loss has gone to NaN!!!")
            if self.losses["total"].isinf():
                self.finalize(self.cfg)
                raise ValueError("Training loss has gone to infinity!!!")
        # Run evaluation to log images to TensorBoard.
        if current_iteration % self.cfg.tensorboard_image_iter == 0:
            data_all = self.test(self.eval_data_loader, mode="val")
            # Log the results to TensorBoard.
            self.log_tensorboard_scalars(data_all, mode="val")
            self.log_tensorboard_images(data_all, mode="val", max_samples=self.cfg.data.val.max_viz_samples)
        # Run validation on val set.
        if current_iteration % self.cfg.validation_iter == 0:
            data_all = self.test(self.eval_data_loader, mode="val")
            # Log the results to TensorBoard.
            self.log_tensorboard_scalars(data_all, mode="val")
            self.log_tensorboard_images(data_all, mode="val", max_samples=self.cfg.data.val.max_viz_samples)

    def _end_of_epoch(self, data, current_epoch, current_iteration):
        r"""Operations to do after an epoch.

        Args:
            data (dict): Data used for the current iteration.
            current_epoch (int): Current number of epoch.
            current_iteration (int): Current epoch number.
        """
        # Run validation on val set.
        if current_epoch % self.cfg.validation_epoch == 0:
            data_all = self.test(self.eval_data_loader, mode="val")
            # Log the results to TensorBoard.
            self.log_tensorboard_scalars(data_all, mode="val")
            self.log_tensorboard_images(data_all, mode="val", max_samples=self.cfg.data.val.max_viz_samples)

    def _get_visualizations(self, data):
        r"""Compute visualization outputs.

        Args:
            data (dict): Data used for the current iteration.
        """
        return None

    def train_step(self, data, last_iter_in_epoch=False):
        r"""One training step.

        Args:
            data (dict): Data used for the current iteration.
        """
        # Set requires_grad flags.
        requires_grad(self.model_module, True)

        # Compute the loss.
        self.timer._time_before_forward()

        autocast_dtype = getattr(self.cfg.trainer.amp_config, 'dtype', 'float16')
        autocast_dtype = torch.bfloat16 if autocast_dtype == 'bfloat16' else torch.float16
        amp_kwargs = {
            'enabled': self.cfg.trainer.amp_config.enabled,
            'dtype': autocast_dtype
        }
        with autocast(**amp_kwargs):
            total_loss = self.model_forward(data)
            # Scale down the loss w.r.t. gradient accumulation iterations.
            total_loss = total_loss / float(self.cfg.trainer.grad_accum_iter)

        # Backpropagate the loss.
        self.timer._time_before_backward()
        self.scaler.scale(total_loss).backward()

        self._extra_step(data)

        # Perform an optimizer step. This enables gradient accumulation when grad_accum_iter is not 1.
        if (self.current_iteration + 1) % self.cfg.trainer.grad_accum_iter == 0 or last_iter_in_epoch:
            self.timer._time_before_step()
            self.scaler.step(self.optim)
            self.scaler.update()
            # Zero out the gradients.
            self.optim.zero_grad(**self.optim_zero_grad_kwargs)

        # Update model average.
        self.timer._time_before_model_avg()
        if self.cfg.trainer.ema_config.enabled:
            self.model.update_average()

        self._detach_losses()
        self.timer._time_before_leave_gen()

    def model_forward(self, data):
        # Model forward.
        output = self.model(data)
        data.update(output)
        # Compute loss.
        self.timer._time_before_loss()
        self._compute_loss(data, mode="train")
        total_loss = self._get_total_loss()
        return total_loss

    def train(self, cfg, data_loader):
        self.progress = self.model_module.progress = self.current_iteration / self.cfg.max_iter

        self.current_epoch = self.checkpointer.resume_epoch or self.current_epoch
        self.current_iteration = self.checkpointer.resume_iteration or self.current_iteration
        if ((self.current_epoch % self.cfg.validation_epoch == 0 or
             self.current_iteration % self.cfg.validation_iter == 0)):
            # Do an initial validation.
            data_all = self.test(self.eval_data_loader, mode="val")
            # Log the results to TensorBoard.
            self.log_tensorboard_scalars(data_all, mode="val")
            self.log_tensorboard_images(data_all, mode="val", max_samples=self.cfg.data.val.max_viz_samples)
        # Train.
        start_epoch = self.checkpointer.resume_epoch or self.current_epoch  # The epoch to start with.
        current_iteration = self.checkpointer.resume_iteration or self.current_iteration  # The starting iteration.

        self.timer.checkpoint_tic()  # start timer
        self.timer.reset_timeout_counter()
        for current_epoch in range(start_epoch, cfg.max_epoch):
            self.start_of_epoch(current_epoch)
            data_loader_wrapper = tqdm(data_loader, desc=f"Training epoch {current_epoch + 1}", leave=False)
            for it, data in enumerate(data_loader_wrapper):
                data = self.start_of_iteration(data, current_iteration)

                self.train_step(data, last_iter_in_epoch=(it == len(data_loader) - 1))

                current_iteration += 1
                data_loader_wrapper.set_postfix(iter=current_iteration)
                if it == len(data_loader) - 1:
                    self.end_of_iteration(data, current_epoch + 1, current_iteration)
                else:
                    self.end_of_iteration(data, current_epoch, current_iteration)
                if current_iteration >= cfg.max_iter:
                    print('Done with training!!!')
                    return

            self.end_of_epoch(data, current_epoch + 1, current_iteration)
        print('Done with training!!!')

    @torch.no_grad()
    def test(self, data_loader, output_dir=None, inference_args=None, mode="test"):
        """The evaluation/inference engine.
        Args:
            data_loader: The data loader.
            output_dir: Output directory to dump the test results.
            inference_args: (unused)
            mode: Evaluation mode {"val", "test"}. Can be other modes, but will only gather the data.
        Returns:
            data_all: A dictionary of all the data.
        """
        if self.cfg.trainer.ema_config.enabled:
            model = self.model.averaged_model
        else:
            model = self.model_module
        model.eval()
        data_loader = tqdm(data_loader, desc="Evaluating", leave=False)
        data_batches = []
        for it, data in enumerate(data_loader):
            data = self.start_of_iteration(data, current_iteration=self.current_iteration)
            output = model.inference(data)
            data.update(output)
            data_batches.append(data)
        # Aggregate the data and process the results.
        data_gather = collate_test_data_batches(data_batches)
        data_all = get_unique_test_data(data_gather, data_gather["idx"])
        tqdm.write(f"Evaluating with {len(data_all['idx'])} samples.")
        # Validate/test.
        if mode == "val":
            self._compute_loss(data_all, mode=mode)
            _ = self._get_total_loss()
        return data_all

    def _get_total_loss(self):
        r"""Return the total loss to be backpropagated.
        """
        total_loss = torch.tensor(0., device=torch.device('cuda'))
        # Iterates over all possible losses.
        for loss_name in self.weights:
            if loss_name in self.losses:
                # Multiply it with the corresponding weight and add it to the total loss.
                total_loss += self.losses[loss_name] * self.weights[loss_name]
        self.losses['total'] = total_loss  # logging purpose
        return total_loss

    def _detach_losses(self):
        r"""Detach all logging variables to prevent potential memory leak."""
        for loss_name in self.losses:
            self.losses[loss_name] = self.losses[loss_name].detach()

    def finalize(self, cfg):
        # Finish the TensorBoard logger.
        if hasattr(self, 'tensorboard_writer') and self.tensorboard_writer is not None:
            self.tensorboard_writer.close()

    # Trainer-specific methods (overrides and additions)
    def init_losses(self, cfg):
        r"""Initialize loss functions. All loss names have weights. Some have criterion modules."""
        self.losses = dict()

        # Mapping from loss names to criterion modules.
        self.criteria = torch.nn.ModuleDict()
        # Mapping from loss names to loss weights.
        self.weights = dict()

        # Trainer-specific loss initialization
        # 支持 dict 和 class 两种 loss_weight 格式
        loss_weight_obj = cfg.trainer.loss_weight
        if hasattr(loss_weight_obj, 'items'):
            # dict 格式
            self.weights = {key: value for key, value in loss_weight_obj.items() if value}
        else:
            # class 格式 - 遍历属性
            for key in dir(loss_weight_obj):
                if not key.startswith('_') and not callable(getattr(loss_weight_obj, key)):
                    value = getattr(loss_weight_obj, key)
                    if value:
                        self.weights[key] = value

        for loss_name, loss_weight in self.weights.items():
            print("Loss {:<20} Weight {}".format(loss_name, loss_weight))
            if loss_name in self.criteria.keys() and self.criteria[loss_name] is not None:
                self.criteria[loss_name].to('cuda')

    def _compute_loss(self, data, mode=None):
        if mode == "train":
            # Compute loss only on randomly sampled rays.
            self.losses["render"] = self.criteria["render"](data["rgb"], data["image_sampled"]) * 3  # FIXME:sumRGB?!
            self.metrics["psnr"] = -10 * torch_F.mse_loss(data["rgb"], data["image_sampled"]).log10()
            if "eikonal" in self.weights.keys():
                self.losses["eikonal"] = eikonal_loss(data["gradients"], outside=data["outside"])
            if "curvature" in self.weights:
                self.losses["curvature"] = curvature_loss(data["hessians"], outside=data["outside"])
        else:
            # Compute loss on the entire image.
            self.losses["render"] = self.criteria["render"](data["rgb_map"], data["image"])
            self.metrics["psnr"] = -10 * torch_F.mse_loss(data["rgb_map"], data["image"]).log10()

    def get_curvature_weight(self, current_iteration, init_weight):
        if "curvature" in self.weights:
            if current_iteration <= self.warm_up_end:
                self.weights["curvature"] = current_iteration / self.warm_up_end * init_weight
            else:
                model = self.model_module
                decay_factor = model.neural_sdf.growth_rate ** (model.neural_sdf.anneal_levels - 1)
                self.weights["curvature"] = init_weight / decay_factor

    def log_tensorboard_scalars(self, data, mode=None):
        if not hasattr(self, 'tensorboard_writer') or self.tensorboard_writer is None:
            return
        # Log scalars (basic info & losses).
        if mode == "train":
            self.tensorboard_writer.add_scalar("optim/lr", self.sched.get_last_lr()[0], self.current_iteration)
            self.tensorboard_writer.add_scalar("time/iteration", self.timer.time_iteration, self.current_iteration)
            self.tensorboard_writer.add_scalar("time/epoch", self.timer.time_epoch, self.current_iteration)
        for key, value in self.losses.items():
            if isinstance(value, torch.Tensor):
                self.tensorboard_writer.add_scalar(f"{mode}/loss/{key}", value.item() if value.numel() == 1 else value.mean().item(), self.current_iteration)
            else:
                self.tensorboard_writer.add_scalar(f"{mode}/loss/{key}", value, self.current_iteration)
        self.tensorboard_writer.add_scalar("iteration", self.current_iteration, self.current_iteration)
        self.tensorboard_writer.add_scalar("epoch", self.current_epoch, self.current_iteration)

        if not hasattr(self, 'tensorboard_writer') or self.tensorboard_writer is None:
            return
        self.tensorboard_writer.add_scalar(f"{mode}/PSNR", self.metrics["psnr"].detach().item(), self.current_iteration)
        self.tensorboard_writer.add_scalar(f"{mode}/s-var", self.model_module.s_var.item(), self.current_iteration)
        if "curvature" in self.weights:
            self.tensorboard_writer.add_scalar(f"{mode}/curvature_weight", self.weights["curvature"], self.current_iteration)
        if "eikonal" in self.weights:
            self.tensorboard_writer.add_scalar(f"{mode}/eikonal_weight", self.weights["eikonal"], self.current_iteration)
        if mode == "train" and self.cfg_gradient.mode == "numerical":
            self.tensorboard_writer.add_scalar(f"{mode}/epsilon", self.model_module.neural_sdf.normal_eps, self.current_iteration)
        if self.cfg.model.object.sdf.encoding.coarse2fine.enabled:
            self.tensorboard_writer.add_scalar(f"{mode}/active_levels", self.model_module.neural_sdf.active_levels, self.current_iteration)

    def log_tensorboard_images(self, data, mode=None, max_samples=None):
        trim_test_samples(data, max_samples=max_samples)

        if not hasattr(self, 'tensorboard_writer') or self.tensorboard_writer is None:
            return
        if mode == "val":
            images_error = (data["rgb_map"] - data["image"]).abs()
            self.tensorboard_writer.add_image(f"{mode}/vis/rgb_target", tensorboard_image(data["image"]), self.current_iteration)
            self.tensorboard_writer.add_image(f"{mode}/vis/rgb_render", tensorboard_image(data["rgb_map"]), self.current_iteration)
            self.tensorboard_writer.add_image(f"{mode}/vis/rgb_error", tensorboard_image(images_error), self.current_iteration)
            self.tensorboard_writer.add_image(f"{mode}/vis/normal", tensorboard_image(data["normal_map"], from_range=(-1, 1)), self.current_iteration)
            self.tensorboard_writer.add_image(f"{mode}/vis/inv_depth", tensorboard_image(1 / (data["depth_map"] + 1e-8) * self.cfg.trainer.depth_vis_scale), self.current_iteration)
            self.tensorboard_writer.add_image(f"{mode}/vis/opacity", tensorboard_image(data["opacity_map"]), self.current_iteration)
