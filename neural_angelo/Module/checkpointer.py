import os
import torch
import threading

from imaginaire.utils.misc import to_cpu
from imaginaire.utils.distributed import is_master, get_rank
from imaginaire.utils.distributed import master_only_print as print


class Checkpointer(object):

    def __init__(self, cfg, model, optim=None, sched=None):
        self.model = model
        self.optim = optim
        self.sched = sched
        self.logdir = cfg.logdir
        self.save_period = cfg.checkpoint.save_period
        self.strict_resume = cfg.checkpoint.strict_resume
        self.iteration_mode = cfg.optim.sched.iteration_mode
        self.resume = False
        self.resume_epoch = self.resume_iteration = None

    def save(self, current_epoch, current_iteration, latest=False):
        r"""Save network weights, optimizer parameters, scheduler parameters to a checkpoint.

        Args:
            current_epoch (int): Current epoch.
            current_iteration (int): Current iteration.
            latest (bool): If ``True``, save it using the name 'latest_checkpoint.pt'.
        """
        checkpoint_file = 'latest_checkpoint.pt' if latest else \
                          f'epoch_{current_epoch:05}_iteration_{current_iteration:09}_checkpoint.pt'
        if is_master():
            save_dict = to_cpu(self._collect_state_dicts())
            save_dict.update(
                epoch=current_epoch,
                iteration=current_iteration,
            )
            # Run the checkpoint saver in a separate thread.
            threading.Thread(
                target=self._save_worker, daemon=False, args=(save_dict, checkpoint_file, get_rank())).start()
        checkpoint_path = self._get_full_path(checkpoint_file)
        return checkpoint_path

    def _save_worker(self, save_dict, checkpoint_file, rank=0):
        checkpoint_path = self._get_full_path(checkpoint_file)
        # Save to local disk.
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(save_dict, checkpoint_path)
        if rank == 0:
            self.write_latest_checkpoint_file(checkpoint_file)
        print('Saved checkpoint to {}'.format(checkpoint_path))

    def _collect_state_dicts(self):
        r"""Collect all the state dicts from network modules to be saved."""
        return dict(
            model=self.model.state_dict(),
            optim=self.optim.state_dict(),
            sched=self.sched.state_dict(),
        )

    def load(self, checkpoint_path=None, resume=False, load_opt=True, load_sch=True, **kwargs):
        r"""Load network weights, optimizer parameters, scheduler parameters from a checkpoint.
        Args:
            checkpoint_path (str): Path to the checkpoint (local file or S3 key).
            resume (bool): if False, only the model weights are loaded. If True, the metadata (epoch/iteration) and
                           optimizer/scheduler (optional) are also loaded.
            load_opt (bool): Whether to load the optimizer state dict (resume should be True).
            load_sch (bool): Whether to load the scheduler state dict (resume should be True).
        """
        # Priority: (1) checkpoint_path (2) latest_path (3) train from scratch.
        self.resume = resume
        # If checkpoint path were not specified, try to load the latest one from the same run.
        if resume and checkpoint_path is None:
            latest_checkpoint_file = self.read_latest_checkpoint_file()
            if latest_checkpoint_file is not None:
                checkpoint_path = self._get_full_path(latest_checkpoint_file)
        # Load checkpoint.
        if checkpoint_path is not None:
            self._check_checkpoint_exists(checkpoint_path)
            self.checkpoint_path = checkpoint_path
            state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            print(f"Loading checkpoint (local): {checkpoint_path}")
            # Load the state dicts.
            print('- Loading the model...')
            self.model.load_state_dict(state_dict['model'], strict=self.strict_resume)
            if resume:
                self.resume_epoch = state_dict['epoch']
                self.resume_iteration = state_dict['iteration']
                self.sched.last_epoch = self.resume_iteration if self.iteration_mode else self.resume_epoch
                if load_opt:
                    print('- Loading the optimizer...')
                    self.optim.load_state_dict(state_dict['optim'])
                if load_sch:
                    print('- Loading the scheduler...')
                    self.sched.load_state_dict(state_dict['sched'])
                print(f"Done with loading the checkpoint (epoch {self.resume_epoch}, iter {self.resume_iteration}).")
            else:
                print('Done with loading the checkpoint.')
            self.eval_epoch = state_dict['epoch']
            self.eval_iteration = state_dict['iteration']
        else:
            # Checkpoint not found and not specified. We will train everything from scratch.
            print('Training from scratch.')
        torch.cuda.empty_cache()

    def _get_full_path(self, file):
        return os.path.join(self.logdir, file)

    def _get_latest_pointer_path(self):
        return self._get_full_path('latest_checkpoint.txt')

    def read_latest_checkpoint_file(self):
        checkpoint_file = None
        latest_path = self._get_latest_pointer_path()
        if os.path.exists(latest_path):
            checkpoint_file = open(latest_path).read().strip()
            if checkpoint_file.startswith("latest_checkpoint:"):  # TODO: for backward compatibility, to be removed
                checkpoint_file = checkpoint_file.split(' ')[-1]
        return checkpoint_file

    def write_latest_checkpoint_file(self, checkpoint_file):
        latest_path = self._get_latest_pointer_path()
        content = f"{checkpoint_file}\n"
        with open(latest_path, "w") as file:
            file.write(content)

    def _check_checkpoint_exists(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'File not found (local): {checkpoint_path}')

    def reached_checkpointing_period(self, timer):
        save_now = torch.cuda.BoolTensor([False])
        if is_master():
            if timer.checkpoint_toc() > self.save_period:
                save_now.fill_(True)
        if save_now:
            if is_master():
                print('checkpointing period!')
        return save_now
