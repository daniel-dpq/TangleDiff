import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict

from multiprocessing import cpu_count
from typing import Callable, List
from time import time
from copy import deepcopy
import numpy as np
import random
import logging
import math
import os

import hydra
from omegaconf import DictConfig
import wandb

from tanglediff.data import (
    SequenceDataset, 
    SequenceBatchCollator,
    SequenceMapper,
)
from tanglediff.diffusion import create_diffusion
from tanglediff.modules import DiT_models
from tanglediff.diffusion.timestep_sampler import create_named_schedule_sampler

def exists(x):
    return x is not None

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def flatten_dict(raw_dict):
    """Flattens a nested dict."""
    flattened = []
    for k, v in raw_dict.items():
        if isinstance(v, dict):
            flattened.extend([
                (f'{k}:{i}', j) for i, j in flatten_dict(v)
            ])
        else:
            flattened.append((k, v))
    return flattened

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def cycle(dl):
    while True:
        for data in dl:
            yield data


class DiffusionTrainer(object):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()

        self.data_cfg = cfg.data
        self.exp_cfg = cfg.experiment
        self.diffusion_cfg = cfg.diffusion
        self.model_cfg = cfg.model

        # setup DDP
        assert torch.cuda.is_available(), "Training currently requires at least one GPU."
        self._init_ddp()

        # for logging results in a folder periodically
        if self.rank == 0:
            os.makedirs(self.exp_cfg.log_dir, exist_ok = True)
            self.logger = self._create_logger(self.exp_cfg.log_dir)
        else:
            self.logger = self._create_logger(None)
        
        # condition
        self.num_classes = len(self.data_cfg.binding_energy_seps) + 1 if exists(self.data_cfg.binding_energy_seps) else None
        assert self.exp_cfg.use_cfg is False or exists(self.num_classes), "CFG requires a terminal condition."
        assert not exists(self.num_classes) or self.exp_cfg.use_cfg, "class_num is specified but not using CFG."
        if exists(self.num_classes):
            self.logger.info(f"Number of classes: {self.num_classes}")
        else:
            self.logger.info("No condition is used.")

        # map sequence to continuous space
        self.seq_mapper = SequenceMapper(
            max_length=None,
            seq_encode_mode=self.exp_cfg.seq_encode_mode,
            logits=self.exp_cfg.logits,
            device=self.device
        )

        # model 
        self.model_name = self.model_cfg.name
        self.model = DiT_models[self.model_name](
            in_channels=self.seq_mapper.channel_dim,
            learn_sigma=self.model_cfg.learn_sigma,
            class_dropout_prob=self.model_cfg.class_dropout_prob,
            self_cond=self.model_cfg.self_cond,
            num_classes=self.num_classes,
        )
        if self.exp_cfg.ckpt:
            strict = True # (not self.terminal_condition) # not strict for terminal_dist
            self.load(self.exp_cfg.ckpt, strict=strict)
        else:
            self.start_step = 0
        self.ema = deepcopy(self.model).to(self.device) # Create an EMA of the model for use after training
        self.requires_grad(self.ema, False)
        self.model = DDP(self.model.to(self.device), device_ids = [self.rank])
        self.diffusion = create_diffusion(**dict(self.diffusion_cfg)) # default: 1000 steps, linear noise schedule
        self.logger.info(f"DiT Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.exp_cfg.train_lr, betas=self.exp_cfg.adam_betas)

        # setup data
        self.load_data()

        # mixed precision training
        self.mixed_precision = self.exp_cfg.mixed_precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        # weight losses based on timestep
        self.timestep_sampler = create_named_schedule_sampler(self.exp_cfg.timestep_sampler, self.diffusion)
        
        # wandb
        if self.exp_cfg.use_wandb and self.rank == 0:
            self._init_wandb()
    
    def _init_ddp(self):
        dist.init_process_group("nccl")
        assert self.exp_cfg.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
        self.rank = dist.get_rank()
        self.device = self.rank % torch.cuda.device_count()
        self.seed = self.data_cfg.seed * dist.get_world_size() + self.rank
        torch.manual_seed(self.seed)
        torch.cuda.set_device(self.device)
        logging.info(f"Starting rank={self.rank}, seed={self.seed}, world_size={dist.get_world_size()}.")

    def _init_wandb(self):
        wandb.init(
            project=self.exp_cfg.project,
            name=self.exp_cfg.name,
            dir=self.exp_cfg.log_dir,
        )
        self.logger.info(f"Initialized wandb project: {self.exp_cfg.project}, run: {wandb.run.id}")

    @torch.no_grad()
    def _update_ema(self, decay=0.9999):
        """
        Step the EMA model towards the current model.
        """
        ema_params = OrderedDict(self.ema.named_parameters())
        model_params = OrderedDict(self.model.module.named_parameters())

        for name, param in model_params.items():
            # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    def requires_grad(self, model, flag=True):
        """
        Set requires_grad flag for all parameters in a model.
        """
        for p in model.parameters():
            p.requires_grad = flag

    def cleanup(self):
        """
        End DDP training.
        """
        dist.destroy_process_group()

    def _create_logger(self, logging_dir):
        """
        Create a logger that writes to a log file and stdout.
        """
        if self.rank == 0:  # real logger
            logging.basicConfig(
                level=logging.INFO,
                format='[\033[34m%(asctime)s\033[0m] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                handlers=[logging.StreamHandler()]
            )
            logger = logging.getLogger(__name__)
        else:  # dummy logger (does nothing)
            logging.info(f"Rank {self.rank} does not create a logger.")
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.CRITICAL)
            logger.addHandler(logging.NullHandler())
        return logger

    def load_data(self):
        self.dataset = SequenceDataset(self.data_cfg)
        collator = SequenceBatchCollator(self.data_cfg)
        self.data_size = len(self.dataset)
        self.sampler = DistributedSampler(
            self.dataset,
            num_replicas=dist.get_world_size(),
            rank=self.rank,
            shuffle=True,
            seed=self.data_cfg.seed
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=int(self.exp_cfg.global_batch_size // dist.get_world_size()),
            shuffle=False,
            sampler=self.sampler,
            num_workers=dist.get_world_size(),
            pin_memory=True,
            drop_last=True,
            collate_fn=collator
        )
        self.logger.info(f"Dataset contains {len(self.dataset.sequences):,} sequences with {len(self.dataset):,} clusters.")
        if exists(self.num_classes):
            classes, counts = np.unique(self.dataset.be_classes, return_counts=True)
            self.logger.info(f"Binding energy classes: {dict(zip(classes, counts))}")

    def sample(self, milestone):
        use_cfg = self.exp_cfg.use_cfg
        method = self.exp_cfg.sample_method

        self.model.eval()
        n_total = self.exp_cfg.num_samples
        c = self.seq_mapper.channel_dim
        l = self.data_cfg.max_length + 2 # add <cls> and <eos>
        batch_size = self.exp_cfg.global_batch_size // dist.get_world_size() * 4
        n_one_gpu = n_total // dist.get_world_size()
        # Sample function:
        if method == 'ddpm':
            sample_fn = self.diffusion.p_sample_loop
        elif method == 'ddim':
            sample_fn = self.diffusion.ddim_sample_loop
        else:
            raise ValueError(f"Invalid sampling method: {method}")
        
        def num_to_groups(num, divisor):
            groups = num // divisor
            remainder = num % divisor
            arr = [divisor] * groups
            if remainder > 0:
                arr.append(remainder)
            return arr

        batch_nums = num_to_groups(n_one_gpu, batch_size)
        samples_one_gpu = []
        masks_one_gpu = []
        for num in batch_nums:
            # Create sampling noise:
            mask = self.dataset.sample_lengths(num).to(self.device) # (B, L), pad to the max length in all training data
            z = torch.randn(num, c, mask.size(1), device=self.device)
            masks_one_gpu.append(mask)
            # Prepare inputs:
            if use_cfg:
                # Setup classifier-free guidance:
                z = torch.cat([z, z], 0)
                y = torch.randint(0, self.num_classes, (num,), device=self.device)
                y_null = torch.tensor([self.num_classes] * num, device=self.device)
                y = torch.cat([y, y_null], 0)
                mask = torch.cat([mask, mask], 0) if exists(mask) else None
                model_kwargs = dict(y=y, cfg_scale=self.exp_cfg.cfg_scale, seq_mask=mask)
                model_forward = self.model.module.forward_with_cfg
            else:
                model_kwargs = dict(y=None, seq_mask=mask)
                model_forward = self.model.module.forward
            # Sample sequences:
            samples = sample_fn(
                model_forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=(self.rank==0), device=self.device
            )
            if use_cfg:
                samples, _ = samples.chunk(2, dim=0)    # Remove null class samples
            samples = samples.transpose(1, 2)       # (B, C, L) -> (B, L, C)
            samples_one_gpu.append(samples)

        # gather all samples
        samples_one_gpu = torch.cat(samples_one_gpu, dim=0)
        all_samples = torch.zeros((dist.get_world_size() * samples_one_gpu.size(0), samples_one_gpu.size(1), samples_one_gpu.size(2)), device=self.device)
        dist.all_gather_into_tensor(all_samples, samples_one_gpu)  

        # gather all samples and masks
        masks_one_gpu = torch.cat(masks_one_gpu, dim=0)
        assert samples_one_gpu.size(0) == masks_one_gpu.size(0) and samples_one_gpu.size(1) == masks_one_gpu.size(1)
        all_masks = torch.zeros((dist.get_world_size() * masks_one_gpu.size(0), masks_one_gpu.size(1)), device=self.device)
        dist.all_gather_into_tensor(all_masks, masks_one_gpu)
        all_seqs = self.seq_mapper.decode(all_samples, batch_mask=all_masks)

        if self.rank == 0:
            self._write_to_fasta(all_seqs, os.path.join(self.exp_cfg.log_dir, f'samples_{milestone}.fasta'))

    def save(self, milestone):
        # checkpoint = {
        #     "model": self.model.module.state_dict(),
        #     "ema": self.ema.state_dict(),
        #     "opt": self.optim.state_dict(),
        #     'scaler': self.scaler.state_dict() if exists(self.scaler) else None,
        # }
        checkpoint = self.model.module.state_dict()
        checkpoint_path = os.path.join(self.exp_cfg.log_dir, f'model-{milestone}.pt')
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load(self, checkpoint_path, strict=True):
        'load right after the model is initialized'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'], strict=strict)
        else:
            self.model.load_state_dict(checkpoint, strict=strict)
        self.start_step = int(checkpoint_path.split('-')[-1].split('.')[0])
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}, strict={strict}")

    def _write_to_fasta(self, seqs: List[str], path: str):
        with open(path, 'w') as f:
            for i, seq in enumerate(seqs):
                f.write(f'>seq_{i}\n')
                f.write(f'{seq}\n')

    def train(self):
        # Prepare models for training:
        self._update_ema(decay=0)  # Ensure EMA is initialized with synced weights
        self.model.train()  # important! This enables embedding dropout for classifier-free guidance
        self.ema.eval()  # EMA model should always be in eval mode

        # Variables for monitoring/logging purposes:
        train_steps = self.start_step
        log_steps = 0
        running_loss = 0
        start_time = time()

        self.logger.info(f"Training for {self.exp_cfg.epochs} epochs...")
        self.logger.info(f'effective batch size: {self.exp_cfg.global_batch_size * self.exp_cfg.accumulate_steps}')
        for epoch in range(self.exp_cfg.epochs):
            self.sampler.set_epoch(epoch)

            for step, batch in enumerate(self.dataloader):
                seq_embed, seq_mask = self.seq_mapper.encode(batch['seqs'])
                seq_embed = seq_embed.transpose(1, 2).to(self.device) # (B, L, C) -> (B, C, L)
                seq_mask = seq_mask.to(self.device)

                loss_mask = seq_mask
                y = batch['be_class'].to(self.device) if exists(self.num_classes) else None

                t, weights = self.timestep_sampler.sample(seq_embed.shape[0], self.device)
                model_kwargs = dict(seq_mask=seq_mask, y=y)

                if step % self.exp_cfg.accumulate_steps != 0:
                    with self.model.no_sync():
                        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                            loss_dict = self.diffusion.training_losses(self.model, seq_embed, t, model_kwargs, loss_mask=loss_mask)
                        if self.exp_cfg.timestep_sampler == 'loss-second-moment':
                            self.timestep_sampler.update_with_local_losses(
                                t, loss_dict["loss"].detach()
                            )
                        loss = (loss_dict['loss']*weights).mean() / self.exp_cfg.accumulate_steps
                        self.scaler.scale(loss).backward()

                else:
                    with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                        loss_dict = self.diffusion.training_losses(self.model, seq_embed, t, model_kwargs, loss_mask=loss_mask)
                    if self.exp_cfg.timestep_sampler == 'loss-second-moment':
                        self.timestep_sampler.update_with_local_losses(
                            t, loss_dict["loss"].detach()
                        )
                    loss = (loss_dict['loss']*weights).mean() / self.exp_cfg.accumulate_steps
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.exp_cfg.max_grad_norm)
                    self.scaler.step(self.optim)
                    self.scaler.update()
                    self.optim.zero_grad()
                    self._update_ema()

                torch.cuda.empty_cache()
                train_steps += 1
                log_steps += 1
                running_loss += loss.item()*self.exp_cfg.accumulate_steps
                if train_steps % self.exp_cfg.log_interval == 0:
                    # Measure training speed:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    # Reduce loss history over all processes:
                    avg_loss = torch.tensor(running_loss / log_steps, device=self.device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()
                    self.logger.info(f"[epoch={(epoch+1):03d}, step={train_steps:07d}] Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    # Log to wandb:
                    if self.exp_cfg.use_wandb and self.rank == 0:
                        wandb.log({
                            'train_loss': avg_loss,
                            'train_steps_per_sec': steps_per_sec,
                        }, step=train_steps)
                    # Reset monitoring variables:
                    running_loss = 0
                    log_steps = 0
                    start_time = time()

                # Save DiT checkpoint:
                if train_steps % self.exp_cfg.save_and_sample_interval == 0 and train_steps > 0:
                    if self.rank == 0:
                        self.save(train_steps)
                    dist.barrier()
                    self.sample(train_steps)
                    dist.barrier()
    
    def cleanup(self):
        """
        End DDP training.
        """
        dist.destroy_process_group()


@hydra.main(config_path='config', config_name='train', version_base=None)
def main(cfg):
    trainer = DiffusionTrainer(cfg)
    trainer.train()
    trainer.cleanup()


if __name__ == "__main__":
    main()
