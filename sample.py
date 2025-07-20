# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from tqdm import tqdm
import os
import math
from omegaconf import  DictConfig
import hydra
import logging

from tanglediff.data import SequenceMapper, SequenceDataset
from tanglediff.modules import DiT_models
from tanglediff.diffusion import create_diffusion


@hydra.main(config_path='config', config_name='sample', version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Run sampling.
    """
    # torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Configures
    data_cfg = cfg.data
    model_cfg = cfg.model
    diffusion_cfg = cfg.diffusion
    sample_cfg = cfg.sample
    lm_cfg = cfg.lm
    condition_cfg = cfg.condition

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = data_cfg.seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup logger:
    if rank == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler()]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logging.info(f"Rank {rank} does not create a logger.")
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.CRITICAL)
        logger.addHandler(logging.NullHandler())
    dist.barrier()

    # Condition
    num_classes = len(data_cfg.binding_energy_seps) + 1
    logger.info(f"Total number of classes: {num_classes}")
    condition = condition_cfg.binding_energy_condition
    assert sample_cfg.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = sample_cfg.cfg_scale > 1.0
    if condition is None or condition == num_classes:
        logger.info("Unconditional sampling")
        condition = num_classes
        assert not using_cfg, "Set cfg_scale to 1.0 for unconditional sampling"
    else:
        logger.info(f"Binding energy class {condition} specified")
        assert 0 <= condition and condition < num_classes, "Invalid binding energy condition"
        assert using_cfg, "Set cfg_scale to > 1.0 for conditional sampling"
    if data_cfg.use_core:
        logger.info("Using cropped data")
    logger.info(f"Sample length: {sample_cfg.sample_length}")
    
    # Save path
    ckpt_path = sample_cfg.ckpt    
    fasta_path = os.path.join(sample_cfg.out_dir, f"samples_{sample_cfg.num_samples}_c{condition}_l{sample_cfg.sample_length}_{sample_cfg.method}{sample_cfg.num_sampling_steps}.fasta".replace(' ', ''))  
    os.makedirs(sample_cfg.out_dir, exist_ok=True)
    if os.path.exists(fasta_path):
        raise FileExistsError(f"{fasta_path} already exists. Please move or delete it.")    

    # neccessary objects
    seq_mapper = SequenceMapper(
        max_length=None,
        seq_encode_mode=lm_cfg.seq_encode_mode,
        logits=lm_cfg.logits,
        device=device
    )

    # Load model:
    model = DiT_models[model_cfg.name](
        in_channels=seq_mapper.channel_dim,
        learn_sigma=model_cfg.learn_sigma,
        self_cond=model_cfg.self_cond,
        class_dropout_prob=model_cfg.class_dropout_prob,
        num_classes=num_classes,
    ).to(device)

    # Load a custom DiT checkpoint from train.py:
    logger.info(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(
        timestep_respacing=str(sample_cfg.num_sampling_steps),
        **dict(diffusion_cfg)
    )
    dist.barrier()

    # Sample function:
    if sample_cfg.method == 'ddpm':
        # assert sample_cfg.num_sampling_steps == 1000, "DDPM only supports 1000 steps"
        sample_fn = diffusion.p_sample_loop
    elif sample_cfg.method == 'ddim':
        sample_fn = diffusion.ddim_sample_loop
    else:
        raise ValueError(f"Invalid sampling method: {sample_cfg.method}")

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = sample_cfg.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(sample_cfg.num_samples / global_batch_size) * global_batch_size)
    logger.info(f"Total number of sequences that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    logger.info(f"Each GPU will sample {samples_needed_this_gpu} sequences with a batch size {n}")
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar

    samples_one_gpu = []
    masks_one_gpu = []
    for _ in pbar:
        # Sample input, mask and condition:
        dataset = SequenceDataset(data_cfg)
        seq_mask = dataset.sample_lengths(n, condition, sample_cfg.sample_length).to(device)
        z = torch.randn(n, seq_mapper.channel_dim, seq_mask.size(1), device=device)

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y = torch.tensor([condition] * n, device=device)
            y_null = torch.tensor([num_classes] * n, device=device)
            y = torch.cat([y, y_null], 0)
            seq_mask = torch.cat([seq_mask, seq_mask], 0)
            model_kwargs = dict(y=y, cfg_scale=sample_cfg.cfg_scale, seq_mask=seq_mask)
            model_forward = model.forward_with_cfg
        else:
            y_null = torch.tensor([num_classes] * n, device=device)
            model_kwargs = dict(y=y_null, seq_mask=seq_mask)
            model_forward = model.forward
        # Sample sequences:
        samples = sample_fn(
            model_forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            seq_mask, _ = seq_mask.chunk(2, dim=0) # Remove null class masks
        samples = samples.transpose(1, 2)       # (B, C, L) -> (B, L, C)
        samples_one_gpu.append(samples)
        masks_one_gpu.append(seq_mask)

    # gather all samples
    samples_one_gpu = torch.cat(samples_one_gpu, dim=0)
    all_samples = torch.zeros((dist.get_world_size() * samples_one_gpu.size(0), samples_one_gpu.size(1), samples_one_gpu.size(2)), device=device)
    dist.all_gather_into_tensor(all_samples, samples_one_gpu)
    # gather all masks
    masks_one_gpu = torch.cat(masks_one_gpu, dim=0)
    assert samples_one_gpu.size(0) == masks_one_gpu.size(0) and samples_one_gpu.size(1) == masks_one_gpu.size(1)
    all_masks = torch.zeros((dist.get_world_size() * masks_one_gpu.size(0), masks_one_gpu.size(1)), device=device, dtype=masks_one_gpu.dtype)
    dist.all_gather_into_tensor(all_masks, masks_one_gpu)
    
    all_seqs = seq_mapper.decode(all_samples, batch_mask=all_masks)    
    
    # Save samples to disk:
    if rank == 0:
        with open(fasta_path, 'w') as f:
            for i, seq in enumerate(all_seqs):
                f.write(f'>seq_{i}\n')
                f.write(f'{seq}\n')
        logger.info(f"Saved samples to {fasta_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
