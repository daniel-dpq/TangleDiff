#!/bin/bash

# unconditional sampling
torchrun --nnodes=1 --nproc_per_node=1 sample.py \
    'sample.num_samples=50' \
    'sample.per_proc_batch_size=50'

# specify a length
torchrun --nnodes=1 --nproc_per_node=1 sample.py \
    'sample.num_samples=50' \
    'sample.per_proc_batch_size=50' \
    'sample.sample_length=60'

# specify a length range
torchrun --nnodes=1 --nproc_per_node=1 sample.py \
    'sample.num_samples=50' \
    'sample.per_proc_batch_size=50' \
    'sample.sample_length=[60, 100]'

# specify a binding energy condition range
# 0: < -160 REU
# 1: -160 ~ -140 REU
# 2: -140 ~ -120 REU
# 3: -120 ~ -100 REU
# 4: -100 ~ 0 REU
# null: do not specify a binding energy condition
torchrun --nnodes=1 --nproc_per_node=1 sample.py \
    'sample.num_samples=50' \
    'sample.per_proc_batch_size=50' \
    'sample.sample_length=[60, 100]' \
    'sample.cfg_scale=4.'   \
    'condition.binding_energy_condition=0'


