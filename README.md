# TangleDiff
TangleDiff is a powerful latent-diffusion-based generative model capbale of designing foldable, diverse and novel protein sequences that form into entangled homodimers. The current version enables the specification of sequence length range and inter-chain binding energy range during generation. Generated sequences by TangleDiff have been successfully made into stress-relaxation hydrogels.

We present here a dataset of 79,890 entangled sequences collected through large-scale sequence screening, ESMFold structure prediction and a setries of data cleaning steps. Sequences and all other information (e.g. functions, organisms, calculated binding energies) are availbale at `data/Uniref50_entangled_homodimer.csv`.

To use TangleDiff, we provide a conda environment for the easy use of TangleDiff. You can install TangleDiff and its dependencies by:

```shell
git clone https://github.com/daniel-dpq/TangleDiff.git
cd TangleDiff
conda env create --name=tanglediff -f environment.yml
conda activate tanglediff
```

Code organization:
* `train.py` - the main script to train the model.
* `sample.py` - the main script to sample entangled sequences.
* `samples/` - generated sequence samples.
* `data/` - training data.
* `weight/` - model weight.
* `config/` - configure files for training and sampling.
* `tanglediff/` - codes to retrain the model


Training
-----------------------------------------------------------------------------------------------------

To training TangleDiff on yourself, first specifiy configures in `config/train.yaml`. Some important configures are listed as:
* `data.data_file` - path to the csv training data file.
* `data.binding_energy_seps` - numbers to divide binding energy bins. Defaults to `[-160, -140, -120, -100]`
* `model.name` - name to specify the model size. Defaults to `DiT-B/2`
* `model.self_cond/` - whether to use self-conditioning. Defaults to `True`
* `diffusion.predict_xstart/` - wether to predict x_start or noise. Defaults to `True`
* `experiment.seq_encode_mode/` - encoder used to encode sequences to the latent. Choices: `esm2_t30_150M_UR50D, esm2_t33_650M_UR50D, esm2_t36_3B_UR50D, VHSE, tape_base, tape_unirep`
* `experiment.global_batch_size/` - global batch size across all GPUs. Defaults to `128`
* `experiment.train_lr/` - learning rate. Defaults to `1e-4`
* `experiment.epochs/` - training epochs. Defaults to `500`
* `experiment.save_and_sample_interval/` - save and sample every `N` step. Defaults to `8000`
* `experiment.log_interval/` - log every `N` step. Defaults to `400`
* `experiment.num_samples/` - number of sequences to sample at each checkpoint.  Defaults to `50`
* `experiment.use_wandb/` - use wandb to log or not.  Defaults to `False`

Specify the number of GPU used to train though `nproc_per_node` argument in `train.sh` and run `./train.sh` to train your own model.


Sampling
-----------------------------------------------------------------------------------------------------

Specifiy configures in `config/sample.yaml` before sampling. If you are using your own model, make sure data, model, diffusion configures are specified the same as your training figures. Some important configures are listed as:
* `condition.binding_energy_condition` - binding energy range. Choices: `0 ~ 4`
* `sample.ckpt` - model checkpoint to load.
* `sample.num_samples/` - number of sequences to sample.
* `sample.per_proc_batch_size/` - batch size per GPU.
* `sample.sample_length/` - sample lengths. can be `null`, an `int` or a `list` range.
* `sample.cfg_scale/` - 
hyperparameter that controls the strength of the guidance toward the conditional signal in classifer free guidance. 1 for unconditional generation.
* `sample.out_dir/` - path to save.

We provide some script examples to run unconditional, length-specified, binding-energy-specified sampling in `./sample.sh`
