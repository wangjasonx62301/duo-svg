#!/bin/bash
#SBATCH -J duo-lm1b                   # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=64000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov          # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --constraint="gpu-mid|gpu-high"
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

# To enable preemption re-loading, set `hydra.run.dir` or 
# `checkpointing.save_dir` explicitly.

export CUDA_VISIBLE_DEVICES=1
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

python -u -m main \
  loader.batch_size=4 \
  loader.eval_batch_size=4 \
  data=lm1b-wrap \
  wandb.name=duo-lm1b-baseline-gpt2-no-curriculum-model-length-1024-fix \
  model=small \
  algo=duo \
  algo.curriculum_start=0 \
  algo.curriculum_end=0 \
  algo.gumbel_tau_log10_start=-3.0 \
  algo.gumbel_tau_log10_end=-3.0 \
  algo.gamma_min=-3.5 \
  algo.gamma_max=-1.75 \
  model.length=1024 \
  trainer.max_steps=100000 \
  