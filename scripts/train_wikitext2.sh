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

python -u -m main \
  loader.batch_size=32 \
  loader.eval_batch_size=8 \
  data=wikitext2 \
  wandb.name=duo-wikitext2 \
  model=small \
  algo=duo \
  model.length=128 \
  algo.gumbel_tau_log10_start=-3.0 \
  algo.gumbel_tau_log10_end=-3.0 \
  algo.gamma_min=-3.55 \
  algo.gamma_max=-1.85 \
  algo.curriculum_start=0 \
  algo.curriculum_end=500000 \
  checkpointing.resume_from_ckpt=false
#   checkpointing.resume_ckpt_path=/home/jasonx62301/for_python/duo/duo/outputs/wikitext2/2025.11.11/143337/checkpoints/last.ckpt \
