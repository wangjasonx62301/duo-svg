#!/bin/bash
#SBATCH -J duo-base                   # Job name
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
finetune_path=/home/jasonx62301/for_python/duo/duo/outputs/duo.ckpt

# Assuming the finetune_path corresponds to the DUO model
# trained for 500K steps with curriculum learning, we train the
# model for 500K more steps.
export CUDA_VISIBLE_DEVICES=2
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

python -u -m main \
  loader.batch_size=4 \
  loader.eval_batch_size=4 \
  data=lm1b-wrap \
  model=small \
  algo=duo_base \
  model.length=1024 \
  wandb.name=duo-lm1b-wrap-finetune-svg-with-emb \
  training.finetune_path=$finetune_path \
  sampling.num_sample_batches=0 \
  trainer.max_steps=10000 