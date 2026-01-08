#!/bin/bash
#SBATCH -J sample_ar                  # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu               # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

checkpoint_path=/home/jasonx62301/for_python/duo/duo/outputs/lm1b/2025.12.22/220532/checkpoints/24-40000.ckpt


# checkpoint_path=/home/jasonx62301/for_python/duo/duo/outputs/lm1b/2025.12.22/221821/checkpoints/24-40000.ckpt

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=2

for steps in 8 16 32 64 128 256 512
do
    python -u -m main \
    mode=sample_eval \
    loader.batch_size=4 \
    loader.eval_batch_size=4 \
    data=lm1b-wrap \
    algo=duo_base \
    model=small \
    model.length=1024 \
    eval.checkpoint_path=$checkpoint_path \
    sampling.num_sample_batches=50 \
    sampling.steps=$steps \
    +wandb.offline=true \
    sampling.noise_removal=greedy
done


