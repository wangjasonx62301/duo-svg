#!/bin/bash
#SBATCH -J zeroshot_duo               # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov          # Request partition
#SBATCH --constraint="[a5000|a6000|a100]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

# checkpoint_path=/home/jasonx62301/for_python/duo/duo/outputs/lm1b_duo/2025.11.25/112858/checkpoints/best.ckpt
checkpoint_path=/home/jasonx62301/for_python/duo/duo/outputs/lm1b_duo/2025.11.25/113046/checkpoints/best.ckpt
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=2

datasets=("ag_news"
          "scientific_papers_pubmed"
          "scientific_papers_arxiv"
          "lambada"
          "wikitext2"
          "wikitext103"
          )
for data in "${datasets[@]}"; do
  echo "$data"
  python -u -m main \
    mode=ppl_eval \
    loader.batch_size=32 \
    loader.eval_batch_size=32 \
    loader.eval_global_batch_size=32 \
    data="$data" \
    data.insert_valid_eos=False \
    model=small \
    algo=duo_base \
    model.length=128 \
    eval.checkpoint_path=$checkpoint_path \
    +wandb.offline=true
done