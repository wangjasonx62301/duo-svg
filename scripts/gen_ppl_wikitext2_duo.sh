checkpoint_path=/home/jasonx62301/for_python/duo/duo/outputs/wikitext2/baseline/164440/checkpoints/best.ckpt

# checkpoint_path=/home/jasonx62301/for_python/duo/duo/outputs/lm1b/2025.11.25/113046/checkpoints/best.ckpt
# steps=64

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

for steps in 64
do
    python -u -m main \
    mode=sample_eval \
    loader.batch_size=4 \
    loader.eval_batch_size=4 \
    data=wikitext2 \
    algo=duo_base \
    model=small \
    model.length=1024 \
    eval.checkpoint_path=$checkpoint_path \
    sampling.num_sample_batches=15 \
    sampling.steps=$steps \
    +wandb.offline=true \
    sampling.noise_removal=greedy
done