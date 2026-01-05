export CUDA_VISIBLE_DEVICES=2
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

python -u -m main \
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
  model=small \
  data=lm1b-wrap \
  wandb.name=sedd-lm1b-baseline \
  algo=sedd \
  model.length=128 \
  eval.compute_generative_perplexity=True \
  sampling.predictor=analytic \
  trainer.max_steps=100000 
