CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python -u -m main \
  loader.batch_size=16 \
  loader.eval_batch_size=8 \
  data=wikitext2 \
  wandb.name=duo-wikitext2 \
  model=small \
  algo=duo \
  model.length=128 \
  algo.gumbel_tau_log10_start=-3.0 \
  algo.gumbel_tau_log10_end=-3.0 \
  algo.gamma_min=-3.5 \
  algo.gamma_max=-1.75 \
  algo.curriculum_start=0 \
  algo.curriculum_end=500000 \
  checkpointing.resume_from_ckpt=/home/jasonx62301/for_python/duo/duo/outputs/wikitext2/2025.11.08/134619/checkpoints/1833-66000.ckpt