#!/bin/bash

set -euo pipefail

###########################################################################################
# === Please modify the following paths according to your environment ===
ckpt_path=/data/yxz/starvla/outputs/gravity_single4il_qwen3gr00t_10k/checkpoints/steps_10000_pytorch_model.pt
gpu_id=5
port=10093
num_inference_timesteps_override=4
# === End of environment variable configuration ===
###########################################################################################

CUDA_VISIBLE_DEVICES=${gpu_id} /data/yxz/conda/envs/starVLA/bin/python \
  deployment/model_server/arx/server_policy_arx.py \
  --ckpt_path "${ckpt_path}" \
  --port "${port}" \
  --use_bf16 \
  --num_inference_timesteps_override "${num_inference_timesteps_override}"
