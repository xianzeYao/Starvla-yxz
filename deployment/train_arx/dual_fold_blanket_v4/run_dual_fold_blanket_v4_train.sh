#!/bin/bash


# Leave the NIC selection commented by default. Enable only after confirming the
# interface names on the current node, otherwise NCCL may fail to bootstrap.
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenGR00T
freeze_module_list=''
base_vlm=/mnt/data-alpha-sg-01/team-camera/home/n84416302/ckpts/base_models/Qwen3-VL-4B-Instruct
config_yaml=./deployment/train_arx/dual_fold_blanket_v4/starvla_train_dual_fold_blanket_v4_qwen3_gr00t.yaml
dataset_root=/mnt/data-alpha-sg-01/team-camera/home/n84416302/dataset
dataset_name=dual_fold_blanket_v4
data_mix=dual_fold_blanket_v4
run_root_dir=/home/n84416302/starVLA-yxz/results
run_id=dual_fold_blanket_v4_qwen3gr00t_50k
num_processes=4
wandb_project=StarVLA_ARX
wandb_entity=yao-xian-ze
# === End of environment variable configuration ===
###########################################################################################

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
dataset_dir=${dataset_root}/${dataset_name}
modality_template=${script_dir}/dual_fold_blanket_v4.modality.json
output_dir=${run_root_dir}/${run_id}

mkdir -p "${output_dir}"

# Keep the dataset-side modality mapping in sync with this training config.
cp "${modality_template}" "${dataset_dir}/meta/modality.json"
cp "$0" "${output_dir}/"
cp "${config_yaml}" "${output_dir}/"
cp "${modality_template}" "${output_dir}/"

/home/n84416302/miniconda3/envs/starvla/bin/accelerate launch \
  --main_process_port 29617 \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${dataset_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.action_type abs_qpos \
  --datasets.vla_data.action_mode abs \
  --datasets.vla_data.include_state False \
  --datasets.vla_data.per_device_batch_size 16 \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 50000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project ${wandb_project} \
  --wandb_entity ${wandb_entity}
