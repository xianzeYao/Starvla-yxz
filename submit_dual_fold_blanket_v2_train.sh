#!/bin/bash
#SBATCH --job-name=train_dual_fold_blanket_v2
#SBATCH --partition=lrc-xlong
#SBATCH --qos=normal
#SBATCH --gres=gpu:h200:4        
#SBATCH --time=24:00:00
#SBATCH --nodes=1               
#SBATCH --ntasks=1              
#SBATCH --cpus-per-task=64      
#SBATCH --mem=512G
#SBATCH --output=/home/n84416302/starVLA-yxz/slurm/train_dual_fold_blanket_v2_%j.out
#SBATCH --error=/home/n84416302/starVLA-yxz/slurm/train_dual_fold_blanket_v2_%j.err

set -euo pipefail

# 跳转到您的项目根目录，也就是包含 deployment 的目录
cd /home/n84416302/starVLA-yxz
mkdir -p slurm
export WANDB_API_KEY=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
# 执行训练脚本
bash deployment/train_arx/dual_fold_blanket_v2/run_dual_fold_blanket_v2_train.sh