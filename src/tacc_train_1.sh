#!/bin/bash
#SBATCH -J dfsr
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH -p rtx-dev
#SBATCH -t 1:00:00
#SBATCH --exclusive

# ====== 环境设置 ======
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/work2/10214/yu_yao/frontera/workflow/dfsr"
mkdir -p "$OUTPUT_DIR"

OUT_FILE="${OUTPUT_DIR}/${TIMESTAMP}_train.out"
ERR_FILE="${OUTPUT_DIR}/${TIMESTAMP}_train.err"
exec > "$OUT_FILE"
exec 2> "$ERR_FILE"

module load python3/3.9.2
source ~/.bashrc

# ====== NCCL & CUDA 环境变量 ======
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# ====== 调试输出（可选） ======
echo "Job started at $(date)"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Current working directory: $(pwd)"

# ====== 启动训练 ======
srun python3 train_dfsr.py --nodes 2 --gpus 4 --epochs 100000
