#!/bin/bash
#SBATCH -J dfsr
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH -p rtx
#SBATCH -t 48:00:00
#SBATCH --exclusive

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/work2/10214/yu_yao/frontera/workflow/dfsr"
mkdir -p "$OUTPUT_DIR"

OUT_FILE="${OUTPUT_DIR}/${TIMESTAMP}_train.out"
ERR_FILE="${OUTPUT_DIR}/${TIMESTAMP}_train.err"

exec > "$OUT_FILE"
exec 2> "$ERR_FILE"

module load python3/3.9.2
source ~/.bashrc

srun python3 train_dfsr.py \
    --nodes 2 \
    --gpus 4 \
    --epochs 100000

