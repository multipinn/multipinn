#!/bin/bash
#SBATCH --job-name=ginto_10
#SBATCH --partition=a800
#SBATCH --nodelist=ngpu09
#SBATCH --output=10_obs
#SBATCH --error=10_obs.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -e

. "/userspace/sva/miniconda3/etc/profile.d/conda.sh"
conda activate my_env_2

export PATH=/usr/local/cuda-11/bin${PATH:+:${PATH}}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9

NUM_GPUS=10

echo "🚀 Starting Multi-GPU PINN Training"
echo "  - $NUM_GPUS GPUs, $NUM_GPUS batches"
echo

# Check GPUs
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [ $GPU_COUNT -lt $NUM_GPUS ]; then
    echo "❌ Error: Need $NUM_GPUS GPUs, found $GPU_COUNT"
    exit 1
fi

echo "✅ Found $GPU_COUNT GPUs, using $NUM_GPUS"

export NUM_GPUS_TRAIN=$NUM_GPUS

nvidia-smi -L
nvcc -V
python -V

export PIP_CACHE_DIR="/userspace/sva/.cache/pip"

torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_ginto.py

echo "✅ Training completed!" 