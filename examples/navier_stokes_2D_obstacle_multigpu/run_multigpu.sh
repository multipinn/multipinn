#!/bin/bash

set -e

NUM_GPUS=${1:-2}

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

torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    examples/navier_stokes_2D_obstacle_multigpu/run_train.py

echo "✅ Training completed!" 