#!/bin/bash
# Configuration file for the Weather4cast 2023 benchmark

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)/../env
export DATA_DIR=$(pwd)/../env/data
export RESULTS_DIR=$(pwd)/../results

# GPU configuration
export CUDA_VISIBLE_DEVICES=0

# Default parameters
export BATCH_SIZE=16
export LEARNING_RATE=0.0001
export MAX_EPOCHS=100

echo "Weather4cast 2023 benchmark configuration loaded." 