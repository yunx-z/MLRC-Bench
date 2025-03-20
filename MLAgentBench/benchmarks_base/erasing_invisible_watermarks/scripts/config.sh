#!/bin/bash

# Test data locations
export TEST_DATA_STEGASTAMP="scripts/test_data/test_images_stegastamp.pkl"
export TEST_DATA_TREERING="scripts/test_data/test_images_treering.pkl"

# Development data locations
export DEV_DATA_STEGASTAMP="env/data/dev_images_stegastamp.pkl"
export DEV_DATA_TREERING="env/data/dev_images_treering.pkl"

# Evaluation parameters
export STRENGTH_DEFAULT=1.0
export BATCH_SIZE=4
export NUM_WORKERS=2

# GPU settings
export CUDA_VISIBLE_DEVICES=0

# Dataset URL
export DATASET_URL="https://drive.google.com/file/d/1Q0Ahhg_wLk3OK15fs_cQZ7_GOkye5acS/view?usp=sharing"

# Specify test file name
TEST_FILE_NAME="test_data.pkl"

# Add any other configuration variables needed for the task 