#!/usr/bin/env python3
# Weather4cast 2023 Benchmark
# Data utilities for the benchmark

import os
import yaml
import h5py
import numpy as np
import torch
import psutil
import GPUtil

def load_config(config_file):
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_cuda_memory_usage(gpus):
    """Get memory usage for specified GPUs."""
    if gpus is None:
        return
    
    print("GPU memory usage:")
    for gpu_id in gpus:
        if gpu_id < 0:
            continue
        gpu = GPUtil.getGPUs()[gpu_id]
        print(f"GPU {gpu_id}: {gpu.memoryUsed:.0f}MB / {gpu.memoryTotal:.0f}MB")
    
    print(f"CPU memory: {psutil.virtual_memory().percent}%")

def tensor_to_submission_file(tensor, params):
    """Convert tensor to submission file format."""
    region = params['region_to_predict']
    year = params['year_to_predict']
    
    # Create submission directory if it doesn't exist
    os.makedirs(f'submission/{year}', exist_ok=True)
    
    # Convert tensor to numpy array
    submission = tensor.cpu().numpy()
    
    # Reshape if needed
    if len(submission.shape) == 4:
        # Add batch dimension if missing
        submission = submission.reshape(1, *submission.shape)
    
    # Create HDF5 file
    with h5py.File(f'submission/{year}/{region}.pred.h5', 'w') as f:
        f.create_dataset('submission', data=submission)
    
    print(f"Submission file created: submission/{year}/{region}.pred.h5")
    print(f"Submission shape: {submission.shape}")

def load_data(data_path, region, split, year, data_type):
    """Load data from HDF5 file."""
    if data_type == 'satellite':
        file_path = os.path.join(data_path, str(year), 'HRIT', f'{region}.{split}.reflbt0.ns.h5')
        with h5py.File(file_path, 'r') as f:
            data = f['REFL-BT'][:]
    elif data_type == 'radar':
        file_path = os.path.join(data_path, str(year), 'OPERA', f'{region}.{split}.rates.crop.h5')
        with h5py.File(file_path, 'r') as f:
            data = f['rates.crop'][:]
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    return data

def create_sequences(data, seq_len, step=1):
    """Create sequences from data."""
    sequences = []
    for i in range(0, len(data) - seq_len + 1, step):
        sequences.append(data[i:i+seq_len])
    return np.array(sequences)

def normalize_data(data, min_val=None, max_val=None):
    """Normalize data to [0, 1] range."""
    if min_val is None:
        min_val = data.min()
    if max_val is None:
        max_val = data.max()
    
    return (data - min_val) / (max_val - min_val + 1e-8)

def denormalize_data(data, min_val, max_val):
    """Denormalize data from [0, 1] range."""
    return data * (max_val - min_val) + min_val 