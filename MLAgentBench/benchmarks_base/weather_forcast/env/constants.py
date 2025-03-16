#!/usr/bin/env python3
# Weather4cast 2023 Benchmark
# Constants for the benchmark

# Paths
DATA_PATH = "data"
MODELS_PATH = "methods/models"
RESULTS_PATH = "results"

# Dataset constants
SATELLITE_CHANNELS = ['IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134', 'VIS006', 'VIS008', 'WV_062', 'WV_073']

# All regions including core and transfer learning regions
REGIONS = [
    # Core regions
    'boxi_0015', 'boxi_0034', 'boxi_0076', 'roxi_0004', 'roxi_0005', 'roxi_0006', 'roxi_0007',
    # Transfer learning regions
    'roxi_0008', 'roxi_0009', 'roxi_0010'
]

# Core regions only
CORE_REGIONS = ['boxi_0015', 'boxi_0034', 'boxi_0076', 'roxi_0004', 'roxi_0005', 'roxi_0006', 'roxi_0007']

# Transfer learning regions
TRANSFER_REGIONS = ['roxi_0008', 'roxi_0009', 'roxi_0010']

# Sequence lengths
INPUT_TIMESTEPS = 4
OUTPUT_TIMESTEPS = 32  # 8 hours prediction (can be set to 16 for 4-hour prediction)

# Training constants
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
MAX_EPOCHS = 100 