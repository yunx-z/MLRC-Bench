# Development phase constants
DEV_DATA_PATH = "data/dev"
RESULTS_PATH = "results"

# Image Quality Metrics Coefficients (from competition)
QUALITY_COEFFICIENTS = {
    'FID': 1.53e-3,
    'CLIP_FID': 5.07e-3,
    'PSNR': -2.22e-3,
    'SSIM': -1.13e-1,
    'NMI': -9.88e-2,
    'LPIPS': 3.41e-1,
    'DELTA_AESTHETICS': 4.50e-2,
    'DELTA_ARTIFACTS': -1.44e-1
}

# Performance thresholds
FPR_THRESHOLD = 0.001  # 0.1% FPR threshold
TPR_TARGET = 1.0  # Target True Positive Rate

# Final score is computed as sqrt(Q^2 + A^2) where:
# Q: normalized image quality score (0.1-0.9)
# A: accuracy metric (TPR@0.1%FPR)

# Image parameters
IMAGE_SIZE = 512
CHANNELS = 3
MAX_BATCH_SIZE = 1  # Process one image at a time due to time constraints

# Paths
DATASET_URL = "https://erasinginvisible.github.io/dataset/"  # Replace with actual dataset URL when available

# Model parameters
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 1e-4 