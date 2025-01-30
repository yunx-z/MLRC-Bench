import os
import sys
import shutil
from pathlib import Path
import random
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance

# Define project root and data paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = "/data2/monmonli/MLAgentBench/MLAgentBench/benchmarks_base/erasing_invisible_watermarks/env/data"

def create_splits(dev_ratio=0.2):
    """
    Create dev and test splits for both tracks
    
    Args:
        dev_ratio: Ratio of data to use for dev set (default: 0.2)
    """
    # Create necessary directories
    splits = ['dev', 'test']
    for split in splits:
        # Create directories for black box track
        os.makedirs(os.path.join(PROJECT_ROOT, "env", "data", split, 'black'), exist_ok=True)
        # Create directories for beige box track methods
        os.makedirs(os.path.join(PROJECT_ROOT, "env", "data", split, 'beige', 'stegastamp'), exist_ok=True)
        os.makedirs(os.path.join(PROJECT_ROOT, "env", "data", split, 'beige', 'treering'), exist_ok=True)
    
    # Handle Beige Box Track - StegaStamp method (0-149)
    beige_dir = os.path.join(DATASET_ROOT, 'beige')
    if os.path.exists(beige_dir):
        stega_images = [f"{i}.png" for i in range(150)]
        dev_stega = random.sample(stega_images, int(len(stega_images) * dev_ratio))
        test_stega = [img for img in stega_images if img not in dev_stega]
        
        # Copy StegaStamp files
        for img in dev_stega:
            src = os.path.join(beige_dir, img)
            dst = os.path.join(PROJECT_ROOT, "env", "data", 'dev', 'beige', 'stegastamp', img)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                
        for img in test_stega:
            src = os.path.join(beige_dir, img)
            dst = os.path.join(PROJECT_ROOT, "env", "data", 'test', 'beige', 'stegastamp', img)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        print(f"Beige Box Track - StegaStamp split complete:")
        print(f"Dev set: {len(dev_stega)} images")
        print(f"Test set: {len(test_stega)} images")

        # Handle Tree-Ring images (150-299)
        tree_images = [f"{i}.png" for i in range(150, 300)]
        dev_tree = random.sample(tree_images, int(len(tree_images) * dev_ratio))
        test_tree = [img for img in tree_images if img not in dev_tree]
        
        # Copy Tree-Ring files
        for img in dev_tree:
            src = os.path.join(beige_dir, img)
            dst = os.path.join(PROJECT_ROOT, "env", "data", 'dev', 'beige', 'treering', img)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                
        for img in test_tree:
            src = os.path.join(beige_dir, img)
            dst = os.path.join(PROJECT_ROOT, "env", "data", 'test', 'beige', 'treering', img)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        print(f"Beige Box Track - Tree-Ring split complete:")
        print(f"Dev set: {len(dev_tree)} images")
        print(f"Test set: {len(test_tree)} images")
    
    # Handle Black Box Track
    black_dir = os.path.join(DATASET_ROOT, 'black')
    if os.path.exists(black_dir):
        all_images = [f"{i}.png" for i in range(300)]
        dev_images = random.sample(all_images, int(len(all_images) * dev_ratio))
        test_images = [img for img in all_images if img not in dev_images]
        
        # Copy files to respective directories
        for img in dev_images:
            src = os.path.join(black_dir, img)
            dst = os.path.join(PROJECT_ROOT, "env", "data", 'dev', 'black', img)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                
        for img in test_images:
            src = os.path.join(black_dir, img)
            dst = os.path.join(PROJECT_ROOT, "env", "data", 'test', 'black', img)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        print(f"\nBlack Box Track split complete:")
        print(f"Dev set: {len(dev_images)} images")
        print(f"Test set: {len(test_images)} images")

def prepare_data():
    """
    Prepare data directories for black box and beige box tracks.
    Following competition structure:
    - Black Box Track: Images watermarked with unknown methods (0-299)
    - Beige Box Track:
        - StegaStamp method (images 0-149)
        - Tree-Ring method (images 150-299)
    
    Source data is located at:
    - Beige Box: /data2/monmonli/MLAgentBench/MLAgentBench/benchmarks_base/erasing_invisible_watermarks/env/data/beige
    - Black Box: /data2/monmonli/MLAgentBench/MLAgentBench/benchmarks_base/erasing_invisible_watermarks/env/data/black
    """
    # Create split directories
    splits_dir = os.path.join(PROJECT_ROOT, "env", "data")
    for split in ['dev', 'test']:
        os.makedirs(os.path.join(splits_dir, split, 'black'), exist_ok=True)
        os.makedirs(os.path.join(splits_dir, split, 'beige', 'stegastamp'), exist_ok=True)
        os.makedirs(os.path.join(splits_dir, split, 'beige', 'treering'), exist_ok=True)
    
    print(f"Created split directories under: {splits_dir}")
    print("\nSource data locations:")
    print(f"- Black Box Track: {os.path.join(DATASET_ROOT, 'black')}")
    print(f"- Beige Box Track: {os.path.join(DATASET_ROOT, 'beige')}")
    
    print("\nNote: For submission, ensure images maintain their original names and format (.png)")
    print("      Images will be evaluated for both quality preservation and watermark removal effectiveness")

def setup_environment():
    """Setup additional environment requirements"""
    results_dir = os.path.join(PROJECT_ROOT, "env", "results")
    os.makedirs(results_dir, exist_ok=True)

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    prepare_data()
    
    # Create the splits from the source data
    create_splits(dev_ratio=0.2)  # 20% for dev, 80% for test
    
    setup_environment() 