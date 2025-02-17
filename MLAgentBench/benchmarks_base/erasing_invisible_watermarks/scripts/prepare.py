"""
Prepare and split the beige box dataset into development and test sets.

Steps:
1. Rename your downloaded beige box folder to 'temp_data' and place it in scripts/
2. This script will:
   - Split images by algorithm (0-149: StegaStamp, 150-299: TreeRing)
   - Further split each algorithm's data into dev/test sets
   - Save processed data in required locations
"""

import os
import pickle
import random
import shutil
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import sys

def create_directories():
    """Create necessary directories for storing split datasets"""
    directories = [
        'scripts/test_data',  # For test set (hidden from AI agents)
        'env/data'           # For development set
    ]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def check_dataset_folder(data_dir):
    """Check if renamed beige box dataset exists with correct images"""
    if not data_dir.exists():
        print("\n" + "="*80)
        print("DATASET NOT FOUND!")
        print("\nStep 1: Download the beige box dataset")
        print("Step 2: Rename the downloaded folder to 'temp_data' and place it in scripts/")
        print("\nExpected structure:")
        print("scripts/")
        print("└── temp_data/")
        print("    └── [numbered images from beige box (0.png to 299.png)]")
        print("\nNote: Images 0-149 are StegaStamp algorithm")
        print("      Images 150-299 are TreeRing algorithm")
        print("="*80 + "\n")
        sys.exit(1)
    
    # Check for numbered images
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(list(data_dir.glob(ext)))
    
    if not image_files:
        print("\n" + "="*80)
        print("NO IMAGES FOUND!")
        print("\nThe scripts/temp_data directory exists but contains no images.")
        print("Please ensure you've copied all numbered images (0.png to 299.png)")
        print("from the beige box dataset into the scripts/temp_data directory.")
        print("="*80 + "\n")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images in the renamed beige box dataset")
    return data_dir

def load_images_from_directory(data_dir):
    """Load and split images by watermarking algorithm"""
    dataset = {'stegastamp': [], 'treering': []}
    
    # Get all image files and sort them by number
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(list(data_dir.glob(ext)))
    
    image_files.sort(key=lambda x: int(x.stem))  # Sort by the numeric part of filename
    print(f"\nProcessing {len(image_files)} total images from beige box dataset")
    
    # Split images by algorithm based on index
    for img_path in image_files:
        try:
            img_num = int(img_path.stem)
            img = Image.open(img_path)
            img = transforms.ToTensor()(img)
            
            # Split based on algorithm (0-149: StegaStamp, 150+: TreeRing)
            if img_num < 150:
                dataset['stegastamp'].append(img)
                print(f"Loaded StegaStamp algorithm image: {img_path.name}")
            else:
                dataset['treering'].append(img)
                print(f"Loaded TreeRing algorithm image: {img_path.name}")
                
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
    
    print(f"\nSplit by algorithm:")
    print(f"- StegaStamp algorithm: {len(dataset['stegastamp'])} images (from 0.png to 149.png)")
    print(f"- TreeRing algorithm: {len(dataset['treering'])} images (from 150.png to 299.png)")
    
    return dataset

def prepare_dataset(data_dir, test_ratio=0.2, seed=42):
    """
    Process the beige box dataset:
    1. Split by algorithm (StegaStamp vs TreeRing)
    2. For each algorithm, split into dev/test sets
    
    Args:
        data_dir: Directory containing the renamed beige box images
        test_ratio: Ratio of images to use for testing
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Create necessary directories
    scripts_dir = Path("scripts")
    test_data_dir = scripts_dir / "test_data"
    env_data_dir = Path("env") / "data"
    
    os.makedirs(test_data_dir, exist_ok=True)
    os.makedirs(env_data_dir, exist_ok=True)
    
    # Verify renamed beige box dataset
    data_dir = check_dataset_folder(data_dir)
    
    # Load and split by algorithm
    dataset = load_images_from_directory(data_dir)
    
    if not dataset['stegastamp'] and not dataset['treering']:
        print("No valid images found in the beige box dataset!")
        sys.exit(1)
    
    # Further split each algorithm's data into dev/test sets
    for watermark_type in ['stegastamp', 'treering']:
        if not dataset[watermark_type]:
            print(f"Warning: No {watermark_type} algorithm images found")
            continue
            
        print(f"\nSplitting {watermark_type} algorithm data into dev/test sets...")
        images = dataset[watermark_type]
        num_test = int(len(images) * test_ratio)
        
        # Randomly select test images
        test_indices = random.sample(range(len(images)), num_test)
        test_images = [images[i] for i in test_indices]
        dev_images = [img for i, img in enumerate(images) if i not in test_indices]
        
        print(f"- Total {watermark_type} images: {len(images)}")
        print(f"- Test set size: {len(test_images)} images")
        print(f"- Development set size: {len(dev_images)} images")
        
        # Save test images (hidden from AI agents)
        test_data = {watermark_type: test_images}
        test_save_path = test_data_dir / f"test_images_{watermark_type}.pkl"
        with open(test_save_path, 'wb') as f:
            pickle.dump(test_data, f)
        print(f"- Saved {watermark_type} test data to: {test_save_path}")
        
        # Save development images
        dev_data = {watermark_type: dev_images}
        dev_save_path = env_data_dir / f"dev_images_{watermark_type}.pkl"
        with open(dev_save_path, 'wb') as f:
            pickle.dump(dev_data, f)
        print(f"- Saved {watermark_type} development data to: {dev_save_path}")

def main():
    print("\n" + "="*80)
    print("BEIGE BOX DATASET PREPARATION")
    print("="*80)
    print("\nThis script will:")
    print("1. Use your renamed beige box dataset (scripts/temp_data folder)")
    print("2. Split images by algorithm (StegaStamp: 0-149, TreeRing: 150-299)")
    print("3. Further split each algorithm's data into:")
    print("   - Development set (in env/data/)")
    print("   - Test set (in scripts/test_data/)")
    print("\nMake sure you've placed the temp_data folder in the scripts directory!")
    print("="*80 + "\n")
    
    # Create necessary directories
    create_directories()
    
    # Set up paths - now looking in scripts/temp_data
    scripts_dir = Path("scripts")
    temp_dir = scripts_dir / "temp_data"
    
    # Process and split dataset
    prepare_dataset(temp_dir)
    
    print("\n" + "="*80)
    print("DATASET PREPARATION COMPLETED!")
    print("- Test data (hidden from AI agents) saved in: scripts/test_data/")
    print("- Development data saved in: env/data/")
    print("="*80 + "\n")

if __name__ == "__main__":
    main() 