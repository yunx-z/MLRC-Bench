import os
import argparse
from pathlib import Path
import torch
from PIL import Image
import json
from tqdm import tqdm
import logging
from WavesRemover import WavesRemover
from datetime import datetime
import time
import numpy as np
from torchvision import transforms
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_experiment(strength, phase):
    """Run watermark removal experiments"""
    logger.info(f"Running waves evaluation on {phase} set with strength {strength}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    method = WavesRemover(strength=strength).to(device)
    logger.info(f"Using device: {device}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results") / f"evaluation_{timestamp}"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize separate metrics for each method
    metrics_dict = {
        'stegastamp': {'overall_score': [], 'watermark_detection': [], 'quality_degradation': []},
        'treering': {'overall_score': [], 'watermark_detection': [], 'quality_degradation': []}
    }
    
    # Set data paths based on phase
    if phase == 'dev':
        data_dir = Path("env/data")
        file_prefix = "dev_images"
    else:
        data_dir = Path("scripts/test_data")
        file_prefix = "test_images"
    
    # Process each watermark type
    for watermark_type in ['stegastamp', 'treering']:
        # Load data from pickle file
        data_path = data_dir / f"{file_prefix}_{watermark_type}.pkl"
        if not data_path.exists():
            logger.warning(f"No data found for {watermark_type} at {data_path}")
            continue
            
        logger.info(f"Processing {watermark_type} images from {data_path}")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            images = data[watermark_type]
        
        # Process each image
        for idx, img_tensor in enumerate(tqdm(images, desc=f"Processing {watermark_type}")):
            try:
                # Convert tensor to PIL for attack
                original_img = transforms.ToPILImage()(img_tensor)
                
                # Process image
                processed_img = method.attack(original_img)
                processed_tensor = transforms.ToTensor()(processed_img).unsqueeze(0).to(device)
                
                # Evaluate
                metrics = method.evaluate(img_tensor.unsqueeze(0).to(device), processed_tensor)
                
                # Save processed image
                output_path = results_dir / watermark_type / f"processed_{idx}.png"
                output_path.parent.mkdir(exist_ok=True, parents=True)
                processed_img.save(output_path)
                
                # Accumulate metrics
                for key in ['overall_score', 'watermark_detection', 'quality_degradation']:
                    metrics_dict[watermark_type][key].append(metrics[key])
                
            except Exception as e:
                logger.error(f"Error processing image {idx} for {watermark_type}: {e}")
                continue
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Print and save results
    print(f"\nResults for {phase} set:")
    print("=" * 60)
    
    for watermark_type, metrics in metrics_dict.items():
        if any(metrics.values()):  # Only print if we have results
            print(f"\n{watermark_type.upper()}:")
            print("-" * 50)
            mean_metrics = {
                key: float(np.mean(values)) if values else 0.0
                for key, values in metrics.items()
            }
            print(f"Overall Score: {mean_metrics['overall_score']:.4f}")
            print(f"Watermark Detection: {mean_metrics['watermark_detection']:.4f}")
            print(f"Quality Degradation: {mean_metrics['quality_degradation']:.4f}")
            
            # Save to JSON
            result_file = results_dir / f"{watermark_type}_results_{timestamp}.json"
            with open(result_file, 'w') as f:
                json.dump({
                    'phase': phase,
                    'watermark_type': watermark_type,
                    'metrics': mean_metrics,
                    'num_images': len(metrics['overall_score'])
                }, f, indent=2)
    
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Run watermark removal experiments')
    parser.add_argument('-p', '--phase', type=str, required=True,
                       choices=['dev', 'test'],
                       help='Evaluation phase (dev or test)')
    parser.add_argument('-s', '--strength', type=float, default=1.0,
                       help='Strength of watermark removal (affects frequency threshold)')
    args = parser.parse_args()
    
    run_experiment(args.strength, args.phase)

if __name__ == "__main__":
    main()