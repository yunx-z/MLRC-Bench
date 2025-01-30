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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ... (keep the imports and other parts the same) ...

def run_experiment(strength, data_dir, results_dir, phase):
    """Run watermark removal experiments"""
    logger.info(f"Running waves evaluation on {phase} set with strength {strength}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    method = WavesRemover(strength=strength).to(device)
    logger.info(f"Using device: {device}")
    
    to_tensor = transforms.ToTensor()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize separate metrics for each track/method
    metrics_dict = {
        'black': {'overall_score': [], 'watermark_detection': [], 'quality_degradation': []},
        'beige_stegastamp': {'overall_score': [], 'watermark_detection': [], 'quality_degradation': []},
        'beige_treering': {'overall_score': [], 'watermark_detection': [], 'quality_degradation': []}
    }
    
    # Process each track
    for track in ['black', 'beige']:
        track_dir = data_dir / track
        if not track_dir.exists():
            continue
            
        methods = ['stegastamp', 'treering'] if track == 'beige' else ['']
        
        for method_type in methods:
            method_dir = track_dir / method_type if method_type else track_dir
            if not method_dir.exists():
                continue
            
            # Determine which metrics list to use
            current_key = track if track == 'black' else f'beige_{method_type}'
            
            image_files = sorted(list(method_dir.glob('*.png')))
            for image_path in tqdm(image_files, desc=f"Processing {track}/{method_type if method_type else ''}"):
                try:
                    original_img = Image.open(str(image_path))
                    original_tensor = to_tensor(original_img).unsqueeze(0).to(device)
                    
                    processed_img = method.attack(original_img)
                    processed_tensor = to_tensor(processed_img).unsqueeze(0).to(device)
                    metrics = method.evaluate(original_tensor, processed_tensor)
                    
                    # Save processed image
                    output_subdir = method_type if method_type else ''
                    output_path = results_dir / track / output_subdir / image_path.name
                    output_path.parent.mkdir(exist_ok=True, parents=True)
                    processed_img.save(output_path)
                    
                    # Accumulate metrics
                    for key in ['overall_score', 'watermark_detection', 'quality_degradation']:
                        metrics_dict[current_key][key].append(metrics[key])
                    
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    continue
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    # Print results for each track/method
    print("\nResults by Track/Method:")
    print("=" * 60)
    
    for track_key, track_metrics in metrics_dict.items():
        if any(track_metrics.values()):  # Only print if we have results
            print(f"\n{track_key.upper()}:")
            print("-" * 50)
            print(f"Overall Score: {np.mean(track_metrics['overall_score']):.4f}")
            print(f"Watermark Detection: {np.mean(track_metrics['watermark_detection']):.4f}")
            print(f"Quality Degradation: {np.mean(track_metrics['quality_degradation']):.4f}")
            
            # Save to separate JSON files
            result_file = results_dir / f"{track_key}_results_{timestamp}.json"
            with open(result_file, 'w') as f:
                json.dump({
                    'phase': phase,
                    'track': track_key,
                    'metrics': {
                        'overall_score': float(np.mean(track_metrics['overall_score'])),
                        'watermark_detection': float(np.mean(track_metrics['watermark_detection'])),
                        'quality_degradation': float(np.mean(track_metrics['quality_degradation']))
                    }
                }, f, indent=2)
    
    print("=" * 60)

# ... (keep the main function the same) ...

def main():
    parser = argparse.ArgumentParser(description='Run watermark removal experiments')
    parser.add_argument('-p', '--phase', type=str, required=True,
                       choices=['dev', 'test'],
                       help='Evaluation phase (dev or test)')
    parser.add_argument('-s', '--strength', type=float, default=1.0,
                       help='Strength of watermark removal (affects frequency threshold)')
    args = parser.parse_args()

    base_path = Path("/data2/monmonli/MLAgentBench/MLAgentBench/benchmarks_base/erasing_invisible_watermarks/env")
    data_dir = base_path / f"data/{args.phase}"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results") / f"evaluation_{timestamp}"
    results_dir.mkdir(exist_ok=True, parents=True)

    run_experiment(args.strength, data_dir, results_dir, args.phase)

if __name__ == "__main__":
    main()