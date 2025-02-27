import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity, normalized_mutual_information
import pickle


def evaluate_method(method, phase, track, track_type=None):
    """
    Process images with the watermark removal method and save results
    
    Args:
        method: The method to evaluate
        phase: 'dev' or 'test'
        track: 'beige' or 'black'
        track_type: 'stegastamp' or 'treering' (only for beige track)

    Returns:
        dict: Average metrics across all processed images
    """
    data_dir = "data/"
    print(f"Data directory: {data_dir}")
    
    if track == "beige":
        data_file = os.path.join(data_dir, f"{phase}_images_{track_type}.pkl")
        print(f"Looking for data file: {data_file}")
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
            images = data[track_type]  # List of tensors
    else:
        raise ValueError(f"Unsupported track: {track}")
    
    # Create output directory
    output_dir = os.path.join("output", phase, track)
    if track_type:
        output_dir = os.path.join(output_dir, track_type)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    for idx, img_tensor in enumerate(images):
        try:
            # Convert tensor to PIL Image without resizing
            img = transforms.ToPILImage()(img_tensor)
            
            # Get original image size
            orig_size = img.size
            
            # Process image with method
            processed_img = method.remove_watermark(img, track_type=track_type)
            
            # Ensure processed image has same size as original
            if processed_img.size != orig_size:
                processed_img = processed_img.resize(orig_size, Image.Resampling.LANCZOS)
            
            # Save processed image
            output_path = os.path.join(output_dir, f"{idx}.png")
            processed_img.save(output_path)
                
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            continue

def get_scores(method, phase, track):
    """
    Calculate evaluation scores for processed images
    
    Args:
        method: The method to evaluate
        phase (str): Either 'dev' or 'test'
        track (str): Track name (e.g., 'beige_stegastamp', 'beige_treering')
        
    Returns:
        dict: Dictionary containing evaluation scores
    """
    # Parse track information
    track_type = "beige"
    track_subtype = track.split("_")[1]
    
    # First ensure images are processed
    evaluate_method(method, phase, track_type, track_subtype)
    
    # Initialize metrics accumulator
    total_metrics = {
        'overall_score': 0.0,
        'watermark_detection': 0.0,
        'quality_degradation': 0.0,
        'psnr': 0.0,
        'ssim': 0.0,
        'nmi': 0.0,
        'mse': 0.0
    }
    valid_count = 0
    
    # Load original images
    if phase == 'dev':
        data_dir = "data"
    else:  # test phase
        data_dir = os.path.join("..", "scripts", "test_data")
    
    data_file = os.path.join(data_dir, f"{phase}_images_{track_subtype}.pkl")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
        images = data[track_subtype]
    
    # Calculate metrics for each image
    output_dir = os.path.join("output", phase, track_type, track_subtype)
    
    for idx, img_tensor in enumerate(images):
        try:
            processed_path = os.path.join(output_dir, f"{idx}.png")
            if not os.path.exists(processed_path):
                continue
                
            # Convert original tensor to numpy
            img_np = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
            
            # Load and convert processed image
            processed_img = Image.open(processed_path).convert('RGB')
            processed_tensor = transforms.ToTensor()(processed_img).unsqueeze(0)
            proc_np = processed_tensor.squeeze(0).permute(1, 2, 0).numpy()
            
            # Calculate basic metrics
            mse = np.mean((img_np - proc_np) ** 2)
            psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
            ssim = structural_similarity(img_np, proc_np, channel_axis=2, data_range=1.0)
            nmi = normalized_mutual_information(
                (img_np * 255).astype(np.uint8),
                (proc_np * 255).astype(np.uint8)
            )
            
            # Quality score (Q) - penalize if image is too similar to original
            similarity_penalty = 0.9 if ssim > 0.95 and psnr > 45 else 0.0
            Q = 0.5 * (1 - ssim) + 0.3 * (1 - psnr/50) + 0.2 * (1 - nmi) + similarity_penalty
            Q = np.clip(Q, 0.1, 0.9)
            
            # Watermark detection score (A)
            def high_freq_diff(img):
                gray = np.mean(img, axis=2)
                freq = np.fft.fft2(gray)
                freq_shift = np.fft.fftshift(freq)
                h, w = gray.shape
                center_h, center_w = h//2, w//2
                high_freq = np.abs(freq_shift[center_h-10:center_h+10, center_w-10:center_w+10])
                return np.mean(high_freq)
            
            orig_hf = high_freq_diff(img_np)
            proc_hf = high_freq_diff(proc_np)
            freq_diff = abs(orig_hf - proc_hf) / orig_hf
            freq_penalty = 0.9 if freq_diff < 0.05 else 0.0
            A = np.clip(freq_diff + freq_penalty, 0.1, 0.9)
            
            # Overall score
            overall_score = np.sqrt(Q**2 + A**2)
            
            metrics = {
                'overall_score': float(overall_score),
                'watermark_detection': float(A),
                'quality_degradation': float(Q),
                'psnr': float(psnr),
                'ssim': float(ssim),
                'nmi': float(nmi),
                'mse': float(mse)
            }
            
            # Save metrics
            metrics_path = os.path.join(output_dir, f"{idx}_metrics.pkl")
            with open(metrics_path, 'wb') as f:
                pickle.dump(metrics, f)
            
            # Accumulate metrics
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            valid_count += 1
                
        except Exception as e:
            print(f"Error calculating metrics for image {idx}: {e}")
            continue
    
    # Calculate averages
    if valid_count > 0:
        for key in total_metrics:
            total_metrics[key] /= valid_count
    
    # Return the required metrics
    return {
        'overall_score': total_metrics['overall_score'],
        'watermark_detection': total_metrics['watermark_detection'],
        'quality_degradation': total_metrics['quality_degradation']
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate watermark removal')
    parser.add_argument('--original', type=str, required=True, help='Directory with original images')
    parser.add_argument('--processed', type=str, required=True, help='Directory with processed images')
    
    args = parser.parse_args()
    
