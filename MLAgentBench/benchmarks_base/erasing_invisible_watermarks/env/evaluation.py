import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity, normalized_mutual_information
import pickle
from MLAgentBench.constants import MLR_BENCH_DIR

def evaluate_image(original_path, processed_path):
    """
    Evaluate the quality between original and processed images
    
    Args:
        original_path (str): Path to original image
        processed_path (str): Path to processed image
            
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    try:
        # Load and convert images to tensors
        transform = transforms.ToTensor()
        original_img = Image.open(original_path).convert('RGB')
        processed_img = Image.open(processed_path).convert('RGB')
        
        original_tensor = transform(original_img).unsqueeze(0)
        processed_tensor = transform(processed_img).unsqueeze(0)
        
        # Convert tensors to numpy arrays
        original_np = original_tensor.squeeze(0).permute(1, 2, 0).numpy()
        processed_np = processed_tensor.squeeze(0).permute(1, 2, 0).numpy()
        
        # Basic metrics
        mse = np.mean((original_np - processed_np) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        ssim = structural_similarity(original_np, processed_np, channel_axis=2, data_range=1.0)
        nmi = normalized_mutual_information(
            (original_np * 255).astype(np.uint8),
            (processed_np * 255).astype(np.uint8)
        )
        
        # Quality score (Q) based on basic metrics
        Q = 0.5 * (1 - ssim) + 0.3 * (1 - psnr/50) + 0.2 * (1 - nmi)
        Q = np.clip(Q, 0.1, 0.9)  # Ensure Q is in [0.1, 0.9] range
        
        # Watermark detection score (A) based on high frequency differences
        def high_freq_diff(img):
            gray = np.mean(img, axis=2)
            freq = np.fft.fft2(gray)
            freq_shift = np.fft.fftshift(freq)
            h, w = gray.shape
            center_h, center_w = h//2, w//2
            high_freq = np.abs(freq_shift[center_h-10:center_h+10, center_w-10:center_w+10])
            return np.mean(high_freq)
        
        orig_hf = high_freq_diff(original_np)
        proc_hf = high_freq_diff(processed_np)
        A = np.clip(abs(orig_hf - proc_hf) / orig_hf, 0.1, 0.9)
        
        # Overall score
        overall_score = np.sqrt(Q**2 + A**2)
        
        return {
            'overall_score': float(overall_score),
            'watermark_detection': float(A),
            'quality_degradation': float(Q),
            'psnr': float(psnr),
            'ssim': float(ssim),
            'nmi': float(nmi),
            'mse': float(mse)
        }
        
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return {
            'overall_score': 0.0,
            'watermark_detection': 0.0,
            'quality_degradation': 0.0,
            'error': str(e)
        }

def evaluate_dataset(original_dir, processed_dir):
    """
    Evaluate all images in a directory
    
    Args:
        original_dir (str): Directory containing original images
        processed_dir (str): Directory containing processed images
        
    Returns:
        dict: Average metrics across all images
    """
    metrics_sum = {
        'overall_score': 0.0,
        'watermark_detection': 0.0,
        'quality_degradation': 0.0,
        'psnr': 0.0,
        'ssim': 0.0,
        'nmi': 0.0,
        'mse': 0.0
    }
    
    count = 0
    for img_name in os.listdir(original_dir):
        if img_name.endswith('.png'):
            original_path = os.path.join(original_dir, img_name)
            processed_path = os.path.join(processed_dir, img_name)
            
            if os.path.exists(processed_path):
                metrics = evaluate_image(original_path, processed_path)
                
                if 'error' not in metrics:
                    for key in metrics_sum:
                        metrics_sum[key] += metrics[key]
                    count += 1
    
    # Calculate averages
    if count > 0:
        for key in metrics_sum:
            metrics_sum[key] /= count
    
    metrics_sum['processed_images'] = count
    return metrics_sum

def evaluate_method(method, phase, track, track_type=None, base_dir=None):
    """
    Evaluate a watermark removal method
    
    Args:
        method: The method to evaluate
        phase: 'dev' or 'test'
        track: 'beige' or 'black'
        track_type: 'stegastamp' or 'treering' (only for beige track)
        base_dir: Base directory path
        
    Returns:
        dict: Average metrics across all processed images
    """
    if base_dir is None:
        base_dir = './'
    
    # Load data from pickle files - different paths for dev and test
    if phase == 'dev':
        data_dir = os.path.join(base_dir, "data")
    else:  # test phase
        data_dir = os.path.join(base_dir, "..", "scripts", "test_data")
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
    output_dir = os.path.join(base_dir, "output", phase, track)
    if track_type:
        output_dir = os.path.join(output_dir, track_type)
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Process each image
    for idx, img_tensor in enumerate(images):
        try:
            # Convert tensor to PIL Image
            img = transforms.ToPILImage()(img_tensor)
            
            # Process image with method
            processed_img = method.remove_watermark(img, track_type=track_type)
            
            # Save processed image
            output_path = os.path.join(output_dir, f"{idx}.png")
            processed_img.save(output_path)
            
            # Convert processed image back to tensor for evaluation
            processed_tensor = transforms.ToTensor()(processed_img).unsqueeze(0)
            
            # Evaluate and save metrics
            metrics = method.evaluate(img_tensor.unsqueeze(0), processed_tensor)
            metrics_path = os.path.join(output_dir, f"{idx}_metrics.pkl")
            with open(metrics_path, 'wb') as f:
                pickle.dump(metrics, f)
            
            # Accumulate metrics if no error
            if 'error' not in metrics:
                for key in total_metrics:
                    if key in metrics:
                        total_metrics[key] += metrics[key]
                valid_count += 1
                
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            continue
    
    # Calculate averages
    if valid_count > 0:
        for key in total_metrics:
            total_metrics[key] /= valid_count
    
    return total_metrics

def get_scores(method, phase, track):
    """
    Get evaluation scores for a method
    
    Args:
        method: The watermark removal method
        phase (str): Either 'dev' or 'test'
        track (str): Track name (e.g., 'black', 'beige_stegastamp', 'beige_treering')
        
    Returns:
        dict: Dictionary containing evaluation scores
    """
    # Parse track information
    if track == "black":
        track_type = "black"
        track_subtype = None
    else:
        track_type = "beige"
        track_subtype = track.split("_")[1]
    
    # Get evaluation results
    results = evaluate_method(method, phase, track_type, track_subtype)
    
    # Return the required metrics
    return {
        'overall_score': results['overall_score'],
        'watermark_detection': results['watermark_detection'],
        'quality_degradation': results['quality_degradation']
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate watermark removal')
    parser.add_argument('--original', type=str, required=True, help='Directory with original images')
    parser.add_argument('--processed', type=str, required=True, help='Directory with processed images')
    
    args = parser.parse_args()
    
    results = evaluate_dataset(args.original, args.processed)
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
