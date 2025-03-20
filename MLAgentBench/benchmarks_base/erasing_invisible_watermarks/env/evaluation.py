import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity, normalized_mutual_information
import pickle


def evaluate_method(method, phase, watermarked_dir, output_dir):
    """
    Process images with the watermark removal method and save results
    
    Args:
        method: The method to evaluate
        phase: 'dev' or 'test'
        watermarked_dir: Directory containing watermarked images
        output_dir: Directory to save processed images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image in watermarked directory
    for img_file in os.listdir(watermarked_dir):
        if not img_file.endswith('.png'):
            continue
            
        try:
            # Load watermarked image
            img_path = os.path.join(watermarked_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            orig_size = img.size
            
            # Process image with method
            processed_img = method.remove_watermark(img)
            
            # Ensure processed image has same size as original
            if processed_img.size != orig_size:
                processed_img = processed_img.resize(orig_size, Image.Resampling.LANCZOS)
            
            # Save processed image with same filename
            output_path = os.path.join(output_dir, img_file)
            processed_img.save(output_path)
                
        except Exception as e:
            print(f"Error processing image {img_file}: {e}")
            continue

def get_scores(method, phase, watermarked_dir, output_dir):
    """
    Calculate evaluation scores for processed images
    
    Args:
        method: The method to evaluate
        phase: 'dev' or 'test'
        watermarked_dir: Directory containing original watermarked images
        output_dir: Directory containing processed images
    """
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
    
    # Process each image pair
    for img_file in os.listdir(watermarked_dir):
        if not img_file.endswith('.png'):
            continue
            
        try:
            # Load original watermarked image
            wm_path = os.path.join(watermarked_dir, img_file)
            wm_img = Image.open(wm_path).convert('RGB')
            wm_tensor = transforms.ToTensor()(wm_img)
            wm_np = wm_tensor.permute(1, 2, 0).numpy()
            
            # Load processed image
            proc_path = os.path.join(output_dir, img_file)
            if not os.path.exists(proc_path):
                continue
                
            proc_img = Image.open(proc_path).convert('RGB')
            proc_tensor = transforms.ToTensor()(proc_img)
            proc_np = proc_tensor.permute(1, 2, 0).numpy()
            
            # Calculate basic metrics
            mse = np.mean((wm_np - proc_np) ** 2)
            psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
            ssim = structural_similarity(wm_np, proc_np, channel_axis=2, data_range=1.0)
            nmi = normalized_mutual_information(
                (wm_np * 255).astype(np.uint8),
                (proc_np * 255).astype(np.uint8)
            )
            
            # Quality score (Q)
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
            
            orig_hf = high_freq_diff(wm_np)
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
            metrics_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_metrics.pkl")
            with open(metrics_path, 'wb') as f:
                pickle.dump(metrics, f)
            
            # Accumulate metrics
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            valid_count += 1
                
        except Exception as e:
            print(f"Error calculating metrics for {img_file}: {e}")
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
