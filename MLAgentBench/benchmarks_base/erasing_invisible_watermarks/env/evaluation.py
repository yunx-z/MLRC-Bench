import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity, normalized_mutual_information
import pickle
import sys
import subprocess
from dev import get_performance_from_jsons, get_quality_from_jsons
import math

QUALITY_METRICS = {
    "legacy_fid": "Legacy FID",
    "clip_fid": "CLIP FID",
    "psnr": "PSNR",
    "ssim": "SSIM",
    "nmi": "Normed Mutual-Info",
    "lpips": "LPIPS",
    "aesthetics": "Delta Aesthetics",
    "artifacts": "Delta Artifacts",
    # "clip_score": "Delta CLIP-Score",
    # "watson": "Watson-DFT",
}

def evaluate_method(method_class, method_name, phase, watermarked_dir, output_dir):
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
    method = method_class(method_name)
    
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

def get_scores(watermarked_dir, output_dir):
    """
    Calculate evaluation scores for processed images using warm-up-kit
    
    Args:
        watermarked_dir: Directory containing original watermarked images
        output_dir: Directory containing processed images
    """
    try:
        # Set environment variables
        os.environ["RESULT_DIR"] = output_dir
        unwatermarked_dir = watermarked_dir.replace("watermarked", "unwatermarked")
        
        # Run the evaluation command
        conda_env_path = os.path.dirname(os.path.dirname(os.path.dirname(sys.executable))) # "~/.conda/envs"
        erasinginvisible_path = os.path.join(conda_env_path, "erasing_invisible_watermarks/bin/erasinginvisible")

        command = [
            erasinginvisible_path,
            "eval",
            "--path",
            output_dir,
            "--w_path",
            watermarked_dir,
            "--uw_path",
            unwatermarked_dir,
        ]

        try:
            result = subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"erasinginvisible execution failed, error message: {e}")
        except FileNotFoundError:
            print(f"Error: Cannot find executable {erasinginvisible_path}\n Please double check the path")
                
        # Use warm-up-kit's own functions to extract metrics
        performance_dict = get_performance_from_jsons(
            os.path.join(output_dir, "org-decode.json"),
            os.path.join(output_dir, "decode.json"),
            "stable_sig"
        )
        
        quality_dict = get_quality_from_jsons(
            os.path.join(output_dir, "org-metric.json"),
            os.path.join(output_dir, "metric.json")
        )
        
        # Get the A metric (TPR@0.1%FPR)
        # Higher watermark_detection score means better watermark removal
        watermark_detection = performance_dict.get('low1000_1', 0.5)  # TPR@0.1%FPR
        
        # Extract all required metrics from quality_dict
        fid_score = quality_dict.get('legacy_fid', [0, 0])[0] if 'legacy_fid' in quality_dict else 0
        clip_fid_score = quality_dict.get('clip_fid', [0, 0])[0] if 'clip_fid' in quality_dict else 0
        psnr_score = quality_dict.get('psnr', [0, 0])[0] if 'psnr' in quality_dict else 0
        ssim_score = quality_dict.get('ssim', [0, 0])[0] if 'ssim' in quality_dict else 0
        nmi_score = quality_dict.get('nmi', [0, 0])[0] if 'nmi' in quality_dict else 0
        lpips_score = quality_dict.get('lpips', [0, 0])[0] if 'lpips' in quality_dict else 0
        delta_aesthetics = quality_dict.get('aesthetics', [0, 0])[0] if 'aesthetics' in quality_dict else 0
        delta_artifacts = quality_dict.get('artifacts', [0, 0])[0] if 'artifacts' in quality_dict else 0
        
        # Define 10th and 90th percentile values for normalization
        # extracted from Figure 14 of WAVES paper (https://arxiv.org/pdf/2401.08573)
        percentiles = {
            'legacy_fid': {'p10': 1.20, 'p90': 53.40},
            'clip_fid': {'p10': 0.28, 'p90': 19.02},
            'psnr': {'p10': -41.97, 'p90': -12.15},
            'ssim': {'p10': -0.98, 'p90': -0.32},
            'nmi': {'p10': -1.71, 'p90': -1.05},
            'lpips': {'p10': 0.02, 'p90': 0.62},
            'aesthetics': {'p10': 0.02, 'p90': 1.82},
            'artifacts': {'p10': -0.00, 'p90': 0.58}
        }
        
        # Simplified normalization function - just maps to [0.1, 0.9] range
        def normalize(value, p10, p90):
            # Clamp the value to the range [p10, p90]
            clamped_value = min(max(value, p10), p90)
            # Normalize to [0.1, 0.9] range
            normalized = 0.1 + 0.8 * (clamped_value - p10) / (p90 - p10)
            return normalized
        
        # Normalize each metric and explicitly invert where needed
        # For metrics where higher is better (PSNR, SSIM, NMI, Delta Artifacts), invert the normalization
        # For metrics where higher is worse (FID, CLIP FID, LPIPS, Delta Aesthetics), use normalization directly
        norm_fid = normalize(fid_score, percentiles['legacy_fid']['p10'], percentiles['legacy_fid']['p90'])
        norm_clip_fid = normalize(clip_fid_score, percentiles['clip_fid']['p10'], percentiles['clip_fid']['p90'])
        norm_psnr = 1.0 - normalize(psnr_score, percentiles['psnr']['p10'], percentiles['psnr']['p90'])
        norm_ssim = 1.0 - normalize(ssim_score, percentiles['ssim']['p10'], percentiles['ssim']['p90'])
        norm_nmi = 1.0 - normalize(nmi_score, percentiles['nmi']['p10'], percentiles['nmi']['p90'])
        norm_lpips = normalize(lpips_score, percentiles['lpips']['p10'], percentiles['lpips']['p90'])
        norm_aesthetics = normalize(delta_aesthetics, percentiles['aesthetics']['p10'], percentiles['aesthetics']['p90'])
        norm_artifacts = 1.0 - normalize(delta_artifacts, percentiles['artifacts']['p10'], percentiles['artifacts']['p90'])
        
        # Calculate Q using the coefficients provided
        # Q = +1.53×10^-3 FID + 5.07×10^-3 CLIP FID - 2.22×10^-3 PSNR - 1.13×10^-1 SSIM
        #    - 9.88×10^-2 NMI + 3.41×10^-1 LPIPS + 4.50×10^-2 ΔAesthetics - 1.44×10^-1 ΔArtifacts
        Q = (
            1.53e-3 * norm_fid +
            5.07e-3 * norm_clip_fid -
            2.22e-3 * norm_psnr -
            1.13e-1 * norm_ssim -
            9.88e-2 * norm_nmi +
            3.41e-1 * norm_lpips +
            4.50e-2 * norm_aesthetics -
            1.44e-1 * norm_artifacts
        )
        
        # Quality degradation is directly mapped to Q
        # Lower Q is better quality (less degradation)
        quality_degradation = Q
        
        # Overall score calculation (higher is better) using the new A and Q metrics
        overall_score = math.sqrt(quality_degradation**2 + watermark_detection**2)

        print("#" * 20)
        print()
        print("# Evaluation results:")
        print("### Watermark Performance:")
        print(f"Accuracy: {performance_dict['acc_1']*100:.2f}%")
        print(f"AUC Score: {performance_dict['auc_1']*100:.2f}%")
        print(f"TPR@0.1%FPR: {performance_dict['low1000_1']*100:.2f}%")
        print(f"TPR@1%FPR Score: {performance_dict['low100_1']*100:.2f}%")
        print()
        print("### Image Quality:")
        for key, value in quality_dict.items():
            if value is None:
                continue
            print(f"{QUALITY_METRICS[key]}: {value[0]:e} +/- {value[1]:e}")
        print()
        print("Warmup kit evaluation completed.")
        print("#" * 20)
        
        return {
            'overall_score': float(overall_score),
            'watermark_detection': float(watermark_detection),
            'quality_degradation': float(quality_degradation),
        }
        
    except Exception as e:
        print(f"Error running warm-up-kit evaluation: {e}")
