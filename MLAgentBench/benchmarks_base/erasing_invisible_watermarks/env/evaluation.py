import torch
import numpy as np
from constants import QUALITY_COEFFICIENTS
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import lpips

class WatermarkEvaluator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure().to(self.device)
        self.fid = FrechetInceptionDistance(feature=2048).to(self.device)
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        
    def compute_quality_score(self, original_img, processed_img):
        """
        Compute normalized quality score Q using competition metrics
        """
        # Handle both single images and batches
        if isinstance(original_img, list):
            # Convert list of images to tensor batch
            original_batch = torch.stack([
                torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0 
                for img in original_img
            ])
            processed_batch = torch.stack([
                torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0 
                for img in processed_img
            ])
            
            original_img = original_batch
            processed_img = processed_batch
        elif not isinstance(original_img, torch.Tensor):
            # Handle single image
            original_img = torch.from_numpy(np.array(original_img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            processed_img = torch.from_numpy(np.array(processed_img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        original_img = original_img.to(self.device)
        processed_img = processed_img.to(self.device)
        
        # Compute metrics for each image in batch
        batch_size = original_img.size(0)
        batch_metrics = []
        
        for i in range(batch_size):
            orig = original_img[i:i+1]
            proc = processed_img[i:i+1]
            
            metrics = {
                'PSNR': self.psnr(proc, orig).item(),
                'SSIM': self.ssim(proc, orig).item(),
                'LPIPS': self.lpips_fn(proc, orig).item(),
            }
            
            # Normalize and combine metrics for this image
            Q = sum(coef * metrics[name] 
                    for name, coef in QUALITY_COEFFICIENTS.items()
                    if name in metrics)
            
            # Clamp to [0.1, 0.9] range as per competition rules
            batch_metrics.append(np.clip(Q, 0.1, 0.9))
        
        # Return average score for the batch
        return np.mean(batch_metrics)
    
    def _compute_fid(self, original_img, processed_img):
        """Helper method to compute FID score"""
        if original_img.size(0) < 2 or processed_img.size(0) < 2:
            return 0.0  # Skip FID for single images
        self.fid.update(original_img, real=True)
        self.fid.update(processed_img, real=False)
        score = self.fid.compute()
        self.fid.reset()
        return score
    
    def compute_accuracy_score(self, watermark_detector, processed_images):
        """
        Compute TPR@0.1%FPR (A metric)
        """
        # Placeholder until watermark detector is implemented
        return 0.5  # Return dummy score for now
    
    def compute_final_score(self, Q, A):
        """
        Compute final score sqrt(Q^2 + A^2)
        """
        return np.sqrt(Q**2 + A**2) 