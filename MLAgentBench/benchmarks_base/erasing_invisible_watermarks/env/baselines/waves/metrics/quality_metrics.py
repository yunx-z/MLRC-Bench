import torch
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import lpips

class QualityMetrics:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure().to(self.device)
        self.fid = FrechetInceptionDistance(feature=2048).to(self.device)
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        
        # Initialize watermark detector thresholds
        self.fpr_threshold = 0.001  # 0.1% FPR
    
    def compute_metrics(self, original, processed):
        """
        Compute all quality metrics for a pair of images
        """
        metrics = {
            'PSNR': self.psnr(processed, original).item(),
            'SSIM': self.ssim(processed, original).item(),
            'LPIPS': self.lpips_fn(processed, original).item(),
        }
        
        if original.size(0) > 1:  # For batches
            metrics['FID'] = self._compute_fid(original, processed)
            
        return metrics
    
    def _compute_fid(self, real_images, fake_images):
        self.fid.update(real_images, real=True)
        self.fid.update(fake_images, real=False)
        score = self.fid.compute()
        self.fid.reset()
        return score

    def compute_accuracy_score(self, watermark_detector, processed_images, original_messages=None):
        """
        Compute TPR@0.1%FPR (A metric)
        Args:
            watermark_detector: Watermark detection model
            processed_images: Batch of processed images
            original_messages: Original watermark messages (if available)
        Returns:
            TPR at 0.1% FPR
        """
        if watermark_detector is None:
            return 0.5  # Default score if no detector available
            
        # Get decoded messages from processed images
        decoded_messages = []
        for img in processed_images:
            decoded = watermark_detector.decode(img)
            decoded_messages.append(decoded)
            
        # Calculate distances between decoded and original messages
        distances = []
        for dec, orig in zip(decoded_messages, original_messages):
            dist = self._compute_message_distance(dec, orig)
            distances.append(dist)
            
        # Determine threshold for 0.1% FPR using clean images
        clean_distances = self._get_clean_image_distances(watermark_detector)
        threshold = np.percentile(clean_distances, 99.9)  # 0.1% FPR threshold
        
        # Calculate TPR
        tpr = np.mean([d <= threshold for d in distances])
        
        return tpr
    
    def _compute_message_distance(self, msg1, msg2):
        """
        Compute distance between two watermark messages
        """
        if isinstance(msg1, torch.Tensor):
            msg1 = msg1.cpu().numpy()
        if isinstance(msg2, torch.Tensor):
            msg2 = msg2.cpu().numpy()
            
        return np.linalg.norm(msg1 - msg2)
    
    def _get_clean_image_distances(self, watermark_detector, n_samples=10000):
        """
        Get distribution of distances for clean (unwatermarked) images
        """
        # This should be implemented based on the competition's specific watermark detector
        # For now, return dummy distances
        return np.random.rand(n_samples) 