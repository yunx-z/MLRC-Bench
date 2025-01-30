import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

class BaseMethod(object):
    def __init__(self, name):
        self.name = name
        
        # Model configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_type = None
        
        # Default parameters
        self.input_size = (256, 256)
        self.batch_size = 32
        self.learning_rate = 1e-4
        
        # Evaluation parameters
        self.metrics = {
            'psnr': True,
            'ssim': True,
            'perceptual': False,
            'detection': False
        }

    def get_name(self):
        return self.name

    def _build_model(self):
        """Build and return the model architecture"""
        raise NotImplementedError

    def _load_model(self, checkpoint_path=None):
        """Load model from checkpoint if provided"""
        if checkpoint_path and os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        self.model.eval()

    def remove_watermark(self, image):
        """Remove watermark from the given image"""
        raise NotImplementedError

    def evaluate(self, original_img, processed_img):
        """Evaluate image quality metrics"""
        metrics = {}
        
        if self.metrics['psnr']:
            metrics['psnr'] = self._compute_psnr(original_img, processed_img)
            
        if self.metrics['ssim']:
            metrics['ssim'] = self._compute_ssim(original_img, processed_img)
            
        return metrics

    def _compute_psnr(self, original_img, processed_img):
        """Compute PSNR metric"""
        original_np = np.array(original_img) / 255.0
        processed_np = np.array(processed_img) / 255.0
        mse = np.mean((original_np - processed_np) ** 2)
        return 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')

    def _compute_ssim(self, original_img, processed_img):
        """Compute SSIM metric"""
        original_np = np.array(original_img) / 255.0
        processed_np = np.array(processed_img) / 255.0
        try:
            return structural_similarity(
                original_np, 
                processed_np, 
                channel_axis=-1,
                data_range=1.0
            )
        except Exception as e:
            print(f"Error calculating SSIM: {e}")
            return 0.0

    def save_model(self, save_path):
        """Save model checkpoint"""
        if self.model is not None:
            torch.save(self.model.state_dict(), save_path)

    def load_model(self, load_path):
        """Load model checkpoint"""
        if os.path.exists(load_path):
            self._load_model(load_path)