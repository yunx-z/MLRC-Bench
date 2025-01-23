import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from .metrics.quality_metrics import QualityMetrics
from .feature_extractors import CLIPExtractor, ResNet18Extractor, KLVAEExtractor
from .utils.image_utils import preprocess_image, postprocess_image
from .distortions import apply_distortions
from .utils.logger import setup_logger

logger = setup_logger(__name__)

class WatermarkRemovalManager:
    """
    WAVES baseline watermark removal using multiple attack strategies
    """
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = QualityMetrics(device=self.device)
        
        # Load feature extractors
        self.clip = CLIPExtractor(device=self.device)
        self.resnet = ResNet18Extractor(device=self.device)
        self.vae = KLVAEExtractor(device=self.device)
        
        # Attack parameters from WAVES
        self.num_steps = 200
        self.alpha_ratio = 0.05
        self.epsilon = 16/255
        
        logger.info("Starting watermark removal...")
        
    def remove_watermark(self, image):
        """Combined watermark removal strategy from WAVES"""
        img_tensor = preprocess_image(image).to(self.device)
        
        # 1. Apply adversarial attack
        adv_img = self._adversarial_attack(img_tensor)
        
        # 2. Apply distortions
        dist_img = apply_distortions(adv_img)
        
        # 3. Optional: Apply regeneration if needed
        # final_img = self._regenerate_image(dist_img)
        
        return postprocess_image(dist_img)
    
    def _adversarial_attack(self, img_tensor):
        """Untargeted adversarial attack using multiple feature extractors"""
        # Initialize x with requires_grad
        x = img_tensor.clone().detach().requires_grad_(True)
        img_tensor = img_tensor.detach()  # Make sure original image doesn't require grad
        
        # Move tensors to device
        x = x.to(self.device)
        img_tensor = img_tensor.to(self.device)
        
        for step in range(self.num_steps):
            # Zero any existing gradients
            if x.grad is not None:
                x.grad.zero_()
            
            # Ensure input is properly shaped for all extractors
            if x.shape[-2:] != (224, 224):
                x_resized = torch.nn.functional.interpolate(x, size=(224, 224), mode='bicubic', align_corners=True)
            else:
                x_resized = x
            
            # Compute total loss instead of individual gradients
            total_loss = 0
            
            # CLIP features
            clip_feat = self.clip(x_resized)
            total_loss = total_loss + F.mse_loss(clip_feat, torch.zeros_like(clip_feat).to(self.device))
            
            # ResNet features
            resnet_feat = self.resnet(x_resized)
            total_loss = total_loss + F.mse_loss(resnet_feat, torch.zeros_like(resnet_feat).to(self.device))
            
            # VAE features
            vae_feat = self.vae(x_resized)
            total_loss = total_loss + F.mse_loss(vae_feat, torch.zeros_like(vae_feat).to(self.device))
            
            # Compute gradient of total loss
            total_loss.backward()
            grad_total = x.grad.clone()
            
            # Update with normalized gradient
            with torch.no_grad():
                grad_norm = torch.norm(grad_total.view(grad_total.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                grad_total = grad_total / (grad_norm + 1e-8)
                
                step_size = self.alpha_ratio * self.epsilon
                x.data = x.data - step_size * grad_total.sign()
                
                # Project back to epsilon ball
                delta = x.data - img_tensor
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                x.data = torch.clamp(img_tensor + delta, 0, 1)
        
        return x.detach() 