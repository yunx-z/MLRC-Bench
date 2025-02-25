import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
from skimage.metrics import structural_similarity, normalized_mutual_information
from .BaseMethod import BaseMethod

class WavesModel(nn.Module):
    """Watermark removal using frequency domain filtering"""
    
    def __init__(self, strength=1):
        super().__init__()
        self.strength = strength
        # Parameters for frequency-based removal
        self.kernel_size = 3
        self.sigma = 1.0
        self.threshold = 0.1 * strength  # Adjust threshold based on strength
        
    def gaussian_kernel(self, size, sigma):
        """Create a 2D Gaussian kernel"""
        x = torch.linspace(-size//2, size//2, size)
        x = x.view(1, -1).repeat(size, 1)
        y = x.t()
        kernel = torch.exp(-(x**2 + y**2)/(2*sigma**2))
        return kernel / kernel.sum()

    def forward(self, x):
        """Forward pass implementing frequency domain filtering"""
        B, C, H, W = x.shape
        device = x.device
        
        # Process each color channel separately
        x_filtered = torch.zeros_like(x)
        for c in range(C):
            x_freq = torch.fft.fft2(x[:,c:c+1])
            x_freq = torch.fft.fftshift(x_freq)
            
            # Create frequency mask
            Y, X = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W))
            R = torch.sqrt(X**2 + Y**2).to(device)
            mask = (R > self.threshold).float()
            mask = mask.view(1, 1, H, W)
            
            # Apply mask and convert back
            x_freq = x_freq * mask
            x_freq = torch.fft.ifftshift(x_freq)
            x_filtered[:,c:c+1] = torch.fft.ifft2(x_freq).real
        
        # Apply Gaussian smoothing
        kernel = self.gaussian_kernel(self.kernel_size, self.sigma).to(device)
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size)
        kernel = kernel.repeat(C, 1, 1, 1)
        
        x_smooth = F.conv2d(
            F.pad(x, (self.kernel_size//2,)*4, mode='reflect'),
            kernel,
            groups=C
        )
        
        # Blend filtered results
        alpha = 0.5
        x_result = alpha * x_filtered + (1 - alpha) * x_smooth
        
        # Color correction
        for c in range(C):
            mean_orig = x[:,c:c+1].mean()
            mean_result = x_result[:,c:c+1].mean()
            std_orig = x[:,c:c+1].std()
            std_result = x_result[:,c:c+1].std()
            
            x_result[:,c:c+1] = (x_result[:,c:c+1] - mean_result) * (std_orig / std_result) + mean_orig
        
        return torch.clamp(x_result, 0, 1)

class MyMethod(BaseMethod):
    """Implementation of watermark removal using frequency domain filtering"""
    
    def __init__(self, name, strength=1.0, **kwargs):
        # Initialize strength before calling super().__init__
        self.strength = strength
        super().__init__(name, **kwargs)
        
        # Override default parameters if needed
        self.input_size = (512, 512)  # Keep original size
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])  # No normalization needed for frequency domain

    def _build_model(self):
        """Initialize the watermark removal model"""
        model = WavesModel(strength=self.strength)
        return model.to(self.device)

    def remove_watermark(self, image, track_type=None):
        """Remove watermark from the given image"""
        # Preprocess
        x = self.preprocess(image)
        
        # Model inference
        with torch.no_grad():
            output = self.model(x)
        
        # Postprocess
        result = self.postprocess(output)
        return result

    def evaluate_advanced_metrics(self, original_img, processed_img):
        """Compute additional evaluation metrics"""
        # Convert images to numpy arrays
        original_np = np.array(original_img) / 255.0
        processed_np = np.array(processed_img) / 255.0
        
        # Compute normalized mutual information
        nmi = normalized_mutual_information(
            (original_np * 255).astype(np.uint8),
            (processed_np * 255).astype(np.uint8)
        )
        
        # Simple quality score (Q) based on basic metrics
        ssim = structural_similarity(original_np, processed_np, channel_axis=2, data_range=1.0)
        mse = np.mean((original_np - processed_np) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        Q = 0.5 * (1 - ssim) + 0.3 * (1 - psnr/50) + 0.2 * (1 - nmi)
        Q = np.clip(Q, 0.1, 0.9)  # Ensure Q is in [0.1, 0.9] range
        
        # Simple watermark detection score (A) based on high frequency differences
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
            'nmi': float(nmi)
        }
            
