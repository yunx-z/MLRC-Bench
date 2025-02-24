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
    
    def __init__(self, strength=1.0, device="cpu"):
        super().__init__()
        self.device = device
        self.strength = strength
        # Parameters for frequency-based removal
        self.kernel_size = 3
        self.sigma = 1.0
        self.threshold = 0.1 * strength  # Adjust threshold based on strength
        
    def to(self, device):
        """Move the model to specified device"""
        self.device = device
        return super().to(device)

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
    
    def __init__(self, name):
        super().__init__(name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set parameters
        self.input_size = (512, 512)  # Keep original size
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor()
        ])  # No normalization needed for frequency domain
        
        # Initialize with different strengths for different watermark types
        self.strength_map = {
            'stegastamp': 1.0,  # Match baseline strength
            'treering': 1.0     # Match baseline strength
        }

    def _build_model(self):
        """Initialize the watermark removal model"""
        model = WavesModel(strength=1.0, device=self.device)
        return model.to(self.device)

    def remove_watermark(self, image, track_type='stegastamp'):
        """Remove watermark with strength adapted to track type"""
        try:
            # Convert PIL image to tensor
            if isinstance(image, Image.Image):
                image = image.convert('RGB')
                image = transforms.ToTensor()(image).unsqueeze(0)
            
            # Move to device
            image = image.to(self.device)
            
            # Adjust model strength
            strength = self.strength_map.get(track_type, 1.0)
            self.model.strength = strength
            self.model.threshold = 0.1 * strength
            
            # Process image
            with torch.no_grad():
                x_result = self.model(image)
            
            # Convert back to PIL image
            result_image = transforms.ToPILImage()(x_result.squeeze().cpu())
            return result_image
            
        except Exception as e:
            print(f"Error in remove_watermark: {e}")
            if isinstance(image, torch.Tensor):
                return transforms.ToPILImage()(image.squeeze().cpu())
            return image

    def evaluate(self, original_tensor, processed_tensor):
        """
        Evaluate the quality between original and processed images
        
        Args:
            original_tensor (torch.Tensor): Original image tensor
            processed_tensor (torch.Tensor): Processed image tensor
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        try:
            # Move tensors to CPU and convert to numpy arrays
            original_np = original_tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
            processed_np = processed_tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
            
            # Basic metrics
            mse = np.mean((original_np - processed_np) ** 2)
            psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
            ssim = structural_similarity(original_np, processed_np, channel_axis=2, data_range=1.0)
            nmi = normalized_mutual_information(
                (original_np * 255).astype(np.uint8),
                (processed_np * 255).astype(np.uint8)
            )
            
            # Simple quality score (Q) based on basic metrics
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