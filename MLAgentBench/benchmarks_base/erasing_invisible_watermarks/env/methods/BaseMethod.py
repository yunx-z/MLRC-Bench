import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import torchvision.transforms as transforms

class BaseMethod(object):
    def __init__(self, name):
        self.name = name
        
        # Model configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()
        self.model_type = None
        
        # Default parameters
        self.input_size = (256, 256)
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.hidden_channels = 64
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Loss functions
        self.content_loss = nn.MSELoss()
        self.perceptual_loss = None
        self.style_loss = None
        
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

    def preprocess(self, image):
        """Convert PIL image to tensor"""
        if isinstance(image, Image.Image):
            tensor = self.transform(image)
            tensor = tensor.unsqueeze(0).to(self.device)
            return tensor
        return image

    def postprocess(self, tensor):
        """Convert tensor to PIL Image"""
        if torch.is_tensor(tensor):
            tensor = tensor.cpu().detach()
            tensor = tensor.squeeze(0)
            # Denormalize
            tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                    torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            tensor = tensor.clamp(0, 1)
            return transforms.ToPILImage()(tensor)
        return tensor

    def attack(self, image):
        """
        Remove watermark from the given image
        
        Args:
            image: PIL Image
            
        Returns:
            PIL Image: Processed image with watermark removed
        """
        try:
            # Preprocess
            x = self.preprocess(image)
            
            # Model inference
            with torch.no_grad():
                output = self.model(x)
            
            # Postprocess
            result = self.postprocess(output)
            return result
            
        except Exception as e:
            print(f"Error in attack: {e}")
            return image

    def remove_watermark(self, image):
        """Alias for attack method"""
        return self.attack(image)

    def evaluate(self, original_img, processed_img):
        """Evaluate image quality metrics"""
        metrics = {}
        
        if self.metrics['psnr']:
            metrics['psnr'] = self._compute_psnr(original_img, processed_img)
            
        if self.metrics['ssim']:
            metrics['ssim'] = self._compute_ssim(original_img, processed_img)
            
        if self.metrics['perceptual']:
            metrics.update(self.evaluate_advanced_metrics(original_img, processed_img))
            
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

    def evaluate_advanced_metrics(self, original_img, processed_img):
        """Compute additional evaluation metrics"""
        metrics = {}
        
        # Implement advanced metrics in child classes
        # Example metrics:
        # - Perceptual loss
        # - Style loss
        # - Content loss
        # - Detection confidence
        
        return metrics

    def train(self, train_data, val_data=None, num_epochs=100):
        """Train the watermark removal model"""
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate
        )
        
        for epoch in range(num_epochs):
            for batch in train_data:
                # Forward pass
                # Loss calculation
                # Backward pass
                # Optimization step
                pass
            
            if val_data is not None:
                # Validation
                pass

    def save_model(self, save_path):
        """Save model checkpoint"""
        if self.model is not None:
            torch.save(self.model.state_dict(), save_path)

    def load_model(self, load_path):
        """Load model checkpoint"""
        if os.path.exists(load_path):
            self._load_model(load_path)