import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

class BaseMethod(object):
    def __init__(self, name):
        self.name = name
        
        # Model configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = "base"
        
        # Default parameters
        self.input_size = (512, 512)
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.hidden_channels = 64
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Build and move model to device
        self.model = self._build_model()
        self.model = self.model.to(self.device)

    def get_name(self):
        return self.name

    def _build_model(self):
        """Build and return a basic model that just returns the input image.
        This is a placeholder - actual watermark removal methods should override this.
        """
        class PassthroughModel(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                # Simply return the input image
                return x
        
        return PassthroughModel()

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
            tensor = tensor.clamp(0, 1)
            return transforms.ToPILImage()(tensor)
        return tensor

    def remove_watermark(self, image, track_type=None):
        """Remove watermark from the given image
        
        This is the main method that should be called to process an image.
        The default implementation handles preprocessing and postprocessing.
        Subclasses should implement _build_model() instead of overriding this.
        
        Args:
            image: PIL Image to process
            track_type: Optional string indicating watermark type ('stegastamp' or 'treering')
            
        Returns:
            PIL Image with watermark removed
        """
        x = self.preprocess(image)
        with torch.no_grad():
            output = self.model(x)
        result = self.postprocess(output)
        return result

    def save_model(self, save_path):
        """Save model checkpoint"""
        if self.model is not None:
            torch.save(self.model.state_dict(), save_path)

    def load_model(self, load_path):
        """Load model checkpoint"""
        if os.path.exists(load_path):
            state_dict = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()