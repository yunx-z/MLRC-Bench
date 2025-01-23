import torch
import torchvision.transforms as transforms
from PIL import Image

def preprocess_image(image):
    """Convert PIL image to tensor and normalize"""
    if isinstance(image, Image.Image):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        # Don't resize here, let the feature extractors handle it
        return transform(image).unsqueeze(0)  # Add batch dimension
    return image

def postprocess_image(tensor):
    """Convert tensor back to PIL image"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension
    return transforms.ToPILImage()(tensor) 