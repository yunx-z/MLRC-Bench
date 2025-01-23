import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as TF

def apply_distortions(img_tensor, strength=0.15):
    """Apply combined distortions from WAVES"""
    # Convert to PIL for some operations
    img_pil = TF.to_pil_image(img_tensor.squeeze(0))
    
    # Apply geometric distortions
    img_pil = apply_geometric(img_pil, strength)
    
    # Apply degradation distortions
    img_pil = apply_degradation(img_pil, strength)
    
    # Convert back to tensor
    img_tensor = TF.to_tensor(img_pil).unsqueeze(0)
    
    # Apply photometric distortions
    img_tensor = apply_photometric(img_tensor, strength)
    
    return img_tensor

def apply_geometric(img, strength):
    """Apply geometric distortions"""
    # Resize
    size = img.size
    scale = 1.0 - strength/2
    new_size = (int(size[0]*scale), int(size[1]*scale))
    img = img.resize(new_size, Image.BILINEAR)
    img = img.resize(size, Image.BILINEAR)
    
    return img

def apply_degradation(img, strength):
    """Apply degradation distortions"""
    # JPEG compression
    quality = int(100 * (1 - strength))
    img.save("temp.jpg", quality=quality)
    img = Image.open("temp.jpg")
    
    # Gaussian blur
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1 - strength)
    
    return img

def apply_photometric(img_tensor, strength):
    """Apply photometric distortions"""
    # Add noise
    noise = torch.randn_like(img_tensor) * strength
    img_tensor = torch.clamp(img_tensor + noise, 0, 1)
    
    # Adjust brightness/contrast
    brightness = 1.0 + (torch.rand(1).item() - 0.5) * strength
    contrast = 1.0 + (torch.rand(1).item() - 0.5) * strength
    img_tensor = TF.adjust_brightness(img_tensor, brightness)
    img_tensor = TF.adjust_contrast(img_tensor, contrast)
    
    return img_tensor 