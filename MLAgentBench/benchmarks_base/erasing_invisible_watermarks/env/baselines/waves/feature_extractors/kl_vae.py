import torch
import torch.nn as nn
import clip

class KLVAEExtractor(nn.Module):
    """
    Using CLIP as feature extractor instead of VAE since we don't have access to the original VAE
    """
    def __init__(self, device=None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Use CLIP instead
        self.model, _ = clip.load('ViT-B/32', device=self.device)
        self.model.eval()
        
    def forward(self, x):
        with torch.no_grad():
            features = self.model.encode_image(x)
            # Normalize features
            features = features / features.norm(dim=-1, keepdim=True)
            return features 