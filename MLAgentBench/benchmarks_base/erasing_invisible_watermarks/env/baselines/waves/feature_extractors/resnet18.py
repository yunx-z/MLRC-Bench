import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Extractor(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.features.to(self.device).eval()
        
    def forward(self, x):
        with torch.no_grad():
            return self.features(x) 