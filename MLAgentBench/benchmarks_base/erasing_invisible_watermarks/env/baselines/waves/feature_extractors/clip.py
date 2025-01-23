import torch
import clip
import torchvision.transforms as transforms

class CLIPExtractor(torch.nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.preprocess = clip.load('ViT-B/32', device=self.device)
        self.model.eval()
        
        # CLIP specific preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                               (0.26862954, 0.26130258, 0.27577711))
        ])
        
    def forward(self, x):
        # Don't use no_grad here since we need gradients
        if x.shape[-2:] != (224, 224):
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bicubic', align_corners=True)
        x = self.transform(x)
        features = self.model.encode_image(x)
        features = features / features.norm(dim=-1, keepdim=True)
        return features 