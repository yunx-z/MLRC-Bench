import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from .connect import Corr_Up

class SiamFC(nn.Module):
    def __init__(self, config,  **kwargs):
        super(SiamFC, self).__init__()
        self.features = None
        self.connect_model = Corr_Up()
        self.zf = None  # for online tracking
        if kwargs['base'] is None:
            self.features = ResNet22W()
        else:
            self.features = kwargs['base']
        self.config = config
        self.model_alphaf = 0
        self.zf = None
        self.features.eval()

    def feature_extractor(self, x):
        return self.features(x)

    def forward(self, x):
        xf = self.feature_extractor(x) * self.config.cos_window
        zf = self.zf
        response = self.connect_model(zf, xf)
        return response

    def update(self, z, lr=0):
        zf_raw = self.feature_extractor(z).detach()
        # Using h_feat, w_feat explicitly for clarity, though ResNet22W should be square
        _, _, h_feat, w_feat = zf_raw.shape

        # Original SiamFC often assumes square template features for this cropping logic
        # For safety, one might use h_feat and w_feat to calculate bg_h, ed_h, bg_w, ed_w separately
        # if non-square features were possible. Here, we'll assume ts = h_feat for simplicity,
        # as ResNet22W produces square features.
        ts = h_feat # Or min(h_feat, w_feat) if they could differ significantly

        padding_val = self.config.padding
        denominator = 2 * (padding_val + 1)

        # Handle potential division by zero if denominator is zero, though unlikely with padding >= 0
        if denominator == 0:
            # This case should ideally not happen with valid padding
            # Default to a central small crop or raise error
            bg = ts // 2 - 1
            ed = ts // 2 + 1
            if ts < 2 : bg = 0; ed = ts # Ensure crop is within bounds if ts is tiny
        else:
            bg = ts // 2 - int(ts // denominator)
            ed = ts // 2 + int(ts // denominator)

        # Ensure bg and ed are valid indices
        bg = max(0, bg)
        ed = min(ts, ed)
        if bg >= ed : # If crop size is zero or negative, try a minimal 1x1 or 2x2 crop if possible
            if ts >= 1:
                 bg = ts // 2
                 ed = bg + 1 # Try to get 1x1
                 if ts >=2 and ts//2 -1 >=0 : # try 2x2 if possible
                     bg = ts//2 -1
                     ed = ts//2 +1
            else: # ts is 0
                bg = 0
                ed = 0


        zf_cropped = zf_raw[:, :, bg:ed, bg:ed] # Apply to both height and width assuming square crop intention

        if self.zf is None:
            self.zf = zf_cropped
        else:
            self.zf = (1 - lr) * self.zf + lr * zf_cropped



