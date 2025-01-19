from methods.BaseMethod import BaseMethod
import torch.nn as nn
from libs.core import load_config
from libs.modeling import make_meta_arch

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)
        
    def get_model(self, cfg):
        """Initialize ActionFormer model"""
        # Create model using meta architecture
        model = make_meta_arch(cfg['model_name'], **cfg['model'])
        
        # Wrap with DataParallel if multiple GPUs
        if len(cfg['devices']) > 1:
            model = nn.DataParallel(model, device_ids=cfg['devices'])
            
        return model

    def run(self, mode):
        """Handle different running modes:
        train: training in dev phase
        valid: validation in dev phase
        test: evaluation in test phase
        """
        
        # Load appropriate config based on mode
        if mode == "train":
            # Get training config
            cfg = load_config("configs/perception_tal_multi_train.yaml")
        elif mode == "valid":
            # Get validation config
            cfg = load_config("configs/perception_tal_multi_valid.yaml")   
        else:  
            # Get test config
            cfg = load_config("configs/perception_tal_multi_test.yaml")
            
        # Set default devices if not specified  
        if 'devices' not in cfg:
            cfg['devices'] = ['cuda:0']
        
        # Initialize model
        model = self.get_model(cfg)
        
        # Load checkpoint for validation/test
        if mode in ["valid", "test"]:
            checkpoint_path = self.get_checkpoint_path()
            model = self.load_checkpoint(model, checkpoint_path)
            
        return model, cfg
