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
        else:
            model = model.cuda()
            
        return model

    def deep_merge(self, dict1, dict2):
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def run(self, mode):
        """Handle different running modes"""
        
        # Load appropriate configs
        if mode == "train":
            paths_cfg = load_config("configs_read_only/train_paths.yaml")
            model_cfg = load_config("configs/core_configs.yaml")
        elif mode == "valid":
            paths_cfg = load_config("configs_read_only/valid_paths.yaml")
            model_cfg = load_config("configs/core_configs.yaml")
        else:  # test mode
            paths_cfg = load_config("configs_read_only/test_paths.yaml")
            model_cfg = load_config("configs/core_configs.yaml")
            
        # Deep merge configs, with model_cfg taking precedence
        cfg = self.deep_merge(paths_cfg, model_cfg)
        
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
