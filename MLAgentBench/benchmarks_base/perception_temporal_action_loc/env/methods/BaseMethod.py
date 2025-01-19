import torch

class BaseMethod:
    def __init__(self, name):
        self.name = name
        
    def get_name(self):
        return self.name

    def get_checkpoint_path(self):
        """Get path to saved checkpoint"""
        return f"./ckpt/{self.name}/model_best.pth.tar"

    def load_checkpoint(self, model, checkpoint_path):
        """Load model weights from checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=lambda storage, loc: storage.cuda()
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def run(self, mode):
        """Main entry point for both training and evaluation
        
        Args:
            mode: str
                "train": training model in dev phase
                "valid": validation in dev phase  
                "test": evaluation in test phase
        Returns:
            (model, config) tuple
        """
        raise NotImplementedError