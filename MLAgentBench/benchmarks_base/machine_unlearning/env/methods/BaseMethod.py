from copy import deepcopy
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class BaseMethod(object):
    def __init__(self, name):
        self.name = name
        
    def get_name(self):
        return self.name
        
    def run(self, original_model, retain_loader, forget_loader, val_loader, phase="dev"):
        """Base method for machine unlearning
        
        Args:
            original_model: The model to be unlearned
            retain_loader: DataLoader for retained training data
            forget_loader: DataLoader for data to be forgotten 
            val_loader: DataLoader for validation data
            phase: One of ["dev", "debug", "test"] - debug returns retrained model copy
            
        Returns:
            The unlearned model
        """
        raise NotImplementedError