from copy import deepcopy
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class BaseMethod(object):
    def __init__(self, name):
        self.name = name
        
    def get_name(self):
        return self.name
        
    def run(self, net, retain_loader, forget_loader, val_loader):
        """Base method for machine unlearning
        
        Args:
            net: The model to be unlearned
            retain_loader: DataLoader for retained training data
            forget_loader: DataLoader for data to be forgotten 
            val_loader: DataLoader for validation data
            
        Returns:
            The unlearned model
        """
        raise NotImplementedError