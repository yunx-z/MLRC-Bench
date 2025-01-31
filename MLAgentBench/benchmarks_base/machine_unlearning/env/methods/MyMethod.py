from copy import deepcopy
import torch
from torch import nn, optim
from methods.BaseMethod import BaseMethod

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)
        
    def run(self, net, retain_loader, forget_loader, val_loader):
        """Unlearning implementation - either finetuning or debug copy
        
        Args:
            net: The model to be unlearned
            retain_loader: DataLoader for retained training data
            forget_loader: DataLoader for data to be forgotten
            val_loader: DataLoader for validation data
            
        Returns:
            The unlearned model
        """ 
        epochs = 1
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001,
                          momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)

        for ep in range(epochs):
            net.train()
            # Process all batches in retain_loader
            for batch_idx, sample in enumerate(retain_loader):
                if isinstance(sample, dict):
                    inputs = sample["image"] 
                    targets = sample["age_group"]
                else:
                    inputs, targets = sample  # For CIFAR format
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            scheduler.step()
        
        net.eval()