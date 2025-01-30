import torch
import torch.nn as nn
from typing import List


def process_support_set(model: nn.Module, 
                        weights: List[torch.Tensor], 
                        X_train: torch.Tensor, 
                        y_train: torch.Tensor, 
                        num_classes: int) -> torch.Tensor:
    """ Process the support set following the Prototypical Networks strategy.

    Args:
        model (nn.Module): Model to be used.
        weights (List[torch.Tensor]): Weights to be used by the model.
        X_train (torch.Tensor): Support set images.
        y_train (torch.Tensor): Support set labels.
        num_classes (int): Number of classes to predict.

    Returns:
        torch.Tensor: Support prototypes.
    """
    # Compute input embeddings
    support_embeddings = model.forward_weights(X_train, weights, 
        embedding=True)
    
    # Compute prototypes
    prototypes = torch.zeros((num_classes, support_embeddings.size(1)), 
        device=weights[0].device)
    for i in range(num_classes):
        mask = y_train == i
        prototypes[i] = (support_embeddings[mask].sum(dim=0) / 
            torch.sum(mask).item())
        
    return prototypes


def process_query_set(model: nn.Module, 
                      weights: List[torch.Tensor], 
                      X_test: torch.Tensor, 
                      prototypes: torch.Tensor) -> torch.Tensor:
    """ Process the query set following the Matching Networks strategy.

    Args:
        model (nn.Module): Model to be used.
        weights (List[torch.Tensor]): Weights to be used by the model.
        X_test (torch.Tensor): Query set images.
        prototypes (torch.Tensor): Support prototypes

    Returns:
        torch.Tensor: Distances to prototypes.
    """
    # Compute input embeddings
    query_embeddings = model.forward_weights(X_test, weights, embedding=True)

    # Create distance matrix (negative predictions)
    distance_matrix = (torch.cdist(query_embeddings.unsqueeze(0), 
        prototypes.unsqueeze(0))**2).squeeze(0) 
    out = -1 * distance_matrix
    
    return out
