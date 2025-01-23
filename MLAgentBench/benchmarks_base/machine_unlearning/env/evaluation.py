"""Evaluation code for machine unlearning methods."""

import os
import time
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader

from metric import compute_logit_scaled_confidence, compute_forget_score_from_confs
from load_cifar_script import get_cifar10_data, get_cifar10_pretrained_models

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_MODELS = 512  # Number of models for evaluation

def _get_confs(net, loader):
    """Returns the confidences of the data in loader extracted from net."""
    confs = []
    targets_list = []
    net.eval()
    with torch.no_grad():
        for data, targets in loader:
            inputs = data.to(DEVICE)
            logits = net(inputs)
            logits = logits.detach().cpu().numpy()
            targets = targets.numpy()
            targets_list.append(targets)
            _, conf = compute_logit_scaled_confidence(logits, targets)
            confs.append(conf)
    confs = np.concatenate(confs, axis=0)
    return confs

def evaluate_model(method, phase: str = "dev"):
    """Evaluate an unlearning method.
    
    Args:
        method: The unlearning method to evaluate
        phase: Either "dev" or "test" phase
        
    Returns:
        Dict containing evaluation metrics
    """
    print(f"Starting evaluation for {method.get_name()} in {phase} phase")
    
    if phase == "dev":
        return _evaluate_dev(method)
    elif phase == "test":
        return _evaluate_test(method)
    else:
        raise ValueError(f"Invalid phase: {phase}")

def _evaluate_dev(method):
    """Development phase evaluation using CIFAR-10."""
    print("Loading CIFAR-10 data and models...")
    
    # Load data and models
    data_loaders = get_cifar10_data()
    pretrained_models = get_cifar10_pretrained_models(device=DEVICE)

    retain_loader = data_loaders["retain"]
    forget_loader = data_loaders["forget"] 
    val_loader = data_loaders["validation"]
    test_loader = data_loaders["testing"]

    original_model = pretrained_models["original"]
    retrained_model = pretrained_models["retrained"]

    # Run unlearning multiple times
    unlearned_confs_forget = []
    unlearned_retain_accs = []
    unlearned_test_accs = []
    unlearned_forget_accs = []

    print(f"Running unlearning {NUM_MODELS} times...")
    for i in range(NUM_MODELS):
        if i % 100 == 0:
            print(f"Processing model {i}/{NUM_MODELS}")
            
        # Clone original model and unlearn
        model = deepcopy(original_model)
        unlearned_model = method.run(
            original_model=model,
            retain_loader=retain_loader,
            forget_loader=forget_loader,
            val_loader=val_loader
        )

        # Get confidences on forget set
        confs_forget = _get_confs(unlearned_model, forget_loader)
        unlearned_confs_forget.append(confs_forget)

        # Compute accuracies
        unlearned_retain_accs.append(_compute_accuracy(unlearned_model, retain_loader))
        unlearned_test_accs.append(_compute_accuracy(unlearned_model, test_loader))
        unlearned_forget_accs.append(_compute_accuracy(unlearned_model, forget_loader))

    unlearned_confs_forget = np.stack(unlearned_confs_forget)

    # Get retrained model metrics
    RAR = _compute_accuracy(retrained_model, retain_loader)
    TAR = _compute_accuracy(retrained_model, test_loader)
    FAR = _compute_accuracy(retrained_model, forget_loader)

    # Get retrained model confidences
    retrained_confs_forget = []
    print("Computing retrained model confidences...")
    
    for i in range(NUM_MODELS):
        confs = _get_confs(retrained_model, forget_loader)
        retrained_confs_forget.append(confs)
    
    retrained_confs_forget = np.stack(retrained_confs_forget)

    # Get unlearned model average metrics
    RAU = np.mean(unlearned_retain_accs)
    TAU = np.mean(unlearned_test_accs)
    FAU = np.mean(unlearned_forget_accs)

    # Compute forget score
    forget_score = compute_forget_score_from_confs(
        unlearned_confs_forget, retrained_confs_forget
    )

    # Compute final score with utility adjustment
    final_score = forget_score * (RAU / RAR) * (TAU / TAR)

    print(f"\nResults:")
    print(f"Forget Score: {forget_score:.4f}")
    print(f"Retain Accuracy Ratio (RAU/RAR): {RAU/RAR:.4f}")
    print(f"Test Accuracy Ratio (TAU/TAR): {TAU/TAR:.4f}") 
    print(f"Final Score: {final_score:.4f}")

    return {
        "total_score": float(final_score),
        "forgetting_quality": float(forget_score),
        "unlearn_retain_acc": float(RAU),
        "unlearn_test_acc": float(TAU),
        "unlearn_forget_acc": float(FAU),
        "retrain_retain_acc": float(RAR),
        "retrain_test_acc": float(TAR),
        "retrain_forget_acc": float(FAR)
    }

def _evaluate_test(method):
    """Test phase evaluation - generates Kaggle submission notebook."""
    print("Test phase evaluation - generating Kaggle submission")
    raise NotImplementedError("Test phase evaluation not implemented")

def _compute_accuracy(model: torch.nn.Module, loader: DataLoader) -> float:
    """Compute accuracy for a model on a data loader."""
    correct = 0
    total = 0
    model.eval()
    
    with torch.no_grad():
        for data, targets in loader:
            images = data.to(DEVICE)
            labels = targets.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return correct / total

def get_score(method, phase: str = "dev") -> float:
    """Get the evaluation score for a method."""
    results = evaluate_model(method, phase)
    return results["total_score"]