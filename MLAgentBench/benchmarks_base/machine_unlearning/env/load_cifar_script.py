import os
import requests

import numpy as np

import torch
from torch.utils.data import DataLoader, Subset, random_split

import torchvision
from torchvision import transforms
from torchvision.models import resnet18


def get_cifar10_data(rng: int = 42) -> dict[str, DataLoader]:
    """Returns a dictionary containing all CIFAR10 data loaders
    required for the unlearn task.
    """
    
    # manual random seed is used for dataset partitioning
    # to ensure reproducible results across runs
    RNG = torch.Generator().manual_seed(rng)
    
    # download and pre-process CIFAR10
    normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=normalize)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)

    # we split held out data into test and validation set
    held_out = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=normalize)
    test_set, val_set = random_split(held_out, [0.5, 0.5], generator=RNG)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=2)

    # download the forget and retain index split
    local_path = "forget_idx.npy"
    if not os.path.exists(local_path):
        response = requests.get(
            "https://storage.googleapis.com/unlearning-challenge/" + local_path)
        open(local_path, "wb").write(response.content)
    forget_idx = np.load(local_path)

    # construct indices of retain from those of the forget set
    forget_mask = np.zeros(len(train_set.targets), dtype=bool)
    forget_mask[forget_idx] = True
    retain_idx = np.arange(forget_mask.size)[~forget_mask]

    # split train set into a forget and a retain set
    forget_set = Subset(train_set, forget_idx)
    retain_set = Subset(train_set, retain_idx)

    forget_loader = DataLoader(
        forget_set, batch_size=128, shuffle=True, num_workers=2)
    retain_loader = DataLoader(
        retain_set, batch_size=128, shuffle=True, num_workers=2, generator=RNG)
    
    return {
        "training": train_loader,
        "testing": test_loader,
        "validation": val_loader,
        "forget": forget_loader,
        "retain": retain_loader
    }


def get_cifar10_pretrained_models(rng: int = 42, device = None) -> dict:
    """Returns a dictionary of the original ResNet18 model, pretrained on 
    the whole of the training set, and the retrained ResNet18 model, which
    was retrained only on the retain set."""
    
    assert rng == 42, "Pretrained models only available for initial random seed = 42."
    
    # Original model
    # download pre-trained weights
    local_path = "weights_resnet18_cifar10.pth"
    if not os.path.exists(local_path):
        response = requests.get(
            "https://storage.googleapis.com/unlearning-challenge/weights_resnet18_cifar10.pth")
        open(local_path, "wb").write(response.content)

    weights_pretrained = torch.load(local_path, map_location=device)

    # load model with pre-trained weights
    model = resnet18(weights=None, num_classes=10)
    model.load_state_dict(weights_pretrained)
    model.to(device)
    model.eval();
    
    # Retrained model
    # download weights of a model trained exclusively on the retain set
    local_path = "retrain_weights_resnet18_cifar10.pth"
    if not os.path.exists(local_path):
        response = requests.get(
            "https://storage.googleapis.com/unlearning-challenge/" + local_path)
        open(local_path, "wb").write(response.content)

    weights_pretrained = torch.load(local_path, map_location=device)

    # load model with pre-trained weights
    rt_model = resnet18(weights=None, num_classes=10)
    rt_model.load_state_dict(weights_pretrained)
    rt_model.to(device)
    rt_model.eval();
    
    return {
        "original": model,
        "retrained": rt_model
    }    