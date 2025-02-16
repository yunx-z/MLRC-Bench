# download and preprocess any large-size data/model if needed

import os
import json
import shutil
import requests
import torch
from torch.utils.data import DataLoader, Subset, random_split

import torchvision
from torchvision import transforms

# Create directories relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dirs = [
   os.path.join(PROJECT_ROOT, "env/data"),
   os.path.join(PROJECT_ROOT, "output"),
   os.path.join(PROJECT_ROOT, "scripts/.kaggle"),  # Add .kaggle directory
]

for dir in dirs:
   os.makedirs(dir, exist_ok=True)

# Setup Kaggle credentials if provided in scripts/.kaggle/kaggle.json
script_kaggle_json = os.path.join(PROJECT_ROOT, "scripts/.kaggle/kaggle.json")
user_kaggle_dir = os.path.expanduser("~/.kaggle")
user_kaggle_json = os.path.join(user_kaggle_dir, "kaggle.json")

if os.path.exists(script_kaggle_json):
    # Create user's .kaggle directory if it doesn't exist
    os.makedirs(user_kaggle_dir, exist_ok=True)
    
    # Copy kaggle.json to user's .kaggle directory
    shutil.copy2(script_kaggle_json, user_kaggle_json)
    
    # Set correct permissions
    os.chmod(user_kaggle_json, 0o600)
    
# Original model
# download pre-trained weights
local_path = "../env/weights_resnet18_cifar10.pth"
if not os.path.exists(local_path):
    response = requests.get(
        "https://storage.googleapis.com/unlearning-challenge/weights_resnet18_cifar10.pth")
    open(local_path, "wb").write(response.content)


# Retrained model
# download weights of a model trained exclusively on the retain set
local_path = "../env/retrain_weights_resnet18_cifar10.pth"
if not os.path.exists(local_path):
    response = requests.get(
        "https://storage.googleapis.com/unlearning-challenge/retrain_weights_resnet18_cifar10.pth")
    open(local_path, "wb").write(response.content)

# manual random seed is used for dataset partitioning
# to ensure reproducible results across runs
RNG = torch.Generator().manual_seed(42)

# download and pre-process CIFAR10
normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


train_set = torchvision.datasets.CIFAR10(
    root="../env/data", train=True, download=True, transform=normalize)

# we split held out data into test and validation set
held_out = torchvision.datasets.CIFAR10(
    root="../env/data", train=False, download=True, transform=normalize)

# download the forget and retain index split
local_path = "../env/forget_idx.npy"
if not os.path.exists(local_path):
    response = requests.get(
        "https://storage.googleapis.com/unlearning-challenge/forget_idx.npy")
    open(local_path, "wb").write(response.content)

os.system("rm ../env/data/cifar-10-python.tar.gz")


print("All preparation steps completed successfully!")
with open("prepared", 'w') as writer:
    pass
