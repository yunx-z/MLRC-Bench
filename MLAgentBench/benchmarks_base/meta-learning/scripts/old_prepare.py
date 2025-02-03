import requests
import os
import json
import zipfile
import shutil
from io import BytesIO

url = "https://codalab.lisn.upsaclay.fr/my/datasets/download/3613416d-a8d7-4bdb-be4b-7106719053f1"

response = requests.get(url, stream=True)
with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall("test_data")

# Define paths
parent_folder = "test_data"
public_data_folder = os.path.join(parent_folder, "public_data")

# Check if the public_data folder exists
if os.path.exists(public_data_folder) and os.path.isdir(public_data_folder):
    # Move each file and subfolder from public_data to its parent (test_data)
    for item in os.listdir(public_data_folder):
        source = os.path.join(public_data_folder, item)
        destination = os.path.join(parent_folder, item)
        
        # Move the file or folder
        shutil.move(source, destination)
    
    # Remove the now-empty public_data folder
    os.rmdir(public_data_folder)

val = ["BCT", "BRD", "CRS", "FLW", "MD_MIX"]

for dataset in val:
        source = f"test_data/{dataset}"
        destination = f"../env/data/{os.path.basename(dataset)}"
        shutil.copytree(source, destination, dirs_exist_ok=True)

# Construct the dictionary in the desired format
val_splits_dict = {
    "meta-train": ["BCT", "BRD", "CRS"],
    "meta-test": ["FLW", "MD_MIX"]
}

os.makedirs("../env/data/info", exist_ok=True)
with open("../env/data/info/meta_splits.txt", "w", encoding="utf-8") as f:
    json.dump(val_splits_dict, f, indent=4)
