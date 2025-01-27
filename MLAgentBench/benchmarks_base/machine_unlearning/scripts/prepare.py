# download and preprocess any large-size data/model if needed

import os
import json
import shutil

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