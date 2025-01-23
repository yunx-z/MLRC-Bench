# download and preprocess any large-size data/model if needed

import os

# Create directories relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dirs = [
   os.path.join(PROJECT_ROOT, "env/data"),
   os.path.join(PROJECT_ROOT, "output"),
]

for dir in dirs:
   os.makedirs(dir, exist_ok=True)