import os
import requests
import zipfile
import shutil
from io import BytesIO

url = "https://codalab.lisn.upsaclay.fr/my/datasets/download/3613416d-a8d7-4bdb-be4b-7106719053f1"

response = requests.get(url, stream=True)
with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall("data")

shutil.copytree("data", "../env/data", dirs_exist_ok=True)
