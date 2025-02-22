import subprocess
import zipfile
import shutil
import os

cmd = ["aicrowd", "dataset", "download", "--challenge", "task-1-next-product-recommendation", "0", "3", "6", "7", "8"]

subprocess.run(cmd, capture_output=True, text=True)

with zipfile.ZipFile('19dd45a8-5f0c-4c95-bf60-506398327251_kdd-2023-ground-truth.zip', 'r') as zip_ref:
    zip_ref.extractall('test_data')

shutil.move('test_data/ground_truth/phase1/gt_task1.csv', '../env/data/dev_labels.csv')
shutil.move('test_data/ground_truth/phase2/gt_task1.csv', 'test_data/test_labels.csv')
shutil.rmtree('test_data/ground_truth')

shutil.move('sessions_test_task1_phase1.csv', '../env/data/dev_features.csv')
shutil.move('sessions_test_task1.csv', 'test_data/test_features.csv')
shutil.move('sessions_train.csv', '../env/data/train.csv')
shutil.move('products_train.csv', '../env/data/products.csv')

os.remove('19dd45a8-5f0c-4c95-bf60-506398327251_kdd-2023-ground-truth.zip')
