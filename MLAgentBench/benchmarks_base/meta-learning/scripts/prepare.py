import openml
import json
import os
import shutil


# Set the cache dir
openml.config.set_root_cache_directory(os.path.expanduser('openml_cache_dir'))

# Set correct path
data_dir = "openml_cache_dir/org/openml/www/datasets"

set_0 = {
    "BRD": 44285,
    "PLK": 44282,
    "FLW": 44283,
    "PLT_VIL": 44286,
    "BCT": 44281,
    "RESISC": 44290,
    "CRS": 44289,
    "TEX": 44288,
    "SPT": 44284,
    "MD_MIX": 44287
}

for name, dataset_id in set_0.items():
    print(f"Downloading {name}...")
    openml.datasets.get_dataset(dataset_id, download_data=True, download_all_files=True)
    cache_path = os.path.join(data_dir, str(dataset_id), f"{name}_Mini")
    shutil.copytree(src=cache_path, dst=f"test_data/{name}", dirs_exist_ok=True)

set_1 = {
    "DOG": 44298,
    "INS_2": 44292,
    "PLT_NET": 44293,
    "MED_LF": 44299,
    "PNU": 44297,
    "RSICB": 44300,
    "APL": 44295,
    "TEX_DTD": 44294,
    "ACT_40": 44291,
    "MD_5_BIS": 44296
}

for name, dataset_id in set_1.items():
    print(f"Downloading {name}...")
    openml.datasets.get_dataset(dataset_id, download_data=True, download_all_files=True)
    cache_path = os.path.join(data_dir, str(dataset_id), f"{name}_Mini")
    shutil.copytree(src=cache_path, dst=f"test_data/{name}", dirs_exist_ok=True)

# Construct the dictionary in the desired format
splits_dict = {
    "meta-train": list(set_0.keys()),
    "meta-test": list(set_1.keys())
}

info_path = os.path.join("test_data", "info")
os.makedirs(info_path, exist_ok=True)

# Write to 'meta_splits.txt'
file_path = os.path.join("test_data", "info", "meta_splits.txt")
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(splits_dict, f, indent=4)

shutil.copytree("test_data", "../env/data", dirs_exist_ok=True)

set_2 = {
    "AWA": 44305,
    "INS": 44306,
    "FNG": 44302,
    "PLT_DOC": 44303,
    "PRT": 44308,
    "RSD": 44307,
    "BTS": 44309,
    "TEX_ALOT": 44304,
    "ACT_410": 44301,
    "MD_6": 44310
}

for name, dataset_id in set_2.items():
    print(f"Downloading {name}...")
    openml.datasets.get_dataset(dataset_id, download_data=True, download_all_files=True)
    cache_path = os.path.join(data_dir, str(dataset_id), f"{name}_Mini")
    shutil.copytree(src=cache_path, dst=f"test_data/{name}", dirs_exist_ok=True)

# Construct the dictionary in the desired format
splits_dict = {
    "meta-train": list(set_0.keys()) + list(set_1.keys()),
    "meta-test": list(set_2.keys())
}

# Write to 'meta_splits.txt'
file_path = os.path.join("test_data", "info", "meta_splits.txt")
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(splits_dict, f, indent=4)
