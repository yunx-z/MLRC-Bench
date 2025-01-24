import openml
import json
import os
import shutil


# Switch to the read-only server
openml.config.server = "http://145.38.195.79/api/v1/xml"

# Set the cache dir
openml.config.set_root_cache_directory(os.path.expanduser('openml_cache_dir'))

# Set correct path
data_dir = "openml_cache_dir/79/195/38/145/datasets"

set_0 = {
    "BRD": 44320,
    "PLK": 44317,
    "FLW": 44318,
    "PLT_VIL": 44321,
    "BCT": 44316,
    "RESISC": 44324,
    "CRS": 44323,
    "TEX": 44322,
    "SPT": 44319,
    "MD_MIX": 44287
}

for name, dataset_id in set_0.items():
    print(f"Downloading {name}...")
    openml.datasets.get_dataset(dataset_id, download_data=True, download_all_files=True)
    if name == "MD_MIX":
        cache_path = os.path.join(data_dir, str(dataset_id), f"{name}_Mini")
    else:
        cache_path = os.path.join(data_dir, str(dataset_id), f"{name}_Extended")
    shutil.copytree(src=cache_path, dst=f"data/{name}", dirs_exist_ok=True)

set_1 = {
    "DOG": 44331,
    "INS_2": 44326,
    "PLT_NET": 44327,
    "MED_LF": 44332,
    "PNU": 44330,
    "RSICB": 44333,
    "APL": 44329,
    "TEX_DTD": 44328,
    "ACT_40": 44325,
    "MD_5_BIS": 44296
}

for name, dataset_id in set_1.items():
    print(f"Downloading {name}...")
    openml.datasets.get_dataset(dataset_id, download_data=True, download_all_files=True)
    if name == "MD_5_BIS":
        cache_path = os.path.join(data_dir, str(dataset_id), f"{name}_Mini")
    else:
        cache_path = os.path.join(data_dir, str(dataset_id), f"{name}_Extended")
    shutil.copytree(src=cache_path, dst=f"data/{name}", dirs_exist_ok=True)

# Construct the dictionary in the desired format
splits_dict = {
    "meta-train": list(set_0.keys()),
    "meta-test": list(set_1.keys())
}

# Write to 'meta_splits.txt'
file_path = os.path.join("data", "meta_splits.txt")
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(splits_dict, f, indent=4)

shutil.copytree("data", "../env/data", dirs_exist_ok=True)

set_2 = {
    "AWA": 44338,
    "INS": 44340,
    "FNG": 44335,
    "PLT_DOC": 44336,
    "PRT": 44342,
    "RSD": 44341,
    "BTS": 44343,
    "TEX_ALOT": 44337,
    "ACT_410": 44334,
    "MD_6": 44310
}

for name, dataset_id in set_2.items():
    print(f"Downloading {name}...")
    openml.datasets.get_dataset(dataset_id, download_data=True, download_all_files=True)
    if name == "MD_6":
        cache_path = os.path.join(data_dir, str(dataset_id), f"{name}_Mini")
    else:
        cache_path = os.path.join(data_dir, str(dataset_id), f"{name}_Extended")
    shutil.copytree(src=cache_path, dst=f"data/{name}", dirs_exist_ok=True)

# Construct the dictionary in the desired format
splits_dict = {
    "meta-train": list(set_0.keys()) + list(set_1.keys()),
    "meta-test": list(set_2.keys())
}

# Write to 'meta_splits.txt'
file_path = os.path.join("data", "meta_splits.txt")
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(splits_dict, f, indent=4)
