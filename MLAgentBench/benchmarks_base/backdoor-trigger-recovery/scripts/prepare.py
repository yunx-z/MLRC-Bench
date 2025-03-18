from datasets import load_dataset, Dataset
import os
import random
import json

# Load dataset
# Replace 'dataset_name' with the actual dataset you want to load
dataset = load_dataset('Zhaorun/CLAS_backdoor_recovery', split='train')

# Shuffle dataset with a fixed seed for reproducibility
random.seed(42)
indices = list(range(len(dataset)))
random.shuffle(indices)

half_size = len(indices) // 2
dev_indices = indices[:half_size]
test_indices = indices[half_size:]

dev_dataset = dataset.select(dev_indices)
test_dataset = dataset.select(test_indices)

# Save datasets to disk
dev_file = 'dev.jsonl'
test_file = 'test.jsonl'

def save_to_file(dataset, filename):
    with open(filename, 'w') as f:
        for example in dataset:
            f.write(json.dumps(example) + '\n')

save_to_file(dev_dataset, dev_file)
save_to_file(test_dataset, test_file)

# Load datasets back from files
def load_from_file(filename):
    with open(filename, 'r') as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_list(data)

dev_dataset_loaded = load_from_file(dev_file)
test_dataset_loaded = load_from_file(test_file)

# Verify the size of loaded datasets
print(f"Original Dev Size: {len(dev_dataset)}, Loaded Dev Size: {len(dev_dataset_loaded)}")
print(f"Original Test Size: {len(test_dataset)}, Loaded Test Size: {len(test_dataset_loaded)}")

os.system(f"cp {dev_file} ../env/data/")
with open("prepared", 'w') as writer:
    pass
