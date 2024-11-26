import argparse
import os
import sys
import time

from importlib.metadata import entry_points

from llm_merging.evaluation import * 
from llm_merging.data import * 
from llm_merging.merging import *

def all_merge_handlers():
    """Enumerate and Load (import) all merge methods."""
    loaded_merges = {
        "my_merge" : MyMerge,
        ## TODO Add more merge methods here
    }
    
    return loaded_merges


if __name__ == "__main__":
    DEFAULT_METHOD_NAME = "my_merge"
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--merging_method", type=str)
    parser.add_argument(
        "--dataset_filepaths", 
        type=str, 
        default=["data/test.csv"], 
        nargs='+'
    )
    parser.add_argument("--output_folder", type=str, default="output")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    loaded_merges = all_merge_handlers()
    merge_method = loaded_merges[args.merging_method](args.merging_method)

    start_time = time.time()
    # Call the merge function. The merged model is stored under merging_method object 
    merge_method.merge()
    
    # Evaluate method on datsets passed in (used for testing)
    evaluate_model(
        merge_method,
        args.dataset_filepaths,
        args.output_folder,
    )
    end_time = time.time()
    runtime = end_time - start_time

    score = get_score()

    base_class = loaded_merges[DEFAULT_METHOD_NAME]
    merge_method_class = loaded_merges[args.merging_method]
    save_evals(args.merging_method, merge_method_class, base_class, score, runtime)
