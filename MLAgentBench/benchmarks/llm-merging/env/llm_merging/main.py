import argparse
import os
import sys
import time

from importlib.metadata import entry_points
from kaggle.api.kaggle_api_extended import KaggleApi

from llm_merging.evaluation import * 
from llm_merging.data import * 
from llm_merging.merging import *

def all_merge_handlers():
    """Enumerate and Load (import) all merge methods."""
    loaded_merges = {
        "MyMerge" : MyMerge,
    }
    
    return loaded_merges


def get_submission_result(competition, idx=0):
    api = KaggleApi()
    api.authenticate()
    
    # Fetch submissions
    submissions = api.competitions_submissions_list(competition)
    
    # Iterate through submissions and print error messages
    latest_submission = submissions[idx]
    if latest_submission["hasPublicScore"]:
        score = float(latest_submission["publicScore"])
        print(f"\nYour merged model scores {score} on the test set!")
    else:
        error_msg = latest_submission["errorDescription"] 
        print(f"\nYour merged model may generate something invalid so the submission does not have a score. Here is the error message from the Kaggle leaderboard:\n\n{error_msg}")
        score = 0
    return score

def get_score():
    submission_path = "output/test.csv"
    competition_name = "llm-merging-competition"
    print("Submitting to Kaggle leaderbord for evaluation on test set ...")
    os.system(f"kaggle competitions submit -c {competition_name} -f {submission_path} -m \"llm-merging\"")
    print("Waiting for Kaggle leaderboard to refresh ...")
    time.sleep(60)
    return get_submission_result(competition_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--merging_method", type=str, default="MyMerge")
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

    # Call the merge function. The merged model is stored under merging_method object 
    merge_method.merge()
    
    # Evaluate method on datsets passed in (used for testing)
    evaluate_model(
        merge_method,
        args.dataset_filepaths,
        args.output_folder,
    )

    get_score()
