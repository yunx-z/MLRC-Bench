import evaluate 
import time
import inspect
import json 
import os 
import pandas as pd

from typing import List, Dict, Any

import torch
from tqdm import tqdm
from torch.utils import data
from kaggle.api.kaggle_api_extended import KaggleApi

from llm_merging.data import *
from MLAgentBench.utils import *

BASE_RUNTIME = 606.0022532939911 # avg over 3 runs on Quadro RTX 8000 GPU 

def convert_dict_of_lists_to_list_of_dicts(dict_of_lists: Dict[Any, List]) -> List[Dict]:
    """
    Args:
        dict_of_lists:

    Returns:
        list_ofDict
    """
    list_of_dicts = []
    for datapoint_values in zip(*dict_of_lists.values()):
        list_of_dicts.append(dict(zip(dict_of_lists, datapoint_values)))
    return list_of_dicts

def collate_fn(batch_of_datapoints: List[Dict]) -> Dict[Any, List]:
    """
    Convert a batch of datapoints into a datapoint that is batched. This is meant to override the default collate function in pytorch and specifically can handle when the value is a list 

    Args:
        batch_ofDatapoints:

    Returns:

    """
    datapoint_batched = {}
    for datapoint in batch_of_datapoints:
        # Gather together all the values per key
        for key, value in datapoint.items():
            if key in datapoint_batched:
                datapoint_batched[key].append(value)
            else:
                datapoint_batched[key] = [value]
    return datapoint_batched


def evaluate_dataset(
    merge_method,
    dataset_filepath: str,
) -> (Dict, List):

    data_loader = data.DataLoader(
        Dataset(dataset_filepath),
        batch_size=1,
        num_workers=0,
        shuffle=False,
        collate_fn=collate_fn
    )

    all_batches = []

    print("Running predictions on test set ...")
    with torch.no_grad():
        for batch in data_loader:
            # There are two types of evaluation models:
            # 1) multiple choice where the model scores each choice and predicts the choice with the highest score 
            # 2) generation where the model generate some output give some input 
            eval_type = batch["eval_type"][0]
            if eval_type == "multiple_choice":
                (
                    predicted_choice,
                    answer_choice_scores,
                ) = merge_method.predict_multiple_choice(batch)

                batch["prediction"] = str(predicted_choice.cpu().numpy().tolist()[0])
                all_batches.extend(convert_dict_of_lists_to_list_of_dicts(batch))
            
            else:
                assert eval_type == "generation"
                (
                    generated_ids, generated_txt
                ) = merge_method.generate(batch
                )
                batch["prediction"] = generated_txt 
                all_batches.extend(convert_dict_of_lists_to_list_of_dicts(batch))

    return all_batches


def evaluate_model(
    merge_method,
    all_dataset_filepaths: List[str],
    output_folder: str,
) -> Dict:   
    output_dir = os.path.join("output", merge_method.get_name())
    prediction_dir = os.path.join(output_dir, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)
    # Save merged model 
    merge_method.save_model(output_dir)

    all_scores = {}

    for dataset_filepath in all_dataset_filepaths:
        dataset_predictions = evaluate_dataset(merge_method, dataset_filepath)
        dp_df = pd.DataFrame(dataset_predictions)
        dp_df["dummy_field"] = 0
        # avoid error "Submission contains null values"
        dp_df['prediction'] = dp_df['prediction'].replace('', 'unknown').fillna('unknown')
        fn = os.path.basename(dataset_filepath)
        dp_df.to_csv(f"{output_folder}/{fn}", columns=["id", "prediction", "dummy_field"], index=False)

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
    lock_file = os.path.expanduser("~/submission.lock")
    score = 0
    while os.path.exists(lock_file):
        print("Another submission is in progress. Waiting...")
        time.sleep(30)  # Wait before checking again
    # Create a lock file
    with open(lock_file, 'w') as f:
        f.write('Locked')
    try:
        print("\nSubmitting to Kaggle leaderbord for evaluation on test set ...")
        os.system(f"kaggle competitions submit -c {competition_name} -f {submission_path} -m \"llm-merging\"")
        print("\nWaiting for Kaggle leaderboard to refresh ...")
        time.sleep(60)
        score = get_submission_result(competition_name)
    finally:
        # Remove the lock file
        os.remove(lock_file)

    return score

def save_evals(merge_method_name, merge_method_class, base_class, score, runtime):
    # save idea, merge_method_name, merge_method_code, feedback, score into a file
    merge_method_code = inspect.getsource(merge_method_class)
    base_method_code = inspect.getsource(base_class)
    idea_file = "idea.txt"
    if os.path.exists(idea_file):
        with open(idea_file, 'r') as reader:
            idea = reader.read()
        feedback, relevance_score = get_llm_feedback(idea, merge_method_code) 
        print(feedback)
    else:
        idea, feedback, relevance_score = None, None, None

    eval_file = "output/idea_evals.json"
    if os.path.exists(eval_file):
        with open(eval_file, 'r') as reader:
            all_evals = json.load(reader)
    else:
        all_evals = {"idea" : idea, "implementations" : []}

    method_total_lines = count_code_lines(merge_method_code)
    base_total_lines = count_code_lines(base_method_code)
    num_diff_line = count_different_lines(base_method_code, merge_method_code)
    eval_result = {
            "merge_method_name" : merge_method_name,
            "performance" : score,
            "relevance_score" : relevance_score, 
            "relative_runtime" : runtime / BASE_RUNTIME,
            "relative_complexity" :  num_diff_line / (2 * base_total_lines),
            "runtime" : runtime,
            "method_total_lines" : method_total_lines,
            "base_total_lines" : base_total_lines,
            "num_diff_line" : num_diff_line,
            "code" : merge_method_code,
            "feedback" : feedback,
            }
    all_evals["implementations"].append(eval_result)
    with open(eval_file, 'w') as writer:
        json.dump(all_evals, writer, indent=2)


