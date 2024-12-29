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

class Dataset(object):
    def __init__(
        self,
        dataset_filepath: str,
    ):
        self.dataset = []
        self.dataset = pd.read_csv(dataset_filepath).to_dict('records')
        for dp in self.dataset:
            if not dp['answer_choices'] or dp['answer_choices'] != dp['answer_choices']:
                del dp['answer_choices']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

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


def evaluate_model(merge_method, phase):
    # Call the merge function. The merged model is stored under merging_method object 
    merge_method.run()

    os.makedirs("output", exist_ok=True)
    output_dir = os.path.join("output", merge_method.get_name())
    prediction_dir = os.path.join(output_dir, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)
    # Save merged model 
    merge_method.save_model(output_dir)

    all_scores = {}

    if phase == 'test':
        dataset_filepath = "data/test.csv"
    else:
        raise ValueError(f"Invalid phase: {phase}")
    dataset_predictions = evaluate_dataset(merge_method, dataset_filepath)
    dp_df = pd.DataFrame(dataset_predictions)
    dp_df["dummy_field"] = 0
    # avoid error "Submission contains null values"
    dp_df['prediction'] = dp_df['prediction'].replace('', 'unknown').fillna('unknown')
    dp_df.to_csv(f"output/test.csv", columns=["id", "prediction", "dummy_field"], index=False)

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


