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
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
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
        for batch in tqdm(data_loader):
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

def evaluate_tinybench(
    model,
    tokenizer,
    tasks_list: List[str] = ["tinyBenchmarks"],
    batch_size: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    
    my_model = HFLM(pretrained=model, tokenizer=tokenizer) 
    del model
    torch.cuda.empty_cache()
    # Run evaluation
    # Set apply_chat_template=True based on https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/discussions/81
    results = evaluator.simple_evaluate(
        model=my_model,
        tasks=tasks_list,
        batch_size=batch_size,
        device=device,
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        verbosity='ERROR',
    )
    
    return results['results']
    

def evaluate_model(merge_method, phase):
    # Call the merge function. The merged model is stored under merging_method object 
    merge_method.run()

    output_dir = os.path.join("output", merge_method.get_name())
    os.makedirs(output_dir, exist_ok=True)

    if phase == 'test':
        dataset_filepath = "data/test.csv"
        dataset_predictions = evaluate_dataset(merge_method, dataset_filepath)
        dp_df = pd.DataFrame(dataset_predictions)
        dp_df["dummy_field"] = 0
        # avoid error "Submission contains null values"
        dp_df['prediction'] = dp_df['prediction'].replace('', 'unknown').fillna('unknown')
        output_file = os.path.join(output_dir, "test.csv")
        dp_df.to_csv(output_file, columns=["id", "prediction", "dummy_field"], index=False)
        print(f"predictions saved to {output_file}")
    elif phase == 'dev':
        result = evaluate_tinybench(merge_method.base_model, merge_method.input_tokenizer)
        output_file = os.path.join(output_dir, "dev.json")
        with open(output_file, 'w') as writer:
            json.dump(result, writer, indent=2)
        print(f"evaluation results on tineBenchmark saved to {output_file}")
    else:
        raise ValueError(f"Invalid phase: {phase}")


def get_submission_result(competition, idx=0):
    api = KaggleApi()
    api.authenticate()
    
    # Fetch submissions
    submissions = api.competitions_submissions_list(competition)
    
    # Iterate through submissions and print error messages
    latest_submission = submissions[idx]
    if latest_submission["hasPublicScore"]:
        score = float(latest_submission["publicScore"])
        print(f"\nYour merged model scores {score} out of 1.00 on the test set!")
    else:
        error_msg = latest_submission["errorDescription"] 
        print(f"\nYour merged model may generate something invalid so the submission does not have a score. Here is the error message from the Kaggle leaderboard:\n\n{error_msg}")
        score = None
    return score

def get_score(merge_method, phase):
    output_dir = os.path.join("output", merge_method.get_name())

    if phase == 'test':
        submission_path = os.path.join(output_dir, "test.csv")
        competition_name = "llm-merging-competition"
        lock_file = os.path.expanduser("~/submission.lock")
        score = None
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
    elif phase == 'dev':
        output_file = os.path.join(output_dir, "dev.json")
        with open(output_file, 'r') as reader:
            result = json.load(reader)
        scores = []
        for bench in result:
            bench_result = result[bench]
            _task_score = None
            if 'acc_norm,none' in bench_result:
                _task_score = bench_result['acc_norm,none']
            elif _task_score is None or 'exact_match,flexible-extract' in bench_result:
                _task_score = bench_result['exact_match,flexible-extract']
            if _task_score is None:
                return None
            else:
                scores.append(_task_score)
        score = sum(scores) / len(scores)
        print(f"\nYour merged model scores {score} out of 1.00 on the dev set!")
    else:
        raise ValueError(f"Invalid phase: {phase}")

    return score


