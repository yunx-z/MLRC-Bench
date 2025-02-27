import pandas as pd
import numpy as np
from functools import lru_cache

from constants import labels_path # the constants will be replaced with held-out test data/models during test phase

@lru_cache(maxsize=1)
def read_test_labels():
    return pd.read_csv(labels_path) 

def evaluate_model(Method, phase):
    # 1. load test input data from dataset_filepath
    # 2. apply the method / model on the whole dev / test data depending on the spcified phase
    # 3. save the results to a file under `./output`
    pred = Method.run()
    pred.to_csv('output/pred.csv')

def get_score(Method, phase):
    # 1. load results from `./output`
    # 2. calculate evaluation metric (either locally or use Kaggle API to submit to the leaderboard)
    # 3. (optional) save sample-level evaluation scores to a file (this may not be possible with Kaggle API evaluation)
    # 4. use `print()` function to print a message informing the evaluation score, which will be visible to LLM agents.
    # 5. return the final score (a single number, the higher the better)
    pred = pd.read_csv('output/pred.csv')
    pred_next_items = pred['next_item_prediction']

    ground_truth = read_test_labels()
    gt_next_items = ground_truth['next_item']
    
    reciprocal_ranks = []
    
    for pred_list, gt_item in zip(pred_next_items, gt_next_items):
        try:
            # Find the rank (index + 1 because it's 1-based)
            rank = pred_list.index(gt_item) + 1
            reciprocal_ranks.append(1 / rank)
        except ValueError:
            # If gt_item is not in pred_list, ignore (equivalent to 0 contribution)
            reciprocal_ranks.append(0)
    
    # Compute Mean Reciprocal Rank (MRR)
    mrr = np.mean(reciprocal_ranks)
    print(f"Final Score: {mrr}")
    return mrr
