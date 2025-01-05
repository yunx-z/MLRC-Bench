from constants import * # the constants will be replaced with held-out test data/models during test phase

# define or import any evaluation util functions here 

def evaluate_model(Method, phase):
    # 1. load test input data from dataset_filepath
    # 2. apply the method / model on the whole dev / test data depending on the spcified phase
    # 3. save the results to a file under `./output`
    pass

def get_score(Method, phase):
    # 1. load results from `./output`
    # 2. calculate evaluation metric (either locally or use Kaggle API to submit to the leaderboard)
    # 3. (optional) save sample-level evaluation scores to a file (this may not be possible with Kaggle API evaluation)
    # 4. use `print()` function to print a message informing the evaluation score, which will be visible to LLM agents.
    # 5. return the final score (a single number)
    pass

