import os
import pandas as pd

def evaluate_model(Method, phase):
    # 1. load test input data
    if phase == "dev":
        data_directory = "./data/dev"
    elif phase == "test":
        data_directory = "./data/test"
    else:
        raise(ValueError('Unrecognized phase. Phase should be either "dev" or "test"'))
    
    # 2. apply the method / model on the whole dev / test data depending on the spcified phase

    # Train model
    Method.train()

    # 3. save the results to a file under `./output`
    Method.run(data_directory)


def get_score(Method, phase):
    # 1. load results from `./output`

    output_file = os.path.join("./output", "predictions.csv")
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Predictions not found at {output_file}")

    df = pd.read_csv(output_file)

    # Ground truth
    y_intra = df["intra_depth"].values
    y_inter = df["inter_depth"].values

    # Predictions
    pred_intra = df["pred_intra"].values
    pred_inter = df["pred_inter"].values

    # 2. calculate evaluation metric (either locally or use Kaggle API to submit to the leaderboard)
    mse_intra = (pred_intra - y_intra) ** 2
    mse_inter = (pred_inter - y_inter) ** 2

    # Average MSEs
    avg_mse_intra = mse_intra.mean()
    avg_mse_inter = mse_inter.mean()

    # Combined metric
    avg_mse_overall = (avg_mse_intra + avg_mse_inter) / 2.0

    # Return negative MSE for "higher is better"
    score = -avg_mse_overall

    # 4. use `print()` function to print a message informing the evaluation score, which will be visible to LLM agents.
    print(f"Evaluation score for phase '{phase}': {score:.4f}")

    # 5. return the final score (a single number, the higher the better)
    return score
