from constants import * # the constants will be replaced with held-out test data/models during test phase
import os
import subprocess
import json

# define or import any evaluation util functions here 
TEST_TASKS = {'dev' : 100, 'test' : 600, 'debug' : 1}

def evaluate_model(method_dir, phase):
    # 1. load test input data from dataset_filepath
    # 2. apply the method / model on the whole dev / test data depending on the spcified phase
    # 3. save the results to a file under `./output`

    directory = os.path.join(os.getcwd(), "methods", method_dir)

    input_data_dir = os.path.join(os.getcwd(), "data")
    
    if phase == 'test':
        config_path = os.path.join(directory, "config.json")

        # Check if config.json exists
        if os.path.isfile(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # Extract 'validation_datasets'
            if "validation_datasets" in config_data:
                num_val_datasets = config_data["validation_datasets"]
                # Multiply by 2
                config_data["validation_datasets"] = num_val_datasets * 2

                # Write back the updated config
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=4)
    
    cmd = [
        "python", "-m", "cdmetadl.run_eval",
        f"--input_data_dir={input_data_dir}",
        f"--submission_dir={directory}",
        "--output_dir_ingestion=ingestion_output",
        "--verbose=False",
        "--overwrite_previous_results=True",
        f"--test_tasks_per_dataset={TEST_TASKS[phase]}"
    ]

    # subprocess.run executes the command in a new process.
    # By default, it prints output to the console. 
    # 'check=True' raises an exception if the command fails (non-zero exit code).
    subprocess.run(cmd, check=True)
   
    
    
    # correct validation_datasets if changed for validation phase
    if phase == 'test':
        config_path = os.path.join(directory, "config.json")
        if os.path.isfile(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            if "validation_datasets" in config_data:
                config_data["validation_datasets"] = num_val_datasets

                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=4)

def get_score(method_dir, phase):
    # 1. load results from `./output`
    # 2. calculate evaluation metric (either locally or use Kaggle API to submit to the leaderboard)
    # 3. (optional) save sample-level evaluation scores to a file (this may not be possible with Kaggle API evaluation)
    # 4. use `print()` function to print a message informing the evaluation score, which will be visible to LLM agents.
    # 5. return the final score (a single number, the higher the better)
    input_data_dir = os.path.join(os.getcwd(), "data")

    cmd = [
        "python", "-m", "cdmetadl.run_scoring",
        f"--input_data_dir={input_data_dir}",
        "--output_dir_ingestion=ingestion_output",
        "--output_dir_scoring=scoring_output",
        "--verbose=False",
        "--overwrite_previous_results=True",
        f"--test_tasks_per_dataset={TEST_TASKS[phase]}"
    ]

    # subprocess.run executes the command in a new process.
    # By default, it prints output to the console. 
    # 'check=True' raises an exception if the command fails (non-zero exit code).
    subprocess.run(cmd, check=True)

    scoring_output = os.path.join(os.getcwd(), "scoring_output", "scores.txt")
    with open(scoring_output, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()

    # Split on the colon, take the second part, and convert to float
    _, raw_score = first_line.split(":", 1)
    score = float(raw_score.strip())
    print(f"Balanced accuracy score: {score}")
    return score
