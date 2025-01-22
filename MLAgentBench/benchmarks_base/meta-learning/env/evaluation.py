from constants import * # the constants will be replaced with held-out test data/models during test phase
import os
import subprocess
import json

# define or import any evaluation util functions here 

def evaluate_model(method_dir, phase):
    # 1. load test input data from dataset_filepath
    # 2. apply the method / model on the whole dev / test data depending on the spcified phase
    # 3. save the results to a file under `./output`

    # wrap up necessary files for submission
    # The file path to 'metadata' (no extension)

    directory = os.path.join(os.getcwd(), "methods", method_dir)
    metadata_path = os.path.join(directory, "metadata")

    # Check if 'metadata' already exists
    if not os.path.isfile(metadata_path):
        # Create an empty file by opening in write mode and closing immediately
        with open(metadata_path, 'w'):
            pass  # Just create and close the file

    if phase == 'dev':
        input_data_dir = os.path.join(os.getcwd(), "data", "val_data")
        config_path = os.path.join(directory, "config.json")

        # Check if config.json exists
        if os.path.isfile(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # Extract 'validation_datasets'
            if "validation_datasets" in config_data:
                num_val_datasets = config_data["validation_datasets"]
                # Multiply by 0.6 and round
                config_data["validation_datasets"] = round(num_val_datasets * 0.6)

                # Write back the updated config
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=4)
    
    cmd = [
        "python", "-m", "cdmetadl.run",
        f"--input_data_dir={input_data_dir}",
        f"--submission_dir={directory}",
        "--output_dir_ingestion=ingestion_output",
        "--output_dir_scoring=scoring_output",
        "--verbose=True",
        "--overwrite_previous_results=True",
        "--test_tasks_per_dataset=10"
    ]

    # subprocess.run executes the command in a new process.
    # By default, it prints output to the console. 
    # 'check=True' raises an exception if the command fails (non-zero exit code).
    subprocess.run(cmd, check=True)
   
    
    
    # correct validation_datasets if changed for validation phase
    if phase == 'dev':
        config_path = os.path.join(directory, "config.json")
        if os.path.isfile(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            if "validation_datasets" in config_data:
                config_data["validation_datasets"] = num_val_datasets

                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=4)
    pass

def get_score(Method, phase):
    # 1. load results from `./output`
    # 2. calculate evaluation metric (either locally or use Kaggle API to submit to the leaderboard)
    # 3. (optional) save sample-level evaluation scores to a file (this may not be possible with Kaggle API evaluation)
    # 4. use `print()` function to print a message informing the evaluation score, which will be visible to LLM agents.
    # 5. return the final score (a single number, the higher the better)
    pass

