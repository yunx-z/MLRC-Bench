""" This is the scoring program written by the organizers. This program also
runs on the Cross-Domain MetaDL competition platform to test your code.

Usage: 

python -m cdmetadl.scoring.scoring \
    --results_dir=output_dir_ingestion \
    --input_data_dir=input_dir \
    --output_dir_scoring=output_dir

* The results directory output_dir_ingestion (e.g. ../../ingestion_output) must
be the directory containing the output from the ingestion program with the 
following files:
    metadata_ingestion
    experimental_settings.txt
	task_{task_id}.predict
 
* The input directory input_dir (e.g. ../../public_data) contains several 
datasets formatted following this instructions: 
https://github.com/ihsaan-ullah/meta-album/tree/master/DataFormat

* The output directory output_dir (e.g. ../../scoring_output) will store the 
computed scores

AS A PARTICIPANT, DO NOT MODIFY THIS CODE.
"""
import os
import datetime
import jinja2
from sys import exit, version
from absl import app, flags

from cdmetadl.helpers.scoring_helpers import *
from cdmetadl.helpers.general_helpers import *
from cdmetadl.ingestion.image_dataset import create_datasets
from cdmetadl.ingestion.data_generator import CompetitionDataLoader

# =============================== BEGIN OPTIONS ===============================
FLAGS = flags.FLAGS

# Random seed
# Any int to be used as random seed for reproducibility
flags.DEFINE_integer("seed", 93, "Random seed.")

# Verbose mode 
# True: show various progression messages (recommended value)
# False: no progression messages are shown
flags.DEFINE_boolean("verbose", True, "Verbose mode.")

# Debug mode
# 0: no debug 
# 1: compute additional scores (recommended value)
# 2: same as 1 + show the Python version and list the directories 
flags.DEFINE_integer("debug_mode", 1, "Debug mode.")

# Private information
# True: the name of the datasets is kept private
# False: all the information is shown (recommended value)
flags.DEFINE_boolean("private_information", False, "Private information flag.")

# Overwrite results flag
# True: the previous output directory is overwritten
# False: the previous output directory is renamed (recommended value)
flags.DEFINE_boolean("overwrite_previous_results", False, 
    "Overwrite results flag.")

# Tesk tasks per dataset
# The total number of test tasks will be num_datasets x test_tasks_per_dataset
flags.DEFINE_integer("test_tasks_per_dataset", 100, 
    "Number of test tasks per dataset.")

# Default location of directories
# If no arguments to scoring.py are provided, these are the directories used. 
flags.DEFINE_string("input_data_dir", "../../public_data", "Path to the " 
    + "directory containing the meta_train and meta_test data.")
flags.DEFINE_string("results_dir","../../ingestion_output", 
    "Path to the output directory for the ingestion program.")
flags.DEFINE_string("output_dir_scoring", "../../scoring_output", 
    "Path to the ourput directory for the scoring program.")

# =============================================================================
# =========================== END USER OPTIONS ================================
# =============================================================================

# Program version
VERSION = 1.1

def scoring(argv) -> None:
    del argv
    
    SEED = FLAGS.seed
    VERBOSE = FLAGS.verbose
    DEBUG_MODE = FLAGS.debug_mode
    PRIVATE_INFORMATION = FLAGS.private_information
    OVERWRITE_PREVIOUS_RESULTS = FLAGS.overwrite_previous_results
    TEST_TASKS_PER_DATASET = FLAGS.test_tasks_per_dataset
    
    vprint(f"Scoring program version: {VERSION}", VERBOSE)
    vprint(f"Using random seed: {SEED}", VERBOSE)
    
    # Define the path to the directory
    results_dir = os.path.abspath(FLAGS.results_dir)
    ref_dir = os.path.abspath(FLAGS.input_data_dir) 
    output_dir = os.path.abspath(FLAGS.output_dir_scoring)
    
    # Show python version and directory structure
    if DEBUG_MODE > 1: 
        print(f"\nPython version: {version}")
        print(f"Using results_dir: {results_dir}")
        print(f"Using output_dir: {output_dir}")
        show_dir(".")
    
    vprint(f"\n{'#'*60}\n{'#'*18} Scoring program starts {'#'*18}\n{'#'*60}\n", 
        VERBOSE)
    
    # Check all the required directories
    vprint("\nChecking directories...", VERBOSE)
    exist_dir(results_dir)
    vprint("[+] Directories", VERBOSE)
    
    # Prepare test generator to access test tasks
    vprint("\nPreparing datasets info...", VERBOSE)
    _, _, test_datasets_info = prepare_datasets_information(ref_dir, 0, SEED, 
        VERBOSE, True)
    vprint("[+] Datasets info", VERBOSE)
    
    vprint("\nInitializing test generator...", VERBOSE)
    test_generator_config = {
        "N": None,
        "min_N": 2,
        "max_N": 20,
        "k": None,
        "min_k": 1,
        "max_k": 20,
        "query_images_per_class": 20
    }
    test_datasets = create_datasets(test_datasets_info)
    test_loader = CompetitionDataLoader(datasets=test_datasets, 
        episodes_config=test_generator_config, seed=SEED, 
        private_info=PRIVATE_INFORMATION, test_generator=True, verbose=VERBOSE)
    meta_test_generator = test_loader.generator
    vprint("[+] Data generator", VERBOSE)
    
    vprint("\nChecking ingestion output...", VERBOSE)
    result_files = os.listdir(results_dir)
    number_of_tasks = sum(".predict" in file for file in result_files)
    if number_of_tasks != len(test_datasets) * TEST_TASKS_PER_DATASET: 
        print(f"[-] There are no enough results in {results_dir}")
        exit(1)
    vprint("\n[+] Ingestion output", VERBOSE)
    
    # Compute scores
    vprint("\nComputing scores...", VERBOSE)
    # Compute the score for each task
    if DEBUG_MODE < 1:
        # Read metric and count the number of tasks
        curr_dir_path = os.path.dirname(os.path.realpath(__file__))
        score_file = os.path.join(curr_dir_path, "scores.txt")
        score_name, scoring_function = get_score(score_file)
        main_score = score_name
        vprint(f"\tUsing score: {score_name}", VERBOSE)
    else:
        main_score = "Normalized Accuracy"
        vprint(f"\tUsing scores: Accuracy, Macro F1 Score, Macro Precision, "
            + f"Macro Recall", VERBOSE)
        
    scores = dict()
    scores_per_dataset = dict()
    scores_per_ways = dict()
    scores_per_shots = dict()
    tasks = list()
    for i, task in enumerate(meta_test_generator(TEST_TASKS_PER_DATASET)):
        vprint(f"\tTask {i} started...", VERBOSE)
        
        # Extract task information
        y_true = task.query_set[1].numpy()
        task_dataset = task.dataset
        task_ways = task.num_ways
        task_shots = task.num_shots
        
        # Load ground truth and predicted labels
        task_name = f"{results_dir}/task_{i+1}"
        y_pred = read_results_file(f"{task_name}.predict")
        vprint("\t\t[+] Information loaded", VERBOSE)
    
        # Compute and store the scores
        if DEBUG_MODE < 1:
            if score_name == "Normalized Accuracy":
                task_scores = scoring_function(y_true, y_pred, task_ways)
            else:
                task_scores = scoring_function(y_true, y_pred)
            task_scores = {score_name: task_scores}
        else:
            task_scores = compute_all_scores(y_true, y_pred, task_ways)
        keys = list(task_scores.keys())
        vprint("\t\t[+] Score(s) computed", VERBOSE)
        
        if task_dataset not in scores_per_dataset:
            scores_per_dataset[task_dataset] = {key: list() for key in keys}
        
        if task_ways not in scores_per_ways:
            scores_per_ways[task_ways] = {key: list() for key in keys}
        
        if task_shots not in scores_per_shots:
            scores_per_shots[task_shots] = {key: list() for key in keys}
        
        for key in keys:
            if key not in scores:
                scores[key] = list()
            current_score = task_scores[key]
            scores[key].append(current_score)
            scores_per_dataset[task_dataset][key].append(current_score)
            scores_per_ways[task_ways][key].append(current_score)
            scores_per_shots[task_shots][key].append(current_score)
            
        tasks.append({
            "dataset": task_dataset,
            "num_ways": task_ways,
            "classes": list(task.original_class_idx),
            "num_shots": task_shots,
            "scores": [round(task_scores[key], 3) for key in keys],
        })
        vprint("\t\t[+] Score(s) stored", VERBOSE)
        vprint(f"\t[+] Task {i} processed", VERBOSE)   
    vprint("[+] Scores computed", VERBOSE)
    
    
    # Save scores
    vprint(f"\nSaving scores...", VERBOSE)

    # Create output dir
    if not OVERWRITE_PREVIOUS_RESULTS:
        timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        mvdir(output_dir, f"{output_dir}_{timestamp}") 
    mkdir(output_dir) 
    
    # Data for html report
    overall_scores = dict()
    scores_grouped_by_dataset = list()
    datasets_heatmaps = dict()
    scores_grouped_by_ways = list()
    ways_heatmaps = dict()
    scores_grouped_by_shots = list()
    shots_heatmaps = dict()
    try:
        plots_dir = os.path.join(output_dir, "plots")
        mkdir(plots_dir)
        with open(os.path.join(output_dir, "scores.txt"), "w") as score_file:
            scores_names = list(scores.keys())
            scores_names_to_save = [name.replace(' ', '_').lower() for name in 
                scores_names]
            
            # Overall scores
            for i, score_name in enumerate(scores_names):
                overall_score, overall_ci = mean_confidence_interval(scores[
                    score_name])
                if score_name == main_score:
                    score_file.write(f"overall_score: {overall_score}\n")
                else:
                    score_file.write(f"overall_{scores_names_to_save[i]}: "
                        + f"{overall_score}\n")
                vprint(f"\tOverall {score_name}: {overall_score:.3f} ± "
                    + f"{overall_ci:.3f}", VERBOSE)
                overall_histogram = create_histogram(scores[score_name], 
                    score_name, f"Overall Frequency Histogram ({score_name})", 
                    f"{plots_dir}/overall_histogram_{scores_names_to_save[i]}") 
                overall_scores[score_name] = {
                    "mean_score": round(overall_score, 3),
                    "ci": round(overall_ci, 3),
                    "histogram": overall_histogram
                }
                
            # Score per dataset     
            vprint("\n\tScores per dataset", VERBOSE)
            keys = sorted(scores_per_dataset.keys(), key = natural_sort)
            for i, dataset in enumerate(keys):
                dataset_info = {
                    "value": dataset,
                    "mean_score": list(),
                    "ci": list()
                }
                for j, score_name in enumerate(scores_names):
                    curr_info = scores_per_dataset[dataset][score_name]
                    score, conf_int = mean_confidence_interval(curr_info)
                    if score_name == main_score:
                        score_file.write(f"dataset_{i+1}: {score}\n")
                    else:
                        score_file.write(f"dataset_{i+1}_"+ 
                            f"{scores_names_to_save[j]}: {score}\n")
                    vprint(f"\t\t{dataset} ({score_name}): {score:.3f} ± "
                        + f"{conf_int:.3f}", VERBOSE)
                    dataset_info["mean_score"].append(round(score, 3))
                    dataset_info["ci"].append(round(conf_int, 3))
                scores_grouped_by_dataset.append(dataset_info)
            for i, score_name in enumerate(scores_names):
                datasets_heatmap = create_heatmap(scores_per_dataset, keys, 
                    keys, score_name, 
                    f"Frequency Heatmap per Dataset ({score_name})", 
                    f"{plots_dir}/heatmap_dataset_{scores_names_to_save[i]}")
                datasets_heatmaps[score_name] = datasets_heatmap
                
            # Score per number of ways      
            vprint("\n\tScores per number of ways", VERBOSE)
            keys = sorted(scores_per_ways.keys(), key = int)
            for n_ways in keys:
                ways_info = {
                    "value": n_ways,
                    "mean_score": list(),
                    "ci": list()
                }
                for j, score_name in enumerate(scores_names):
                    curr_info = scores_per_ways[n_ways][score_name]
                    score, conf_int = mean_confidence_interval(curr_info)
                    if j == 0:
                        ways_info["tasks"] = len(curr_info)
                    score_file.write(f"{n_ways}_ways_"
                        + f"{scores_names_to_save[j]}: {score}\n")
                    vprint(f"\t\t{n_ways}-ways ({score_name}): {score:.3f} ± "
                        + f"{conf_int:.3f}", VERBOSE)
                    ways_info["mean_score"].append(round(score, 3))
                    ways_info["ci"].append(round(conf_int, 3))
                scores_grouped_by_ways.append(ways_info)
            for i, score_name in enumerate(scores_names):    
                ways_heatmap = create_heatmap(scores_per_ways, keys, 
                    [f"{key}-ways" for key in keys], score_name, 
                    f"Frequency Heatmap per Number of Ways ({score_name})", 
                    f"{plots_dir}/heatmap_way_{scores_names_to_save[i]}")
                ways_heatmaps[score_name] = ways_heatmap
                
            # Score per number of shots        
            vprint("\n\tScores per number of shots", VERBOSE)
            keys = sorted(scores_per_shots.keys(), key = int)
            for k_shots in keys:
                shots_info = {
                    "value": k_shots,
                    "mean_score": list(),
                    "ci": list()
                }
                for j, score_name in enumerate(scores_names):
                    curr_info = scores_per_shots[k_shots][score_name]
                    score, conf_int = mean_confidence_interval(curr_info)
                    if j == 0:
                        shots_info["tasks"] = len(curr_info)
                    score_file.write(f"{k_shots}_shot_"
                        + f"{scores_names_to_save[j]}: {score}\n")
                    vprint(f"\t\t{k_shots}-shots ({score_name}): {score:.3f} ±"
                        + f" {conf_int:.3f}", VERBOSE)
                    shots_info["mean_score"].append(round(score, 3))
                    shots_info["ci"].append(round(conf_int, 3))
                scores_grouped_by_shots.append(shots_info)
            for i, score_name in enumerate(scores_names):    
                shots_heatmap = create_heatmap(scores_per_shots, keys, 
                    [f"{key}-shot" for key in keys], score_name, 
                    f"Frequency Heatmap per Number of Shots ({score_name})", 
                    f"{plots_dir}/heatmap_shot_{scores_names_to_save[i]}")
                shots_heatmaps[score_name] = shots_heatmap
                
            # Global metadata information
            metadata_file = os.path.join(results_dir, "metadata_ingestion")
            if os.path.exists(metadata_file):
                metadata = load_yaml(metadata_file)
            tasks_per_dataset = metadata["Tasks per dataset"]
            total_datasets = metadata["Number of test datasets"]
            score_file.write(f"duration: {metadata['Total execution time']}")
    except Exception as e:
        print("[-] Scores could not be saved")
        raise Exception(f"Error while saving scores. Detailed error:{repr(e)}")
    vprint(f"[+] Scores saved", VERBOSE)
    
    # Create HTML report
    vprint(f"\nCreating HTML report...", VERBOSE)
    try:
        subs = jinja2.Environment(
            loader=jinja2.FileSystemLoader(os.path.dirname(__file__))
        ).get_template("template.html").render(
            title="Results Report",
            scores_names=scores_names,
            overall_scores=overall_scores,
            tasks_per_dataset=tasks_per_dataset,
            total_datasets=total_datasets,
            scores_grouped_by_dataset=scores_grouped_by_dataset,
            datasets_heatmaps=datasets_heatmaps,
            scores_grouped_by_ways=scores_grouped_by_ways,
            ways_heatmaps=ways_heatmaps,
            scores_grouped_by_shots=scores_grouped_by_shots,
            shots_heatmaps=shots_heatmaps,
            tasks=tasks
        )

        html_file = os.path.join(output_dir, "detailed_results.html")
        with open(html_file, 'w', encoding="utf-8") as f: 
            f.write(subs)    
    except Exception as e:
        print(f"Error while creating HTML report. Detailed error: {repr(e)}")
    vprint(f"[+] HTML report created", VERBOSE)
    
    vprint(f"\n{'#'*60}\n{'#'*10} Scoring program finished successfully "
        + f"{'#'*11}\n{'#'*60}\n", VERBOSE)
    
    print("Your detailed results are available in this file: "
          + f"{FLAGS.output_dir_scoring}/detailed_results.html")


if __name__ == "__main__":
    app.run(scoring)
    