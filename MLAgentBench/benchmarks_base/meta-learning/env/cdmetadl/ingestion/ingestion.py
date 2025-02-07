""" This is the ingestion program written by the organizers. This program also
runs on the Cross-Domain MetaDL competition platform to test your code.

Usage: 

python -m cdmetadl.ingestion.ingestion \
    --input_data_dir=input_dir \
    --output_dir_ingestion=output_dir \
    --submission_dir=submission_dir

* The input directory input_dir (e.g. ../../public_data) contains several 
datasets formatted following this instructions: 
https://github.com/ihsaan-ullah/meta-album/tree/master/DataFormat

* The output directory output_dir (e.g. ../../ingestion_output) will store 
the predicted labels of the meta-testing phase, the metadata, the meta-trained 
learner, and the logs:
    logs/                   <- Directory containing all the meta-training logs
    model/                  <- Meta-trained learner (Output of Learner.save())
    metadata_ingestion      <- Metadata from the ingestion program
	task_{task_id}.predict  <- Predicter probabilities for each meta-test task

* The code directory submission_dir (e.g. ../../baselines/random) must 
contain your code submission model.py, it can also contain other helpers and
configuration files.

AS A PARTICIPANT, DO NOT MODIFY THIS CODE.
"""
import os
import time 
import datetime
import numpy as np
from sys import path, exit, version
from absl import app, flags
from torch.utils.data import DataLoader

from cdmetadl.helpers.ingestion_helpers import *
from cdmetadl.helpers.general_helpers import *
from cdmetadl.ingestion.image_dataset import ImageDataset, create_datasets
from cdmetadl.ingestion.data_generator import CompetitionDataLoader
from cdmetadl.ingestion.competition_logger import Logger


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
# 1: run the code normally, but limits the time to MAX_TIME (recommended value)
# 2: same as 1 + list the directories 
# 3: same as 2, but it checks if PyTorch recognizes the GPU (Only for debug)
flags.DEFINE_integer("debug_mode", 0, "Debug mode.")

# Image size
# Int specifying the image size for all generators (recommended value 128)
flags.DEFINE_integer("image_size", 128, "Image size.")

# Overwrite results flag
# True: the previous output directory is overwritten
# False: the previous output directory is renamed (recommended value)
flags.DEFINE_boolean("overwrite_previous_results", False, 
    "Overwrite results flag.")

# Time budget
# Maximum time in seconds PER TESTING TASK
flags.DEFINE_integer("max_time", 1000, "Max time in seconds per test task.") 

# Tesk tasks per dataset
# The total number of test tasks will be num_datasets x test_tasks_per_dataset
flags.DEFINE_integer("test_tasks_per_dataset", 100,
    "Number of test tasks per dataset.")

# Default location of directories
# If no arguments to ingestion.py are provided, these are the directories used. 
flags.DEFINE_string("input_data_dir", "../../public_data", "Path to the " 
    + "directory containing the meta_train and meta_test data.")
flags.DEFINE_string("output_dir_ingestion","../../ingestion_output", 
    "Path to the output directory for the ingestion program.")
flags.DEFINE_string("submission_dir", "../../baselines/random",
    "Path to the directory containing the solution to use.")

# =============================================================================
# =========================== END USER OPTIONS ================================
# =============================================================================

# Program version
VERSION = 1.1

def ingestion(argv) -> None:
    total_time_start = time.time() 
    
    del argv
    
    SEED = FLAGS.seed
    VERBOSE = FLAGS.verbose
    DEBUG_MODE = FLAGS.debug_mode
    IMG_SIZE = FLAGS.image_size
    OVERWRITE_PREVIOUS_RESULTS = FLAGS.overwrite_previous_results
    MAX_TIME = FLAGS.max_time
    TEST_TASKS_PER_DATASET = FLAGS.test_tasks_per_dataset

    vprint(f"Ingestion program version: {VERSION}", VERBOSE)
    vprint(f"Using random seed: {SEED}", VERBOSE)
    
    # Define the path to the directories
    input_dir = os.path.abspath(FLAGS.input_data_dir)
    output_dir = os.path.abspath(FLAGS.output_dir_ingestion)
    submission_dir = os.path.abspath(FLAGS.submission_dir)

    # Show python version and directory structure
    # print(f"\nPython version: {version}")
    # print("\n\n")
    # os.system("nvidia-smi")
    # print("\n\n")
    # os.system("nvcc --version")
    # print("\n\n")
    # os.system("pip list")
    # print("\n\n")
    
    if DEBUG_MODE >= 2: 
        print(f"Using input_dir: {input_dir}")
        print(f"Using output_dir: {output_dir}")
        print(f"Using submission_dir: {submission_dir}")
        show_dir(".")
    
    if DEBUG_MODE == 3:
        join_list = lambda info: "\n".join(info)
        gpu_settings = ["\n----- GPU settings -----"]
        gpu_info = get_torch_gpu_environment()
        gpu_settings.extend(gpu_info)
        print(join_list(gpu_settings))
        exit(0)
        
    # Import your model
    path.insert(1, submission_dir)
    try:
        from model import MyMetaLearner, MyLearner
    except:
        print(f"MyMetaLearner and MyLearner not found in {submission_dir}"
            + "/model.py")
        exit(1)        
        
    vprint(f"\n{'#'*60}\n{'#'*17} Ingestion program starts {'#'*17}\n{'#'*60}", 
        VERBOSE)
    
    # Check all the required directories
    vprint("\nChecking directories...", VERBOSE)
    exist_dir(input_dir)
    exist_dir(submission_dir)
    vprint("[+] Directories", VERBOSE)

    # Define the configuration for the generators
    vprint("\nDefining generators config..", VERBOSE)
    train_data_format = "task"
    batch_size = 16
    validation_datasets = None
    
    train_generator_config = {
        "N": 5,
        "min_N": None,
        "max_N": None,
        "k": None,
        "min_k": 1,
        "max_k": 20,
        "query_images_per_class": 20
    }
 
    valid_generator_config = {
        "N": None,
        "min_N": 2,
        "max_N": 20,
        "k": None,
        "min_k": 1,
        "max_k": 20,
        "query_images_per_class": 20
    }
    
    test_generator_config = {
        "N": None,
        "min_N": 2,
        "max_N": 20,
        "k": None,
        "min_k": 1,
        "max_k": 20,
        "query_images_per_class": 20
    }
    
    config_file = os.path.join(submission_dir, "config.json")
    if os.path.exists(config_file):
        user_config = load_json(config_file)
        if "train_data_format" in user_config:
            train_data_format = user_config["train_data_format"]
        if "batch_size" in user_config:
            batch_size = user_config["batch_size"]
            if batch_size is not None and batch_size < 1:
                print("[-] Defining generators config: batch_size cannot be " +
                      f"less than 1. Received: {batch_size}")
                exit(1)
        if "validation_datasets" in user_config:
            validation_datasets = user_config["validation_datasets"]
            if validation_datasets is not None and validation_datasets > 9:
                print("[-] Defining generators config: validation_datasets " +
                      "cannot be greater than 9. Received: "
                      + f"{validation_datasets}")
                exit(1)
        if "train_config" in user_config:
            train_generator_config.update(user_config["train_config"])
        if "valid_config" in user_config:
            valid_generator_config.update(user_config["valid_config"])
    
    vprint("[+] Generators config", VERBOSE)
    
    vprint("\nPreparing datasets info...", VERBOSE)
    (train_datasets_info, valid_datasets_info, 
     test_datasets_info) = prepare_datasets_information(input_dir, 
        validation_datasets, SEED, VERBOSE)
    vprint("[+] Datasets info", VERBOSE)
    
    # Initialize genetators
    vprint("\nInitializing data generators...", VERBOSE)
    # Train generator
    if train_data_format == "task":
        train_datasets = create_datasets(train_datasets_info, IMG_SIZE)
        train_loader = CompetitionDataLoader(datasets=train_datasets, 
            episodes_config=train_generator_config, seed=SEED, verbose=VERBOSE)
        meta_train_generator = train_loader.generator
        train_classes = train_generator_config["N"]
        total_classes = sum([len(dataset.idx_per_label) for dataset in 
            train_datasets])
    else:
        g = torch.Generator()
        g.manual_seed(SEED)
        train_dataset = ImageDataset(train_datasets_info, IMG_SIZE)
        meta_train_generator = lambda batches: iter(cycle(batches, 
            DataLoader(dataset=train_dataset, batch_size=batch_size, 
            shuffle=True, num_workers=2, generator=g)))
        train_classes = len(train_dataset.idx_per_label)
        total_classes = train_classes
    vprint("\t[+] Train generator", VERBOSE)
    
    # Valid generator
    if len(valid_datasets_info) > 0:
        valid_datasets = create_datasets(valid_datasets_info, IMG_SIZE)
        valid_loader = CompetitionDataLoader(datasets=valid_datasets, 
            episodes_config=valid_generator_config, seed=SEED, verbose=VERBOSE)
        meta_valid_generator = valid_loader.generator
        vprint("\t[+] Valid generator", VERBOSE)
    else:
        meta_valid_generator = None
        vprint("\t[!] Valid generator is None", VERBOSE)

    # Test generator
    test_datasets = create_datasets(test_datasets_info, IMG_SIZE)
    test_loader = CompetitionDataLoader(datasets=test_datasets, 
        episodes_config=test_generator_config, seed=SEED, test_generator=True, 
        verbose=VERBOSE)
    meta_test_generator = test_loader.generator
    vprint("\t[+] Test generator", VERBOSE)
    
    vprint("[+] Data generators", VERBOSE)
    
    # Create output dir
    if not OVERWRITE_PREVIOUS_RESULTS:
        timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        mvdir(output_dir, f"{output_dir}_{timestamp}") 
    mkdir(output_dir) 
    
    # Create logs dir and initialize logger
    logs_dir = f"{output_dir}/logs"
    mkdir(logs_dir)
    logger = Logger(logs_dir)

    # Create output model dir
    model_dir = f"{output_dir}/model"
    mkdir(model_dir)
    
    # Save/print experimental settings
    join_list = lambda info: "\n".join(info)
    
    gpu_settings = ["\n----- GPU settings -----"]
    gpu_info = get_torch_gpu_environment()
    gpu_settings.extend(gpu_info)
    
    data_settings = [
        "\n----- Data settings -----",
        f"# Train datasets: {len(train_datasets_info)}",
        f"# Validation datasets: {len(valid_datasets_info)}",
        f"# Test datasets: {len(test_datasets_info)}",
        f"Image size: {IMG_SIZE}",
        f"Random seed: {SEED}"
    ]
    
    if train_data_format == "task":
        train_settings = [
            "\n----- Train settings -----",
            f"Train data format: {train_data_format}",
            f"N-way: {train_loader.n_way}",
            f"Minimum ways: {train_loader.min_ways}",
            f"Maximum ways: {train_loader.max_ways}",
            f"k-shot: {train_loader.k_shot}",
            f"Minimum shots: {train_loader.min_shots}",
            f"Maximum shots: {train_loader.max_shots}",
            f"Query size: {train_loader.query_size}"
        ]
    else:
        train_settings = [
            "\n----- Train settings -----",
            f"Train data format: {train_data_format}",
            f"Batch size: {batch_size}"
        ]
    
    if len(valid_datasets_info) > 0:
        validation_settings = [
            "\n----- Validation settings -----",
            f"N-way: {valid_loader.n_way}",
            f"Minimum ways: {valid_loader.min_ways}",
            f"Maximum ways: {valid_loader.max_ways}",
            f"k-shot: {valid_loader.k_shot}",
            f"Minimum shots: {valid_loader.min_shots}",
            f"Maximum shots: {valid_loader.max_shots}",
            f"Query size: {valid_loader.query_size}"
        ]
         
    test_settings = [
        "\n----- Test settings -----",
        f"N-way: {test_loader.n_way}",
        f"Minimum ways: {test_loader.min_ways}",
        f"Maximum ways: {test_loader.max_ways}",
        f"k-shot: {test_loader.k_shot}",
        f"Minimum shots: {test_loader.min_shots}",
        f"Maximum shots: {test_loader.max_shots}",
        f"Query size: {test_loader.query_size}"
    ]
    
    if len(valid_datasets_info) > 0:
        all_settings = [
            f"\n{'*'*9} Experimental settings {'*'*9}",
            join_list(gpu_settings),
            join_list(data_settings),
            join_list(train_settings),
            join_list(validation_settings),
            join_list(test_settings),
            f"\n{'*'*41}"
        ]
    else:
        all_settings = [
            f"\n{'*'*9} Experimental settings {'*'*9}",
            join_list(gpu_settings),
            join_list(data_settings),
            join_list(train_settings),
            join_list(test_settings),
            f"\n{'*'*41}"
        ]

    experimental_settings = join_list(all_settings)
    vprint(experimental_settings, VERBOSE)
    experimental_settings_file = f"{logs_dir}/experimental_settings.txt"
    with open(experimental_settings_file, "w") as f:
        f.writelines(experimental_settings)

    # Meta-train
    vprint("\nMeta-training your meta-learner...", VERBOSE)
    meta_training_start = time.time()
    meta_learner = MyMetaLearner(train_classes, total_classes, logger)
    learner = meta_learner.meta_fit(meta_train_generator, meta_valid_generator)
    meta_training_time = time.time() - meta_training_start
    learner.save(model_dir)
    vprint("[+] Meta-learner meta-trained", VERBOSE)
    
    # Meta-test
    vprint("\nMeta-testing your learner...", VERBOSE)
    meta_testing_start = time.time()
    for i, task in enumerate(meta_test_generator(TEST_TASKS_PER_DATASET)):
        vprint(f"\tTask {i+1} started...", VERBOSE)
        learner = MyLearner() 
        learner.load(model_dir)   

        support_set = (task.support_set[0], task.support_set[1], 
            task.support_set[2], task.num_ways, task.num_shots)
        
        task_start = time.time()
        predictor = learner.fit(support_set)
        vprint("\t\t[+] Learner trained", VERBOSE)
        
        y_pred = predictor.predict(task.query_set[0])
        vprint("\t\t[+] Labels predicted", VERBOSE)
        
        task_time = time.time() - task_start
        if DEBUG_MODE >= 1:
            if task_time > MAX_TIME:
                print(f"\t\t[-] Task {i+1} exceeded the maximum time allowed. "
                    + f"Max time {MAX_TIME}, execution time {task_time}")
                exit(1)
        
        file_name = f"{output_dir}/task_{i+1}"
        fmt = "%f" if len(y_pred.shape) == 2 else "%d"
        np.savetxt(f"{file_name}.predict", y_pred, fmt=fmt)
        vprint("\t\t[+] Predictions saved", VERBOSE)
        
        vprint(f"\t[+] Task {i+1} finished in {task_time} seconds", VERBOSE)   
    vprint("[+] Learner meta-tested", VERBOSE)
    meta_testing_time = time.time() - meta_testing_start
    
    total_time = time.time() - total_time_start
    with open(f"{output_dir}/metadata_ingestion", "w") as global_metadata_file:
        global_metadata_file.write(f"Total execution time: {total_time}\n")
        global_metadata_file.write(f"Meta-train time: {meta_training_time}\n")
        global_metadata_file.write(f"Meta-test time: {meta_testing_time}\n")
        global_metadata_file.write("Number of test datasets: "
            + f"{len(test_datasets_info)}\n")
        global_metadata_file.write("Tasks per dataset: "
            + f"{TEST_TASKS_PER_DATASET}")
    vprint(f"\nOverall time spent: {total_time} seconds", VERBOSE)
        
    vprint(f"\n{'#'*60}\n{'#'*9} Ingestion program finished successfully "
        + f"{'#'*10}\n{'#'*60}", VERBOSE)


if __name__ == "__main__":	
    app.run(ingestion)
