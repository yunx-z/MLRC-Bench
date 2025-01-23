""" Combine the ingestion and scoring processes. 

Usage: 
    python -m cdmetadl.run \
        --input_data_dir=../public_data \
        --submission_dir=../baselines/random \
        --overwrite_previous_results=True \
        --test_tasks_per_dataset=10
    
AS A PARTICIPANT, DO NOT MODIFY THIS CODE. 
"""

from shlex import split
from subprocess import call

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 93, "Random seed.")

flags.DEFINE_boolean("verbose", True, "Verbose mode.")

flags.DEFINE_integer("debug_mode", 1, "Debug mode.")

flags.DEFINE_integer("image_size", 128, "Image size.")

flags.DEFINE_integer("max_time", 1000, 
    "Maximum time in seconds per testing task.")

flags.DEFINE_boolean("overwrite_previous_results", False, 
    "Overwrite results flag.")

flags.DEFINE_integer("test_tasks_per_dataset", 100, 
    "Number of test tasks per dataset.")

flags.DEFINE_boolean("private_information", False, "Private information flag.")

flags.DEFINE_string("input_data_dir", "../public_data", "Path to the directory" 
    + " containing the meta_train and meta_test data.")

flags.DEFINE_string("output_dir_ingestion","../ingestion_output", 
    "Path to the output directory for the ingestion program.")

flags.DEFINE_string("submission_dir", "../baselines/random", 
    "Path to the directory containing the solution to use.")

flags.DEFINE_string("output_dir_scoring", "../scoring_output", 
    "Path to the ourput directory for the scoring program.")


def main(argv) -> None:
    """ Runs the ingestion and scoring programs sequentially, as they are 
    handled in CodaLab.
    """
    del argv
    
    seed = FLAGS.seed
    verbose = FLAGS.verbose
    debug_mode = FLAGS.debug_mode
    image_size = FLAGS.image_size
    max_time = FLAGS.max_time
    overwrite_previous_results = FLAGS.overwrite_previous_results
    test_tasks_per_dataset = FLAGS.test_tasks_per_dataset
    private_information = FLAGS.private_information
    input_data_dir = FLAGS.input_data_dir
    output_dir_ingestion = FLAGS.output_dir_ingestion
    submission_dir = FLAGS.submission_dir
    output_dir_scoring = FLAGS.output_dir_scoring
    
    command_ingestion = "python -m cdmetadl.ingestion.ingestion " \
        + f"--seed={seed} " \
        + f"--verbose={verbose} " \
        + f"--debug_mode={debug_mode} " \
        + f"--image_size={image_size} " \
        + f"--overwrite_previous_results={overwrite_previous_results} " \
        + f"--max_time={max_time} " \
        + f"--test_tasks_per_dataset={test_tasks_per_dataset} " \
        + f"--input_data_dir={input_data_dir} " \
        + f"--output_dir_ingestion={output_dir_ingestion} " \
        + f"--submission_dir={submission_dir}"

    command_scoring = "python -m cdmetadl.scoring.scoring " \
        + f"--seed={seed} " \
        + f"--verbose={verbose} " \
        + f"--debug_mode={debug_mode} " \
        + f"--private_information={private_information} " \
        + f"--overwrite_previous_results={overwrite_previous_results} " \
        + f"--test_tasks_per_dataset={test_tasks_per_dataset} " \
        + f"--input_data_dir={input_data_dir} " \
        + f"--results_dir={output_dir_ingestion} " \
        + f"--output_dir_scoring={output_dir_scoring}"
        
    cmd_ing = split(command_ingestion)
    cmd_sco = split(command_scoring)
    
    call(cmd_ing)
    call(cmd_sco)


if __name__ == "__main__":
    app.run(main)
