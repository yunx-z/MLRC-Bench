# Setup

The MLAgentBench package can be installed with
```
pip install -e .
```

Install dependencies with python 3.10 by running 
```
bash install.sh
```

set `MLR_BENCH_DIR="/path/to/MLAgentBench/on/your/local/machine"` in `MLAgentBench/constants.py`

(Optional) For Kaggle datasets, you need to set up Kaggle API and authentication (~/.kaggle/kaggle.json) as described [here](https://www.kaggle.com/docs/api). You may also need to provide manual consent to the rules of specific competitions by following the prompts. 

# Tasks

Each task is a folder in `MLAgentBench/benchmarks_base/`, under which the `env/` folder contains files that the research agent will see at the beginning, and `script/` folder contains additional hidden files such as `prepare.py` for downloading data.

# Instructions for Refactoring:

Steps:
- Fork this github repo to your own github space.
- Complete steps in Setup Section for the MLAgentBench packages.
- Create a new task folder under `MLAgentBench/benchmarks/benchmarks_base`, following the [template](https://github.com/yunx-z/MLAgentBench/tree/main/MLAgentBench/benchmarks_base/base-competition). 
- Submit a pull request.

Here are the commands to test your newly added tasks:
```
# prepare conda environment and data
cd MLAgentBench/benchmarks_base/${TASK_NAME}/scripts/
conda env create -f environment.yml
conda activate ${TASK_NAME}
python prepare.py

# evaluate baseline method on validation set
cd ../env
python main.py -m my_method -p dev

# evaluate baseline method on test set
source config.sh
cp ../scripts/${TEST_FILE_NAME} data/ # prepare test data
cp ../scripts/test_constants.py constants.py # prepare test-time configuration
python main.py -m my_method -p test
```

add labels of dev/test set to ref/${TASK_NAME}. Don't put them under env/ folder otherwise LLM agents can "see" them.

specify `TEST_FILE_NAME="test_data_file_name_here"` in scripts/config.sh and put test_data_file under scripts/
