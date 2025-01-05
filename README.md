# Instructions for Refactoring:

Students: create a new folder under `MLAgentBench/benchmarks/`

Commands to test your newly added tasks:

conda activate ${TASK_NAME}
cd MLAgentBench/benchmarks/${TASK_NAME}/scripts/
python prepare.py
cd ../env
python main.py -m my_method -p dev
python main.py -m my_method -p test

add labels of dev/test set to ref/${TASK_NAME}
set `export MLR_BENCH_DIR="/path/to/MLAgentBench"` in ~/.bashrc and `source ~/.bashrc`

specify `TEST_FILE_NAME="test_data_file_name_here"` in scripts/config.sh and put test_data_file under scripts/

add the runtime of the new tasks in `MLAgentBench/constants.py` (Yunxiang will do it)


# Setup

The MLAgentBench package can be installed with
```
pip install -e .
```

Install dependencies with python 3.10 by running 
```
bash install.sh
```

For Kaggle datasets, you need to set up Kaggle API and authentication (~/.kaggle/kaggle.json) as described [here](https://www.kaggle.com/docs/api). You may also need to provide manual consent to the rules of specific competitions by following the prompts. 

# Tasks

Each task is a folder in `MLAgentBench/benchmarks/`, under which the `env/` folder contains files that the research agent will see at the beginning, and `script/` folder contains additional hidden files such as `prepare.py` for downloading data.
