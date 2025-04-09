# MLRC-Bench: Can Language Agents Solve Machine Learning Research Challenges?
MLRC-Bench ([paper](https://drive.google.com/file/d/14ol9OVNm2d1SKdNQkLTztH6pbyhrHyW8/view?usp=sharing)) is a benchmark designed to quantify how effectively language agents can tackle challenging Machine Learning (ML) Research Competition problems by proposing and implementing novel ideas into code. 

![](main.pdf)

The first release of our benchmark includes 7 tasks adapted from recent Machine Learning conference competitions.

![](table.pdf)

# Setup


Create a conda enviroment with your TASK_NAME.

Then install the MLAgentBench package with
```
pip install -e .
pip install openai
```

Install dependencies with python 3.10 by running 
```
bash install.sh
```


(Optional) For Kaggle datasets, you need to set up Kaggle API and authentication (~/.kaggle/kaggle.json) as described [here](https://www.kaggle.com/docs/api). You may also need to provide manual consent to the rules of specific competitions by following the prompts. 

# Tasks

Each task is a folder in `MLAgentBench/benchmarks_base/`, under which the `env/` folder contains files that the research agent will see at the beginning, and `script/` folder contains additional hidden files such as `prepare.py` for downloading data.

# Instructions for Refactoring:

Steps:
- Fork this github repo to your own github space.
- Complete steps in Setup Section for the MLAgentBench packages.
- Create a new task folder under `MLAgentBench/benchmarks_base`, following the [template](https://github.com/yunx-z/MLAgentBench/tree/main/MLAgentBench/benchmarks_base/base-competition).
- add runtime and performance of your baseline method in `MLAgentBench/constants.py` (Repeat your run multiple times to ensure consistency; the score should remain relatively stable across runs.)
- Submit a pull request.

Here are the commands to test your newly added tasks:
```
# prepare conda environment and data
cd MLAgentBench/benchmarks_base/${TASK_NAME}/scripts/
conda env create -f environment.yml
conda activate ${TASK_NAME}
# We will install MLAgentBench and openai packages in the newly created conda environment
python prepare.py

# evaluate baseline method on validation set
cd ../env
python main.py -m my_method -p dev

# evaluate baseline method on test set
cp -r ../scripts/test_data/* data/ # prepare test data (updated)
cp ../scripts/test_constants.py constants.py # prepare test-time configuration
python main.py -m my_method -p test
```

Also if possible, please include a `background.txt`  file under scripts  folder with excerpt from relevant papers or technical reports written by competition participants (besides baseline paper) containing description and core code for relevant methods. See [this](https://github.com/yunx-z/MLAgentBench/blob/main/MLAgentBench/benchmarks_base/llm-merging/scripts/background.txt) for an example on llm-merging task. This info will be used to inspire LLM agents for better solutions.

The goal of refactored code is to achieve the following requirements:

Basically this command stays constant:
`python main.py -m my_method -p dev/test`
and then any code that could deal with evaluation metrics should be read_only and need to make sure read_only files don’t contain stuff that are necessary for training and that the agent could need to modify for their implementation.

Others:
- The LLM agent will be able to “see” all files under `env/` folder so make sure not to put any test-time information (including test data and model name used in test phases) there to avoid LLM agent “cheating”.
- Also put all test data under `scripts/test_data`
- Your code should not attempt to access internet. Any pretrained models, datasets should be downloaded beforehand by `prepare.py`.

# Acknowledgements
This repo is based on [MLAgentBench](https://github.com/snap-stanford/MLAgentBench). 
