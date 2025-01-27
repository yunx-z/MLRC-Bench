## Neurips 2023 Machine Unlearning Competition

## Dataset Attribution
This project uses CIFAR-10 dataset for training and validation, loaded from torchvision.datasets.CIFAR10.

## Code Attribution
For the dev phase, the code is based on:
https://www.kaggle.com/code/mgorinova/machine-unlearning-evaluation-on-cifar-10

For the test phase, the code is based on:
https://www.kaggle.com/code/eleni30fillou/run-unlearn-finetune

## Setup
You need to manually accept the competition terms and conditions on kaggle.\\
To be able to use kaggle API, follow the steps and place it in your home directory, you could also place it in 'scripts/.kaggle/kaggle.json' and prepare.py will load it from there.

```bash
# Create conda environment
cd MLAgentBench/MLAgentBench/benchmarks_base/machine_unlearning/scripts/
conda env create -f environment.yml
conda init
source ~/.bashrc
conda activate machine_unlearning

# Run prepare.py
python prepare.py

# Install AgentBench requirements
cd ../../../..
pip install -e .
bash install.sh

# Run main.py
cd MLAgentBench/benchmarks_base/machine_unlearning/env/
python main.py -m my_method -p dev

# Run evaluation on test set
cp -r ../scripts/test_data/* data/
python main.py -m my_method -p test

# Deactivate and delete conda environment
conda deactivate
conda env remove -n machine_unlearning
conda clean --all
```
## Results:
* Time taken to train and validate = 542.0315222740173, .05367534933511104, 505.67512345314026
* Score on validation = 0.053832025891711176, 505.4854168891907, 0.05515877249475864
* Time taken on test = Test time is based on kaggle submission time.
* Score on test =  0.0588217216, 0.0639509074, 0.0605025745 (Score on original baseline code: 0.0583642010)
**Note**: Training and testing was ran thrice. Results in MLAgentBench/constants.py are the average of the three runs.
