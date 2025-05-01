TASK=$1
echo """#!/bin/bash

#SBATCH --account=wangluxy1
#SBATCH --job-name=baseline-${TASK}       # Name of the job
#SBATCH --output=logs/gl/baseline-${TASK}-%j.log   # File to which the output will be written
#SBATCH --error=logs/gl/baseline-${TASK}-%j.log     # File to which the error will be written
#SBATCH --time=01-00:00:00           # Wall time limit of the job (e.g., 1 hour)
#SBATCH --partition=gpu           # Partition (or queue) name
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --ntasks=1                # Number of tasks, typically set to 1 for single GPU jobs
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --mem=46GB                 # Amount of memory per node (e.g., 16 GB)

echo \"My job ID is \$SLURM_JOB_ID\"
echo \"Running on host \$(hostname)\"
echo \"Starting at \$(date)\"

source ~/.bashrc
source /etc/profile.d/http_proxy.sh
export TQDM_DISABLE=1 # required by erasing_invisible_watermarks
module load gcc/14.1.0 # required by perception_temporal_action_loc

TASK=${TASK}
method_name=\"my_method\" # baseline name
BASE_DIR=MLAgentBench/benchmarks_base_exp/\${TASK}
EXP_DIR=\${BASE_DIR}/\${SLURM_JOB_ID}
PYTHON_PATH=\"/home/yunxiang/.conda/envs/\${TASK}/bin/python\"


mkdir -p \${BASE_DIR}/env/output
mkdir -p \${EXP_DIR}/env
mkdir -p \${EXP_DIR}/scripts
bash scripts/copy_read_only_files.sh MLAgentBench/benchmarks_base/\${TASK}/scripts/read_only_files.txt MLAgentBench/benchmarks_base/\${TASK}/env \${EXP_DIR}/env  
ln -s \$(realpath MLAgentBench/benchmarks_base/\${TASK}/scripts)/* \${EXP_DIR}/scripts
cp -r MLAgentBench/benchmarks_base/\${TASK}/* \${EXP_DIR} 2>/dev/null
cd \${EXP_DIR}/env

echo \"evaluating \${TASK} on dev set\"
\${PYTHON_PATH} main.py -m \${method_name} -p dev

# if [[ \"\${TASK}\" != \"machine_unlearning\" ]]; then
echo \"evaluating \${TASK} on test set\"
mkdir -p data/
# cp -r ../scripts/test_data/* data/
ln -s \$(realpath ../scripts/test_data)/* data/ 2>/dev/null
cp ../scripts/test_constants.py constants.py
# \${PYTHON_PATH} main.py -m \${method_name} -p test
# fi
# append idea_evals to env
cd - > /dev/null

cat \${EXP_DIR}/env/output/idea_evals.json
mv \${EXP_DIR}/env/output/idea_evals.json \${BASE_DIR}/env/output/\${SLURM_JOB_ID}-idea_evals.json
# rm -rf \${EXP_DIR}

echo \"Finished at \$(date)\"
""" > scripts/slurm.sh

sbatch scripts/slurm.sh
