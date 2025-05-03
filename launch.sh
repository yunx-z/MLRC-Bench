echo "============ launch.sh Started at: $(date) ==============="

source ~/.bashrc
conda activate mlab

TASK_ENV=$1 # llm-merging
MODEL=$2 # o1-mini, model for implementing the idea
GPU_ID=$3 # 1
IDEA_PROPOSAL_MODEL=$4
OPTION=$5 # rag/0/1/...

if [[ "${TASK_ENV}" == "weather_forcast" ]]; then
	MAX_HOURS=10
	MAX_API_COST=20
	MAX_STEPS=100
else
	MAX_HOURS=5
	MAX_API_COST=10
	MAX_STEPS=50
fi

# ANCHORS=("ties" "dare" "emrmerging")
# PYTHON_PATH="/home/yunxiang/.conda/envs/${TASK_ENV}/bin/python"
PYTHON_PATH="/opt/anaconda3/envs/mlab"

if [ "$OPTION" ]; then
	TASK=${TASK_ENV}--${OPTION}--${IDEA_PROPOSAL_MODEL}
else
	TASK=${TASK_ENV}
fi
echo "${MODEL} is implementing the ${OPTION}-th idea on ${TASK_ENV} proposed by ${IDEA_PROPOSAL_MODEL}"

MAX_TIME=$((${MAX_HOURS} * 60 * 60)) # how many seconds
echo "max steps: ${MAX_STEPS}; max run time: ${MAX_TIME} seconds; max api cost: ${MAX_API_COST}"

# TASK_ENV=$(echo "${TASK}" | awk -F'--' '{print $1}') # split by '--' to extract TASK name
RUN_ID=$(date +"%m%d%H%M%S")_$$
ERROR_LOG_DIR="logs/error/${TASK}/${MODEL}"
LOG_DIR="logs/${TASK}/${MODEL}/${RUN_ID}"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/run.log"
WORKSPACE_DIR="workspace/${TASK}/${MODEL}/${RUN_ID}"
BENCHMARK_DIR="MLAgentBench/benchmarks_base"

if [ ! -d ${BENCHMARK_DIR}/${TASK} ]; then
	echo "${TASK} env does not exists! please run scripts/init_env.sh first"
	echo "============ launch.sh Exited at: $(date) ==============="
	exit 1
fi
# remove __pycache__ directories
find ${BENCHMARK_DIR}/${TASK}/env/ -name "__pycache__" -type d -exec rm -r {} + 2>/dev/null

# --feedback-llm-name this llm is used for generating post-hoc explanation (summary) of the method implemented by code
echo ${LOG_FILE}
nohup python -u -m MLAgentBench.runner \
	--python ${PYTHON_PATH} \
	--task ${TASK} \
	--device ${GPU_ID} \
	--log-dir ${LOG_DIR} \
	--work-dir ${WORKSPACE_DIR} \
	--llm-name ${MODEL} \
	--edit-script-llm-name ${MODEL} \
	--feedback-llm-name "o1" \
	--fast-llm-name ${MODEL} \
	--max-api-cost ${MAX_API_COST} \
	--agent-max-steps ${MAX_STEPS} \
	--max-time ${MAX_TIME} \
	> ${LOG_FILE} 2>&1  
#	--agent-type "Agent" \

cleanup() {
	rm -rf ${WORKSPACE_DIR} 
	# remove big files from traces (including checkpoints!)
	find "${LOG_DIR}/env_log/traces" -type f -size +10M -exec rm {} \; 2>/dev/null

	if [[ "${TASK_ENV}" == "meta-learning" ]]; then 
		rm -rf ${LOG_DIR}/env_log/traces/step_*_files/ingestion_output
	elif [[ "${TASK_ENV}" == "erasing_invisible_watermarks" || "${TASK_ENV}" == "product-recommendation" ]]; then
		rm -rf ${LOG_DIR}/env_log/traces/step_*_files/output/
	fi

	bash scripts/copy_idea_eval.sh "${LOG_DIR}"
}

error_file="${LOG_DIR}/env_log/error.txt"
if [ -f "$error_file" ]; then
   echo "Error found in trial ${LOG_DIR}"
   echo "-----------------------------"
   cat "$error_file"
   echo "-----------------------------"
   echo "Moving error directory from ${LOG_DIR} to ${ERROR_LOG_DIR}"
   mkdir -p ${ERROR_LOG_DIR}
   mv ${LOG_DIR} ${ERROR_LOG_DIR} 
   cleanup
   echo "============ launch.sh Exited at: $(date) ==============="
   exit 1
fi


# select the best-performing method/snapshot based on dev and evaluate on test
IDEA_EVAL_FILE="${WORKSPACE_DIR}/${TASK}/output/idea_evals.json"
# Check if the JSON file exists
if [ ! -f "$IDEA_EVAL_FILE" ]; then
    cleanup
    echo "Error: File '$IDEA_EVAL_FILE' does not exist."
    echo "============ launch.sh Exited at: $(date) ==============="
    exit 1
fi

read -r step method_name < <(jq -r '
  .implementations
  | map(select(.phase == "dev"))
  | max_by(.performance) 
  | if . then "\(.step) \(.method_name)" else "null null" end
' "$IDEA_EVAL_FILE")

# Check if a valid implementation was found
if [[ "$step" == "null" || "$method_name" == "null" ]]; then
  cleanup
  echo "No implementation with phase 'dev' found in '$IDEA_EVAL_FILE'."
  echo "============ launch.sh Exited at: $(date) ==============="
  exit 1
fi

cp ${IDEA_EVAL_FILE} ${LOG_DIR}/env_log/ # dev evals
if [[ "${TASK_ENV}" != "machine_unlearning" ]]; then
	# TODO: for machine-unlearning, skip below and manually go back to login node for kaggle submission since compute nodes are not allowed access to public internet
	SNAPSHOT_DIR="${LOG_DIR}/env_log/traces/step_${step}_files"
	# prep for test
	mkdir -p ${SNAPSHOT_DIR}/data/ # some benchmark folder has empty data/, so data/ dir does not exists in snapshot
	# cp -r ${BENCHMARK_DIR}/${TASK}/scripts/test_data/* ${SNAPSHOT_DIR}/data/
	ln -s $(realpath ${BENCHMARK_DIR}/${TASK}/scripts/test_data)/* ${SNAPSHOT_DIR}/data/ 2>/dev/null
	# first copy read-only files
	bash scripts/copy_read_only_files.sh ${BENCHMARK_DIR}/${TASK}/scripts/read_only_files.txt ${BENCHMARK_DIR}/${TASK}/env ${SNAPSHOT_DIR}
	# then cover test_constants.py. This needs to be done AFTER copy_read_only_files!
	cp ${BENCHMARK_DIR}/${TASK}/scripts/test_constants.py ${SNAPSHOT_DIR}/constants.py

	cd ${SNAPSHOT_DIR}
	echo "evaluating snapshot with best dev performance on test set: ${SNAPSHOT_DIR}"
	CUDA_VISIBLE_DEVICES=${GPU_ID} nohup ${PYTHON_PATH} main.py -m ${method_name} -p test > ${method_name}-test.log 2>&1

	cp output/idea_evals.json ../../test_idea_evals.json # test evals
	# clean up
	cd - > /dev/null
	bash scripts/remove_read_only_files.sh ${BENCHMARK_DIR}/${TASK}/scripts/read_only_files.txt ${SNAPSHOT_DIR}
fi

cleanup
echo "============ launch.sh Finished at: $(date) ==============="
