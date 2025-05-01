echo "============ init_env.sh Started at: $(date) ==============="

source ~/.bashrc
conda activate mlab

TASK_ENV=$1 # llm-merging
MODEL=$2 # o1-mini, model for implementing the idea
GPU_ID=$3 # 1
IDEA_PROPOSAL_MODEL=$4
OPTION=$5 # rag/0/1/...

MAX_HOURS=5
MAX_API_COST=10
MAX_STEPS=50
# ANCHORS=("ties" "dare" "emrmerging")

if [ "$OPTION" ]; then
	TASK=${TASK_ENV}--${OPTION}--${IDEA_PROPOSAL_MODEL}
else
	TASK=${TASK_ENV}
fi

MAX_TIME=$((${MAX_HOURS} * 60 * 60)) # how many seconds

BENCHMARK_DIR="MLAgentBench/benchmarks"

find . -name "__pycache__" -type d -exec rm -r {} +
# update with the latest base benchmark folder
if [ ! -d ${BENCHMARK_DIR}/${TASK} ]; then
	echo "Initializing task environment for ${TASK}"
	echo "Please wait before launching the next worker!"
	mkdir -p ${BENCHMARK_DIR}/${TASK}/env
	mkdir -p ${BENCHMARK_DIR}/${TASK}/scripts
	bash scripts/copy_read_only_files.sh MLAgentBench/benchmarks_base/${TASK_ENV}/scripts/read_only_files.txt MLAgentBench/benchmarks_base/${TASK_ENV}/env ${BENCHMARK_DIR}/${TASK}/env
	ln -s $(realpath MLAgentBench/benchmarks_base/${TASK_ENV}/scripts)/* ${BENCHMARK_DIR}/${TASK}/scripts
	cp -r MLAgentBench/benchmarks_base/${TASK_ENV}/env/* ${BENCHMARK_DIR}/${TASK}/env 2>/dev/null

	if [ "$OPTION" ]; then
		prompt_file="${BENCHMARK_DIR}/${TASK}/scripts/research_problem.txt"
		# make a hard copy for prompt_file
		cp --dereference $prompt_file $prompt_file.tmp && mv $prompt_file.tmp $prompt_file

		if [[ ${OPTION} == "rag" ]]; then
			IDEA_FILE="${BENCHMARK_DIR}/${TASK}/scripts/background.txt"
			if [ ! -f "$IDEA_FILE" ]; then
			    echo "Error: File '$IDEA_FILE' does not exist."
			    exit 1
			fi
			IDEA=$(cat ${IDEA_FILE})
		else
			# extract the idea from json
			IDEA_FILE="../CoI-Agent/results/${TASK_ENV}/${IDEA_PROPOSAL_MODEL}/${OPTION}/result.json"
			if [ ! -f "$IDEA_FILE" ]; then
			    echo "Error: File '$IDEA_FILE' does not exist."
			    exit 1
			fi
			IDEA=$(jq -r ".idea" "$IDEA_FILE")
		fi

		echo """

## Research Ideas
You will be provided with some research ideas on this problem from a machine learning expert. You’re encouraged to draw inspiration from these ideas when implementing your own method.

	${IDEA}
	""" >> $prompt_file
		echo "${IDEA}" > ${BENCHMARK_DIR}/${TASK}/env/idea.txt
	fi
	echo "environment initialization done!"
else
	echo "task environment exists for ${TASK}"
	echo "skip initializing and more workers can be launched!"
fi


