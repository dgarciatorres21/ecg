#!/bin/bash
#SBATCH --job-name=evaluate_official
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00  # each job is now much faster
#SBATCH --array=0-3      # launch 4 jobs, with IDs 0, 1, 2, 3
#SBATCH --output=/users/lip24dg/ecg/HPC/logs_nnunet/evaluate_official_%x_%A_%a.out
#SBATCH --error=/users/lip24dg/ecg/HPC/logs_nnunet/evaluate_official_%x_%A_%a.err

if [ -z "$1" ] || ( [ "$1" != "12L" ] && [ "$1" != "LL" ] ); then
    echo "ERROR: You must provide a valid model type as an argument."
    echo "Usage: sbatch evaluate_official_parallel.sh 12L"
    echo "   or: sbatch evaluate_official_parallel.sh LL"
    exit 1
fi

MODEL_TYPE=$1

# dynamically update the job name
scontrol update jobid=${SLURM_JOB_ID} jobname=evaluate_${MODEL_TYPE}

# setup
module load Anaconda3/2024.02-1
source activate unet

# configuration
export nnUNet_raw="/mnt/parscratch/users/lip24dg/data/Generated_data"
export nnUNet_preprocessed="/mnt/parscratch/users/lip24dg/data/Generated_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/parscratch/users/lip24dg/data/Generated_data/nnUNet_results"

# dynamically set paths based on the model type
if [ "$MODEL_TYPE" == "12L" ]; then
    DATASET_ID=7
    DATASET_NAME="Dataset00${DATASET_ID}_ecg_12L"
    TEST_SETS_BASE_DIR="/mnt/parscratch/users/lip24dg/data/Generated_data/structured_test_sets_12L"
else # ll model
    DATASET_ID=8
    DATASET_NAME="Dataset00${DATASET_ID}_LL"
    TEST_SETS_BASE_DIR="/mnt/parscratch/users/lip24dg/data/Generated_data/structured_test_sets_LL"
fi

# map the slurm array task id to the test set name (parallelization)
TEST_SETS_ARRAY=("test_clean" "test_scanner" "test_physical" "test_chaos")
CURRENT_TEST_SET=${TEST_SETS_ARRAY[$SLURM_ARRAY_TASK_ID]}

if [ -z "$CURRENT_TEST_SET" ]; then
    echo "ERROR: Invalid SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}. Exiting."
    exit 1
fi

# construct paths for this specific job
PRED_DIR="${nnUNet_results}/${DATASET_NAME}/predictions/${CURRENT_TEST_SET}"
GT_DIR="${TEST_SETS_BASE_DIR}/${CURRENT_TEST_SET}/labelsTs"
OUTPUT_JSON="${nnUNet_results}/${DATASET_NAME}/evaluation/summary_${CURRENT_TEST_SET}.json"
DATASET_JSON_FILE="$nnUNet_raw/${DATASET_NAME}/dataset.json"
PLANS_JSON_FILE="$nnUNet_results/${DATASET_NAME}/nnUNetTrainer__nnUNetPlans__2d/plans.json"

# evaluation for this job's assigned test set
echo "--- Starting Evaluation for ${MODEL_TYPE} on Test Set: ${CURRENT_TEST_SET}"
mkdir -p "$(dirname "$OUTPUT_JSON")"

nnUNetv2_evaluate_folder \
    -o "${OUTPUT_JSON}" \
    -djfile "${DATASET_JSON_FILE}" \
    -pfile "${PLANS_JSON_FILE}" \
    "${GT_DIR}" \
    "${PRED_DIR}"

echo "--- Evaluation for ${CURRENT_TEST_SET} complete."
