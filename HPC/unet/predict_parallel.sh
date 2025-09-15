#!/bin/bash
#SBATCH --job-name=nnunet_predict_parallel
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=/users/lip24dg/ecg/HPC/logs_nnunet/predict/%x_%A_%a.out
#SBATCH --error=/users/lip24dg/ecg/HPC/logs_nnunet/predict/%x_%A_%a.err

set -e

# USAGE CHECK
if [ "$#" -ne 2 ]; then
    echo "ERROR: You must provide a model type AND a test set name."
    echo "Usage: sbatch predict_parallel.sh <MODEL_TYPE> <TEST_SET_NAME>"
    echo "Example: sbatch predict_parallel.sh LL test_clean"
    exit 1
fi

MODEL_TYPE=$1
TEST_SET_NAME=$2

# Dynamically update the job name
scontrol update jobid=${SLURM_JOB_ID} jobname=predict_${MODEL_TYPE}_${TEST_SET_NAME}

# SETUP
module load Anaconda3/2024.02-1
source activate unet

# CONFIGURATION
export nnUNet_raw="/mnt/parscratch/users/lip24dg/data/Generated_data"
export nnUNet_preprocessed="/mnt/parscratch/users/lip24dg/data/Generated_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/parscratch/users/lip24dg/data/Generated_data/nnUNet_results"


if [ "$MODEL_TYPE" == "12L" ]; then
    DATASET_ID=4
    DATASET_NAME="Dataset00${DATASET_ID}_ecg_12L"
    # Point to the 12L-specific test set directory
    TEST_SETS_BASE_DIR="/mnt/parscratch/users/lip24dg/data/Generated_data/structured_test_sets_12L"
else # This will be the "LL" model
    DATASET_ID=5
    DATASET_NAME="Dataset00${DATASET_ID}_13L"
    # Point to the LL-specific test set directory
    TEST_SETS_BASE_DIR="/mnt/parscratch/users/lip24dg/data/Generated_data/structured_test_sets_13L"
fi

# FOLDER DEFINITIONS
ORIGINAL_INPUT_DIR="${TEST_SETS_BASE_DIR}/${TEST_SET_NAME}/imagesTs"
FINAL_OUTPUT_DIR="${nnUNet_results}/${DATASET_NAME}/predictions/${TEST_SET_NAME}"

TEMP_BASE_DIR="/mnt/parscratch/users/lip24dg/data/temp_predict"
TEMP_INPUT_DIR="${TEMP_BASE_DIR}/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_input"
TEMP_OUTPUT_DIR="${TEMP_BASE_DIR}/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_output"

mkdir -p "${TEMP_INPUT_DIR}" "${TEMP_OUTPUT_DIR}" "${FINAL_OUTPUT_DIR}"

# SHARDING LOGIC
# First, check if the source directory actually exists
if [ ! -d "${ORIGINAL_INPUT_DIR}" ]; then
    echo "FATAL ERROR: Input directory not found: ${ORIGINAL_INPUT_DIR}"
    exit 1
fi

FILES=($(ls "${ORIGINAL_INPUT_DIR}"))
NUM_FILES=${#FILES[@]}
NUM_SHARDS=${SLURM_ARRAY_JOB_COUNT:-20}

shard_size=$(( (NUM_FILES + NUM_SHARDS - 1) / NUM_SHARDS ))
start_index=$(( SLURM_ARRAY_TASK_ID * shard_size ))
end_index=$(( start_index + shard_size - 1 ))

echo "This shard will process up to ${shard_size} files from index ${start_index}."
for (( i=start_index; i<=end_index && i<NUM_FILES; i++ )); do
    cp "${ORIGINAL_INPUT_DIR}/${FILES[i]}" "${TEMP_INPUT_DIR}/"
done

# PREDICTION ON THE SHARD
echo "--- Starting Prediction on Shard"
nnUNetv2_predict \
    -d ${DATASET_ID} \
    -i "${TEMP_INPUT_DIR}" \
    -o "${TEMP_OUTPUT_DIR}" \
    -c 2d \
    --save_probabilities

# MOVE RESULTS AND CLEAN UP
mv "${TEMP_OUTPUT_DIR}"/*.nii.gz "${FINAL_OUTPUT_DIR}/"
rm -rf "${TEMP_INPUT_DIR}" "${TEMP_OUTPUT_DIR}"

echo "--- Shard ${SLURM_ARRAY_TASK_ID} for ${MODEL_TYPE} on ${TEST_SET_NAME} finished successfully!"