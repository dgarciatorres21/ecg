#!/bin/bash
#SBATCH --job-name=nnunet_predict_cpu
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --array=0-49
#SBATCH --output=/users/lip24dg/ecg/HPC/logs_nnunet/predict_cpu/%x_%A_%a.out
#SBATCH --error=/users/lip24dg/ecg/HPC/logs_nnunet/predict_cpu/%x_%A_%a.err
#SBATCH --mail-user=dgarcia3@sheffield.ac.uk
#SBATCH --mail-type=FAIL,END

# This command ensures the script will exit immediately if any command fails
set -e

# USAGE CHECK
if [ "$#" -ne 2 ]; then
    echo "ERROR: You must provide a model type AND a test set name."
    echo "Usage: sbatch predict_parallel_cpu.sh <MODEL_TYPE> <TEST_SET_NAME>"
    echo "Example: sbatch predict_parallel_cpu.sh 12L test_clean"
    exit 1
fi

MODEL_TYPE=$1
TEST_SET_NAME=$2

# Dynamically update the job name for better tracking in squeue
scontrol update jobid=${SLURM_JOB_ID} jobname=predict_cpu_${MODEL_TYPE}_${TEST_SET_NAME}

# SETUP
echo "--- Setting up environment for Parallel CPU Prediction"
module load Anaconda3/2024.02-1
source activate unet

# CONFIGURATION
export nnUNet_raw="/mnt/parscratch/users/lip24dg/data/Generated_data"
export nnUNet_preprocessed="/mnt/parscratch/users/lip24dg/data/Generated_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/parscratch/users/lip24dg/data/Generated_data/nnUNet_results"

# CPU USAGE
export CUDA_VISIBLE_DEVICES=""
echo "CUDA_VISIBLE_DEVICES is set to empty, forcing CPU usage."

# Dynamically set paths and IDs based on the model type
if [ "${MODEL_TYPE}" == "12L" ]; then
    DATASET_ID=4
    DATASET_NAME="Dataset00${DATASET_ID}_ecg_12L"
    TEST_SETS_BASE_DIR="/mnt/parscratch/users/lip24dg/data/Generated_data/structured_test_sets_12L"
else # This will be the "LL" model
    DATASET_ID=5
    DATASET_NAME="Dataset00${DATASET_ID}_13L"
    TEST_SETS_BASE_DIR="/mnt/parscratch/users/lip24dg/data/Generated_data/structured_test_sets_13L"
fi

# FOLDER DEFINITIONS
ORIGINAL_INPUT_DIR="${TEST_SETS_BASE_DIR}/${TEST_SET_NAME}/imagesTs"
FINAL_OUTPUT_DIR="${nnUNet_results}/${DATASET_NAME}/predictions/${TEST_SET_NAME}"

# Create temporary, unique directories for this specific job shard
TEMP_BASE_DIR="/mnt/parscratch/users/lip24dg/data/temp_predict"
TEMP_INPUT_DIR="${TEMP_BASE_DIR}/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_input"
TEMP_OUTPUT_DIR="${TEMP_BASE_DIR}/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_output"

mkdir -p "${TEMP_INPUT_DIR}" "${TEMP_OUTPUT_DIR}" "${FINAL_OUTPUT_DIR}"

# SHARDING LOGIC
if [ ! -d "${ORIGINAL_INPUT_DIR}" ]; then
    echo "FATAL ERROR: Input directory not found: ${ORIGINAL_INPUT_DIR}"
    exit 1
fi

FILES=($(ls "${ORIGINAL_INPUT_DIR}"))
NUM_FILES=${#FILES[@]}
NUM_SHARDS=${SLURM_ARRAY_JOB_COUNT:-100}

# Ceiling division to calculate shard size
shard_size=$(( (NUM_FILES + NUM_SHARDS - 1) / NUM_SHARDS ))
start_index=$(( SLURM_ARRAY_TASK_ID * shard_size ))
end_index=$(( start_index + shard_size - 1 ))

echo "This shard will process up to ${shard_size} files from index ${start_index}."
for (( i=start_index; i<=end_index && i<NUM_FILES; i++ )); do
    # Use -L to dereference symbolic links if necessary, though cp is usually fine
    cp "${ORIGINAL_INPUT_DIR}/${FILES[i]}" "${TEMP_INPUT_DIR}/"
done

# PREDICTION ON THE SHARD
echo "--- Starting Prediction on Shard using CPU"
nnUNetv2_predict \
    -d ${DATASET_ID} \
    -i "${TEMP_INPUT_DIR}" \
    -o "${TEMP_OUTPUT_DIR}" \
    -c 2d \
    --save_probabilities \
    -device cpu

# MOVE RESULTS AND CLEAN UP
echo "--- Moving results and cleaning up temporary files"
# This moves only the final masks to the shared output folder
mv "${TEMP_OUTPUT_DIR}"/*.nii.gz "${FINAL_OUTPUT_DIR}/"
# Clean up all temporary files and folders for this shard
rm -rf "${TEMP_INPUT_DIR}" "${TEMP_OUTPUT_DIR}"

echo "--- CPU Shard ${SLURM_ARRAY_TASK_ID} for ${MODEL_TYPE} on ${TEST_SET_NAME} finished successfully!"
