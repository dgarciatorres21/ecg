#!/bin/bash
#SBATCH --job-name=evaluate_per_lead
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=18:00:00
#SBATCH --array=0-3 # Launch 4 parallel jobs for clean, scanner, physical, chaos
#SBATCH --output=/users/lip24dg/ecg/HPC/logs_nnunet/eval_per_lead_%x_%A_%a.out
#SBATCH --error=/users/lip24dg/ecg/HPC/logs_nnunet/eval_per_lead_%x_%A_%a.err

if [ -z "$1" ] || ( [ "$1" != "12L" ] && [ "$1" != "LL" ] ); then
    echo "ERROR: You must provide a valid model type as an argument."
    echo "Usage: sbatch evaluate_per_lead.sh 12L"
    exit 1
fi

MODEL_TYPE=$1
scontrol update jobid=${SLURM_JOB_ID} jobname=eval_per_lead_${MODEL_TYPE}

# SETUP
module load Anaconda3/2024.02-1
source activate unet

# CONFIGURATION
export nnUNet_raw="/mnt/parscratch/users/lip24dg/data/Generated_data"
export nnUNet_results="/mnt/parscratch/users/lip24dg/data/Generated_data/nnUNet_results"

# Dynamically set paths based on model type
if [ "$MODEL_TYPE" == "12L" ]; then
    DATASET_ID=7
    DATASET_NAME="Dataset00${DATASET_ID}_ecg_12L"
    TEST_SETS_BASE_DIR="/mnt/parscratch/users/lip24dg/data/Generated_data/structured_test_sets_12L"
else # LL model
    DATASET_ID=8
    DATASET_NAME="Dataset00${DATASET_ID}_LL"
    TEST_SETS_BASE_DIR="/mnt/parscratch/users/lip24dg/data/Generated_data/structured_test_sets_LL"
fi

# Map Array ID to Test Set
TEST_SETS_ARRAY=("test_clean" "test_scanner" "test_physical" "test_chaos")
CURRENT_TEST_SET=${TEST_SETS_ARRAY[$SLURM_ARRAY_TASK_ID]}

# Construct paths for this specific job
PRED_DIR="${nnUNet_results}/${DATASET_NAME}/predictions/${CURRENT_TEST_SET}"
ORIGINAL_GT_DIR="${TEST_SETS_BASE_DIR}/${CURRENT_TEST_SET}/labelsTs"
ORIGINAL_IMG_DIR="${TEST_SETS_BASE_DIR}/${CURRENT_TEST_SET}/imagesTs"
CSV_OUT="${PRED_DIR}/evaluation_per_lead_partial.csv"

TEMP_GT_DIR="${PRED_DIR}/temp_synced_labelsTs"
TEMP_IMG_DIR="${PRED_DIR}/temp_synced_imagesTs"

# STAGE 1: Synchronize the data folders
echo "--- Stage 1: Synchronizing temp folders for ${CURRENT_TEST_SET}"
rm -rf "${TEMP_GT_DIR}" "${TEMP_IMG_DIR}"
mkdir -p "${TEMP_GT_DIR}" "${TEMP_IMG_DIR}"

if [ ! -d "${PRED_DIR}" ]; then echo "ERROR: Prediction dir not found: ${PRED_DIR}"; exit 1; fi

predicted_files=$(ls "${PRED_DIR}" | grep ".nii.gz")
echo "Found $(echo "$predicted_files" | wc -l) prediction files. Synchronizing..."

for pred_file in $predicted_files; do
    original_img_file=$(echo "$pred_file" | sed 's/\.nii\.gz$/_0000.nii.gz/')
    gt_path="${ORIGINAL_GT_DIR}/${pred_file}"
    img_path="${ORIGINAL_IMG_DIR}/${original_img_file}"
    if [ -f "$gt_path" ]; then cp "$gt_path" "${TEMP_GT_DIR}/"; fi
    if [ -f "$img_path" ]; then cp "$img_path" "${TEMP_IMG_DIR}/"; fi
done
echo "Synchronization complete."

# STAGE 2: Run Evaluation using the synchronized folders
echo -e "\n--- Stage 2: Evaluating predictions for ${CURRENT_TEST_SET}"
PYTHON_SCRIPT_PATH="/users/lip24dg/ecg/code-unet/evaluate_per_lead.py"

python "${PYTHON_SCRIPT_PATH}" \
    --pred_dir "${PRED_DIR}" \
    --gt_dir "${TEMP_GT_DIR}" \
    --img_dir "${TEMP_IMG_DIR}" \
    --csv_out "${CSV_OUT}"

# STAGE 3: Clean up
echo -e "\n--- Stage 3: Cleaning up temporary directories"
rm -rf "${TEMP_GT_DIR}" "${TEMP_IMG_DIR}"
echo "Cleanup complete."