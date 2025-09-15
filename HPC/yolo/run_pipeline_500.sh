#!/bin/bash
#SBATCH --job-name=yolo_train_pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/users/lip24dg/ecg/HPC/logs_pipeline/%A_output.txt
#SBATCH --error=/users/lip24dg/ecg/HPC/logs_pipeline/%A_error.txt
#SBATCH --mail-user=dgarcia3@sheffield.ac.uk
#SBATCH --mail-type=FAIL,END

# DIAGNOSTICS
echo "========================================"
echo "YOLO Training Pipeline Job started on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "========================================"

# 1. VALIDATE SCRIPT ARGUMENT
BUCKET_TYPE=$1
if [ -z "$BUCKET_TYPE" ]; then
    echo "FATAL ERROR: No bucket type specified for training. Usage: sbatch script.sh <bucket_name>"
    exit 1
fi
echo "--- Processing generated data for bucket: ${BUCKET_TYPE}"

# SETUP
echo "Setting up the job environment..."
module load Anaconda3/2024.02-1
source activate yolo
echo "Conda environment 'yolo' activated."
mkdir -p /users/lip24dg/ecg/HPC/logs_pipeline

# CENTRALIZED AND DYNAMIC PATH CONFIGURATION
PROJECT_DIR="/users/lip24dg/ecg"
YOLO_SCRIPTS_DIR="${PROJECT_DIR}/ecg-yolo"
# BASE_OUTPUT_DIR="/mnt/parscratch/users/lip24dg/data/final_dataset_augmented"
BASE_OUTPUT_DIR="/mnt/parscratch/users/lip24dg/data/dataset"

# Dynamical paths based on the BUCKET_TYPE
# Input directory for JSON/PNG files
# CONVERSION_INPUT_DIR="${BASE_OUTPUT_DIR}/Generated_Images_${BUCKET_TYPE^}"
# Output directory for the generated YOLO label files (.txt)
# LABEL_OUTPUT_DIR="${BASE_OUTPUT_DIR}/yolo_labels_${BUCKET_TYPE}"
# Output directory for the final split dataset (train/valid/test)
# SPLIT_DATA_OUTPUT_DIR="${BASE_OUTPUT_DIR}/yolo_split_data_${BUCKET_TYPE}"

CONVERSION_INPUT_DIR="${BASE_OUTPUT_DIR}/Generated_Images"
# Output directory for the generated YOLO label files (.txt)
LABEL_OUTPUT_DIR="${BASE_OUTPUT_DIR}/yolo_labels"
# Output directory for the final split dataset (train/valid/test)
SPLIT_DATA_OUTPUT_DIR="${BASE_OUTPUT_DIR}/yolo_split_data"

# Print paths for easy debugging
echo "Source Data Directory : ${CONVERSION_INPUT_DIR}"
echo "YOLO Labels Directory : ${LABEL_OUTPUT_DIR}"
echo "Split Data Directory  : ${SPLIT_DATA_OUTPUT_DIR}"

# Ensure the source directory exists before proceeding
if [ ! -d "$CONVERSION_INPUT_DIR" ]; then
    echo "FATAL ERROR: Source data directory not found at ${CONVERSION_INPUT_DIR}"
    exit 1
fi

# PIPELINE EXECUTION

# Step 1: Convert JSON annotations to YOLO format for the specified bucket
echo "--- Step 1: Converting JSON to YOLO format for bucket '${BUCKET_TYPE}'"
python3 "${YOLO_SCRIPTS_DIR}/convert_to_yolo.py" \
    --data-dir "${CONVERSION_INPUT_DIR}" \
    --output-dir "${LABEL_OUTPUT_DIR}"

if [ $? -ne 0 ]; then
    echo "ERROR: Step 1 (convert_to_yolo) failed. Exiting."
    exit 1
fi

# Step 2: Split data into train/valid/test sets
echo "--- Step 2: Splitting data"
# Note: The label source for this step is the output from the previous step.
python3 "${YOLO_SCRIPTS_DIR}/split_data.py" \
    --image-source-dir "${CONVERSION_INPUT_DIR}" \
    --label-source-dir "${LABEL_OUTPUT_DIR}" \
    --output-dir "${SPLIT_DATA_OUTPUT_DIR}"

if [ $? -ne 0 ]; then
    echo "ERROR: Step 2 (split_data) failed. Exiting."
    exit 1
fi

# Step 3: Train the YOLOv8 model
echo "--- Step 3: Training the model"
python "${YOLO_SCRIPTS_DIR}/Train.py"
if [ $? -ne 0 ]; then
    echo "ERROR: Step 3 (training) failed. Exiting."
    exit 1
fi

# Step 4: Find latest model
echo "--- Finding the latest training run directory..."

# Find the highest number suffix from the directory names
# This handles both 'yolo_ecg_model' (no number) and 'yolo_ecg_model2', 'yolo_ecg_model10', etc.
LATEST_RUN_NUM=$(ls -d ${RUNS_DIR}/yolo_ecg_model* | grep -o '[0-9]*$' | sort -n | tail -1)

# Construct the name of the latest run directory
if [ -z "$LATEST_RUN_NUM" ]; then
    # This handles the case where the first run is just 'yolo_ecg_model'
    LATEST_RUN_DIR_NAME="yolo_ecg_model"
else
    LATEST_RUN_DIR_NAME="yolo_ecg_model${LATEST_RUN_NUM}"
fi

# Construct the full path to the best model from the latest run
BEST_MODEL_PATH="${RUNS_DIR}/${LATEST_RUN_DIR_NAME}/weights/best.pt"
VIS_OUTPUT_DIR="/users/lip24dg/data/yolo_runs/${LATEST_RUN_DIR_NAME}/test_predictions"
echo "Found latest model path: ${BEST_MODEL_PATH}"


# Step 5: Test the best trained model
# The training script saves the best model in a predictable path.
# We assume the run name in Train.py is 'yolo_ecg_model'.
echo "--- Step 5: Testing the best model"
if [ ! -f "${BEST_MODEL_PATH}" ]; then
   echo "ERROR: Could not find the trained model at ${BEST_MODEL_PATH}. Skipping test step."
else
   python "${YOLO_SCRIPTS_DIR}/Test.py" \
       --model-path "${BEST_MODEL_PATH}" \
       --image-dir "${TEST_IMAGES_DIR}" \
       --output-dir "${VIS_OUTPUT_DIR}" \
       --conf 0.5
   if [ $? -ne 0 ]; then
       echo "WARNING: Step 5 (testing) failed. Training was successful, but testing encountered an error."
   fi
fi

# Step 6: Evaluate standard metrics (P, R, mAP) per class
echo "--- Step 6: Evaluating standard per-class metrics"
if [ ! -f "${BEST_MODEL_PATH}" ]; then
    echo "ERROR: Model file not found at '${BEST_MODEL_PATH}'. Skipping evaluation."
else
    # Pass the Bash variable to the Python script as an argument
    python "${YOLO_SCRIPTS_DIR}/evaluate_model.py" --model-path "${BEST_MODEL_PATH}"
fi


# Step 7: Evaluate advanced IoU metrics per class
echo "--- Step 7: Evaluating advanced IoU per-class metrics"
if [ ! -f "${BEST_MODEL_PATH}" ]; then
    echo "ERROR: Model file not found at '${BEST_MODEL_PATH}'. Skipping IoU calculation."
else
    # Pass the same Bash variable to the other Python script
    python "${YOLO_SCRIPTS_DIR}/iou_calculation.py" --model-path "${BEST_MODEL_PATH}"
fi

# Step 8: Prepare Dataset for nnU-Net
echo "--- Step 8: Cropping detected leads to create nnU-Net dataset"

# Run the cropping script
python "${YOLO_SCRIPTS_DIR}/crop_leads_for_nnunet.py" \
    --model-path "${BEST_MODEL_PATH}" \
    --image-source-dir "${NNUNET_SOURCE_IMAGES_DIR}" \
    --output-dir "${NNUNET_CROPPED_OUTPUT_DIR}" \
    --conf 0.7 # Use a slightly higher confidence to ensure high-quality crops

if [ $? -ne 0 ]; then
    echo "WARNING: Step 8 (nnU-Net data preparation) failed."
fi