#!/bin/bash
#SBATCH --job-name=yolo_train_pipeline_fast
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --mem=32G                  
#SBATCH --cpus-per-task=8       
#SBATCH --time=24:00:00  
#SBATCH --output=/users/lip24dg/ecg/HPC/logs_yolo/%A_output.txt
#SBATCH --error=/users/lip24dg/ecg/HPC/logs_yolo/%A_error.txt
#SBATCH --mail-user=dgarcia3@sheffield.ac.uk
#SBATCH --mail-type=FAIL,END

# --- diagnostics ---
echo "========================================="
echo "YOLO 12L Training Pipeline Job started on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "========================================="

# --- 1. validate script argument ---
BUCKET_TYPE=$1
if [ -z "$BUCKET_TYPE" ]; then
    echo "FATAL ERROR: No bucket type specified for training. Usage: sbatch script.sh <bucket_name>"
    exit 1
fi
echo "Processing 12L generated data for bucket: ${BUCKET_TYPE}"

# --- setup ---
echo "Setting up the job environment..."
module load Anaconda3/2024.02-1
source activate yolo
echo "Conda environment 'yolo' activated."
mkdir -p /users/lip24dg/ecg/HPC/logs_yolo

# --- path configuration ---
PROJECT_DIR="/users/lip24dg/ecg"
YOLO_SCRIPTS_DIR="${PROJECT_DIR}/ecg-yolo"
BASE_INPUT_DIR="/mnt/parscratch/users/lip24dg/data/final_dataset_augmented"
BASE_OUTPUT_DIR="/mnt/parscratch/users/lip24dg/data/final_dataset_augmented_12L"
RUNS_DIR="/users/lip24dg/ecg/ecg-yolo/runs_12L" # Path for 12L runs

CONVERSION_INPUT_DIR="${BASE_INPUT_DIR}/Generated_Images_${BUCKET_TYPE}"
LABEL_OUTPUT_DIR="${BASE_OUTPUT_DIR}/yolo_labels_${BUCKET_TYPE}"
SPLIT_DATA_OUTPUT_DIR="${BASE_OUTPUT_DIR}/yolo_split_data_${BUCKET_TYPE}"
TEST_IMAGES_DIR="${SPLIT_DATA_OUTPUT_DIR}/test/images"
NNUNET_CROPPED_OUTPUT_DIR="${BASE_OUTPUT_DIR}/nnunet_cropped_output_${BUCKET_TYPE}"

# --- print paths for easy debugging ---
echo "Source Data Directory : ${CONVERSION_INPUT_DIR}"
echo "YOLO Labels Directory : ${LABEL_OUTPUT_DIR}"
echo "Split Data Directory  : ${SPLIT_DATA_OUTPUT_DIR}"

if [ ! -d "$CONVERSION_INPUT_DIR" ]; then
    echo "FATAL ERROR: Source data directory not found at ${CONVERSION_INPUT_DIR}"
    exit 1
fi

# --- pipeline execution ---

echo "--- Step 1: Converting JSON to YOLO format for bucket '${BUCKET_TYPE}' ---"
python3 "${YOLO_SCRIPTS_DIR}/convert_to_yolo_12L.py" \
    --data-dir "${CONVERSION_INPUT_DIR}" \
    --output-dir "${LABEL_OUTPUT_DIR}"

if [ $? -ne 0 ]; then
    echo "ERROR: Step 1 (convert_to_yolo_12L) failed. Exiting."
    exit 1
fi

echo "--- Step 2: Splitting data ---"
python3 "${YOLO_SCRIPTS_DIR}/split_data.py" \
    --image-source-dir "${CONVERSION_INPUT_DIR}" \
    --label-source-dir "${LABEL_OUTPUT_DIR}" \
    --output-dir "${SPLIT_DATA_OUTPUT_DIR}"

if [ $? -ne 0 ]; then
    echo "ERROR: Step 2 (split_data) failed. Exiting."
    exit 1
fi

echo "--- Step 3: Training the 12L model ---"
python "${YOLO_SCRIPTS_DIR}/Train_12L.py"
if [ $? -ne 0 ]; then
    echo "ERROR: Step 3 (training) failed. Exiting."
    exit 1
fi

# --- POST-TRAINING STEPS ---

echo "--- Step 4: Finding the latest training run directory..."
LATEST_RUN_NUM=$(ls -d ${RUNS_DIR}/yolo_ecg_model_12L* | grep -o '[0-9]*
 | sort -n | tail -1)
if [ -z "$LATEST_RUN_NUM" ]; then
    LATEST_RUN_DIR_NAME="yolo_ecg_model_12L"
else
    LATEST_RUN_DIR_NAME="yolo_ecg_model_12L${LATEST_RUN_NUM}"
fi
BEST_MODEL_PATH="${RUNS_DIR}/${LATEST_RUN_DIR_NAME}/weights/best.pt"
VIS_OUTPUT_DIR="/users/lip24dg/data/yolo_runs_12L/${LATEST_RUN_DIR_NAME}/test_predictions_${BUCKET_TYPE}"
echo "Found latest model path: ${BEST_MODEL_PATH}"

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
       echo "WARNING: Step 5 (testing) failed."
   fi
fi

echo "--- Step 6: Evaluating standard per-class metrics"
if [ ! -f "${BEST_MODEL_PATH}" ]; then
    echo "ERROR: Model file not found at '${BEST_MODEL_PATH}'. Skipping evaluation."
else
    python "${YOLO_SCRIPTS_DIR}/evaluate_model.py" --model-path "${BEST_MODEL_PATH}"
fi

echo "--- Step 7: Evaluating advanced IoU per-class metrics"
if [ ! -f "${BEST_MODEL_PATH}" ]; then
    echo "ERROR: Model file not found at '${BEST_MODEL_PATH}'. Skipping IoU calculation."
else
    python "${YOLO_SCRIPTS_DIR}/iou_calculation.py" --model-path "${BEST_MODEL_PATH}"
fi

echo "--- Step 8: Cropping detected leads to create nnU-Net dataset"
if [ ! -f "${BEST_MODEL_PATH}" ]; then
    echo "ERROR: Model file not found at '${BEST_MODEL_PATH}'. Skipping nnU-Net preparation."
else
    python "${YOLO_SCRIPTS_DIR}/crop_leads_for_nnunet.py" \
        --model-path "${BEST_MODEL_PATH}" \
        --image-source-dir "${CONVERSION_INPUT_DIR}" \
        --output-dir "${NNUNET_CROPPED_OUTPUT_DIR}" \
        --conf 0.7
    if [ $? -ne 0 ]; then
        echo "WARNING: Step 8 (nnU-Net data preparation) failed."
    fi
fi

echo "========================================="
echo "Full 12L pipeline completed successfully for bucket: ${BUCKET_TYPE}"
echo "========================================="
