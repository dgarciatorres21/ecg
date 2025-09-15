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

# diagnostics
echo "========================================="
echo "YOLO Training Pipeline Job started on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "========================================="

# 1. validate script argument
BUCKET_TYPE=$1
if [ -z "$BUCKET_TYPE" ]; then
    echo "FATAL ERROR: No bucket type specified for training. Usage: sbatch script.sh <bucket_name>"
    exit 1
fi
echo "--- Processing generated data for bucket: ${BUCKET_TYPE}"

# setup
echo "Setting up the job environment..."
module load Anaconda3/2024.02-1
source activate yolo
echo "Conda environment 'yolo' activated."
mkdir -p /users/lip24dg/ecg/HPC/logs_pipeline

# centralized and dynamic path configuration
PROJECT_DIR="/users/lip24dg/ecg"
YOLO_SCRIPTS_DIR="${PROJECT_DIR}/ecg-yolo"
# base_output_dir="/mnt/parscratch/users/lip24dg/data/final_dataset_augmented"
BASE_OUTPUT_DIR="/mnt/parscratch/users/lip24dg/data/dataset"

# dynamical paths based on the bucket_type
# input directory for json/png files
# conversion_input_dir="${base_output_dir}/generated_images_${bucket_type^}"
# output directory for the generated yolo label files (.txt)
# label_output_dir="${base_output_dir}/yolo_labels_${bucket_type}"
# output directory for the final split dataset (train/valid/test)
# split_data_output_dir="${base_output_dir}/yolo_split_data_${bucket_type}"

CONVERSION_INPUT_DIR="${BASE_OUTPUT_DIR}/Generated_Images"
# output directory for the generated yolo label files (.txt)
LABEL_OUTPUT_DIR="${BASE_OUTPUT_DIR}/yolo_labels"
# output directory for the final split dataset (train/valid/test)
SPLIT_DATA_OUTPUT_DIR="${BASE_OUTPUT_DIR}/yolo_split_data"

# print paths for easy debugging
echo "Source Data Directory : ${CONVERSION_INPUT_DIR}"
echo "YOLO Labels Directory : ${LABEL_OUTPUT_DIR}"
echo "Split Data Directory  : ${SPLIT_DATA_OUTPUT_DIR}"

# ensure the source directory exists before proceeding
if [ ! -d "$CONVERSION_INPUT_DIR" ]; then
    echo "FATAL ERROR: Source data directory not found at ${CONVERSION_INPUT_DIR}"
    exit 1
fi

# pipeline execution

# step 1: convert json annotations to yolo format for the specified bucket
echo "--- Step 1: Converting JSON to YOLO format for bucket '${BUCKET_TYPE}'"
python3 "${YOLO_SCRIPTS_DIR}/convert_to_yolo.py" \
    --data-dir "${CONVERSION_INPUT_DIR}" \
    --output-dir "${LABEL_OUTPUT_DIR}"

if [ $? -ne 0 ]; then
    echo "ERROR: Step 1 (convert_to_yolo) failed. Exiting."
    exit 1
fi

# step 2: split data into train/valid/test sets
echo "--- Step 2: Splitting data"
# note: the label source for this step is the output from the previous step.
python3 "${YOLO_SCRIPTS_DIR}/split_data.py" \
    --image-source-dir "${CONVERSION_INPUT_DIR}" \
    --label-source-dir "${LABEL_OUTPUT_DIR}" \
    --output-dir "${SPLIT_DATA_OUTPUT_DIR}"

if [ $? -ne 0 ]; then
    echo "ERROR: Step 2 (split_data) failed. Exiting."
    exit 1
fi

# step 3: train the yolov8 model
echo "--- Step 3: Training the model"
python "${YOLO_SCRIPTS_DIR}/Train.py"
if [ $? -ne 0 ]; then
    echo "ERROR: Step 3 (training) failed. Exiting."
    exit 1
fi

# step 4: find latest model
echo "--- Finding the latest training run directory..."

# find the highest number suffix from the directory names
# this handles both 'yolo_ecg_model' (no number) and 'yolo_ecg_model2', 'yolo_ecg_model10', etc.
LATEST_RUN_NUM=$(ls -d ${RUNS_DIR}/yolo_ecg_model* | grep -o '[0-9]*$' | sort -n | tail -1)

# construct the name of the latest run directory
if [ -z "$LATEST_RUN_NUM" ]; then
    # this handles the case where the first run is just 'yolo_ecg_model'
    LATEST_RUN_DIR_NAME="yolo_ecg_model"
else
    LATEST_RUN_DIR_NAME="yolo_ecg_model${LATEST_RUN_NUM}"
fi

# construct the full path to the best model from the latest run
BEST_MODEL_PATH="${RUNS_DIR}/${LATEST_RUN_DIR_NAME}/weights/best.pt"
VIS_OUTPUT_DIR="/users/lip24dg/data/yolo_runs/${LATEST_RUN_DIR_NAME}/test_predictions"
echo "Found latest model path: ${BEST_MODEL_PATH}"


# step 5: test the best trained model
# the training script saves the best model in a predictable path.
# we assume the run name in train.py is 'yolo_ecg_model'.
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

# step 6: evaluate standard metrics (p, r, map) per class
echo "--- Step 6: Evaluating standard per-class metrics"
if [ ! -f "${BEST_MODEL_PATH}" ]; then
    echo "ERROR: Model file not found at '${BEST_MODEL_PATH}'. Skipping evaluation."
else
    # pass the bash variable to the python script as an argument
    python "${YOLO_SCRIPTS_DIR}/evaluate_model.py" --model-path "${BEST_MODEL_PATH}"
fi


# step 7: evaluate advanced iou metrics per class
echo "--- Step 7: Evaluating advanced IoU per-class metrics"
if [ ! -f "${BEST_MODEL_PATH}" ]; then
    echo "ERROR: Model file not found at '${BEST_MODEL_PATH}'. Skipping IoU calculation."
else
    # pass the same bash variable to the other python script
    python "${YOLO_SCRIPTS_DIR}/iou_calculation.py" --model-path "${BEST_MODEL_PATH}"
fi

# step 8: prepare dataset for nnu-net
echo "--- Step 8: Cropping detected leads to create nnU-Net dataset"

# run the cropping script
python "${YOLO_SCRIPTS_DIR}/crop_leads_for_nnunet.py" \
    --model-path "${BEST_MODEL_PATH}" \
    --image-source-dir "${NNUNET_SOURCE_IMAGES_DIR}" \
    --output-dir "${NNUNET_CROPPED_OUTPUT_DIR}" \
    --conf 0.7 # use a slightly higher confidence to ensure high-quality crops

if [ $? -ne 0 ]; then
    echo "WARNING: Step 8 (nnU-Net data preparation) failed."
fi
