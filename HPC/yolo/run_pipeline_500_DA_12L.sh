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

# --- DIAGNOSTICS ---
echo "========================================"
echo "YOLO Training Pipeline Job started on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "========================================"

# --- SETUP ---
echo "Setting up the job environment..."
module load Anaconda3/2024.02-1
source activate yolo
echo "Conda environment 'yolo' activated."
mkdir -p /users/lip24dg/ecg/HPC/logs_yolo

PROJECT_DIR="/users/lip24dg/ecg"
YOLO_SCRIPTS_DIR="${PROJECT_DIR}/ecg-yolo"
BASE_INPUT_DIR="/mnt/parscratch/users/lip24dg/data/final_dataset_augmented"
BASE_OUTPUT_DIR="/mnt/parscratch/users/lip24dg/data/final_dataset_augmented_12L"

# --- Dynamical paths based on the BUCKET_TYPE ---
# Input directory for JSON/PNG files
# CONVERSION_INPUT_DIR="${BASE_INPUT_DIR}/Generated_Images_${BUCKET_TYPE}"
# Output directory for the generated YOLO label files (.txt)
LABEL_OUTPUT_DIR="${BASE_OUTPUT_DIR}/yolo_labels_${BUCKET_TYPE}"
# # Output directory for the final split dataset (train/valid/test)
# SPLIT_DATA_OUTPUT_DIR="${BASE_OUTPUT_DIR}/yolo_split_data_${BUCKET_TYPE}"

# --- Print paths for easy debugging ---
# echo "Source Data Directory : ${CONVERSION_INPUT_DIR}"
echo "YOLO Labels Directory : ${LABEL_OUTPUT_DIR}"
# echo "Split Data Directory  : ${SPLIT_DATA_OUTPUT_DIR}"

# if [ ! -d "$CONVERSION_INPUT_DIR" ]; then
#     echo "FATAL ERROR: Source data directory not found at ${CONVERSION_INPUT_DIR}"
#     exit 1
# fi

# --- PIPELINE EXECUTION ---

# # Step 1: Convert JSON annotations to YOLO format for the specified bucket
# echo "--- Step 1: Converting JSON to YOLO format for bucket '${BUCKET_TYPE}' ---"
# python3 "${YOLO_SCRIPTS_DIR}/convert_to_yolo_12L.py" \
#     --data-dir "${CONVERSION_INPUT_DIR}" \
#     --output-dir "${LABEL_OUTPUT_DIR}"

# if [ $? -ne 0 ]; then
#     echo "ERROR: Step 1 (convert_to_yolo) failed. Exiting."
#     exit 1
# fi

# # Step 2: Split data into train/valid/test sets
# echo "--- Step 2: Splitting data ---"
# # Note: The label source for this step is the output from the previous step.
# python3 "${YOLO_SCRIPTS_DIR}/split_data.py" \
#     --image-source-dir "${CONVERSION_INPUT_DIR}" \
#     --label-source-dir "${LABEL_OUTPUT_DIR}" \
#     --output-dir "${SPLIT_DATA_OUTPUT_DIR}"

# if [ $? -ne 0 ]; then
#     echo "ERROR: Step 2 (split_data) failed. Exiting."
#     exit 1
# fi

# echo "========================================"
# echo "Pipeline completed successfully for bucket: ${BUCKET_TYPE}"
# echo "========================================"

echo "========================================"
echo "Start training"
echo "========================================"

# Step 3: Train the YOLOv8 model
echo "--- Step 3: Training the model ---"
python "${YOLO_SCRIPTS_DIR}/Train_12L.py"
if [ $? -ne 0 ]; then
    echo "ERROR: Step 3 (training) failed. Exiting."
    exit 1
fi

echo "========================================"
echo "Finish training"
echo "========================================"