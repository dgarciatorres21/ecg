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
echo "YOLO Training Pipeline Job started on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "========================================="

# --- setup ---
echo "Setting up the job environment..."
module load Anaconda3/2024.02-1
source activate yolo
echo "Conda environment 'yolo' activated."
mkdir -p /users/lip24dg/ecg/HPC/logs_yolo

PROJECT_DIR="/users/lip24dg/ecg"
YOLO_SCRIPTS_DIR="${PROJECT_DIR}/ecg-yolo"
BASE_INPUT_DIR="/mnt/parscratch/users/lip24dg/data/final_dataset_augmented"
BASE_OUTPUT_DIR="/mnt/parscratch/users/lip24dg/data/final_dataset_augmented_12L"

# --- dynamical paths based on the bucket_type ---
# input directory for json/png files
# conversion_input_dir="${base_input_dir}/generated_images_${bucket_type}"
# output directory for the generated yolo label files (.txt)
LABEL_OUTPUT_DIR="${BASE_OUTPUT_DIR}/yolo_labels_${BUCKET_TYPE}"
# # output directory for the final split dataset (train/valid/test)
# split_data_output_dir="${base_output_dir}/yolo_split_data_${bucket_type}"

# --- print paths for easy debugging ---
# echo "source data directory : ${conversion_input_dir}"
# echo "yolo labels directory : ${label_output_dir}"
# echo "split data directory  : ${split_data_output_dir}"

# if [ ! -d "$conversion_input_dir" ]; then
#     echo "fatal error: source data directory not found at ${conversion_input_dir}"
#     exit 1
# fi

# --- pipeline execution ---

# # step 1: convert json annotations to yolo format for the specified bucket
# echo "--- step 1: converting json to yolo format for bucket '${bucket_type}' ---"
# python3 "${yolo_scripts_dir}/convert_to_yolo_12l.py" \
#     --data-dir "${conversion_input_dir}" \
#     --output-dir "${label_output_dir}"

# if [ $? -ne 0 ]; then
#     echo "error: step 1 (convert_to_yolo) failed. exiting."
#     exit 1
# fi

# # step 2: split data into train/valid/test sets
# echo "--- step 2: splitting data ---"
# # note: the label source for this step is the output from the previous step.
# python3 "${yolo_scripts_dir}/split_data.py" \
#     --image-source-dir "${conversion_input_dir}" \
#     --label-source-dir "${label_output_dir}" \
#     --output-dir "${split_data_output_dir}"

# if [ $? -ne 0 ]; then
#     echo "error: step 2 (split_data) failed. exiting."
#     exit 1
# fi

# echo "========================================="
# echo "pipeline completed successfully for bucket: ${bucket_type}"
# echo "========================================="

echo "========================================="
echo "Start training"
echo "========================================="

# step 3: train the yolov8 model
echo "--- Step 3: Training the model ---"
python "${YOLO_SCRIPTS_DIR}/Train_12L.py"
if [ $? -ne 0 ]; then
    echo "ERROR: Step 3 (training) failed. Exiting."
    exit 1
fi

echo "========================================="
echo "Finish training"
echo "========================================="
