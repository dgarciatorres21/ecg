#!/bin/bash
#SBATCH --job-name=ecg_crop_12L
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-29
#SBATCH --time=04:00:00
#SBATCH --output=/users/lip24dg/ecg/HPC/logs_crop/%A_%a_12L_out.txt
#SBATCH --error=/users/lip24dg/ecg/HPC/logs_crop/%A_%a_12L_err.txt
#SBATCH --mail-user=dgarcia3@sheffield.ac.uk
#SBATCH --mail-type=FAIL,END

# diagnostics
echo "========================================="
echo "12-Lead Cropping Job started on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID}, Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "========================================="

# setup
echo "Setting up the job environment..."
module load Anaconda3/2024.02-1
source activate /users/lip24dg/.conda/envs/ecg
echo "Conda environment 'ecg' activated."
echo "-----------------------------------------"

# centralized path configuration
SCRIPT_DIR="/users/lip24dg/ecg/ecg-image-generator"
DATA_SOURCE_DIR="/mnt/parscratch/users/lip24dg/data/1.0.3/records500"

BASE_INPUT_DIR="/mnt/parscratch/users/lip24dg/data/final_dataset_augmented/"
BASE_OUTPUT_DIR="/mnt/parscratch/users/lip24dg/data/final_dataset_augmented_12L"
LOG_DIR="/users/lip24dg/ecg/HPC/logs_crop"

# options: "Clean", "Scanner", "Physical", "Chaos" 
TARGET_BUCKET="Chaos" 

# derived paths will now correctly use the new base_output_dir
GENERATED_IMAGE_DIR="${BASE_INPUT_DIR}/Generated_Images_${TARGET_BUCKET}"
GENERATED_MASK_DIR="${BASE_INPUT_DIR}/Generated_Masks_${TARGET_BUCKET}"
CROPPED_IMAGE_DIR="${BASE_OUTPUT_DIR}/Cropped_Images_${TARGET_BUCKET}"
CROPPED_MASK_DIR="${BASE_OUTPUT_DIR}/Cropped_Masks_${TARGET_BUCKET}"

mkdir -p $CROPPED_IMAGE_DIR
mkdir -p $CROPPED_MASK_DIR
mkdir -p $LOG_DIR

# stage 1: audit source data (runs only on the first job)
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo "--- STAGE 1: AUDITING SOURCE DATA (Job 0 Only)"
    python3 "${SCRIPT_DIR}/audit_wfdb_records.py" --directory "$DATA_SOURCE_DIR"
fi

# stage 2: crop leads from images and masks
echo "--- STAGE 2: CROPPING LEADS FOR BUCKET: ${TARGET_BUCKET}"

MODEL_PATH="/users/lip24dg/ecg/ecg-yolo/runs_12L/yolo_ecg_model_12L3/weights/best.pt"

if [ ! -f "$MODEL_PATH" ]; then
    echo "FATAL ERROR: 12-Lead YOLO model not found at $MODEL_PATH. Halting cropping."
else
    echo "Using 12-Lead model found at: $MODEL_PATH"
    TOTAL_JOBS=30 # must match #sbatch --array
    
    python3 "${SCRIPT_DIR}/crop_leads.py" \
        --model-path "$MODEL_PATH" \
        --image-source-dir "$GENERATED_IMAGE_DIR" \
        --image-output-dir "$CROPPED_IMAGE_DIR" \
        --mask-source-dir "$GENERATED_MASK_DIR" \
        --mask-output-dir "$CROPPED_MASK_DIR" \
        --job-id ${SLURM_ARRAY_TASK_ID} \
        --total-jobs ${TOTAL_JOBS} \
        --conf 0.6
fi

echo "-----------------------------------------"
echo "Job finished with exit code $?"
echo "Job finished at $(date)"
echo "========================================="
