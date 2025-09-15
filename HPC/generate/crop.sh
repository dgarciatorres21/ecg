#!/bin/bash
#SBATCH --job-name=ecg_crop
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-29
#SBATCH --time=04:00:00
#SBATCH --output=/users/lip24dg/ecg/HPC/logs_crop/%A_%a_12L_out.txt
#SBATCH --error=/users/lip24dg/ecg/HPC/logs_crop/%A_%a_12L_err.txt
#SBATCH --mail-user=dgarcia3@sheffield.ac.uk
#SBATCH --mail-type=FAIL,END

# DIAGNOSTICS
echo "========================================"
echo "12-Lead Cropping Job started on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID}, Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "========================================"

# SETUP
echo "Setting up the job environment..."
module load Anaconda3/2024.02-1
source activate /users/lip24dg/.conda/envs/ecg
echo "Conda environment 'ecg' activated."
echo "-----------------------------------------"

# CENTRALIZED PATH CONFIGURATION
SCRIPT_DIR="/users/lip24dg/ecg/ecg-image-generator"
DATA_SOURCE_DIR="/mnt/parscratch/users/lip24dg/data/1.0.3/records500"

BASE_INPUT_DIR="/mnt/parscratch/users/lip24dg/data/final_dataset_augmented"
# BASE_INPUT_DIR="/mnt/parscratch/users/lip24dg/data/dataset/"
BASE_OUTPUT_DIR="/mnt/parscratch/users/lip24dg/data/final_dataset_augmented"
LOG_DIR="/users/lip24dg/ecg/HPC/logs_crop"

# Target Bucket
# Options: "Clean", "Scanner", "Physical", "Chaos"
if [ -z "$1" ]; then
    echo "FATAL ERROR: No target bucket provided."
    echo "Usage: sbatch $0 [Clean|Scanner|Physical|Chaos]"
    exit 1
fi

TARGET_BUCKET="$1"
echo "--- Target bucket set to: ${TARGET_BUCKET} ---"

# Derived paths will now correctly use the new BASE_OUTPUT_DIR
GENERATED_IMAGE_DIR="${BASE_INPUT_DIR}/Generated_Images_${TARGET_BUCKET}"
GENERATED_MASK_DIR="${BASE_INPUT_DIR}/Generated_Masks_${TARGET_BUCKET}"
CROPPED_IMAGE_DIR="${BASE_OUTPUT_DIR}/Cropped_Images_${TARGET_BUCKET}"
CROPPED_MASK_DIR="${BASE_OUTPUT_DIR}/Cropped_Masks_${TARGET_BUCKET}"
SYNC_FILE="${BASE_OUTPUT_DIR}/.setup_done_${TARGET_BUCKET}"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    # This is Job 0: The Setup Task
    echo "This is Job 0: Performing one-time directory cleanup for bucket: ${TARGET_BUCKET}"
    # First, remove any old sync file from a previous run
    rm -f "$SYNC_FILE"

    DIRECTORIES_TO_RESET=(
        "$CROPPED_IMAGE_DIR"
        "$CROPPED_MASK_DIR"
    )

    for dir in "${DIRECTORIES_TO_RESET[@]}"; do
        if [ -d "$dir" ]; then
            echo "Deleting existing directory: $dir"
            rm -rf "$dir"
        fi
        echo "Creating fresh directory: $dir"
        mkdir -p "$dir"
    done

    echo "--- Job 0 has finished preparing directories. Creating sync file."
    # Create the empty flag file to signal completion
    touch "$SYNC_FILE"

else
    # This is other Worker Job (1-)
    echo "Job ${SLURM_ARRAY_TASK_ID}: Waiting for Job 0 to create the sync file..."
    # The '!' means 'not'. So, "while not file exists"
    while [ ! -f "$SYNC_FILE" ]; do
        # Wait for 5 seconds before checking again to avoid spamming the filesystem
        sleep 5
    done
    
    echo "Job ${SLURM_ARRAY_TASK_ID}: Sync file found. Proceeding with task."
fi

# This directory can be created by all jobs, as mkdir -p is safe and non-destructive.
mkdir -p "$LOG_DIR"


# STAGE 1: AUDIT SOURCE DATA
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo "--- STAGE 1: AUDITING SOURCE DATA (Job 0 Only)"
    python3 "${SCRIPT_DIR}/audit_wfdb_records.py" --directory "$DATA_SOURCE_DIR"
fi

# STAGE 2: CROP LEADS FROM IMAGES AND MASKS
echo "--- STAGE 2: CROPPING LEADS FOR BUCKET: ${TARGET_BUCKET} ---"

MODEL_PATH="/users/lip24dg/ecg/ecg-yolo/runs/yolo_ecg_model4/weights/best.pt"

if [ ! -f "$MODEL_PATH" ]; then
    echo "FATAL ERROR: 12-Lead YOLO model not found at $MODEL_PATH. Halting cropping."
else
    echo "Using 12-Lead model found at: $MODEL_PATH"
    TOTAL_JOBS=30 # Must match #SBATCH --array
    
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
echo "========================================"