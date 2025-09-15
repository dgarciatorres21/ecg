#!/bin/bash
#SBATCH --job-name=ecg_generate_clean
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-29
#SBATCH --time=06:00:00
#SBATCH --output=/users/lip24dg/ecg/HPC/logs_500/%A_%a.out
#SBATCH --error=/users/lip24dg/ecg/HPC/logs_500/%A_%a.err
#SBATCH --mail-user=dgarcia3@sheffield.ac.uk
#SBATCH --mail-type=FAIL,END

# DIAGNOSTICS
echo "========================================"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID}, Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "--- Generating CLEAN data bucket ---"
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
BASE_OUTPUT_DIR="/mnt/parscratch/users/lip24dg/data/final_dataset_augmented"
LOG_DIR="/users/lip24dg/ecg/HPC/logs_500"

# Define the bucket type statically for this script
BUCKET_TYPE="Clean"

# CHANGED: Derive paths using the BUCKET_TYPE variable for consistency
GENERATED_IMAGE_DIR="${BASE_OUTPUT_DIR}/Generated_Images_${BUCKET_TYPE}"
GENERATED_MASK_DIR="${BASE_OUTPUT_DIR}/Generated_Masks_${BUCKET_TYPE}"

TOTAL_JOBS=30

# STAGE 1: GENERATE IMAGES AND MASKS (CLEAN)
echo "--- STAGE 1: GENERATING IMAGES AND MASKS ---"
python3 "${SCRIPT_DIR}/run_generation_500.py" \
    --script-to-run "${SCRIPT_DIR}/gen_ecg_images_from_data_batch.py" \
    --data-root-dir "$DATA_SOURCE_DIR" \
    --image-output-dir "$GENERATED_IMAGE_DIR" \
    --mask-output-dir "$GENERATED_MASK_DIR" \
    --job-id ${SLURM_ARRAY_TASK_ID} \
    --total-jobs ${TOTAL_JOBS} \
    --lead_bbox \
    --store_config \
    --generate_masks


# STAGE 2: AUDIT SOURCE DATA (Optional - runs only on the first job)
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo "--- STAGE 2: AUDITING SOURCE DATA (Job 0 Only) ---"
    python3 "${SCRIPT_DIR}/audit_wfdb_records.py" --directory "$DATA_SOURCE_DIR"
fi
# FINAL DIAGNOSTICS
echo "-----------------------------------------"
echo "Job finished with exit code $?"
echo "Job finished at $(date)"
echo "========================================"