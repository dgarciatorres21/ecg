#!/bin/bash
#SBATCH --job-name=ecg_augment
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-29
#SBATCH --time=12:00:00
#SBATCH --output=/users/lip24dg/ecg/HPC/logs_augment/%A_%a.out
#SBATCH --error=/users/lip24dg/ecg/HPC/logs_augment/%A_%a.err
#SBATCH --mail-user=dgarcia3@sheffield.ac.uk
#SBATCH --mail-type=FAIL,END

# diagnostics
echo "========================================"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID}, Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "========================================"

# step 1: validate script argument
# this script requires one argument: the type of bucket to generate.
BUCKET_TYPE=$1
if [ -z "$BUCKET_TYPE" ]; then
    echo "FATAL ERROR: No bucket type specified."
    echo "Usage: sbatch generate_augmented_data.sh [scanner|physical|chaos]"
    exit 1
fi
echo "--- Generating augmentation bucket of type: ${BUCKET_TYPE}"

# setup
echo "Setting up the job environment..."
module load Anaconda3/2024.02-1
source activate /users/lip24dg/.conda/envs/ecg
echo "Conda environment 'ecg' activated. Using Python: $(which python)"
echo "-----------------------------------------"

# centralized path configuration
SCRIPT_DIR="/users/lip24dg/ecg/ecg-image-generator"
DATA_SOURCE_DIR="/mnt/parscratch/users/lip24dg/data/1.0.3/records500"
BASE_OUTPUT_DIR="/mnt/parscratch/users/lip24dg/data/final_dataset_augmented" # using a new base dir for clarity
LOG_DIR="/users/lip24dg/ecg/HPC/logs_augment"

mkdir -p $LOG_DIR

# step 2: configure bucket-specific settings
# based on the script argument, set the output directories and augmentation flags.

# default empty array for augmentation flags
AUGMENT_FLAGS=()

case $BUCKET_TYPE in
  "scanner")
    GENERATED_IMAGE_DIR="${BASE_OUTPUT_DIR}/Generated_Images_Scanner"
    GENERATED_MASK_DIR="${BASE_OUTPUT_DIR}/Generated_Masks_Scanner"
    AUGMENT_FLAGS=(
        "--augment"
        "--rotate" "5"
        "--noise" "50"
        "--random_grid_color"
    )
    ;;
  "physical")
    GENERATED_IMAGE_DIR="${BASE_OUTPUT_DIR}/Generated_Images_Physical"
    GENERATED_MASK_DIR="${BASE_OUTPUT_DIR}/Generated_Masks_Physical"
    AUGMENT_FLAGS=(
        "--wrinkles"
        "--num_creases_vertically" "15"
    )
    ;;
  "chaos")
    GENERATED_IMAGE_DIR="${BASE_OUTPUT_DIR}/Generated_Images_Chaos"
    GENERATED_MASK_DIR="${BASE_OUTPUT_DIR}/Generated_Masks_Chaos"
    AUGMENT_FLAGS=(
        "--fully_random"
        "--rotate" "5"
        "--noise" "50"
        "--random_grid_color"
        "--wrinkles"
        "--num_creases_vertically" "15"
    )
    ;;
  *)
    echo "FATAL ERROR: Invalid bucket type '${BUCKET_TYPE}'."
    echo "Valid options are: scanner, physical, chaos"
    exit 1
    ;;
esac

# create the specific directories for this run
mkdir -p $GENERATED_IMAGE_DIR
mkdir -p $GENERATED_MASK_DIR

echo "Outputting images to: ${GENERATED_IMAGE_DIR}"
echo "Outputting masks to:  ${GENERATED_MASK_DIR}"

# step 3: run the python script with the correct configuration
TOTAL_JOBS=30 # must match #sbatch --array

python3 "${SCRIPT_DIR}/run_generation_500.py" \
    --script-to-run "${SCRIPT_DIR}/gen_ecg_images_from_data_batch.py" \
    --data-root-dir "$DATA_SOURCE_DIR" \
    --image-output-dir "$GENERATED_IMAGE_DIR" \
    --mask-output-dir "$GENERATED_MASK_DIR" \
    --job-id ${SLURM_ARRAY_TASK_ID} \
    --total-jobs ${TOTAL_JOBS} \
    --lead_bbox \
    --store_config \
    --generate_masks \
    "${AUGMENT_FLAGS[@]}" # this correctly expands the array of flags

# final diagnostics
echo "-----------------------------------------"
echo "Job finished with exit code $?"
echo "Job finished at $(date)"
echo "========================================"
