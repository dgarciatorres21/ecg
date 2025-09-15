#SBATCH --job-name=yolo_data_prep
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --array=0-6
#SBATCH --output=/users/lip24dg/ecg/HPC/logs_yolo/prep_%A_%a.out # %A=jobID, %a=arrayID
#SBATCH --error=/users/lip24dg/ecg/HPC/logs_yolo/prep_%A_%a.err
#SBATCH --mail-user=dgarcia3@sheffield.ac.uk
#SBATCH --mail-type=FAIL,END

# DIAGNOSTICS
echo "========================================"
echo "Data Prep Job Array started on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID}, Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "========================================"

# SETUP
echo "Setting up environment..."
module load Anaconda3/2024.02-1
source activate yolo

# DEFINE BUCKETS
# Create a bash array of your bucket names
BUCKETS=("Clean" "Scanner" "Physical" "Chaos")
# Select the bucket for this specific job in the array
BUCKET_TYPE=${BUCKETS[$SLURM_ARRAY_TASK_ID]}
echo "--- Processing data for bucket: ${BUCKET_TYPE} ---"

# PATH CONFIGURATION
YOLO_SCRIPTS_DIR="/users/lip24dg/ecg/ecg-yolo"
BASE_OUTPUT_DIR="/mnt/parscratch/users/lip24dg/data/final_dataset_augmented"

# Use a conditional to handle the different source path for the 'Clean' data
if [ "$BUCKET_TYPE" == "Clean" ]; then
    CONVERSION_INPUT_DIR="/mnt/parscratch/users/lip24dg/data/dataset/Generated_Images"
else
    CONVERSION_INPUT_DIR="${BASE_OUTPUT_DIR}/Generated_Images_${BUCKET_TYPE}"
fi

LABEL_OUTPUT_DIR="${BASE_OUTPUT_DIR}/yolo_labels_${BUCKET_TYPE}"
SPLIT_DATA_OUTPUT_DIR="${BASE_OUTPUT_DIR}/yolo_split_data_${BUCKET_TYPE}"

echo "Source Dir: ${CONVERSION_INPUT_DIR}"
echo "Label Dir:  ${LABEL_OUTPUT_DIR}"
echo "Split Dir:  ${SPLIT_DATA_OUTPUT_DIR}"

# EXECUTION
echo "--- Step 1: Converting JSON to YOLO format ---"
python3 "${YOLO_SCRIPTS_DIR}/convert_to_yolo.py" \
    --data-dir "${CONVERSION_INPUT_DIR}" \
    --output-dir "${LABEL_OUTPUT_DIR}"

echo "--- Step 2: Splitting data ---"
python3 "${YOLO_SCRIPTS_DIR}/split_data.py" \
    --image-source-dir "${CONVERSION_INPUT_DIR}" \
    --label-source-dir "${LABEL_OUTPUT_DIR}" \
    --output-dir "${SPLIT_DATA_OUTPUT_DIR}"

echo "--- Data prep for ${BUCKET_TYPE} complete. ---"