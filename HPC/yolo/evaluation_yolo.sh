#slurm configuration
#SBATCH --job-name=yolo_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/users/lip24dg/ecg/HPC/logs_yolo/%x_%A.out
#SBATCH --error=/users/lip24dg/ecg/HPC/logs_yolo/%x_%A.err
#SBATCH --mail-user=dgarcia3@sheffield.ac.uk
#SBATCH --mail-type=FAIL,END

#1. argument validation
MODEL_TYPE=$1
DATASET_TYPE=$2

scontrol update jobid=${SLURM_JOB_ID} name="eval_${MODEL_TYPE}_${DATASET_TYPE}"

echo "--- Starting UNIFIED YOLO Model Evaluation --"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Model Type: ${MODEL_TYPE}, Dataset Type: ${DATASET_TYPE}"
echo "-----------------------------------------"

if [ -z "$MODEL_TYPE" ] || [ -z "$DATASET_TYPE" ]; then
    echo "FATAL ERROR: Missing arguments."
    echo "Usage: sbatch $0 <model_type> <dataset_type>"
    echo "  <model_type>:   'LL' or '12L'"
    echo "  <dataset_type>: 'clean', 'chaos', 'physical', or 'scanner'"
    exit 1
fi

#2. configuration
EVAL_SCRIPT="/users/lip24dg/ecg/ecg-yolo/evaluation.py"

# model ll config
MODEL_LL="/users/lip24dg/ecg/ecg-yolo/runs/yolo_ecg_model3/weights/best.pt"
DATA_YAML_LL="/users/lip24dg/ecg/ecg-yolo/data.yaml"

# model 12l config
MODEL_12L="/users/lip24dg/ecg/ecg-yolo/runs_12L/yolo_ecg_model_12L3/weights/best.pt"
DATA_YAML_12L="/users/lip24dg/ecg/ecg-yolo/data_12L.yaml"

declare -A BASE_PATHS
BASE_PATHS=(
    ["LL_clean"]="/mnt/parscratch/users/lip24dg/data/final_dataset_augmented/yolo_split_data_Clean"
    ["LL_chaos"]="/mnt/parscratch/users/lip24dg/data/final_dataset_augmented/yolo_split_data_Chaos"
    ["LL_physical"]="/mnt/parscratch/users/lip24dg/data/final_dataset_augmented/yolo_split_data_Physical"
    ["LL_scanner"]="/mnt/parscratch/users/lip24dg/data/final_dataset_augmented/yolo_split_data_Scanner"
    
    ["12L_clean"]="/mnt/parscratch/users/lip24dg/data/final_dataset_augmented_12L/yolo_split_data_Clean"
    ["12L_chaos"]="/mnt/parscratch/users/lip24dg/data/final_dataset_augmented_12L/yolo_split_data_Chaos"
    ["12L_physical"]="/mnt/parscratch/users/lip24dg/data/final_dataset_augmented_12L/yolo_split_data_Physical"
    ["12L_scanner"]="/mnt/parscratch/users/lip24dg/data/final_dataset_augmented_12L/yolo_split_data_Scanner"
)

# test directories are now always the same relative path
TEST_DIR_RELATIVE="test/images"

#activate conda environment
echo "Activating Conda environment..."
module load Anaconda3/2024.02-1
source activate yolo

#3. dynamically configuration
SELECTED_MODEL_PATH=""
SELECTED_DATA_YAML=""
SELECTED_BASE_PATH=""

if [ "$MODEL_TYPE" == "LL" ]; then
    echo "Selecting configuration for LL Model."
    SELECTED_MODEL_PATH=$MODEL_LL
    SELECTED_DATA_YAML=$DATA_YAML_LL
    SELECTED_BASE_PATH=${BASE_PATHS["LL_${DATASET_TYPE}"]}
elif [ "$MODEL_TYPE" == "12L" ]; then
    echo "Selecting configuration for 12L Model."
    SELECTED_MODEL_PATH=$MODEL_12L
    SELECTED_DATA_YAML=$DATA_YAML_12L
    SELECTED_BASE_PATH=${BASE_PATHS["12L_${DATASET_TYPE}"]}
else
    echo "FATAL ERROR: Invalid model type '${MODEL_TYPE}'. Choose 'LL' or '12L'."
    exit 1
fi

# validate that the dataset key was found
if [ -z "$SELECTED_BASE_PATH" ]; then
    echo "FATAL ERROR: Invalid dataset type '${DATASET_TYPE}' for model '${MODEL_TYPE}'."
    exit 1
fi

echo "Model Path: ${SELECTED_MODEL_PATH}"
echo "Data YAML: ${SELECTED_DATA_YAML}"
echo "Base Path: ${SELECTED_BASE_PATH}"
echo "Test Dir (Relative): ${TEST_DIR_RELATIVE}"
echo "-----------------------------------------"

#4. execute the unified evaluation script
python "$EVAL_SCRIPT" \
    --model-path "$SELECTED_MODEL_PATH" \
    --data-yaml "$SELECTED_DATA_YAML" \
    --base-path "$SELECTED_BASE_PATH" \
    --test-dir "$TEST_DIR_RELATIVE"

echo "-----------------------------------------"
echo "Evaluation for ${MODEL_TYPE} model on ${DATASET_TYPE} dataset complete."
