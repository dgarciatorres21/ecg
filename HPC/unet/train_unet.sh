#!/bin/bash
#SBATCH --job-name=train_unet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --array=0-4
#SBATCH --output=${LOG_DIR}/%x_%A_%a.out
#SBATCH --error=${LOG_DIR}/%x_%A_%a.err

# usage check
if [ "$#" -ne 1 ]; then
    echo "ERROR: You must provide a model type."
    echo "Usage: sbatch $0 <MODEL_TYPE>"
    echo "Example: sbatch $0 12L"
    exit 1
fi

MODEL_TYPE=$1

# configuration based on model_type
if [ "$MODEL_TYPE" == "12L" ]; then
    DATASET_ID=7
    JOB_NAME="train_D7_12L_folds"
    LOG_DIR="logs_train_unet"
elif [ "$MODEL_TYPE" == "LL" ]; then
    DATASET_ID=8
    JOB_NAME="train_D8_LL_folds"
    LOG_DIR="logs_train_unet"
else
    echo "ERROR: Invalid model type '$MODEL_TYPE'. Choose '12L', or 'LL'."
    exit 1
fi


# dynamic job name update
# this command will only work when the script is run via sbatch
if [ -n "$SLURM_JOB_ID" ]; then
    scontrol update jobid=${SLURM_JOB_ID} jobname=${JOB_NAME}
fi

# setup
mkdir -p $LOG_DIR
module load Anaconda3/2024.02-1
source /opt/apps/testapps/common/software/staging/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate unet

# set nnu-net paths
export nnUNet_raw="/mnt/parscratch/users/lip24dg/data/Generated_data"
export nnUNet_preprocessed="/mnt/parscratch/users/lip24dg/data/Generated_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/parscratch/users/lip24dg/data/Generated_data/nnUNet_results"

export nnUNet_compile="0"

# run training
# the $slurm_array_task_id variable is set by slurm when this script is run as a job array.
# if it's empty, it means the script was likely not run via sbatch.
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "ERROR: This script must be submitted as a job to SLURM using 'sbatch'."
    echo "The SLURM_ARRAY_TASK_ID is not set."
    exit 1
fi

echo "Starting PARALLEL training for Dataset ${DATASET_ID} (${MODEL_TYPE}), Fold ${SLURM_ARRAY_TASK_ID}..."
nnUNetv2_train $DATASET_ID 2d $SLURM_ARRAY_TASK_ID

echo "Training for Fold ${SLURM_ARRAY_TASK_ID} finished!"
