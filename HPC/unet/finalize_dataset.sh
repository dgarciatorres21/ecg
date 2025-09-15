#!/bin/bash
#SBATCH --job-name=finalize_dataset
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=/users/lip24dg/ecg/HPC/logs_nnunet_prep/finalize_%A.out
#SBATCH --error=/users/lip24dg/ecg/HPC/logs_nnunet_prep/finalize_%A.err

# usage check
if [ -z "$1" ] || ( [ "$1" != "12L" ] && [ "$1" != "LL" ] ); then
    echo "ERROR: You must provide a valid model type as an argument."
    echo "Usage: sbatch finalize_dataset.sh 12L"
    echo "   or: sbatch finalize_dataset.sh LL"
    exit 1
fi

MODEL_TYPE=$1

# setup
echo "--- Setting up environment for Finalizing Dataset"
module load Anaconda3/2024.02-1
source activate unet

# dynamically set the dataset id and name based on the argument
if [ "$MODEL_TYPE" == "12L" ]; then
    DATASET_ID=7
    DATASET_NAME="Dataset00${DATASET_ID}__12L"
else # this will be the "LL" model
    DATASET_ID=8
    DATASET_NAME="Dataset00${DATASET_ID}__LL"
fi

# define paths
PYTHON_SCRIPT_PATH="/users/lip24dg/ecg/code-unet/create_json.py"
DATASET_DIR="/mnt/parscratch/users/lip24dg/data/Generated_data/${DATASET_NAME}"


echo "Running script to create dataset.json for ${DATASET_DIR}"

python "${PYTHON_SCRIPT_PATH}" \
    --dataset_dir "${DATASET_DIR}" \
    --channel_names '{"0": "ecg_lead"}' \
    --labels '{"background": 0, "foreground": 1}'

echo "Finalization complete."
