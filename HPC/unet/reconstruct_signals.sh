#!/bin/bash
#SBATCH --job-name=reconstruct_signals_D3
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --partition=sheffield
#SBATCH --output=/users/lip24dg/ecg/HPC/logs_pipeline/reconstruct_D3_%A.out
#SBATCH --error=/users/lip24dg/ecg/HPC/logs_pipeline/reconstruct_D3_%A.err

# setup
module load Anaconda3/2024.02-1
source /opt/apps/testapps/common/software/staging/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate unet

# paths here
DATA_PATH="/mnt/parscratch/users/lip24dg/data/Generated_data/"
# the folder containing the prediction masks from nnU-net
PREDICTIONS_FOLDER="${DATA_PATH}/ecg_predictions_D3_ensemble"
# the folder where the final 1d signal csv files will be saved
OUTPUT_CSV_FOLDER="${DATA_PATH}/reconstructed_signals_D3"
# python script
PYTHON_SCRIPT_PATH="/users/lip24dg/ecg/code-unet/reconstruct_signals.py"

echo "Running signal reconstruction..."
python "${PYTHON_SCRIPT_PATH}" \
    --input_dir "${PREDICTIONS_FOLDER}" \
    --output_dir "${OUTPUT_CSV_FOLDER}"

echo "Reconstruction complete."
