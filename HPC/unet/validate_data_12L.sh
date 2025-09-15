#!/bin/bash
#SBATCH --job-name=validate_data
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=/users/lip24dg/ecg/HPC/logs_nnunet/validate_data_%A_out.txt
#SBATCH --error=/users/lip24dg/ecg/HPC/logs_nnunet/validate_data_%A_err.txt

# SETUP
echo "--- Setting up environment for Data Validation"
module load Anaconda3/2024.02-1
source activate unet

# DEFINE SCRIPT PATH
PYTHON_SCRIPT_PATH="/users/lip24dg/ecg/code-unet/validate_data_pairs_12L.py"

echo "Running validation script..."

# EXECUTE SCRIPT
python "${PYTHON_SCRIPT_PATH}"

echo "Validation complete."
