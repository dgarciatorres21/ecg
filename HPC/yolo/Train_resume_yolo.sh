#!/bin/bash
# --- SLURM CONFIGURATION FOR RESUMING GPU TRAINING ---
#
#SBATCH --job-name=yolo_train_resume
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00      # <-- Give it plenty of time to finish
#SBATCH --output=/users/lip24dg/ecg/HPC/logs_yolo/resume_%A.out # <-- New log file name
#SBATCH --error=/users/lip24dg/ecg/HPC/logs_yolo/resume_%A.err   # <-- New log file name
#SBATCH --mail-user=dgarcia3@sheffield.ac.uk
#SBATCH --mail-type=END # Get an email when it successfully finishes

# --- DIAGNOSTICS ---
echo "========================================"
echo "YOLO Training RESUME Job started on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "========================================"

# --- SETUP ---
echo "Setting up environment..."
module load Anaconda3/2024.02-1
source activate yolo

# --- EXECUTION ---
echo "--- Resuming model training ---"
# Call your NEW resume script
python "/users/lip24dg/ecg/ecg-yolo/Train_resume.py"
if [ $? -ne 0 ]; then
    echo "ERROR: Resume script failed."
    exit 1
fi

echo "--- Training finished successfully. ---"