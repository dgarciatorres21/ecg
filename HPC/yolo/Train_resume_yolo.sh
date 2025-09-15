#!/bin/bash
# --- slurm configuration for resuming gpu training ---
#
#SBATCH --job-name=yolo_train_resume
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/users/lip24dg/ecg/HPC/logs_yolo/resume_%A.out
#SBATCH --error=/users/lip24dg/ecg/HPC/logs_yolo/resume_%A.err
#SBATCH --mail-user=dgarcia3@sheffield.ac.uk
#SBATCH --mail-type=END # get an email when it successfully finishes

# --- diagnostics ---
echo "========================================="
echo "YOLO Training RESUME Job started on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "========================================="

# --- setup ---
echo "Setting up environment..."
module load Anaconda3/2024.02-1
source activate yolo

# --- execution ---
echo "--- Resuming model training ---"
# call your new resume script
python "/users/lip24dg/ecg/ecg-yolo/Train_resume.py"
if [ $? -ne 0 ]; then
    echo "ERROR: Resume script failed."
    exit 1
fi

echo "--- Training finished successfully. ---"
