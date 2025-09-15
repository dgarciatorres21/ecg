#!/bin/bash
#SBATCH --job-name=yolo_train_final
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/users/lip24dg/ecg/HPC/logs_yolo/train_%A.out
#SBATCH --error=/users/lip24dg/ecg/HPC/logs_yolo/train_%A.err
#SBATCH --mail-user=dgarcia3@sheffield.ac.uk
#SBATCH --mail-type=FAIL,END

#DIAGNOSTICS
echo "========================================"
echo "YOLO Training Job started on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "========================================"

#SETUP
echo "Setting up environment..."
module load Anaconda3/2024.02-1
source activate yolo

#EXECUTION
echo "--- Starting model training"
python "/users/lip24dg/ecg/ecg-yolo/Train.py"
if [ $? -ne 0 ]; then
    echo "ERROR: Training script failed."
    exit 1
fi

echo "--- Training finished successfully."