#!/bin/bash
#SBATCH --job-name=ecg_demo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=/users/lip24dg/ecg/HPC/logs_pipeline/demo_%A.out
#SBATCH --error=/users/lip24dg/ecg/HPC/logs_pipeline/demo_%A.err

# SETUP
module load Anaconda3/2024.02-1
source activate unet

# SET nnU-Net PATHS (Required by nnUNetv2_predict)
export nnUNet_raw="/mnt/parscratch/users/lip24dg/data/Generated_data"
export nnUNet_preprocessed="/mnt/parscratch/users/lip24dg/data/Generated_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/parscratch/users/lip24dg/data/Generated_data/nnUNet_results"

# RUN THE DEMO SCRIPT
PYTHON_SCRIPT_PATH="/users/lip24dg/ecg/code-unet/create_demo.py"

python "${PYTHON_SCRIPT_PATH}" \
    --input_dir "/users/lip24dg/ecg/code-unet/demo/input/" \
    --output_dir "/users/lip24dg/ecg/code-unet/demo/output/" \
    --dataset_id 5

echo "Demo script finished."

# Step 3: Run the Demo
Execute the SLURM script:
```bash
sbatch run_demo.sh