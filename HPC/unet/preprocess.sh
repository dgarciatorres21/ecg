#!/bin/bash
#SBATCH --job-name=nnUNet_preprocess
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=sheffield
#SBATCH --output=/users/lip24dg/ecg/HPC/logs_nnunet_prep/prepro_%x_%A.out
#SBATCH --error=/users/lip24dg/ecg/HPC/logs_nnunet_prep/prepro_%x_%A.err
#SBATCH --mail-user=dgarcia3@sheffield.ac.uk
#SBATCH --mail-type=FAIL,END

if [ -z "$1" ] || ( [ "$1" != "12L" ] && [ "$1" != "LL" ] ); then
    echo "ERROR: You must provide a valid model type as an argument."
    echo "Usage: sbatch preprocess.sh 12L"
    echo "   or: sbatch preprocess.sh LL"
    exit 1
fi

MODEL_TYPE=$1

# Dynamically update the job name for better tracking
scontrol update jobid=${SLURM_JOB_ID} jobname=preprocess_${MODEL_TYPE}

# SETUP
echo "--- Setting up environment for Preprocessing"
module load Anaconda3/2024.02-1
source activate unet

# SET nnU-Net PATHS
export nnUNet_raw="/mnt/parscratch/users/lip24dg/data/Generated_data"
export nnUNet_preprocessed="/mnt/parscratch/users/lip24dg/data/Generated_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/parscratch/users/lip24dg/data/Generated_data/nnUNet_results"

# Dynamically set the Dataset ID based on the argument
if [ "$MODEL_TYPE" == "12L" ]; then
    DATASET_ID=7
else # This will be the "LL" model
    DATASET_ID=8
fi

echo "--- Starting nnU-Net v2 Planning and Preprocessing for Dataset ${DATASET_ID} (${MODEL_TYPE})"

nnUNetv2_plan_and_preprocess -d ${DATASET_ID} --verify_dataset_integrity

# Added a safety check for robustness
if [ $? -ne 0 ]; then
    echo "ERROR: nnUNetv2_plan_and_preprocess failed for Dataset ${DATASET_ID}."
    exit 1
fi

echo "Preprocessing for Dataset ${DATASET_ID} finished successfully!"