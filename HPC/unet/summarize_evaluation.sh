#!/bin/bash
#SBATCH --job-name=summarize_eval
#SBATCH --partition=sheffield
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --output=/users/lip24dg/ecg/HPC/logs_nnunet/summarize_%x_%A.out
#SBATCH --error=/users/lip24dg/ecg/HPC/logs_nnunet/summarize_%x_%A.err

# usage check
if [ -z "$1" ] || ( [ "$1" != "12L" ] && [ "$1" != "LL" ] ); then
    echo "ERROR: You must provide a valid model type."
    echo "Usage: sbatch summarize_evaluation.sh 12L"
    exit 1
fi

MODEL_TYPE=$1
scontrol update jobid=${SLURM_JOB_ID} jobname=summarize_${MODEL_TYPE}

# setup
module load Anaconda3/2024.02-1
source activate unet

# dynamically set the evaluation directory path
if [ "$MODEL_TYPE" == "12L" ]; then
    DATASET_ID=7
    DATASET_NAME="Dataset00${DATASET_ID}_ecg_12L"
else # ll model
    DATASET_ID=8
    DATASET_NAME="Dataset00${DATASET_ID}_LL"
fi

export EVAL_DIR="/mnt/parscratch/users/lip24dg/data/Generated_data/nnUNet_results/${DATASET_NAME}/evaluation"

# summary
echo "--- Final Summary for ${MODEL_TYPE} Model:"
python -c 'import json; import os; import pandas as pd;
EVAL_DIR = os.environ.get("EVAL_DIR")
TEST_SETS = ["test_clean", "test_scanner", "test_physical", "test_chaos"]
results = []
for ts in TEST_SETS:
    json_path = os.path.join(EVAL_DIR, f"summary_{ts}.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
            dice = data["mean"]["1"]["Dice"]
            results.append({"Test Set": ts, "Mean Dice": f"{dice:.4f}"})
if results:
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
else:
    print("No evaluation results found to summarize.")
'
