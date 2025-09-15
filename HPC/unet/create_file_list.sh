#!/bin/bash
#SBATCH --job-name=create_file_list
#SBATCH --partition=sheffield
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --output=logs_nnunet_prep/create_list_%A.out
#SBATCH --error=logs_nnunet_prep/create_list_%A.err

# usage check
if [ -z "$1" ] || ( [ "$1" != "12L" ] && [ "$1" != "LL" ] ); then
    echo "ERROR: You must provide a valid model type as an argument."
    echo "Usage: sbatch create_file_list.sh 12L"
    echo "   or: sbatch create_file_list.sh LL"
    exit 1
fi

MODEL_TYPE=$1

# setup
module load Anaconda3/2024.02-1
source activate unet

OUTPUT_DIR="/users/lip24dg/ecg/code-unet/"
mkdir -p "${OUTPUT_DIR}"

if [ "$MODEL_TYPE" == "12L" ]; then
    echo "--- Generating master file list for 12L dataset"
    export DATA_SOURCES_JSON='{
        "clean": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented_12L/Cropped_Images_Clean",
        "scanner": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented_12L/Cropped_Images_Scanner",
        "physical": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented_12L/Cropped_Images_Physical",
        "chaos": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented_12L/Cropped_Images_Chaos"
    }'
    export OUTPUT_FILENAME="${OUTPUT_DIR}/all_ecg_ids_12L.txt"
else # this will be the "LL" model
    echo "--- Generating master file list for LL dataset"
    export DATA_SOURCES_JSON='{
        "clean": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented/Cropped_Images_Clean",
        "scanner": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented/Cropped_Images_Scanner",
        "physical": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented/Cropped_Images_Physical",
        "chaos": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented/Cropped_Images_Chaos"
    }'
    export OUTPUT_FILENAME="${OUTPUT_DIR}/all_ecg_ids_LL.txt"
fi

python << 'EOF'
import os
import json

# read the data sources and output filename from environment variables
data_sources_json = os.environ.get("DATA_SOURCES_JSON")
output_filename = os.environ.get("OUTPUT_FILENAME")
data_sources = json.loads(data_sources_json)

all_ecg_ids = set()
for source_key, image_dir in data_sources.items():
    if not os.path.isdir(image_dir):
        print(f"Warning: Directory not found, skipping: {image_dir}")
        continue
    all_images = [f for f in os.listdir(image_dir) if f.endswith(".png")]
    for fname in all_images:
        all_ecg_ids.add("_".join(fname.split("_")[:-1]))

with open(output_filename, "w") as f:
    for ecg_id in sorted(list(all_ecg_ids)):
        f.write(f"{ecg_id}\n")

# this print statement is now safe because the shell is not interpreting it.
# we can use single quotes without escaping them.
print(f"Master list '{output_filename}' created with {len(all_ecg_ids)} unique ECG IDs.")
EOF
