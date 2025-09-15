import os
import sys
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import argparse
import cv2
import random
import json

def get_config(model_type):
    if model_type == "12L":
        config = {
            "sources": {
                "clean":    {"images": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented_12L/Cropped_Images_Clean", "masks": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented_12L/Cropped_Masks_Clean"},
                "scanner":  {"images": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented_12L/Cropped_Images_Scanner", "masks": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented_12L/Cropped_Masks_Scanner"},
                "physical": {"images": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented_12L/Cropped_Images_Physical", "masks": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented_12L/Cropped_Masks_Physical"},
                "chaos":    {"images": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented_12L/Cropped_Images_Chaos", "masks": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented_12L/Cropped_Masks_Chaos"}
            },
            "dataset_id": 7,
            "dataset_name_suffix": "_12L",
            "test_set_suffix": "_12L"
        }
    elif model_type == "LL":
        config = {
            "sources": {
                "clean":    {"images": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented/Cropped_Images_Clean", "masks": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented/Cropped_Masks_Clean"},
                "scanner":  {"images": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented/Cropped_Images_Scanner", "masks": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented/Cropped_Masks_Scanner"},
                "physical": {"images": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented/Cropped_Images_Physical", "masks": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented/Cropped_Masks_Physical"},
                "chaos":    {"images": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented/Cropped_Images_Chaos", "masks": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented/Cropped_Masks_Chaos"}
            },
            "dataset_id": 8,
            "dataset_name_suffix": "_LL",
            "test_set_suffix": "_LL"
        }
    else:
        raise ValueError(f"Unknown model_type provided: {model_type}")
    
    config["dataset_name"] = f"Dataset{config['dataset_id']:03d}_{config['dataset_name_suffix']}"
    return config

def convert_to_nifti(image_path, mask_path, out_image_path, out_mask_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None or img.size == 0 or msk.size == 0:
            return False
        msk[msk > 0] = 1
        img_np = img[np.newaxis].astype(np.float32)
        msk_np = msk[np.newaxis].astype(np.uint8)
        img_sitk = sitk.GetImageFromArray(img_np)
        msk_sitk = sitk.GetImageFromArray(msk_np)
        sitk.WriteImage(img_sitk, out_image_path)
        sitk.WriteImage(msk_sitk, out_mask_path)
        return True
    except Exception:
        if os.path.exists(out_image_path): os.remove(out_image_path)
        if os.path.exists(out_mask_path): os.remove(out_mask_path)
        return False

def main():
    parser = argparse.ArgumentParser(description="Prepare a SHARD of the ECG dataset for nnU-Net.")
    parser.add_argument("--model-type", type=str, required=True, choices=['12L', 'LL'], help="The model type to prepare data for ('12L' or 'LL').")
    parser.add_argument("--file-list", type=str, required=True, help="Path to the master text file of all unique ECG IDs.")
    parser.add_argument("--num-shards", type=int, required=True, help="The total number of parallel jobs.")
    parser.add_argument("--shard-id", type=int, required=True, help="The ID of this specific job.")
    parser.add_argument("--output-root", type=str, required=True, help="Output root for nnU-Net raw datasets.")
    parser.add_argument("--test-split", type=float, default=0.2, help="Fraction of the full dataset to use for testing.")
    args = parser.parse_args()

    # --- Dynamically get all configuration based on the model type ---
    config = get_config(args.model_type)
    DATA_SOURCES = config["sources"]
    TARGET_DATASET_NAME = config["dataset_name"]

    train_dataset_dir = os.path.join(args.output_root, TARGET_DATASET_NAME)
    imagesTr_dir = os.path.join(train_dataset_dir, "imagesTr")
    labelsTr_dir = os.path.join(train_dataset_dir, "labelsTr")
    os.makedirs(imagesTr_dir, exist_ok=True)
    os.makedirs(labelsTr_dir, exist_ok=True)

    test_sets_root_dir = os.path.join(args.output_root, f"structured_test_sets{config['test_set_suffix']}")
    test_dirs = {key: {"imagesTs": os.path.join(test_sets_root_dir, f"test_{key}", "imagesTs"), "labelsTs": os.path.join(test_sets_root_dir, f"test_{key}", "labelsTs")} for key in DATA_SOURCES.keys()}
    for key in test_dirs: 
        os.makedirs(test_dirs[key]["imagesTs"], exist_ok=True)
        os.makedirs(test_dirs[key]["labelsTs"], exist_ok=True)

    # GATHER FILE MAPPINGS
    ecg_to_images_map = {}
    for source_key, paths in DATA_SOURCES.items():
        image_dir = paths["images"]
        if not os.path.isdir(image_dir): continue
        for fname in sorted([f for f in os.listdir(image_dir) if f.endswith(".png")]):
            ecg_id = '_'.join(fname.split('_')[:-1])
            if ecg_id not in ecg_to_images_map: ecg_to_images_map[ecg_id] = {}
            if source_key not in ecg_to_images_map[ecg_id]: ecg_to_images_map[ecg_id][source_key] = []
            ecg_to_images_map[ecg_id][source_key].append(fname)
    
    # READ AND SHARD THE MASTER LIST OF UNIQUE ECG IDs
    with open(args.file_list, 'r') as f:
        all_ecg_ids = [line.strip() for line in f.readlines() if line.strip()]
    
    random.seed(42)
    random.shuffle(all_ecg_ids)

    num_total_ids = len(all_ecg_ids)
    shard_size = int(np.ceil(num_total_ids / args.num_shards))
    start_index = args.shard_id * shard_size
    end_index = min(start_index + shard_size, num_total_ids)
    my_ecg_ids = all_ecg_ids[start_index:end_index]
    
    print(f"Job {args.shard_id}/{args.num_shards}: Processing {len(my_ecg_ids)} ECGs for model type {args.model_type}.")

    # SPLIT JOB ECG IDs INTO TRAIN/TEST
    split_index = int(len(my_ecg_ids) * (1 - args.test_split))
    train_ecgs = my_ecg_ids[:split_index]
    test_ecgs = my_ecg_ids[split_index:]

    # PROCESS AND SAVE THE TRAINING SET SLICE FOR THIS SHARD
    for ecg_id in tqdm(train_ecgs, desc=f"Shard {args.shard_id} Training Set"):
        for source_key, files in ecg_to_images_map.get(ecg_id, {}).items():
            for img_file in files:
                img_path = os.path.join(DATA_SOURCES[source_key]["images"], img_file)
                mask_path = os.path.join(DATA_SOURCES[source_key]["masks"], img_file)
                if not os.path.exists(mask_path): continue
                out_img = os.path.join(imagesTr_dir, img_file.replace(".png", "_0000.nii.gz"))
                out_msk = os.path.join(labelsTr_dir, img_file.replace(".png", ".nii.gz"))
                convert_to_nifti(img_path, mask_path, out_img, out_msk)
    
    # PROCESS AND SAVE THE TEST SET SLICE FOR THIS SHARD
    for ecg_id in tqdm(test_ecgs, desc=f"Shard {args.shard_id} Test Sets"):
        for source_key, files in ecg_to_images_map.get(ecg_id, {}).items():
            for img_file in files:
                img_path = os.path.join(DATA_SOURCES[source_key]["images"], img_file)
                mask_path = os.path.join(DATA_SOURCES[source_key]["masks"], img_file)
                if not os.path.exists(mask_path): continue
                dest_img_dir = test_dirs[source_key]["imagesTs"]
                dest_lbl_dir = test_dirs[source_key]["labelsTs"]
                out_img = os.path.join(dest_img_dir, img_file.replace(".png", "_0000.nii.gz"))
                out_msk = os.path.join(dest_lbl_dir, img_file.replace(".png", ".nii.gz"))
                convert_to_nifti(img_path, mask_path, out_img, out_msk)

    print(f"\nShard {args.shard_id} processing complete.")

if __name__ == "__main__":
    main()