import os
import sys
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import argparse
import cv2
import random
import json
import shutil

# new: top-level configuration
# define your four data sources. the 'key' will be used for the test folder name.
DATA_SOURCES = {
    "clean": {
        "images": "/mnt/parscratch/users/lip24dg/data/final_dataset/Cropped_Images",
        "masks": "/mnt/parscratch/users/lip24dg/data/final_dataset/Cropped_Masks"
    },
    "scanner": {
        "images": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented/Cropped_Images_Scanner",
        "masks": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented/Cropped_Masks_Scanner"
    },
    "physical": {
        "images": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented/Cropped_Images_Physical",
        "masks": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented/Cropped_Masks_Physical"
    },
    "chaos": {
        "images": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented/Cropped_Images_Chaos",
        "masks": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented/Cropped_Masks_Chaos"
    }
}

# the single nnu-net dataset id for the combined training set
TARGET_DATASET_ID = 3
TARGET_DATASET_NAME = f"Dataset{TARGET_DATASET_ID:03d}_ecg_robust" # a more descriptive name

def convert_to_nifti(image_path, mask_path, out_image_path, out_mask_path):
    # (this function remains unchanged)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or msk is None: return False
    msk[msk > 0] = 1
    img_sitk = sitk.GetImageFromArray(img[np.newaxis].astype(np.float32))
    msk_sitk = sitk.GetImageFromArray(msk[np.newaxis].astype(np.uint8))
    img_sitk.SetSpacing((1.0, 1.0, 1.0))
    msk_sitk.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(img_sitk, out_image_path)
    sitk.WriteImage(msk_sitk, out_mask_path)
    return True

def process_ecg_subset(ecg_ids, ecg_to_images_map, args, dest_img_dir, dest_lbl_dir, set_name):
    # (this helper function remains unchanged)
    total_cases = 0
    for ecg_id in tqdm(ecg_ids, desc=f"Processing {set_name}"):
        for img_file in ecg_to_images_map[ecg_id]:
            mask_file = img_file
            img_path = os.path.join(args.cropped_images, img_file)
            mask_path = os.path.join(args.cropped_masks, mask_file)
            if not os.path.exists(mask_path): continue
            out_img = os.path.join(dest_img_dir, img_file.replace(".png", "_0000.nii.gz"))
            out_msk = os.path.join(dest_lbl_dir, mask_file.replace(".png", ".nii.gz"))
            ok = convert_to_nifti(img_path, mask_path, out_img, out_msk)
            if ok: total_cases += 1
    return total_cases

def main():
    parser = argparse.ArgumentParser(description="Prepare and split a multi-source ECG dataset for nnU-Net.")
    parser.add_argument("--output-root", type=str, required=True, help="Output root for nnU-Net raw datasets and separate test sets.")
    parser.add_argument("--test-split", type=float, default=0.2, help="Fraction of data to use for the test sets.")
    args = parser.parse_args()

    # 1. main nnu-net training directory
    train_dataset_dir = os.path.join(args.output_root, TARGET_DATASET_NAME)
    imagesTr_dir = os.path.join(train_dataset_dir, "imagesTr")
    labelsTr_dir = os.path.join(train_dataset_dir, "labelsTr")
    os.makedirs(imagesTr_dir, exist_ok=True)
    os.makedirs(labelsTr_dir, exist_ok=True)

    # 2. separate test set directories (outside the nnu-net folder)
    test_sets_root_dir = os.path.join(args.output_root, "structured_test_sets")
    test_dirs = {}
    for key in DATA_SOURCES.keys():
        test_dirs[key] = {
            "imagesTs": os.path.join(test_sets_root_dir, f"test_{key}", "imagesTs"),
            "labelsTs": os.path.join(test_sets_root_dir, f"test_{key}", "labelsTs")
        }
        os.makedirs(test_dirs[key]["imagesTs"], exist_ok=True)
        os.makedirs(test_dirs[key]["labelsTs"], exist_ok=True)
    
    print(f"Training data will be saved to: {train_dataset_dir}")
    print(f"Test sets will be saved to: {test_sets_root_dir}")

    # gather and split all data
    ecg_to_images = {}
    all_ecg_ids = set()

    print("\nGathering all files from all sources...")
    for source_key, paths in DATA_SOURCES.items():
        image_dir = paths["images"]
        all_images = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
        for fname in all_images:
            ecg_id = '_'.join(fname.split('_')[:-1])
            all_ecg_ids.add(ecg_id)
            # store where this specific file can be found
            if ecg_id not in ecg_to_images: ecg_to_images[ecg_id] = {}
            if source_key not in ecg_to_images[ecg_id]: ecg_to_images[ecg_id][source_key] = []
            ecg_to_images[ecg_id][source_key].append(fname)

    all_ecg_ids = sorted(list(all_ecg_ids))
    random.seed(42)
    random.shuffle(all_ecg_ids)

    split_index = int(len(all_ecg_ids) * (1 - args.test_split))
    train_ecgs = all_ecg_ids[:split_index]
    test_ecgs = all_ecg_ids[split_index:]
    print(f"Total unique ECG IDs: {len(all_ecg_ids)}, Training IDs: {len(train_ecgs)}, Testing IDs: {len(test_ecgs)}")

    # process and save training set
    train_cases = 0
    print("\nProcessing combined training set...")
    for ecg_id in tqdm(train_ecgs, desc="Training Set"):
        for source_key, files in ecg_to_images.get(ecg_id, {}).items():
            for img_file in files:
                img_path = os.path.join(DATA_SOURCES[source_key]["images"], img_file)
                mask_path = os.path.join(DATA_SOURCES[source_key]["masks"], img_file)
                if not os.path.exists(mask_path): continue
                
                out_img = os.path.join(imagesTr_dir, img_file.replace(".png", "_0000.nii.gz"))
                out_msk = os.path.join(labelsTr_dir, img_file.replace(".png", ".nii.gz"))
                ok = convert_to_nifti(img_path, mask_path, out_img, out_msk)
                if ok: train_cases += 1
    
    # process and save separate test sets
    print("\nProcessing separate test sets...")
    for ecg_id in tqdm(test_ecgs, desc="Test Sets"):
        for source_key, files in ecg_to_images.get(ecg_id, {}).items():
            for img_file in files:
                img_path = os.path.join(DATA_SOURCES[source_key]["images"], img_file)
                mask_path = os.path.join(DATA_SOURCES[source_key]["masks"], img_file)
                if not os.path.exists(mask_path): continue

                dest_img_dir = test_dirs[source_key]["imagesTs"]
                dest_lbl_dir = test_dirs[source_key]["labelsTs"]
                
                out_img = os.path.join(dest_img_dir, img_file.replace(".png", "_0000.nii.gz"))
                out_msk = os.path.join(dest_lbl_dir, img_file.replace(".png", ".nii.gz"))
                convert_to_nifti(img_path, mask_path, out_img, out_msk)

    # generate dataset.json for the training set
    generate_dataset_json(train_dataset_dir, train_cases, {"0": "ecg_lead"}, {"background": 0, "foreground": 1})

    print(f"\nProcessing complete. Created {train_cases} training cases.")

if __name__ == "__main__":
    main()
