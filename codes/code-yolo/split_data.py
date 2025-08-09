import os
import random
import shutil
from tqdm import tqdm

# --- CONFIGURATION ---

# 1. The directory where your original .png images are located.
#    This should be the output folder from your image generator script.
# IMAGE_SOURCE_DIR = 'C:/Users/dgarc/Desktop/Dissertation/ecg-image-kit/codes/ecg-image-generator/outputData/Generated_data'
# IMAGE_SOURCE_DIR = '/mnt/parscratch/users/lip24dg/data/1.0.3/records100'
IMAGE_SOURCE_DIR = '/mnt/parscratch/users/lip24dg/data/Generated_data/records500'
# 2. The directory where your .txt YOLO labels are located.
# LABEL_SOURCE_DIR = 'C:/Users/dgarc/Desktop/Dissertation/ecg-image-kit/Data/yolo_labels'
LABEL_SOURCE_DIR = '/users/lip24dg/ecg/data/yolo_labels'
# 3. The new top-level directory where the split data (train/valid/test) will be organized.
#    This folder will be created by the script.
# OUTPUT_DIR = 'C:/Users/dgarc/Desktop/Dissertation/ecg-image-kit/Data/yolo_split_data'
# OUTPUT_DIR = 'C:/users/lip24dg/ecg/data/data/yolo_split_data'
OUTPUT_DIR = '/mnt/parscratch/users/lip24dg/data/yolo_split_data'


# Define the split ratios.
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
# TEST_RATIO is implicitly 0.15

# --- SCRIPT ---

def split_data():
    """
    Splits the generated PNG images and their corresponding TXT labels
    into train, valid, and test sets.
    """
    # Use the clear variable names from the configuration
    image_source_dir = IMAGE_SOURCE_DIR
    label_source_dir = LABEL_SOURCE_DIR

    if not os.path.isdir(image_source_dir):
        print(f"FATAL ERROR: Image source directory not found at '{image_source_dir}'.")
        print("Please check the IMAGE_SOURCE_DIR path in the configuration.")
        return

    if not os.path.isdir(label_source_dir):
        print(f"FATAL ERROR: Label source directory not found at '{label_source_dir}'.")
        print("Please run the 'convert_to_yolo_v2.py' script first and check the LABEL_SOURCE_DIR path.")
        return

    # Create the necessary output directories
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'labels'), exist_ok=True)

    # Get a list of all image files from the correct directory
    all_images = [f for f in os.listdir(image_source_dir) if f.endswith('.png')]
    
    if not all_images:
        print(f"FATAL ERROR: No .png files were found in '{image_source_dir}'.")
        print("Cannot split the data. Please verify your image source directory.")
        return

    random.shuffle(all_images) # Shuffle for random distribution

    # Calculate split indices
    num_images = len(all_images)
    train_end = int(num_images * TRAIN_RATIO)
    valid_end = train_end + int(num_images * VALID_RATIO)

    # Assign files to splits
    train_files = all_images[:train_end]
    valid_files = all_images[train_end:valid_end]
    test_files = all_images[valid_end:]

    splits = {
        'train': train_files,
        'valid': valid_files,
        'test': test_files
    }

    print("Splitting files into train, valid, and test sets...")
    for split_name, file_list in splits.items():
        print(f"Processing {split_name} set ({len(file_list)} files)...")
        
        for image_filename in tqdm(file_list, desc=f"Copying {split_name} files"):
            # Define source paths
            src_image_path = os.path.join(image_source_dir, image_filename)
            
            label_filename = os.path.splitext(image_filename)[0] + '.txt'
            src_label_path = os.path.join(label_source_dir, label_filename)

            # Define destination paths
            dest_image_path = os.path.join(OUTPUT_DIR, split_name, 'images', image_filename)
            dest_label_path = os.path.join(OUTPUT_DIR, split_name, 'labels', label_filename)

            # Copy the files
            if os.path.exists(src_image_path) and os.path.exists(src_label_path):
                shutil.copyfile(src_image_path, dest_image_path)
                shutil.copyfile(src_label_path, dest_label_path)
            else:
                if not os.path.exists(src_label_path):
                    print(f"\nWarning: Label for {image_filename} not found at '{src_label_path}'. Skipping file.")
                if not os.path.exists(src_image_path):
                     print(f"\nWarning: Image {image_filename} not found at '{src_image_path}'. Skipping file.")
                
    print("\nData splitting complete!")
    print(f"Split data is located in: '{OUTPUT_DIR}'")

if __name__ == '__main__':
    split_data()