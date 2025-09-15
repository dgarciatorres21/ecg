import os
import random
import shutil
import argparse
from tqdm import tqdm

TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
# TEST_RATIO = 0.15


def split_data(image_source_dir, label_source_dir, output_dir):

    if not os.path.isdir(image_source_dir):
        print(f"FATAL ERROR: Image source directory not found at '{image_source_dir}'.")
        return

    if not os.path.isdir(label_source_dir):
        print(f"FATAL ERROR: Label source directory not found at '{label_source_dir}'.")
        return

    # Create the necessary output directories
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

    all_images = [f for f in os.listdir(image_source_dir) if f.endswith('.png')]
    
    if not all_images:
        print(f"FATAL ERROR: No .png files were found in '{image_source_dir}'.")
        return

    random.shuffle(all_images)

    num_images = len(all_images)
    train_end = int(num_images * TRAIN_RATIO)
    valid_end = train_end + int(num_images * VALID_RATIO)

    train_files = all_images[:train_end]
    valid_files = all_images[train_end:valid_end]
    test_files = all_images[valid_end:]

    splits = {'train': train_files, 'valid': valid_files, 'test': test_files}

    print("Splitting files into train, valid, and test sets...")
    for split_name, file_list in splits.items():
        print(f"Processing {split_name} set ({len(file_list)} files)...")
        
        for image_filename in tqdm(file_list, desc=f"Copying {split_name} files"):
            src_image_path = os.path.join(image_source_dir, image_filename)
            label_filename = os.path.splitext(image_filename)[0] + '.txt'
            src_label_path = os.path.join(label_source_dir, label_filename)

            dest_image_path = os.path.join(output_dir, split_name, 'images', image_filename)
            dest_label_path = os.path.join(output_dir, split_name, 'labels', label_filename)

            if os.path.exists(src_image_path) and os.path.exists(src_label_path):
                shutil.copyfile(src_image_path, dest_image_path)
                shutil.copyfile(src_label_path, dest_label_path)
            else:
                if not os.path.exists(src_label_path):
                    print(f"\nWarning: Label for {image_filename} not found at '{src_label_path}'. Skipping file.")
                if not os.path.exists(src_image_path):
                     print(f"\nWarning: Image {image_filename} not found at '{src_image_path}'. Skipping file.")
                
    print("\nData splitting complete!")
    print(f"Split data is located in: '{output_dir}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split image and label data into train, valid, and test sets.")
    parser.add_argument('--image-source-dir', type=str, required=True, help="Directory containing the source .png images.")
    parser.add_argument('--label-source-dir', type=str, required=True, help="Directory containing the source .txt label files.")
    parser.add_argument('--output-dir', type=str, required=True, help="The top-level directory where the split data will be saved.")
    args = parser.parse_args()
    
    split_data(args.image_source_dir, args.label_source_dir, args.output_dir)