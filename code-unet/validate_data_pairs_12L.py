import os

# this dictionary should point to the dataset you want to validate.
DATA_SOURCES = {
    "clean": {
        "images": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented_12L/Cropped_Images_Clean",
        "masks": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented_12L/Cropped_Masks_Clean"
    },
    "scanner": {
        "images": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented_12L/Cropped_Images_Scanner",
        "masks": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented_12L/Cropped_Masks_Scanner"
    },
    "physical": {
        "images": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented_12L/Cropped_Images_Physical",
        "masks": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented_12L/Cropped_Masks_Physical"
    },
    "chaos": {
        "images": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented_12L/Cropped_Images_Chaos",
        "masks": "/mnt/parscratch/users/lip24dg/data/final_dataset_augmented_12L/Cropped_Masks_Chaos"
    }
}

def validate_pairs():
    print("--- Starting Validation: Searching for images with missing masks ---")
    orphaned_images_count = 0
    
    for source_name, paths in DATA_SOURCES.items():
        image_dir = paths["images"]
        mask_dir = paths["masks"]
        
        print(f"\nChecking source: {source_name}...")
        
        if not os.path.isdir(image_dir):
            print(f"  Warning: Image directory not found: {image_dir}")
            continue
        if not os.path.isdir(mask_dir):
            print(f"  Warning: Mask directory not found: {mask_dir}")
            continue
            
        image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
        
        for image_filename in image_files:
            # the mask filename is assumed to be identical to the image filename.
            mask_path = os.path.join(mask_dir, image_filename)
            
            if not os.path.exists(mask_path):
                orphaned_image_path = os.path.join(image_dir, image_filename)
                print(f"  MISSING MASK for image: {orphaned_image_path}")
                orphaned_images_count += 1
                
    print("\n-----------------------------------------------------------------")
    if orphaned_images_count == 0:
        print("Validation PASSED: All images have a corresponding mask.")
    else:
        print(f"Validation FAILED: Found {orphaned_images_count} images with missing masks.")
    print("-----------------------------------------------------------------")

if __name__ == "__main__":
    validate_pairs()