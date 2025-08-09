import os
import subprocess
from tqdm import tqdm

# --- Configuration ---
# This gets the absolute path to the directory containing this script (ecg-image-kit)
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
PROJECT_DIR = os.path.dirname(BASE_DIR)

# Build all other paths based on that main directory
SCRIPT_TO_RUN = os.path.join(PROJECT_DIR, 'codes', 'ecg-image-generator', 'gen_ecg_images_from_data_batch.py')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'codes', 'ecg-image-generator', 'OutputData', 'Generated_data')
# SCRIPT_TO_RUN = r'/shared/chen_cv/User/lip24dg/ecg-image-generator/gen_ecg_images_from_data_batch_v3.py'
# OUTPUT_DIR = r'/shared/chen_cv/User/lip24dg/ecg-image-generator/OutputData/Generated_data'
# DATA_ROOT_DIR = '/mnt/parscratch/users/lip24dg/data/1.0.3/records100'


# This is the top-level directory we will search for data files.
DATA_ROOT_DIR = os.path.join(PROJECT_DIR, 'codes', 'ecg-image-generator', 'sampleData', 'PTB_XL_data', 'records100')
# DATA_ROOT_DIR = r'X:\chen_cv\Shared\Public\cardiac\physionet.org\files\ptb-xl\1.0.3\records100'
# DATA_ROOT_DIR = r'/shared/chen_cv/Public/cardiac/physionet.org/files/ptb-xl/1.0.3/records100'

# --- Script Logic ---
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"--- Configuration ---")
    print(f"Searching for data in: {DATA_ROOT_DIR}")
    print(f"Image Output Directory: {OUTPUT_DIR}")
    print(f"---------------------")

    # ====================================================================
    # == NEW LOGIC: Walk through all subfolders to find .dat files      ==
    # ====================================================================
    all_file_paths = []
    print("Scanning all subdirectories for .dat files... (This may take a moment)")
    # os.walk will visit every folder and subfolder inside DATA_ROOT_DIR
    for root, dirs, files in os.walk(DATA_ROOT_DIR):
        for filename in files:
            # We only care about the .dat files
            if filename.endswith('.dat'):
                # Construct the full path to the file and add it to our list
                full_path = os.path.join(root, filename)
                all_file_paths.append(full_path)
    # ====================================================================

    if not all_file_paths:
        print(f"\nFATAL ERROR: Found 0 '.dat' files in '{DATA_ROOT_DIR}' and its subfolders.")
        print("Please check that the data files are actually in this folder structure.")
        return

    print(f"Found {len(all_file_paths)} records to process.")

    # --- LIMIT THE FILES ---
    files_to_process = all_file_paths[:2000]
    print(f"Limiting run to the first {len(files_to_process)} files as requested.")

    # Loop through each full file path we found
    for input_file_path in tqdm(files_to_process, desc="Generating ECG Images"): #all_file_paths
        
        # Get just the filename (e.g., '00001_lr.dat') from the full path
        base_filename = os.path.basename(input_file_path)
        
        # Create the name for the output image (e.g., '00001_lr.png')
        output_image_name = os.path.splitext(base_filename)[0] + '.png'
        output_image_path = os.path.join(OUTPUT_DIR, output_image_name)

        if os.path.exists(output_image_path):
            continue

        command = [
            'python'
            , SCRIPT_TO_RUN
            , '-i'
            , input_file_path  # Pass the full path to the .dat file
            , '-o'
            , OUTPUT_DIR
            , '--num_leads'
            , 'twelve'
            , '--lead_bbox'
            , '--store_config'
        ]
        
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"\n\n--- ERROR ---")
            print(f"The script failed while processing file: {input_file_path}")
            print(f"Error message from child script:\n{e.stderr}")
            print("-------------")
            break

    print("\nGeneration complete!")

if __name__ == '__main__':
    main()