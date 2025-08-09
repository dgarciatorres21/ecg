import os
import subprocess
from tqdm import tqdm
import argparse

# --- Configuration ---
# Use absolute paths for reliability on the HPC.
# These paths should point to your directories on the cluster.
SCRIPT_TO_RUN = '/users/lip24dg/ecg/ecg-image-generator/gen_ecg_images_from_data_batch.py'
OUTPUT_DIR = '/mnt/parscratch/users/lip24dg/data/Generated_data/records500/'
DATA_ROOT_DIR = '/mnt/parscratch/users/lip24dg/data/1.0.3/records500'


# --- Argument Parsing ---
def get_args():
    """
    Parses command-line arguments.
    Handles both job array mode (with --job-id, --total-jobs)
    and test mode (with --limit).
    """
    parser = argparse.ArgumentParser(
        description="Process a chunk of ECG files as part of a job array or a limited test run."
    )
    # Arguments for the SLURM job array
    parser.add_argument(
        '--job-id', 
        type=int, 
        help='The SLURM_ARRAY_TASK_ID of this job (e.g., 0, 1, 2...).'
    )
    parser.add_argument(
        '--total-jobs', 
        type=int, 
        help='The total number of jobs in the array (e.g., 22).'
    )
    # New optional argument for testing
    parser.add_argument(
        '--limit', 
        type=int, 
        help='For testing: limit processing to the first N files. Overrides job array logic.'
    )
    return parser.parse_args()


# --- Main Script Logic ---
def main():
    # Get the arguments passed from the command line
    args = get_args()

    print(f"--- Configuration ---")
    print(f"Searching for data in: {DATA_ROOT_DIR}")
    print(f"Image Output Directory: {OUTPUT_DIR}")
    print(f"---------------------")

    # Ensure the output directory exists before we start
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Walk through all subfolders to find .dat files
    all_file_paths = []
    print("Scanning all subdirectories for .dat files...")
    for root, dirs, files in os.walk(DATA_ROOT_DIR):
        for filename in files:
            if filename.endswith('.dat'):
                full_path = os.path.join(root, filename)
                all_file_paths.append(full_path)
    
    # Sort the list to ensure consistent processing order across all jobs
    all_file_paths.sort()

    # --- MODE SELECTION LOGIC ---
    if args.limit is not None:
        # --- TEST MODE ---
        print(f"--- TEST MODE ACTIVATED: Limiting to the first {args.limit} files. ---")
        all_file_paths = all_file_paths[:args.limit]
        # For a simple test, we can treat it as a single job
        args.job_id = 0
        args.total_jobs = 1
    elif args.job_id is None or args.total_jobs is None:
        # --- ERROR MODE ---
        print("\nFATAL ERROR: In full run mode, --job-id and --total-jobs are required.")
        print("To run a test on a few files, use the --limit <number> argument instead.")
        return

    # Check if any files were found after potential limiting
    if not all_file_paths:
        print(f"\nFATAL ERROR: Found 0 '.dat' files to process in '{DATA_ROOT_DIR}'.")
        return

    print(f"Found a total of {len(all_file_paths)} records to process for this run.")

    # --- JOB ARRAY LOGIC TO SPLIT THE FILES ---
    # This logic now works for both full runs and test mode.
    # In test mode, total_jobs is 1, so this job gets all the files.
    num_files = len(all_file_paths)
    # Perform ceiling division to calculate the size of each chunk
    chunk_size = (num_files + args.total_jobs - 1) // args.total_jobs
    
    # Calculate the start and end index for this specific job's chunk
    start_index = args.job_id * chunk_size
    end_index = min(start_index + chunk_size, num_files)
    
    # Select the slice of files for this specific job
    files_to_process = all_file_paths[start_index:end_index]

    print(f"--- Job {args.job_id + 1}/{args.total_jobs} ---")
    print(f"This job will process {len(files_to_process)} files (from index {start_index} to {end_index - 1}).")
    print("---------------------")

    # If this specific job has no files to process (can happen at the end of a list), exit gracefully.
    if not files_to_process:
        print(f"Job {args.job_id} has no files to process. This is normal for trailing jobs. Exiting.")
        return

    # Loop through the assigned chunk of files
    for input_file_path in tqdm(files_to_process, desc=f"Job {args.job_id} Progress"):
        
        base_filename = os.path.basename(input_file_path)
        output_image_name = os.path.splitext(base_filename)[0] + '.png'
        output_image_path = os.path.join(OUTPUT_DIR, output_image_name)

        # Skip if the output file already exists (allows for easy resuming)
        if os.path.exists(output_image_path):
            continue

        command = [
            'python3',
            SCRIPT_TO_RUN,
            '-i', input_file_path,
            '-o', OUTPUT_DIR,
            '--num_leads', 'twelve',
            '--lead_bbox',
            '--store_config'
        ]
        
        try:
            # Using capture_output=True to catch errors, but not streaming stdout
            # Set to False if you want the child script's output in your main log
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"\n\n--- ERROR in Job {args.job_id} ---")
            print(f"The script failed while processing file: {input_file_path}")
            print(f"STDOUT from child script:\n{e.stdout}")
            print(f"STDERR from child script:\n{e.stderr}")
            print("-----------------------------------")
            # Break to stop this job, but other array jobs will continue.
            break

    print(f"\nGeneration for Job {args.job_id} complete!")

if __name__ == '__main__':
    main()