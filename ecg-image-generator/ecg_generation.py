import os
import subprocess
from tqdm import tqdm
import argparse
import sys

def get_args():
    """
    Parses command-line arguments. This will parse the arguments it knows and
    leave the rest for the downstream script.
    """
    parser = argparse.ArgumentParser(
        description="Manager script: Determines which files to process and passes all other settings to the worker script."
    )
    # Arguments for this script (the "Manager")
    parser.add_argument('--script-to-run', type=str, required=True)
    parser.add_argument('--data-root-dir', type=str, required=True)
    parser.add_argument('--image-output-dir', type=str, required=True)
    parser.add_argument('--mask-output-dir', type=str, required=True)
    parser.add_argument('--job-id', type=int)
    parser.add_argument('--total-jobs', type=int)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--start-index', type=int)
    parser.add_argument('--num-files', type=int)
    
    args, unknown_args = parser.parse_known_args() # This tells the parser to not crash on arguments it doesn't recognize.
    return args, unknown_args


def main():
    # --- Capture both known and unknown arguments ---
    args, unknown_args = get_args()

    print(f"--- Manager Script Started ---")
    print(f"Data Source Directory: {args.data_root_dir}")
    print(f"Image Output Directory: {args.image_output_dir}")
    if unknown_args:
        print(f"Passing through unknown args to worker: {unknown_args}")
    print(f"------------------------------")

    os.makedirs(args.image_output_dir, exist_ok=True)
    os.makedirs(args.mask_output_dir, exist_ok=True)

    all_file_paths = []
    print(f"Scanning {args.data_root_dir} for all matching .dat and .hea file pairs...")
    for root, _, files in os.walk(args.data_root_dir):
        for filename in files:
            if filename.endswith('.dat'):
                dat_path = os.path.join(root, filename)
                hea_path = os.path.splitext(dat_path)[0] + '.hea'
                if os.path.exists(hea_path):
                    all_file_paths.append(dat_path)
    all_file_paths.sort()
    total_files_found = len(all_file_paths)
    print(f"Found a total of {total_files_found} valid records.")

    files_to_process = []
    if args.start_index is not None and args.num_files is not None:
        print(f"--- RANGE MODE: Processing {args.num_files} files from index {args.start_index}. ---")
        start, end = args.start_index, args.start_index + args.num_files
        files_to_process = all_file_paths[start:end]
    elif args.limit is not None:
        print(f"--- TEST MODE: Limiting to the first {args.limit} files. ---")
        files_to_process = all_file_paths[:args.limit]
        args.job_id = 0
    elif args.job_id is not None and args.total_jobs is not None:
        print(f"--- JOB ARRAY MODE: Calculating chunk for job {args.job_id}/{args.total_jobs}. ---")
        chunk_size = (total_files_found + args.total_jobs - 1) // args.total_jobs
        start, end = args.job_id * chunk_size, min(args.job_id * chunk_size + chunk_size, total_files_found)
        files_to_process = all_file_paths[start:end]
        print(f"This job will process files from index {start} to {end - 1}.")
    else:
        print("Error: No mode selected.")
        return

    if not files_to_process:
        print("No files selected for processing. Exiting.")
        return
        
    print(f"This run will process {len(files_to_process)} files.")
    print("-------------------------------------------------")

    job_desc = f"Job {args.job_id}" if args.job_id is not None else "Run"
    for input_file_path in tqdm(files_to_process, desc=f"{job_desc} Progress"):
        base_filename = os.path.basename(input_file_path)
        output_image_name = os.path.splitext(base_filename)[0] + '.png'
        output_image_path = os.path.join(args.image_output_dir, output_image_name)

        if os.path.exists(output_image_path):
            continue

        # Build the command for the worker script
        command = [
            sys.executable,
            args.script_to_run,
            '-i', input_file_path,
            '-o', args.image_output_dir,
            '--mask-output-dir', args.mask_output_dir,
        ]
        
        command.extend(unknown_args)
        
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"\n\n--- ERROR in {job_desc} ---")
            print(f"Failed on file: {input_file_path}")
            print(f"STDOUT from worker:\n{e.stdout}")
            print(f"STDERR from worker:\n{e.stderr}")
            print("-----------------------------------")
            break

    print(f"\nGeneration for {job_desc} complete!")


if __name__ == '__main__':
    main()
