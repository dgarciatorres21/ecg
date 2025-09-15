import os
import argparse
from collections import defaultdict

# Scans a directory for ECG records and reports on the completeness of .dat/.hea pairs.
def audit_wfdb_records(directory_path: str):
    # Check if the directory exists before proceeding
    if not os.path.isdir(directory_path):
        print(f"--- FATAL ERROR ---")
        print(f"Directory not found at: '{directory_path}'")
        print("Please provide a valid path.")
        return

    print(f"--- Starting WFDB Record Audit of Directory: {directory_path} ---\n")
    # Dictionary to hold the records and their found extensions
    records = defaultdict(set)

    target_extensions = {'.dat', '.hea'}
    
    # --- Step 1: Recursively scan the directory and collect file info ---
    print("Scanning for .dat and .hea files...")
    for root, _, files in os.walk(directory_path):
        for filename in files:
            base, ext = os.path.splitext(filename)
            if ext in target_extensions:
                record_key = os.path.join(root, base)
                records[record_key].add(ext)

    if not records:
        print("\n--- Audit Complete ---")
        print("No .dat or .hea files were found in the specified directory.")
        return

    # --- Step 2: Analyze the collected data ---
    total_counts = defaultdict(int)
    incomplete_records = []
    
    expected_set = {'.dat', '.hea'}

    # Sort the record keys for consistent output
    sorted_record_keys = sorted(records.keys())

    for record_key in sorted_record_keys:
        found_extensions = records[record_key]
        
        # Increment total counts
        for ext in found_extensions:
            total_counts[ext] += 1
            
        # Check if the record is incomplete
        missing_extensions = expected_set - found_extensions
        if missing_extensions:
            # Store the record key and a sorted list of what's missing
            incomplete_records.append((record_key, sorted(list(missing_extensions))))

    # --- Step 3: Print a clear and readable report ---
    print("\n--- Audit Report ---")
    print("\n[1] Total File Counts:")
    print(f"  - Found {total_counts['.dat']} '.dat' files.")
    print(f"  - Found {total_counts['.hea']} '.hea' files.")

    print("\n[2] Record Completeness Check (.dat/.hea pairs):")
    if not incomplete_records:
        print("  ✅ Success! All records are complete (have both .dat and .hea files).")
    else:
        print(f"  ❌ Found {len(incomplete_records)} incomplete record(s).")
        print("   (A record is a file that has a .dat or .hea, but not both)")
        print("-" * 30)
        for record_key, missing in incomplete_records:
            relative_record_path = os.path.relpath(record_key, directory_path)
            print(f"  Record: {relative_record_path:<40} -> MISSING: {', '.join(missing)}")
        print("-" * 30)
        
    print("\n--- Audit Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audit an ECG dataset to count .dat/.hea files and check for missing pairs for each record."
    )
    parser.add_argument(
        '--directory', 
        type=str, 
        required=True, 
        help="The path to the dataset directory to audit."
    )
    args = parser.parse_args()
    
    audit_wfdb_records(args.directory)