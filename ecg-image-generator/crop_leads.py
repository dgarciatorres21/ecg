import os
import argparse
import cv2
from ultralytics import YOLO
from tqdm import tqdm

def main():
    # --- 1. Argument Parser (with new arguments) ---
    parser = argparse.ArgumentParser(description="Use YOLOv8 to crop ECG leads from both realistic images and their corresponding masks.")
    parser.add_argument('--model-path', type=str, required=True, help='Path to the best.pt model.')
    parser.add_argument('--image-source-dir', type=str, required=True, help='Path to source realistic images (X).')
    parser.add_argument('--image-output-dir', type=str, required=True, help='Path to save cropped realistic images.')
    parser.add_argument('--mask-source-dir', type=str, required=True, help='Path to source mask images (Y).')
    parser.add_argument('--mask-output-dir', type=str, required=True, help='Path to save cropped mask images.')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold.')
    parser.add_argument('--job-id', type=int, help='SLURM_ARRAY_TASK_ID of this job.')
    parser.add_argument('--total-jobs', type=int, help='Total number of jobs in the array.')
    parser.add_argument('--limit', type=int, help='For local testing: process only the first N files.')
    
    args = parser.parse_args()

    # --- 2. Configuration & Setup ---
    os.makedirs(args.image_output_dir, exist_ok=True)
    os.makedirs(args.mask_output_dir, exist_ok=True)
    
    # --- Error messages ---
    try:
        model = YOLO(args.model_path)
        print(f"Successfully loaded YOLO model from: {args.model_path}")
    except Exception as e:
        print(f"\n--- FATAL ERROR ---")
        print(f"Failed to load YOLO model from path: '{args.model_path}'")
        print("Please check that the path is correct and the model file is not corrupted.")
        print(f"Underlying error: {e}")
        return # Exit the script if the model can't be loaded

    # --- 3. Get and Split the File List ---
    print(f"Scanning for all source images in: {args.image_source_dir}")
    all_image_files = sorted([f for f in os.listdir(args.image_source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not all_image_files:
        print(f"FATAL ERROR: No images found in {args.image_source_dir}"); return

    # --- Logic to select files based on the chosen mode ---
    files_to_process = []
    
    # PRIORITY 1: Local Test Mode
    if args.limit is not None:
        print(f"--- TEST MODE: Limiting to the first {args.limit} files. ---")
        files_to_process = all_image_files[:args.limit]
        job_desc = "Test Run"
    
    # PRIORITY 2: HPC Job Array Mode
    elif args.job_id is not None and args.total_jobs is not None:
        print(f"--- JOB ARRAY MODE: Job {args.job_id} of {args.total_jobs} ---")
        num_files = len(all_image_files)
        chunk_size = (num_files + args.total_jobs - 1) // args.total_jobs
        start_index = args.job_id * chunk_size
        end_index = min(start_index + chunk_size, num_files)
        files_to_process = all_image_files[start_index:end_index]
        job_desc = f"Job {args.job_id}"
    
    else:
        print("\n--- FATAL ERROR: No processing mode specified. ---")
        print("Please provide either --limit (for local testing) or both --job-id and --total-jobs (for HPC).")
        return

    if not files_to_process:
        print("No files assigned to this job. Exiting.")
        return

    print(f"This run will process {len(files_to_process)} files.")
    print("-------------------------------------------------")
    
    # --- 4. Process the Assigned Chunk ---
    total_cropped_images = 0
    for image_file in tqdm(files_to_process, desc=f"{job_desc} Cropping"):
        image_path = os.path.join(args.image_source_dir, image_file)
        original_image = cv2.imread(image_path)
        if original_image is None: continue

        mask_path = os.path.join(args.mask_source_dir, image_file)
        if not os.path.exists(mask_path):
            tqdm.write(f"  - Warning: Mask not found for {image_file}, skipping.") # Use tqdm.write to avoid breaking the progress bar
            continue
        mask_image = cv2.imread(mask_path)
        if mask_image is None: continue

        # Run prediction ONCE on the realistic image
        results = model.predict(source=image_path, conf=args.conf, verbose=False)
        boxes = results[0].boxes
        base_name = os.path.splitext(image_file)[0]

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            
            # Crop the realistic image
            cropped_lead_image = original_image[y1:y2, x1:x2]
            img_output_filename = f"{base_name}_{class_name}.png"
            img_output_path = os.path.join(args.image_output_dir, img_output_filename)
            cv2.imwrite(img_output_path, cropped_lead_image)
            
            # Crop the mask image using the SAME coordinates
            cropped_mask_image = mask_image[y1:y2, x1:x2]
            # --- For nnU-Net: the mask and image must have the same name ---
            mask_output_filename = f"{base_name}_{class_name}_mask.png"
            mask_output_path = os.path.join(args.mask_output_dir, img_output_filename)
            cv2.imwrite(mask_output_path, cropped_mask_image)

            total_cropped_images += 1

    print(f"\n--- {job_desc} Complete ---")
    print(f"Created {total_cropped_images} cropped image/mask pairs.")

if __name__ == '__main__':
    main()