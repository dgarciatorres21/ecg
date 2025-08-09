# --- File: crop_leads_for_nnunet.py (Job Array Version) ---

import os
import argparse
from ultralytics import YOLO
import cv2
from tqdm import tqdm

def main():
    # --- 1. Argument Parser ---
    # Add arguments for the job array
    parser = argparse.ArgumentParser(description="Use YOLOv8 to crop ECG leads. Part of a job array.")
    parser.add_argument('--model-path', type=str, required=True, help='Path to the best.pt model.')
    parser.add_argument('--image-source-dir', type=str, required=True, help='Path to source images.')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to save cropped images.')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold.')
    # Arguments passed from the SLURM job array
    parser.add_argument('--job-id', type=int, required=True, help='SLURM_ARRAY_TASK_ID of this job.')
    parser.add_argument('--total-jobs', type=int, required=True, help='Total number of jobs in the array.')
    args = parser.parse_args()

    # --- 2. Configuration & Setup ---
    MODEL_PATH = args.model_path
    IMAGE_SOURCE_DIR = args.image_source_dir
    OUTPUT_DIR = args.output_dir
    CONFIDENCE_THRESHOLD = args.conf

    # ... (Path validation and model loading are the same) ...
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = YOLO(MODEL_PATH)

    # --- 3. Get and Split the File List ---
    print(f"--- Job {args.job_id} of {args.total_jobs} ---")
    print("Scanning for all source images...")
    all_image_files = sorted([f for f in os.listdir(IMAGE_SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not all_image_files:
        print(f"FATAL ERROR: No images found in {IMAGE_SOURCE_DIR}"); return

    num_files = len(all_image_files)
    print(f"Found {num_files} total images. Assigning chunk to this job...")

    # Perform ceiling division to calculate chunk size
    chunk_size = (num_files + args.total_jobs - 1) // args.total_jobs
    start_index = args.job_id * chunk_size
    end_index = min(start_index + chunk_size, num_files)
    
    # Get the specific slice of files for this job
    files_to_process = all_image_files[start_index:end_index]

    print(f"This job will process {len(files_to_process)} files (from index {start_index} to {end_index - 1}).")
    
    # --- 4. Process the Assigned Chunk ---
    total_cropped_images = 0
    # Use the job ID in the tqdm description for clear logging
    for image_file in tqdm(files_to_process, desc=f"Job {args.job_id} Cropping"):
        image_path = os.path.join(IMAGE_SOURCE_DIR, image_file)
        original_image = cv2.imread(image_path)
        if original_image is None: continue

        results = model.predict(source=image_path, conf=CONFIDENCE_THRESHOLD, verbose=False)
        boxes = results[0].boxes
        base_name = os.path.splitext(image_file)[0]

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            
            cropped_lead_image = original_image[y1:y2, x1:x2]
            
            output_filename = f"{base_name}_{class_name}.png"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            cv2.imwrite(output_path, cropped_lead_image)
            total_cropped_images += 1

    print(f"\n--- Job {args.job_id} Complete ---")
    print(f"Created {total_cropped_images} cropped images.")

if __name__ == '__main__':
    main()