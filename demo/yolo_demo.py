# File: run_demo.py

import os
from ultralytics import YOLO

# MODEL_TYPE can be 'LL' or '12L'
MODEL_TYPE_TO_RUN = '12L'

# Model Paths
MODELS = {
    'LL': 'C:/Users/dgarc/Desktop/Dissertation/ecg-image-kit/codes/code-yolo/models/runs/yolo_ecg_model4/weights/best.pt',
    '12L': 'C:/Users/dgarc/Desktop/Dissertation/ecg-image-kit/codes/code-yolo/models/runs_12L/yolo_ecg_model_12L3/weights/best.pt'
}

# Directory to the Image Datasets
INPUT_DIR = "C:/Users/dgarc/Desktop/Dissertation/ecg-image-kit/codes/demo/data/yolo/input"
# Directory to save the output images
OUTPUT_DIR = "C:/Users/dgarc/Desktop/Dissertation/ecg-image-kit/codes/demo/data/yolo/output/demo_predictions"


print("Starting Local YOLO Model Demo")

# Select the model
selected_model_path = MODELS.get(MODEL_TYPE_TO_RUN)

print(f"Model Type: {MODEL_TYPE_TO_RUN}")
print("-----------------------------------------")
print(f"Using Model: {selected_model_path}")
print(f"Looking for Images in: {INPUT_DIR}")
print("-----------------------------------------")


# 3. VALIDATION AND SETUP
# Check if the model path is valid
if not selected_model_path or not os.path.exists(selected_model_path):
    print(f"FATAL ERROR: Model file not found at '{selected_model_path}'")
    print("Please check the 'MODELS' dictionary in the configuration section.")
    exit()

# Check if the image directory exists
if not INPUT_DIR or not os.path.isdir(INPUT_DIR):
    print(f"FATAL ERROR: Image directory not found at '{INPUT_DIR}'")
    print("Please check the 'INPUT_DIR' variable in the configuration section.")
    exit()

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Results will be saved in the '{OUTPUT_DIR}' folder.\n")


# 4. EXECUTE THE DEMO
# Load the model
try:
    model = YOLO(selected_model_path)
    class_names = model.names
except Exception as e:
    print(f"FATAL ERROR: Could not load the YOLO model. Error: {e}")
    exit()

# Get a list of all files in the input directory
all_files = os.listdir(INPUT_DIR)
image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not image_files:
    print(f"FATAL ERROR: No image files (.png, .jpg, .jpeg) found in '{INPUT_DIR}'")
    exit()
    
print(f"Found {len(image_files)} images to process.")

# Loop through the discovered image files
for image_filename in image_files:
    # Construct the full path to the current image
    full_image_path = os.path.join(INPUT_DIR, image_filename)

    if not os.path.exists(full_image_path):
        print(f"WARNING: Image '{image_filename}' not found. Skipping.\n")
        continue

    print(f"Processing: {image_filename}")
    
    # Run prediction
    results = model.predict(full_image_path, verbose=False)
    
    result = results[0]
    
    # Define the path for the output image
    output_path = os.path.join(OUTPUT_DIR, f"predicted_{image_filename}")
    
    # Print found objects to the console
    if len(result.boxes) == 0:
        print("  No objects detected.")
    else:
        print(f"  Detected {len(result.boxes)} objects:")
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = class_names[class_id]
            confidence = box.conf[0]
            coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
            print(f"    - Class: {class_name}, Confidence: {confidence:.2f}, Coords: {[round(c) for c in coords]}")

    # Save the image with bounding boxes
    result.save(filename=output_path)
    print(f"  > Saved annotated image to: {output_path}\n")

print("Demo Complete")