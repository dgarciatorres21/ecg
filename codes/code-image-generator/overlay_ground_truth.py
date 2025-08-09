# overlay_ground_truth.py
import cv2
import wfdb
import numpy as np
import os
import json # <<< MODIFICATION: Import the json library

# --- Configuration ---
# 1. File paths (assumes files are in the same directory as the script)
IMAGE_PATH = 'SampleData\GeneratedData\00001_lr-0.png'
RECORD_NAME = 'SampleData\GeneratedData\00001_lr'
OUTPUT_IMAGE_PATH = 'final_overlay_image.png'

# <<< MODIFICATION START: Define the path to the JSON file >>>
BOXES_JSON_PATH = 'bounding_boxes.json'
# <<< MODIFICATION END >>>

# 2. ECG Paper Scaling Parameters
PIXELS_PER_MV = 118

# --- Main Script Logic ---

def load_bounding_boxes(json_path):
    """Loads bounding box data from a JSON file."""
    if not os.path.exists(json_path):
        print(f"Error: Bounding box file not found at '{json_path}'")
        return None
    try:
        with open(json_path, 'r') as f:
            boxes = json.load(f)
        print(f"Successfully loaded bounding boxes from '{json_path}'")
        return boxes
    except Exception as e:
        print(f"Error: Failed to parse JSON file '{json_path}'. Reason: {e}")
        return None

def overlay_waveforms():
    """
    Loads the detection image and ECG data, then overlays the ground truth
    waveforms inside the detected bounding boxes.
    """
    # <<< MODIFICATION START: Load boxes from the file instead of using a hardcoded dict >>>
    BOUNDING_BOXES = load_bounding_boxes(BOXES_JSON_PATH)
    if BOUNDING_BOXES is None:
        return # Stop execution if boxes could not be loaded
    # <<< MODIFICATION END >>>

    # 1. Load the image with the detection boxes
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file not found at '{IMAGE_PATH}'")
        return
    image = cv2.imread(IMAGE_PATH)
    print(f"Loaded image: '{IMAGE_PATH}'")

    # 2. Load the ECG signal data using the WFDB library
    try:
        record = wfdb.rdrecord(RECORD_NAME)
        print(f"Loaded ECG record: '{RECORD_NAME}'")
    except Exception as e:
        print(f"Error: Could not read ECG record '{RECORD_NAME}'. Reason: {e}")
        return

    # Extract signal properties
    signals = record.p_signal
    lead_names = record.sig_name
    gain = record.adc_gain[0]
    baseline = record.baseline[0]
    total_samples = record.sig_len

    # 3. Iterate through each detected box and draw the corresponding waveform
    for lead_label, box in BOUNDING_BOXES.items():
        signal_name = 'II' if lead_label == 'II_long' else lead_label
        try:
            signal_index = lead_names.index(signal_name)
            signal = signals[:, signal_index]
        except ValueError:
            print(f"Warning: Lead '{signal_name}' not found in the record. Skipping.")
            continue
            
        x_min, y_min, x_max, y_max = box
        box_width = x_max - x_min
        isoelectric_y = y_min + (y_max - y_min) / 2
        
        samples_to_draw = total_samples if lead_label == 'II_long' else int(total_samples / 4)
        
        points = []
        for i in range(samples_to_draw):
            pixel_x = int(x_min + (i / (samples_to_draw - 1)) * box_width)
            adc_value = signal[i]
            voltage_mv = (adc_value - baseline) / gain
            pixel_offset = voltage_mv * PIXELS_PER_MV
            pixel_y = int(isoelectric_y - pixel_offset)
            points.append((pixel_x, pixel_y))
            
        cv2.polylines(image, [np.array(points)], isClosed=False, color=(255, 255, 0), thickness=2)

    # 4. Save the final image
    cv2.imwrite(OUTPUT_IMAGE_PATH, image)
    print(f"\nSuccessfully created overlay image: '{OUTPUT_IMAGE_PATH}'")

if __name__ == '__main__':
    try:
        import cv2, wfdb, numpy, json
    except ImportError:
        print("Required libraries not found. Please install them using:")
        print("pip install opencv-python wfdb numpy")
    else:
        overlay_waveforms()