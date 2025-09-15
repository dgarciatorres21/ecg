# diagnose_json.py
import os
import json
from tqdm import tqdm

DATA_DIR = '../ecg-image-generator/outputData/Generated_data'

def convert_corners_to_xywh(corners_dict):
    """Helper function to get bbox"""
    points = list(corners_dict.values())
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x_min, y_min = min(x_coords), min(y_coords)
    x_max, y_max = max(x_coords), max(y_coords)
    width = x_max - x_min
    height = y_max - y_min
    return [x_min, y_min, width, height]

def find_bad_annotations():
    json_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    
    print(f"Diagnosing {len(json_files)} .json files...\n")
    
    found_errors = False
    for json_filename in tqdm(json_files, desc="Checking files"):
        json_path = os.path.join(DATA_DIR, json_filename)

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            img_width = data['width']
            img_height = data['height']
        except Exception:
            continue

        if 'leads' not in data:
            continue

        for lead_info in data['leads']:
            if 'lead_bounding_box' in lead_info:
                corners = lead_info['lead_bounding_box']
                x_min, y_min, w, h = convert_corners_to_xywh(corners)

                # the crucial check
                if (x_min + w) > img_width or (y_min + h) > img_height:
                    print(f"--- ERROR FOUND in {json_filename} ---")
                    print(f"  Image dimensions: width={img_width}, height={img_height}")
                    print(f"  Lead '{lead_info.get('lead_name', 'N/A')}' has an out-of-bounds box:")
                    print(f"  Box ends at X={x_min+w}, Y={y_min+h}\n")
                    found_errors = True
    
    if not found_errors:
        print("\nDiagnosis complete. No out-of-bounds bounding boxes were found.")
    else:
        print("\nDiagnosis complete. Found one or more files with errors.")

if __name__ == '__main__':
    find_bad_annotations()
