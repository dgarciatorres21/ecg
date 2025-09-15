import os
import json
import argparse
from PIL import Image
from tqdm import tqdm

CLASS_MAP = {
    'I': 0, 'II': 1, 'III': 2, 'aVR': 3, 'aVL': 4, 'aVF': 5,
    'V1': 6, 'V2': 7, 'V3': 8, 'V4': 9, 'V5': 10, 'V6': 11, 'L': 12
}

rhythm_WIDTH_THRESHOLD = 1000

def convert_corners_to_xywh(corners_dict):
    points = list(corners_dict.values())
    y_coords = [p[0] for p in points]
    x_coords = [p[1] for p in points]
    x_min, y_min = min(x_coords), min(y_coords)
    x_max, y_max = max(x_coords), max(y_coords)
    width = x_max - x_min
    height = y_max - y_min
    return [x_min, y_min, width, height]

def convert_bbox_to_yolo(bbox, img_width, img_height):
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2) / img_width
    y_center = (y_min + h / 2) / img_height
    width_norm = w / img_width
    height_norm = h / img_height
    return x_center, y_center, width_norm, height_norm

def process_json_annotations(data_dir, yolo_labels_dir):
    os.makedirs(yolo_labels_dir, exist_ok=True)
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    print(f"Found {len(json_files)} .json files in '{data_dir}'. Starting conversion...")
    
    for json_filename in tqdm(json_files, desc="Converting JSON to YOLO"):
        base_filename = os.path.splitext(json_filename)[0]
        json_path = os.path.join(data_dir, json_filename)
        output_txt_path = os.path.join(yolo_labels_dir, base_filename + '.txt')

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            img_width = data['width']
            img_height = data['height']
        except (KeyError, json.JSONDecodeError, FileNotFoundError) as e:
            print(f"\nWarning: Could not process {json_filename}. Error: {e}. Skipping.")
            continue

        if 'leads' not in data:
            continue

        yolo_lines = []
        for lead_info in data['leads']:
            if 'lead_bounding_box' in lead_info and 'lead_name' in lead_info:
                lead_name = lead_info['lead_name']
                corners = lead_info['lead_bounding_box']
                
                xywh_bbox = convert_corners_to_xywh(corners)
                class_id = -1
                
                if lead_name == 'II':
                    box_width = xywh_bbox[2]
                    if box_width > rhythm_WIDTH_THRESHOLD:
                        class_id = CLASS_MAP['L']
                    else:
                        class_id = CLASS_MAP['II']
                elif lead_name in CLASS_MAP:
                    class_id = CLASS_MAP[lead_name]
                
                if class_id != -1:
                    x_center, y_center, w_norm, h_norm = convert_bbox_to_yolo(xywh_bbox, img_width, img_height)
                    yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        with open(output_txt_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

    print("\nConversion complete!")
    print(f"YOLO label files have been saved to: {yolo_labels_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert JSON annotations to YOLOv8 format.")
    parser.add_argument('--data-dir', type=str, required=True, help="Directory containing the source .json files.")
    parser.add_argument('--output-dir', type=str, required=True, help="Directory where the output .txt label files will be saved.")
    args = parser.parse_args()
    
    process_json_annotations(args.data_dir, args.output_dir)