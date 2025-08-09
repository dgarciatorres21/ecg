import os
import json
import argparse
from PIL import Image, ImageDraw
from tqdm import tqdm

def generate_masks(input_dir, output_dir):
    """
    Scans the input directory for .json files and generates corresponding
    segmentation masks.
    """
    os.makedirs(output_dir, exist_ok=True)
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

    if not json_files:
        print(f"Error: No .json files found in '{input_dir}'. Exiting.")
        return

    print(f"Found {len(json_files)} .json files in '{input_dir}'.")
    print(f"Generating masks and saving to '{output_dir}'...")

    for json_filename in tqdm(json_files, desc="Generating Masks"):
        json_path = os.path.join(input_dir, json_filename)
        base_filename = os.path.splitext(json_filename)[0]
        
        # Mask will have the same name as the JSON, but with a .png extension
        mask_image_path = os.path.join(output_dir, f"{base_filename}.png")

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            img_width = data['width']
            img_height = data['height']

            mask_image = Image.new('L', (img_width, img_height), 0) # 0 = black
            draw = ImageDraw.Draw(mask_image)

            for lead_info in data.get('leads', []):
                if 'lead_bounding_box' in lead_info:
                    corners = lead_info['lead_bounding_box'].values()
                    y_coords = [p[0] for p in corners]
                    x_coords = [p[1] for p in corners]
                    
                    x_min, y_min = min(x_coords), min(y_coords)
                    x_max, y_max = max(x_coords), max(y_coords)

                    draw.rectangle([(x_min, y_min), (x_max, y_max)], fill=255) # 255 = white

            mask_image.save(mask_image_path)

        except Exception as e:
            print(f"\nWarning: Could not process {json_filename}. Error: {e}. Skipping.")
            continue

    print("\nMask generation complete!")

def main():
    parser = argparse.ArgumentParser(description="Generate segmentation masks from ECG JSON data.")
    parser.add_argument('--input-dir', type=str, required=True, 
                        help='Directory containing the generated .json files.')
    parser.add_argument('--output-dir', type=str, required=True, 
                        help='Directory where the mask images will be saved.')
    args = parser.parse_args()
    
    generate_masks(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main()