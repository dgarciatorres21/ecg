import os
import sys
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd
import argparse

def vectorize_mask(mask_np):

    height, width = mask_np.shape
    reconstructed_signal = np.zeros(width)

    for x in range(width):
        # Find the y-coordinates of all non-zero pixels in the current column
        y_coords = np.where(mask_np[:, x] > 0)[0]
        
        if len(y_coords) > 0:
            # Calculate the mean y-coordinate (center of mass)
            reconstructed_signal[x] = np.mean(y_coords)
        else:
            # If there's a gap in the signal, mark it as NaN (Not a Number)
            reconstructed_signal[x] = np.nan
            
    return reconstructed_signal

def scale_signal(raw_signal_px, dpi, mm_per_mv, mm_per_second):

    # Conversion factors
    px_per_inch = dpi
    mm_per_inch = 25.4
    px_per_mm = px_per_inch / mm_per_inch

    # Invert the y-axis because image coordinates start from the top
    height = 180
    raw_signal_px = height/2 - raw_signal_px

    # Scale to clinical units
    voltage_mv = (raw_signal_px / px_per_mm) / mm_per_mv
    
    time_seconds = (np.arange(len(raw_signal_px)) / px_per_mm) / mm_per_second
    
    return time_seconds, voltage_mv

def main():
    parser = argparse.ArgumentParser(description="Reconstruct 1D signals from 2D nnU-Net prediction masks.")
    parser.add_argument('-i', '--input_dir', required=True, help="Path to the directory with the predicted .nii.gz masks.")
    parser.add_argument('-o', '--output_dir', required=True, help="Path to the directory where the output .csv files will be saved.")
    args = parser.parse_args()

    predicted_masks_dir = args.input_dir
    output_csv_dir = args.output_dir
    IMAGE_DPI = 300
    MM_PER_MV = 10
    MM_PER_SECOND = 25

    os.makedirs(output_csv_dir, exist_ok=True)
    
    if not os.path.exists(predicted_masks_dir):
        print(f"Error: Prediction directory not found at {predicted_masks_dir}", file=sys.stderr)
        return

    for mask_filename in tqdm(sorted(os.listdir(predicted_masks_dir)), desc="Reconstructing Signals"):
        if not mask_filename.endswith('.nii.gz'):
            continue
        
        try:
            mask_path = os.path.join(predicted_masks_dir, mask_filename)
            mask_np = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).squeeze()
            
            raw_pixel_signal = vectorize_mask(mask_np)
            time_s, voltage_mv = scale_signal(raw_pixel_signal, IMAGE_DPI, MM_PER_MV, MM_PER_SECOND)

            df = pd.DataFrame({'time_seconds': time_s, 'voltage_mv': voltage_mv})
            df.interpolate(method='linear', inplace=True)
            df.fillna(method='bfill', inplace=True)
            df.fillna(method='ffill', inplace=True)

            output_filename = mask_filename.replace('.nii.gz', '.csv')
            output_path = os.path.join(output_csv_dir, output_filename)
            df.to_csv(output_path, index=False)

        except Exception as e:
            print(f"Could not process {mask_filename}. Error: {e}", file=sys.stderr)

    print(f"\nSignal reconstruction complete. CSV files saved in: {output_csv_dir}")

if __name__ == "__main__":
    main()