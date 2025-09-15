import os
import sys
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd
import argparse

def calculate_dice(pred_mask, gt_mask):
    # Calculates the Dice score between two binary masks
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"Shape mismatch: pred {pred_mask.shape} vs gt {gt_mask.shape}")
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)
    intersection = np.sum(pred_mask * gt_mask)
    volume_sum = np.sum(pred_mask) + np.sum(gt_mask)
    if volume_sum == 0: return 1.0
    return 2.0 * intersection / volume_sum

def calculate_mse(signal_A, signal_B):
    # Calculates the Mean Squared Error between two signals.
    return np.mean((signal_A - signal_B) ** 2)

def calculate_snr(signal, noise):
    # Calculates the Signal to Noise Ratio in dB.
    power_signal = np.mean(signal ** 2)
    power_noise = np.mean(noise ** 2)
    if power_noise < 1e-9: return float('inf')
    if power_signal < 1e-9: return -float('inf')
    return 10 * np.log10(power_signal / power_noise)

def main():
    # original_images_dir = "/mnt/parscratch/users/lip24dg/data/Generated_data/Dataset003_ecg/imagesTs"
    # ground_truth_masks_dir = "/mnt/parscratch/users/lip24dg/data/Generated_data/Dataset003_ecg/labelsTs"
    # predicted_masks_dir = "/mnt/parscratch/users/lip24dg/data/Generated_data/ecg_predictions_D3_ensemble"
    # output_csv_path = "/mnt/parscratch/users/lip24dg/data/Generated_data/evaluation_results_per_lead.csv" # New output file
    
    # --- Paths from arguments ---
    parser = argparse.ArgumentParser(description="Calculate per-lead evaluation metrics for nnU-Net predictions.")
    parser.add_argument('--pred_dir', required=True, help="Directory with the single-lead prediction .nii.gz files.")
    parser.add_argument('--gt_dir', required=True, help="Directory with the single-lead ground truth .nii.gz masks.")
    parser.add_argument('--img_dir', required=True, help="Directory with the single-lead original .nii.gz images.")
    parser.add_argument('--csv_out', required=True, help="Full path for the output .csv results file.")
    args = parser.parse_args()

    predicted_masks_dir = args.pred_dir
    ground_truth_masks_dir = args.gt_dir
    original_images_dir = args.img_dir
    output_csv_path = args.csv_out

    results = []
    
    if not os.path.exists(predicted_masks_dir):
        print(f"Error: Prediction directory not found at {predicted_masks_dir}", file=sys.stderr)
        return

    for pred_mask_filename in tqdm(sorted(os.listdir(predicted_masks_dir)), desc="Evaluating Predictions"):
        if not pred_mask_filename.endswith('.nii.gz'):
            continue
        
        try:
            # Parse lead name from filename
            base_name_no_ext = pred_mask_filename.replace('.nii.gz', '')
            parts = base_name_no_ext.split('_')
            lead_name = parts[-1]
            # Reconstruct the original image filename from the prediction filename
            original_img_filename = pred_mask_filename.replace('.nii.gz', '_0000.nii.gz')
            gt_mask_filename = pred_mask_filename

            pred_mask_path = os.path.join(predicted_masks_dir, pred_mask_filename)
            original_img_path = os.path.join(original_images_dir, original_img_filename)
            gt_mask_path = os.path.join(ground_truth_masks_dir, gt_mask_filename)

            if not (os.path.exists(original_img_path) and os.path.exists(gt_mask_path)):
                print(f"Warning: Skipping {pred_mask_filename}, a required file was not found.")
                continue

            pred_mask_np = sitk.GetArrayFromImage(sitk.ReadImage(pred_mask_path)).squeeze()
            original_signal_np = sitk.GetArrayFromImage(sitk.ReadImage(original_img_path)).squeeze()
            gt_mask_np = sitk.GetArrayFromImage(sitk.ReadImage(gt_mask_path)).squeeze()

            dice = calculate_dice(pred_mask_np, gt_mask_np)
            
            true_signal = original_signal_np * gt_mask_np
            reconstructed_signal = original_signal_np * pred_mask_np
            noise_signal = reconstructed_signal - true_signal
            
            mse = calculate_mse(reconstructed_signal, true_signal)
            snr = calculate_snr(true_signal, noise_signal)

            results.append({
                'filename': pred_mask_filename, 
                'lead': lead_name,
                'dice': dice, 
                'mse': mse, 
                'snr_db': snr
            })

        except Exception as e:
            print(f"Could not process {pred_mask_filename}. Error: {e}", file=sys.stderr)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv_path, index=False)
        
        # Per-lead calculations
        print("\n--- Custom Evaluation Summary on Test Set (Per Lead) ---")
        
        per_lead_summary = df.groupby('lead').agg({
            'dice': ['mean', 'std'],
            'mse': ['mean', 'std'],
            'snr_db': ['mean', 'std']
        })
        
        # Print summary table
        print(per_lead_summary)
        print(f"\nDetailed per-file results saved to: {output_csv_path}")
    else:
        print("No results were generated. Check paths and file names.")

if __name__ == "__main__":
    main()