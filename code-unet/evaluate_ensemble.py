import os
import sys
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd

def calculate_dice(pred_mask, gt_mask):
    # Calculates the Dice score between two binary masks.
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
    original_images_dir = "/mnt/parscratch/users/lip24dg/data/Generated_data/Dataset003_ecg/imagesTs"
    ground_truth_masks_dir = "/mnt/parscratch/users/lip24dg/data/Generated_data/Dataset003_ecg/labelsTs"
    predicted_masks_dir = "/mnt/parscratch/users/lip24dg/data/Generated_data/ecg_predictions_D3_ensemble"
    output_csv_path = "/mnt/parscratch/users/lip24dg/data/Generated_data/evaluation_results_D3_ensemble.csv"

    results = []
    
    if not os.path.exists(predicted_masks_dir):
        print(f"Error: Prediction directory not found at {predicted_masks_dir}", file=sys.stderr)
        return

    for pred_mask_filename in tqdm(sorted(os.listdir(predicted_masks_dir)), desc="Evaluating Predictions"):
        if not pred_mask_filename.endswith('.nii.gz'):
            continue
        
        try:
            # Reconstruct the original image filename from the prediction filename
            base_name, extension = os.path.splitext(pred_mask_filename)
            original_img_filename = f"{base_name}_0000{extension}"
            
            gt_mask_filename = pred_mask_filename # Ground truth mask has the same name as the prediction mask

            pred_mask_path = os.path.join(predicted_masks_dir, pred_mask_filename)
            original_img_path = os.path.join(original_images_dir, original_img_filename)
            gt_mask_path = os.path.join(ground_truth_masks_dir, gt_mask_filename)

            if not os.path.exists(original_img_path):
                print(f"Warning: Skipping {pred_mask_filename}, original image not found at {original_img_path}")
                continue
            if not os.path.exists(gt_mask_path):
                print(f"Warning: Skipping {pred_mask_filename}, ground truth not found at {gt_mask_path}")
                continue

            # DATA LOADING
            pred_mask_np = sitk.GetArrayFromImage(sitk.ReadImage(pred_mask_path)).squeeze()
            original_signal_np = sitk.GetArrayFromImage(sitk.ReadImage(original_img_path)).squeeze()
            gt_mask_np = sitk.GetArrayFromImage(sitk.ReadImage(gt_mask_path)).squeeze()

            # METRIC CALCULATIONS
            dice = calculate_dice(pred_mask_np, gt_mask_np)
            
            true_signal = original_signal_np * gt_mask_np
            reconstructed_signal = original_signal_np * pred_mask_np
            noise_signal = reconstructed_signal - true_signal
            
            mse = calculate_mse(reconstructed_signal, true_signal)
            snr = calculate_snr(true_signal, noise_signal)

            results.append({
                'filename': pred_mask_filename, 'dice': dice, 'mse': mse, 'snr_db': snr
            })

        except Exception as e:
            print(f"Could not process {pred_mask_filename}. Error: {e}", file=sys.stderr)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv_path, index=False)
        print("\n--- Custom Evaluation Summary on Test Set (Ensemble) ---")
        print(f"Mean Dice: {df['dice'].mean():.4f} (+/- {df['dice'].std():.4f})")
        print(f"Mean MSE:  {df['mse'].mean():.4f} (+/- {df['mse'].std():.4f})")
        print(f"Mean SNR:  {df['snr_db'].mean():.2f} dB (+/- {df['snr_db'].std():.2f} dB)")
        print(f"\nDetailed results saved to: {output_csv_path}")
    else:
        print("No results were generated. Check paths and file names.")

if __name__ == "__main__":
    main()
