import numpy as np
import SimpleITK as sitk
import os

def calculate_mse(signal_A, signal_B):
    # calculates the mean squared error between two signals
    return np.mean((signal_A - signal_B) ** 2)

def calculate_snr(signal, noise):
    # calculates the signal to noise ratio
    power_signal = np.mean(signal ** 2)
    power_noise = np.mean(noise ** 2)
    if power_noise == 0:
        return float('inf') # infinite snr if there is no noise
    return 10 * np.log10(power_signal / power_noise)


# --- define paths ---
original_images_dir = '/path/to/your/imagesTs'
ground_truth_masks_dir = '/path/to/your/labelsTs'
predicted_masks_dir = '/path/to/your/predictions'

all_mse = []
all_snr = []

# loop through all predicted masks
for pred_mask_filename in os.listdir(predicted_masks_dir):
    # 1. load the predicted mask
    pred_mask_itk = sitk.ReadImage(os.path.join(predicted_masks_dir, pred_mask_filename))
    pred_mask_np = sitk.GetArrayFromImage(pred_mask_itk).astype(float)

    # 2. load the original image (signal)
    original_image_itk = sitk.ReadImage(os.path.join(original_images_dir, pred_mask_filename.replace('.nii.gz', '_0000.nii.gz')))
    original_signal_np = sitk.GetArrayFromImage(original_image_itk).astype(float)

    # 3. load the ground truth mask
    gt_mask_itk = sitk.ReadImage(os.path.join(ground_truth_masks_dir, pred_mask_filename))
    gt_mask_np = sitk.GetArrayFromImage(gt_mask_itk).astype(float)

    # reconstruct the signal using the predicted mask
    reconstructed_signal = original_signal_np * pred_mask_np

    # define the "true" signal (original signal only where the ground truth mask is)
    true_signal_component = original_signal_np * gt_mask_np
    
    # calculate mse between the reconstructed signal and the true signal component
    mse = calculate_mse(reconstructed_signal, true_signal_component)
    all_mse.append(mse)

    # calculate snr
    noise_component = reconstructed_signal - true_signal_component
    snr = calculate_snr(true_signal_component, noise_component)
    all_snr.append(snr)
    
    print(f"Case: {pred_mask_filename} -> MSE: {mse:.4f}, SNR: {snr:.4f} dB")


print(f"\n--- Average Results ---")
print(f"Average MSE: {np.mean(all_mse):.4f}")
print(f"Average SNR: {np.mean(all_snr):.4f} dB")
