import os
import sys
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd
import argparse
import subprocess
import matplotlib.pyplot as plt

# --- Helper Functions ---
def vectorize_mask(mask_np):
    # Converts a 2D binary mask into a 1D signal
    height, width = mask_np.shape
    reconstructed_signal = np.zeros(width)
    for x in range(width):
        y_coords = np.where(mask_np[:, x] > 0)[0]
        if len(y_coords) > 0:
            reconstructed_signal[x] = np.mean(y_coords)
        else:
            reconstructed_signal[x] = np.nan
    return reconstructed_signal

def scale_signal(raw_signal_px, dpi, mm_per_mv, mm_per_second, image_height):
    # Converts the raw pixel signal to clinical units
    px_per_mm = dpi / 25.4
    raw_signal_px = image_height / 2 - raw_signal_px # Invert y-axis
    voltage_mv = (raw_signal_px / px_per_mm) / mm_per_mv
    time_seconds = (np.arange(len(raw_signal_px)) / px_per_mm) / mm_per_second
    return time_seconds, voltage_mv

def main():
    # paths as arguments
    parser = argparse.ArgumentParser(description="Full demo pipeline: predict, reconstruct, and plot ECG signals.")
    parser.add_argument('-i', '--input_dir', required=True, help="Path to the directory with a few sample .nii.gz images.")
    parser.add_argument('-o', '--output_dir', required=True, help="Path to a directory where all demo results will be saved.")
    parser.add_argument('-d', '--dataset_id', required=True, type=int, help="The Dataset ID of your trained model (e.g., 5).")
    args = parser.parse_args()

    # Setup Paths
    DATASET_ID = args.dataset_id
    input_images_dir = args.input_dir
    base_output_dir = args.output_dir
    
    predictions_dir = os.path.join(base_output_dir, "predictions")
    reconstructed_csv_dir = os.path.join(base_output_dir, "reconstructed_csvs")
    final_plots_dir = os.path.join(base_output_dir, "final_plots")

    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(reconstructed_csv_dir, exist_ok=True)
    os.makedirs(final_plots_dir, exist_ok=True)

    print(f"--- Starting Demo for Dataset {DATASET_ID} ---")
    print(f"Input Images: {input_images_dir}")
    print(f"Output will be saved in: {base_output_dir}")

    # STAGE 1: Run nnU-Net Prediction
    print("\n--- [Stage 1/3] Running nnU-Net Prediction... ---")
    try:
        cmd = [
            "nnUNetv2_predict",
            "-i", input_images_dir,
            "-o", predictions_dir,
            "-d", str(DATASET_ID),
            "-c", "2d",
            "--save_probabilities" # Use ensemble by default
        ]
        subprocess.run(cmd, check=True)
        print("Prediction successful.")
    except Exception as e:
        print(f"ERROR: nnU-Net prediction failed. Make sure your environment variables are set.", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)

    # STAGE 2: Reconstruct Signals from Masks
    print("\n--- [Stage 2/3] Reconstructing 1D Signals from Masks... ---")
    # ECG Grid Parameters
    IMAGE_DPI, MM_PER_MV, MM_PER_SECOND = 300, 10, 25

    for mask_filename in tqdm(sorted(os.listdir(predictions_dir)), desc="Reconstructing"):
        if not mask_filename.endswith('.nii.gz'): continue
        
        mask_path = os.path.join(predictions_dir, mask_filename)
        mask_np = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).squeeze()
        
        raw_pixel_signal = vectorize_mask(mask_np)
        # We need the original image height for scaling, let's assume it from the mask
        image_height = mask_np.shape[0]
        
        time_s, voltage_mv = scale_signal(raw_pixel_signal, IMAGE_DPI, MM_PER_MV, MM_PER_SECOND, image_height)

        df = pd.DataFrame({'time_seconds': time_s, 'voltage_mv': voltage_mv})
        df = df.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

        output_path = os.path.join(reconstructed_csv_dir, mask_filename.replace('.nii.gz', '.csv'))
        df.to_csv(output_path, index=False)

    print("Reconstruction successful.")

    # STAGE 3: Generate Comparison Plots
    print("\n--- [Stage 3/3] Generating Final Comparison Plots... ---")
    for csv_filename in tqdm(sorted(os.listdir(reconstructed_csv_dir)), desc="Plotting"):
        if not csv_filename.endswith('.csv'): continue

        # Reconstruct filenames for all three components
        base_name = csv_filename.replace('.csv', '')
        original_img_filename = f"{base_name}_0000.nii.gz"
        mask_filename = f"{base_name}.nii.gz"
        
        original_img_path = os.path.join(input_images_dir, original_img_filename)
        mask_path = os.path.join(predictions_dir, mask_filename)
        csv_path = os.path.join(reconstructed_csv_dir, csv_filename)

        if not (os.path.exists(original_img_path) and os.path.exists(mask_path)):
            continue

        # Load all data
        original_img_np = sitk.GetArrayFromImage(sitk.ReadImage(original_img_path)).squeeze()
        mask_np = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).squeeze()
        df_signal = pd.read_csv(csv_path)

        # Create the 3-panel plot
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        fig.suptitle(f"Demonstration for: {base_name}", fontsize=16)

        # Panel 1: Original Image
        axes[0].imshow(original_img_np, cmap='gray')
        axes[0].set_title("1. Original Input Image")
        axes[0].axis('off')

        # Panel 2: Mask Overlay
        axes[1].imshow(original_img_np, cmap='gray')
        axes[1].imshow(mask_np, cmap='Reds', alpha=0.5) # Overlay mask in semi-transparent red
        axes[1].set_title("2. Predicted Mask")
        axes[1].axis('off')

        # Panel 3: Reconstructed Waveform
        axes[2].plot(df_signal['time_seconds'], df_signal['voltage_mv'], color='red')
        axes[2].set_title("3. Reconstructed 1D Signal")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("Voltage (mV)")
        axes[2].grid(True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the final plot
        output_plot_path = os.path.join(final_plots_dir, f"{base_name}_demo.png")
        plt.savefig(output_plot_path, dpi=150)
        plt.close()

    print(f"\n--- Demo Complete! ---")
    print(f"Final plots are saved in: {final_plots_dir}")

if __name__ == "__main__":
    main()