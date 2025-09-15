import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import random

def main():
    parser = argparse.ArgumentParser(description="Generates comparison plots of predicted vs. ground truth ECG signals.")
    parser.add_argument('--pred_dir', required=True, help="Directory with the reconstructed PREDICTED signal CSVs.")
    parser.add_argument('--gt_dir', required=True, help="Directory with the reconstructed GROUND TRUTH signal CSVs.")
    parser.add_argument('--output_dir', required=True, help="Directory where the output plot .png files will be saved.")
    parser.add_argument('--num_plots', type=int, default=20, help="Number of random comparison plots to generate.")
    args = parser.parse_args()

    predicted_csv_dir = args.pred_dir
    ground_truth_csv_dir = args.gt_dir
    output_plot_dir = args.output_dir
    num_plots_to_generate = args.num_plots

    os.makedirs(output_plot_dir, exist_ok=True)

    # find common files to compare
    pred_files = {f for f in os.listdir(predicted_csv_dir) if f.endswith('.csv')}
    gt_files = {f for f in os.listdir(ground_truth_csv_dir) if f.endswith('.csv')}
    common_files = sorted(list(pred_files.intersection(gt_files)))

    if not common_files:
        print("Error: No matching signal files found between the prediction and ground truth directories.", file=sys.stderr)
        return

    if len(common_files) > num_plots_to_generate:
        print(f"Randomly selecting {num_plots_to_generate} out of {len(common_files)} signals to plot...")
        files_to_plot = random.sample(common_files, num_plots_to_generate)
    else:
        print(f"Plotting all {len(common_files)} available signals...")
        files_to_plot = common_files

    # --- loop through the sample and generate plots ---
    for csv_filename in tqdm(files_to_plot, desc="Generating Plots"):
        try:
            pred_path = os.path.join(predicted_csv_dir, csv_filename)
            gt_path = os.path.join(ground_truth_csv_dir, csv_filename)

            # load the data
            df_pred = pd.read_csv(pred_path)
            df_gt = pd.read_csv(gt_path)

            # --- create the plot ---
            plt.figure(figsize=(15, 5)) # create a wide figure to see the signal clearly
            
            plt.plot(df_gt['time_seconds'], df_gt['voltage_mv'], label='Ground Truth', color='blue', linewidth=1.5)
            plt.plot(df_pred['time_seconds'], df_pred['voltage_mv'], label='Predicted Signal', color='red', linestyle='--', linewidth=1.5)
            
            plt.title(f"Signal Comparison: {csv_filename.replace('.csv', '')}")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Voltage (mV)")
            plt.legend()
            plt.grid(True, linestyle=':')
            
            # save the plot to a file
            output_plot_path = os.path.join(output_plot_dir, csv_filename.replace('.csv', '.png'))
            plt.savefig(output_plot_path, dpi=150)
            
            # close the plot to free up memory (critical in a loop)
            plt.close()

        except Exception as e:
            print(f"Could not process {csv_filename}. Error: {e}", file=sys.stderr)

    print(f"\nPlot generation complete. {len(files_to_plot)} plots saved in: {output_plot_dir}")


if __name__ == "__main__":
    main()
