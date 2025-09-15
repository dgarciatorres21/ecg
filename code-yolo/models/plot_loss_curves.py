import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_loss_subplots(run_directory):
    """
    Loads results.csv and creates separate subplot images for training and validation losses.
    """
    
    # --- 1. Find and Load the Data ---
    results_csv_path = "C:/Users/dgarc/Desktop/Dissertation/ecg-image-kit/codes/code-yolo/models/runs_12L/yolo_ecg_model_12L3/results.csv"
    # results_csv_path = "C:/Users/dgarc/Desktop/Dissertation/ecg-image-kit/codes/code-yolo/models/runs/yolo_ecg_model3/results.csv"
    
    if not os.path.exists(results_csv_path):
        print(f"Error: 'results.csv' not found in the directory: {run_directory}")
        return

    print(f"Loading data from: {results_csv_path}")
    df = pd.read_csv(results_csv_path)
    df.columns = df.columns.str.strip() # Clean up column names

    # --- 2. Plot Training Loss Subplots ---
    print("Generating training loss subplot image...")
    
    # Create a figure with 1 row and 3 columns of subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Training Loss Curves per Component', fontsize=20, fontweight='bold')

    # Column names for the training losses
    train_losses = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']
    plot_titles = ['Box Loss (Localization)', 'Class Loss (Classification)', 'DFL Loss (Boundary Confidence)']
    colors = ['#D55E00', '#0072B2', '#009E73'] # Orange, Blue, Green

    for i, (loss_col, title, color) in enumerate(zip(train_losses, plot_titles, colors)):
        ax = axes[i]
        ax.plot(df['epoch'], df[loss_col], color=color, linewidth=2.5, marker='o', markersize=4, label='Loss')
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        # Only add y-label to the first plot
        if i == 0:
            ax.set_ylabel('Loss Value', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    
    train_output_filename = 'training_loss_subplots.png'
    plt.savefig(train_output_filename, dpi=300)
    print(f"✅ Training loss graph saved as '{train_output_filename}'")
    plt.close(fig)

    # --- 3. Plot Validation Loss Subplots ---
    print("Generating validation loss subplot image...")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Validation Loss Curves per Component', fontsize=20, fontweight='bold')

    val_losses = ['val/box_loss', 'val/cls_loss', 'val/dfl_loss']

    for i, (loss_col, title, color) in enumerate(zip(val_losses, plot_titles, colors)):
        ax = axes[i]
        # Use a dashed line for validation to distinguish it visually
        ax.plot(df['epoch'], df[loss_col], color=color, linewidth=2.5, linestyle='--', marker='o', markersize=4, label='Loss')
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        if i == 0:
            ax.set_ylabel('Loss Value', fontsize=12)
            
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    val_output_filename = 'validation_loss_subplots.png'
    plt.savefig(val_output_filename, dpi=300)
    print(f"✅ Validation loss graph saved as '{val_output_filename}'")
    plt.close(fig)


if __name__ == '__main__':
    # --- IMPORTANT: EDIT THIS PATH ---
    # Change this to the specific run folder you want to analyze.
    run_directory = '/users/lip24dg/ecg/ecg-yolo/runs_12L/yolo_ecg_model_12L3'
    
    plot_loss_subplots(run_directory)