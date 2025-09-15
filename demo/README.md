# ECG Analysis Demos

This directory contains demonstrations for two different models used in ECG analysis: a YOLO model for object detection on ECG images, and a UNET model for image segmentation.

## YOLO Demo

The YOLO demo runs inference on sample ECG images to detect various components. 

### Prerequisites

- Python 3
- `ultralytics` and `opencv-python` packages installed. You can install them using pip:
  ```
  pip install ultralytics opencv-python
  ```

### How to Run

1. **Place your ECG images** (in `.png`, `.jpg`, or `.jpeg` format) into the `data/yolo/input` directory.

2. **Run the demo script** from within the `demo` directory:
   ```bash
   python yolo_demo.py
   ```

3. **Check the output**: The script will process the images and save the annotated results in the `data/yolo/output/demo_predictions` directory.

## UNET Demo

The UNET demo performs image segmentation on ECG images. This demo is more complex and requires the `nnU-Net` framework to be installed and configured.

### Prerequisites

- Python 3
- `nnU-Net` installed and configured. Please refer to the official `nnU-Net` documentation for installation instructions.
- Your python environment with `nnU-Net` and other required packages (like `SimpleITK`, `pandas`, `matplotlib`) should be activated.

### How to Run

1. **Place your input images** (in `.nii.gz` format) into the `data/unet/input` directory.

2. **Run the local demo script** from within the `demo` directory:
   ```bash
   bash run_unet_demo_local.sh
   ```
   *Note: You may need to edit `run_unet_demo_local.sh` to activate your specific Python environment if it's not already active in your shell.*

3. **Check the output**: The script will perform prediction, reconstruct the 1D signal, and save plots in the `data/unet/output` directory. You will find subdirectories for `predictions`, `reconstructed_csvs`, and `final_plots`.
