# ECG Project

This project provides a complete workflow for processing ECG images by integrating synthetic ECG creation, YOLOv8 for object detection, and nnU-Net for segmentation. The image below visualizes each stage of this pipeline.
<div align="center">
  <img src="images/pipeline.png" alt="pipeline" width="500">
</div>

## Getting Started

### Prerequisites

- Anaconda or Miniconda
- Python 3.10
- Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/dgarciatorres21/ecg.git
    cd ecg
    ```

2.  **Set up the environments:**
    This project requires separate Conda environments for different components.

    - **Image Generation Environment:**
      ```bash
      conda create --name ecg python=3.10 -y
      conda activate ecg
      pip install -r ecg-image-generator/requirements.txt
      ```
    - **YOLOv8 Environment:**
      ```bash
      conda env create -f code-yolo/environment.yml
      conda activate yolo
      ```
    - **nnU-Net Environment:**
      ```bash
      conda create --name unet python=3.10 -y
      conda activate unet
      pip install -r code-unet/requirements.txt
      ```

## Usage

### 1. Generating ECG Images

To generate a batch of clean ECG images, you can use the HPC script:

```bash
sbatch HPC/generator/generate_data.sh
```

To generate augmented images, specify the augmentation type:

```bash
# Example for 'scanner' type augmentation
sbatch HPC/generator/generate_augmented_data.sh scanner
```

### 2. Training the YOLOv8 Model

To train the YOLOv8 model, use the training script:

```bash
# Activate the correct conda environment first
conda activate yolo
python code-yolo/Train.py
```

### 3. Running the Demo

A demo script is available to test the pipeline:

```bash
python demo/yolo_demo.py --input /path/to/input/image
```

## Project Structure

-   `ecg-image-generator/`: Scripts for generating synthetic ECG images. This is a modification of the original [ecg-image-kit](https://github.com/alphanumericslab/ecg-image-kit).
-   `code-yolo/`: Contains all code for the YOLOv8 object detection model.
-   `code-unet/`: Contains all code for the nnU-Net segmentation model.
-   `demo/`: Scripts and sample data for demonstrating the models.
-   `HPC/`: Shell scripts for running jobs on a High-Performance Computing (HPC) cluster.
    -   `generator/`:
        -   `generate_data.sh`: This script generates a "clean" version of the ECG image dataset. It runs a Python script to convert ECG data into images and corresponding masks without applying any visual augmentations. It also performs an audit of the source data. The following image is an example of a **"clean"** ECG:
            <div align="center">
              <img src="images/clean_example.png" alt="clean" width="300">
            </div>
        -   `generate_augmented_data.sh`: This script generates augmented versions of the ECG dataset. It takes an argument that specifies the type of augmentation to apply, such as "scanner" (simulating scanner noise and rotation), "physical" (simulating wrinkled paper), or "chaos" (a combination of augmentations). Based on the chosen type, it passes different flags to the same underlying Python generation script. The following images are example of each inperfection:

            -   **"Scanner":**
            <div align="center">
              <img src="images/scanner_imperfections_example.png" alt="clean" width="300">
            </div>
            
            -   **"Physical":**
            <div align="center">
              <img src="images/physical_imperfections_example.png" alt="clean" width="300">
            </div>
            
            -   **"Chaos":**
            <div align="center">
              <img src="images/chaos_imperfections_example.png" alt="clean" width="300">
            </div>
    -   `yolo/`: Scripts for running the YOLOv8 pipeline on the HPC cluster. The pipeline can be executed in two ways:
        -   **All-in-One Pipelines:** These scripts run the entire workflow from data preparation to evaluation.
            -   `yolo_pipeline.sh`: Runs the full pipeline on the clean, baseline dataset.
            -   `yolo_pipeline_DA_exp.sh`: Runs the full pipeline on augmented datasets (both 12-lead and long lead).
            -   `yolo_pipeline_DA_12L.sh`: Runs the full pipeline on the 12-lead augmented dataset.
        -   **Step-by-Step Execution:** For more granular control, the pipeline can be run in separate stages.
            1.  `prepare_data_yolo.sh`: Prepares the data by converting annotations and splitting into train/validation/test sets.
            2.  `train_yolo.sh`: Trains the YOLOv8 model on the prepared data.
            3.  `evaluation_yolo.sh`: Runs evaluation scripts on the trained model to assess its performance.
