# ECG Project

This project provides a complete workflow for processing ECG images by integrating synthetic ECG creation, YOLOv8 for object detection, nnU-Net for segmentation. The image below visualizes each stage of this pipeline.
<div align="center">
  <img src="images/pipeline.png" alt="pipeline" width="500">
</div>

## Directories
*   **ecg-image-generator**:
Contains scripts for generating ECG images from data. This is a modification of the original [ecg-image-kit](https://github.com/alphanumericslab/ecg-image-kit).
*   **code-unet**: Contains code related to the nnU-Net model.
*   **code-yolo**: Contains code related to the YOLOv8 model for object detection on ECG images.
*   **demo**: Contains scripts and data for demonstrating the YOLOv8 and nnU-Net models.
  
*   **HPC**: Contains scripts for running code on a High-Performance Computing cluster.

