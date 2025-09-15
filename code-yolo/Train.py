from ultralytics import YOLO
import os

def main():
    model = YOLO('yolov8n.pt') 

    print("Starting YOLOv8 training")

    results = model.train(
        # data='C:/Users/dgarc/Desktop/Dissertation/ecg-image-kit/Data/data.yaml',
        data='/users/lip24dg/ecg/ecg-yolo/data.yaml',
        epochs=100,              # Number of epochs to train for
        imgsz=640,               # Image size
        batch=16,                 # Batch size
        patience=20,             # Early stopping patience
        project='/users/lip24dg/ecg/ecg-yolo/runs',
        name='yolo_ecg_model',    # Name for training run folder
        device=[0, 1]
    )

    print("Training finished!")
    print(f"Best model and results saved in: {results.save_dir}")

if __name__ == '__main__':
    main()