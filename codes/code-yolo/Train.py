# --- File: train_yolov8.py (The new training script) ---
from ultralytics import YOLO

def main():
    # --- Configuration ---
    # Load a pre-trained model. 
    # 'yolov8n.pt' is the smallest and fastest model, great for starting.
    # The library will automatically adapt the model's final layer to your 13 classes.
    model = YOLO('yolov8n.pt') 

    # --- Train the model ---
    # The .train() method handles everything: data loading, augmentations, training loop,
    # validation, and saving checkpoints.
    print("Starting YOLOv8 training...")
    results = model.train(
        data='C:/Users/dgarc/Desktop/Dissertation/ecg-image-kit/Data/data.yaml',   # Path to your data configuration file
        epochs=100,              # Number of epochs to train for (100 is a good starting point)
        imgsz=416,               # Image size (same as your original setup)
        batch=8,                 # Batch size
        name='yolo_ecg_model'    # A name for the training run folder
    )

    print("Training finished!")
    # The best model is automatically saved as 'best.pt' inside the run folder.
    # For example: 'runs/detect/yolo_ecg_model/weights/best.pt'
    print(f"Best model and results saved in: {results.save_dir}")

if __name__ == '__main__':
    main()