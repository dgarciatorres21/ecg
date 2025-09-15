# --- file: train_yolov8.py ---
from ultralytics import YOLO
import os

def main():
    model = YOLO('yolov8n.pt') 

    print("Starting YOLOv8 training")
    results = None
    try:
        results = model.train(
            # data='c:/users/dgarc/desktop/dissertation/ecg-image-kit/data/data.yaml',   # path to your data configuration file
            data='/users/lip24dg/ecg/ecg-yolo/data_12L.yaml',
            epochs=100,              # number of epochs to train for
            imgsz=640,               # image size
            batch=16,                 # batch size
            patience=20,             # early stopping patience
            project='/users/lip24dg/ecg/ecg-yolo/runs_12L',
            name='yolo_ecg_model_12L',    # name for training run folder
            device=[0, 1]
        )
    except Exception as e:
        print(f"An error occurred during training: {e}")

    print("Training finished!")
    
    try:
        if results and results.save_dir:
            # the best model is automatically saved as 'best.pt' inside the run folder.
            print(f"Best model and results saved in: {results.save_dir}")
        else:
            save_path = os.path.join('/users/lip24dg/ecg/ecg-yolo/runs_12L', 'yolo_ecg_model_12L')
            print(f"Training complete. Results were likely saved in a directory like: {save_path}")
            print("(Note: In multi-GPU mode, the results object may not be returned to the main script.)")
            
    except AttributeError:
        print("Could not retrieve the save directory from the results object (common in multi-GPU mode).")

if __name__ == '__main__':
    main()
