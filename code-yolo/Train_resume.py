from ultralytics import YOLO
import os

def main():
    # load the specific checkpoint you want to resume from
    model_checkpoint_path = '/users/lip24dg/ecg/ecg-yolo/runs/yolo_ecg_model4/weights/last.pt'
    
    if not os.path.exists(model_checkpoint_path):
        print(f"FATAL ERROR: Checkpoint file not found at {model_checkpoint_path}")
        print("Please find the correct run folder and update the path in this script.")
        return

    print(f"Attempting to resume training from checkpoint: {model_checkpoint_path}")
    model = YOLO(model_checkpoint_path) 

    # train the model
    print("Continuing YOLOv8 training...")
    
    # the trainer will automatically read the epoch number and optimizer state from the checkpoint.
    results = model.train(
        resume=True
    )

    print("Training finished!")
    try:
        if results and results.save_dir:
            print(f"Best model and results saved in: {results.save_dir}")
    except AttributeError:
        print("Could not retrieve save directory (common in multi-GPU mode). Check the original run folder.")

if __name__ == '__main__':
    main()
