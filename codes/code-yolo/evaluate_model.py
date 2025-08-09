# --- File: evaluate_model.py (Final Robust Version) ---

import os
import argparse
from ultralytics import YOLO

def main():
    # --- 1. Set up Argument Parser ---
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLOv8 model on the test set.")
    parser.add_argument(
        '--model-path', 
        type=str, 
        required=True, 
        help='Absolute path to the best.pt model file.'
    )
    args = parser.parse_args()

    # --- 2. Configuration ---
    MODEL_PATH = args.model_path
    DATA_CONFIG_YAML = '/users/lip24dg/ecg/ecg-yolo/data.yaml'

    print(f"--- Using model from: {MODEL_PATH} ---")

    # --- 3. Load Model ---
    if not os.path.exists(MODEL_PATH):
        print(f"FATAL ERROR: Model file not found at '{MODEL_PATH}'")
        return
        
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- 4. Run Validation on the TEST set ---
    print(f"--- Evaluating model on the test set for per-class metrics ---")
    try:
        # Use verbose=True to see the official YOLOv8 output table as a backup
        metrics = model.val(data=DATA_CONFIG_YAML, split='test', plots=False, verbose=True)
    except Exception as e:
        print(f"An error occurred during model validation: {e}")
        return
    
    # --- 5. Print Overall Metrics (already works, but good to keep) ---
    print("\n--- Overall Performance Metrics Summary ---")
    map50_95 = metrics.box.map
    map50 = metrics.box.map50
    map75 = metrics.box.map75
    
    print(f"mAP@.50-.95 (Primary): {map50_95:.4f}")
    print(f"mAP@.50 (Accuracy):     {map50:.4f}")
    print(f"mAP@.75 (Stricter):     {map75:.4f}")
    
    # --- 6. Extract and Display Per-Class Metrics ---
    print("\n--- Custom Per-Class Performance Table ---")
    
    class_names = model.names
    
    per_class_data = []
    # Loop through the class indices and names
    for class_id, class_name in class_names.items():
        try:
            map50_per_class = metrics.box.maps[class_id][0]
            p = metrics.box.p[class_id].mean()
            r = metrics.box.r[class_id].mean()
            
            per_class_data.append({
                "Class ID": class_id,
                "Class Name": class_name,
                "Precision": f"{p:.4f}",
                "Recall": f"{r:.4f}",
                "mAP@.50": f"{map50_per_class:.4f}"
            })
        except (IndexError, KeyError) as e:
            print(f"Warning: Could not process metrics for class ID {class_id} ('{class_name}'). Error: {e}")

    # --- 7. Attempt to print with Pandas, with a fallback ---
    try:
        import pandas as pd
        df = pd.DataFrame(per_class_data)
        print(df.to_string(index=False))
    except ImportError:
        print("\nWarning: 'pandas' not found. Printing raw data instead.")
        print("To install: pip install pandas")
        # Fallback to printing a simple header and the list of dictionaries
        if per_class_data:
            header = per_class_data[0].keys()
            print("\t".join(header))
            for row in per_class_data:
                print("\t".join(map(str, row.values())))
    except Exception as e:
        print(f"An error occurred while formatting the table with pandas: {e}")
        print("Raw data:", per_class_data)


if __name__ == '__main__':
    main()