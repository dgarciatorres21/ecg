import os
import argparse
import torch
import yaml
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
from collections import defaultdict
from iou_metrics import box_iou, complete_box_iou 

def get_ground_truth(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            boxes.append([int(parts[0])] + [float(p) for p in parts[1:]])
    return boxes

def yolo_to_xyxy(yolo_box, img_width, img_height):
    _, x_c, y_c, w, h = yolo_box
    x1 = (x_c - w / 2) * img_width; y1 = (y_c - h / 2) * img_height
    x2 = (x_c + w / 2) * img_width; y2 = (y_c + h / 2) * img_height
    return torch.tensor([x1, y1, x2, y2])

def main():
    parser = argparse.ArgumentParser(description="Run a complete evaluation (mAP & IoU-family) for a YOLOv8 model on a specific test set.")
    parser.add_argument('--model-path', type=str, required=True, help='Absolute path to the best.pt model file.')
    parser.add_argument('--data-yaml', type=str, required=True, help='Path to the main data.yaml file (used for class names).')
    parser.add_argument('--base-path', type=str, required=True, help='The absolute base path of the dataset structure.')
    parser.add_argument('--test-dir', type=str, required=True, help='Relative path to the test image directory from the base_path.')
    args = parser.parse_args()

    print(f"\n===================================================================")
    print(f"--- Starting Unified Evaluation ---")
    print(f"--- Model: {args.model_path}")
    print(f"--- Dataset Base: {args.base_path}")
    print(f"--- Test Images: {os.path.join(args.base_path, args.test_dir)}")
    print(f"===================================================================\n")

    # PART 1: mAP, Precision, Recall Evaluation
    print("\n--- 1. Calculating mAP, Precision & Recall using model.val() ---")
    
    temp_config = {
        'path': args.base_path,
        'train': 'train/images',
        'val': 'valid/images',
        'test': args.test_dir,
    }
    with open(args.data_yaml, 'r') as f:
        main_data_config = yaml.safe_load(f)
        temp_config['nc'] = main_data_config['nc']
        temp_config['names'] = main_data_config['names']

    temp_yaml_path = 'temp_eval_config.yaml'
    with open(temp_yaml_path, 'w') as f:
        yaml.dump(temp_config, f)

    model = YOLO(args.model_path)
    try:
        metrics = model.val(data=temp_yaml_path, split='test', plots=False, verbose=True)
        print("\n--- Overall Performance Metrics ---")
        print(f"mAP@.50-.95: {metrics.box.map:.4f}")
        print(f"mAP@.50:     {metrics.box.map50:.4f}")
        print(f"mAP@.75:     {metrics.box.map75:.4f}")

        results_data = []
        for i, name in model.names.items():
            results_data.append({
                "Class ID": i, "Class Name": name,
                "Precision": f"{metrics.box.p[i]:.4f}",
                "Recall": f"{metrics.box.r[i]:.4f}",
                "mAP@.50": f"{metrics.box.ap50_per_class[i]:.4f}"
            })
        df_map = pd.DataFrame(results_data)
        print("\n--- Per-Class Performance ---")
        print(df_map.to_string(index=False))

    except Exception as e:
        print(f"\nFATAL ERROR during model.val(): {e}")
        print("Please check that paths are correct and the dataset structure has train/valid/test folders.")
    finally:
        if os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)

    # PART 2: IoU-Family Metrics Evaluation
    print("\n\n--- 2. Calculating IoU-Family Localization Metrics (Manual) ---")
    
    test_img_dir = os.path.join(args.base_path, args.test_dir)
    test_label_dir = test_img_dir.replace('/images', '/labels')

    per_class_metrics = defaultdict(lambda: {'ious': [], 'gious': [], 'dious': [], 'cious': []})
    image_files = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in tqdm(image_files, desc="Calculating IoU Metrics"):
        image_path = os.path.join(test_img_dir, image_file)
        label_path = os.path.join(test_label_dir, os.path.splitext(image_file)[0] + '.txt')

        gt_boxes_yolo = get_ground_truth(label_path)
        if not gt_boxes_yolo:
            continue

        results = model.predict(image_path, verbose=False)
        preds = results[0]
        img_height, img_width = preds.orig_shape

        for gt_box_yolo in gt_boxes_yolo:
            gt_class_id = gt_box_yolo[0]
            gt_box_xyxy = yolo_to_xyxy(gt_box_yolo, img_width, img_height)

            best_iou = -1
            best_pred_box = None
            for box in preds.boxes:
                if int(box.cls) == gt_class_id:
                    pred_iou = box_iou(box.xyxy[0], gt_box_xyxy)
                    if pred_iou > best_iou:
                        best_iou = pred_iou
                        best_pred_box = box.xyxy[0]
            
            if best_pred_box is not None:
                iou, giou, diou, ciou = complete_box_iou(best_pred_box, gt_box_xyxy)
                per_class_metrics[gt_class_id]['ious'].append(iou.item())
                per_class_metrics[gt_class_id]['gious'].append(giou.item())
                per_class_metrics[gt_class_id]['dious'].append(diou.item())
                per_class_metrics[gt_class_id]['cious'].append(ciou.item())

    results_data_iou = []
    for class_id, metrics_dict in sorted(per_class_metrics.items()):
        if metrics_dict['ious']:
            results_data_iou.append({
                "Class ID": class_id,
                "Class Name": model.names[class_id],
                "Avg IoU": f"{sum(metrics_dict['ious']) / len(metrics_dict['ious']):.4f}",
                "Avg GIU": f"{sum(metrics_dict['gious']) / len(metrics_dict['gious']):.4f}",
                "Avg DIU": f"{sum(metrics_dict['dious']) / len(metrics_dict['dious']):.4f}",
                "Avg CIU": f"{sum(metrics_dict['cious']) / len(metrics_dict['cious']):.4f}",
                "Count": len(metrics_dict['ious'])
            })

    if results_data_iou:
        df_iou = pd.DataFrame(results_data_iou)
        print("\n--- Average Localization Metrics Per Class ---")
        print(df_iou.to_string(index=False))
    else:
        print("\nNo matched predictions found to calculate IoU-family metrics.")
    
    print("\n--- Unified Evaluation Complete ---")

if __name__ == "__main__":
    main()
