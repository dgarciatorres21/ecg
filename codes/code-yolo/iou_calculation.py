import os
import torch
from ultralytics import YOLO
from tqdm import tqdm
import yaml
import pandas as pd
from collections import defaultdict
from iou_metrics import complete_box_iou # Assuming iou_metrics.py is in the same directory

def get_ground_truth(label_path):
    # (This function remains the same)
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            boxes.append([int(parts[0])] + [float(p) for p in parts[1:]])
    return boxes

def yolo_to_xyxy(yolo_box, img_width, img_height):
    # (This function remains the same)
    _, x_c, y_c, w, h = yolo_box
    x1 = (x_c - w / 2) * img_width; y1 = (y_c - h / 2) * img_height
    x2 = (x_c + w / 2) * img_width; y2 = (y_c + h / 2) * img_height
    return torch.tensor([x1, y1, x2, y2])

def main():
    MODEL_PATH = '/users/lip24dg/runs/detect/yolo_ecg_model10/weights/best.pt'
    DATA_CONFIG_YAML = '/users/lip24dg/ecg/ecg-yolo/data.yaml'

    with open(DATA_CONFIG_YAML, 'r') as f:
        data_config = yaml.safe_load(f)
    
    base_path = data_config['path']
    test_img_dir = os.path.join(base_path, data_config['test'])
    test_label_dir = os.path.join(base_path, 'test', 'labels')

    model = YOLO(MODEL_PATH)
    class_names = model.names

    # Use defaultdict to easily handle lists for each class
    per_class_metrics = defaultdict(lambda: {'ious': [], 'gious': [], 'dious': [], 'cious': []})

    image_files = [f for f in os.listdir(test_img_dir) if f.endswith('.png')]

    for image_file in tqdm(image_files, desc="Calculating Per-Class IoU Metrics"):
        image_path = os.path.join(test_img_dir, image_file)
        label_path = os.path.join(test_label_dir, os.path.splitext(image_file)[0] + '.txt')

        if not os.path.exists(label_path): continue

        results = model.predict(image_path, verbose=False)
        preds = results[0]
        img_height, img_width = preds.orig_shape
        gt_boxes_yolo = get_ground_truth(label_path)

        for gt_box_yolo in gt_boxes_yolo:
            gt_class_id = gt_box_yolo[0]
            gt_box_xyxy = yolo_to_xyxy(gt_box_yolo, img_width, img_height)

            best_iou = -1
            best_pred_box = None
            for box in preds.boxes:
                if int(box.cls) == gt_class_id:
                    # (Matching logic remains the same)
                    from iou_metrics import box_iou
                    pred_iou = box_iou(box.xyxy[0], gt_box_xyxy)
                    if pred_iou > best_iou:
                        best_iou = pred_iou
                        best_pred_box = box.xyxy[0]
            
            if best_pred_box is not None:
                iou, giou, diou, ciou = complete_box_iou(best_pred_box, gt_box_xyxy)
                # Append results to the correct class list
                per_class_metrics[gt_class_id]['ious'].append(iou.item())
                per_class_metrics[gt_class_id]['gious'].append(giou.item())
                per_class_metrics[gt_class_id]['dious'].append(diou.item())
                per_class_metrics[gt_class_id]['cious'].append(ciou.item())

    # --- Format and Print the Results ---
    print("\n--- Average Localization Metrics Per Class ---")
    
    results_data = []
    for class_id, metrics_dict in sorted(per_class_metrics.items()):
        if metrics_dict['ious']: # Only process if we found matches for this class
            avg_iou = sum(metrics_dict['ious']) / len(metrics_dict['ious'])
            avg_giou = sum(metrics_dict['gious']) / len(metrics_dict['gious'])
            avg_diou = sum(metrics_dict['dious']) / len(metrics_dict['dious'])
            avg_ciou = sum(metrics_dict['cious']) / len(metrics_dict['cious'])
            
            results_data.append({
                "Class ID": class_id,
                "Class Name": class_names[class_id],
                "Avg IoU": f"{avg_iou:.4f}",
                "Avg GIoU": f"{avg_giou:.4f}",
                "Avg DIoU": f"{avg_diou:.4f}",
                "Avg CIoU": f"{avg_ciou:.4f}",
                "Count": len(metrics_dict['ious'])
            })

    if results_data:
        df = pd.DataFrame(results_data)
        print(df.to_string(index=False))
    else:
        print("No matched predictions found to calculate metrics.")

if __name__ == "__main__":
    main()