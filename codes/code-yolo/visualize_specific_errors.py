# --- File: find_all_errors_with_debug.py ---
# This is a slightly modified version of find_errors_on_validation_set_v2.py
# with added print statements for debugging.

import os
import cv2
from ultralytics import YOLO

# ... (Keep the calculate_iou and draw_box functions the same)
def calculate_iou(box1, box2):
    x1_inter, y1_inter = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2_inter, y2_inter = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def draw_box(image, box, label, color, thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(image, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def main():
    # Configuration
    MODEL_PATH = 'runs/detect/yolo_ecg_model10/weights/best.pt'
    DATA_DIR = '../../Data/ecg_yolo_data'
    IMAGE_SET_TO_ANALYZE = 'train'
    IMAGE_DIR = os.path.join(DATA_DIR, 'images', IMAGE_SET_TO_ANALYZE)
    LABEL_DIR = os.path.join(DATA_DIR, 'labels', IMAGE_SET_TO_ANALYZE)
    OUTPUT_DIR = 'mistake_examples/train'
    IOU_THRESHOLD = 0.45  # Slightly more lenient IoU
    CONFIDENCE_THRESHOLD = 0.1 # Very low confidence to catch everything

    # Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    COLOR_GT = (0, 255, 0); COLOR_PRED_CORRECT = (255, 0, 0)
    COLOR_PRED_MISCLASS = (0, 165, 255); COLOR_FN_MISS = (0, 0, 255)

    # Load Model and Class Info
    model = YOLO(MODEL_PATH)
    class_names = model.names
    name_to_id = {v: k for k, v in class_names.items()}
    v_ids_to_check = {name_to_id.get(f'V{i}') for i in range(2, 7)}
    v4_id, v5_id = name_to_id.get('V4'), name_to_id.get('V5')
    
    print(f"Analyzing images in: '{IMAGE_DIR}' with lenient thresholds.")
    
    # Main Loop
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(IMAGE_DIR, image_file)
        label_path = os.path.join(LABEL_DIR, os.path.splitext(image_file)[0] + '.txt')

        # Load GT and Predictions
        image_h, image_w = cv2.imread(image_path).shape[:2]
        gt_boxes = []; errors_in_file = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split(); class_id = int(parts[0])
                x_c, y_c, b_w, b_h = map(float, parts[1:])
                x1, y1 = (x_c-b_w/2)*image_w, (y_c-b_h/2)*image_h
                x2, y2 = (x_c+b_w/2)*image_w, (y_c+b_h/2)*image_h
                gt_boxes.append({'box': [x1, y1, x2, y2], 'class_id': class_id, 'status': 'missed'})

        results = model.predict(source=image_path, conf=CONFIDENCE_THRESHOLD, verbose=False)
        pred_boxes = [{'box': b.xyxy[0].cpu().numpy(), 'class_id': int(b.cls[0].item()), 'conf': b.conf[0].item(), 'matched': False} for b in results[0].boxes]

        # Match logic
        for gt in gt_boxes:
            matches = [(iou, p_idx) for p_idx, p in enumerate(pred_boxes) if not p['matched'] and (iou := calculate_iou(gt['box'], p['box'])) > IOU_THRESHOLD]
            if not matches: continue
            best_iou, best_pred_idx = max(matches, key=lambda x: x[0])
            pred_boxes[best_pred_idx]['matched'] = True
            gt['matched_pred'] = pred_boxes[best_pred_idx]
            gt['status'] = 'correct' if gt['class_id'] == pred_boxes[best_pred_idx]['class_id'] else 'misclassified'

        # Check for target errors and add debug info
        for gt in gt_boxes:
            if gt['status'] == 'missed' and gt['class_id'] in v_ids_to_check:
                errors_in_file.append(f"MISSED {class_names[gt['class_id']]}")
            elif gt['status'] == 'misclassified' and (gt['class_id'], gt['matched_pred']['class_id']) in [(v5_id, v4_id), (v4_id, v5_id)]:
                errors_in_file.append(f"MISCLASSIFIED GT:{class_names[gt['class_id']]} as PRED:{class_names[gt['matched_pred']['class_id']]}")
        
        if errors_in_file:
            print(f"\nFound errors in {image_file}: {', '.join(errors_in_file)}")
            # Visualization logic (same as before)
            vis_image = cv2.imread(image_path)
            for gt in gt_boxes:
                if gt['status'] == 'correct':
                    draw_box(vis_image, gt['matched_pred']['box'], f"{class_names[gt['class_id']]} {gt['matched_pred']['conf']:.2f}", COLOR_PRED_CORRECT)
                elif gt['status'] == 'misclassified':
                    draw_box(vis_image, gt['box'], f"GT: {class_names[gt['class_id']]}", COLOR_GT)
                    draw_box(vis_image, gt['matched_pred']['box'], f"{class_names[gt['matched_pred']['class_id']]} {gt['matched_pred']['conf']:.2f}", COLOR_PRED_MISCLASS)
                elif gt['status'] == 'missed' and gt['class_id'] in v_ids_to_check:
                    draw_box(vis_image, gt['box'], f"MISSED: {class_names[gt['class_id']]}", COLOR_FN_MISS)
            
            cv2.imwrite(os.path.join(OUTPUT_DIR, image_file), vis_image)

    print(f"\n\nAnalysis complete. Check the '{OUTPUT_DIR}' folder.")

if __name__ == '__main__':
    main()