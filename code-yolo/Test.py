import os
from ultralytics import YOLO
from PIL import Image
import cv2

def main():
    MODEL_PATH = 'C:/Users/dgarc/Desktop/Dissertation/ecg-image-kit/codes/runs/detect/yolo_ecg_model5/weights/best.pt'
    
    TEST_IMAGE_DIR = 'C:/Users/dgarc/Desktop/Dissertation/ecg-image-kit/data/yolo_split_data/test/images'
    OUTPUT_VIS_DIR = 'test_predictions_yolov8'
    CONFIDENCE_THRESHOLD = 0.5

    # create output directory
    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

    # load the trained yolov8 model
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Please make sure the model path is correct: '{MODEL_PATH}'")
        return

    # run inference on all test images
    image_files = [f for f in os.listdir(TEST_IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images in '{TEST_IMAGE_DIR}'")
    print(f"Running inference and saving results to '{OUTPUT_VIS_DIR}'...")

    for image_file in image_files:
        image_path = os.path.join(TEST_IMAGE_DIR, image_file)
        # perform prediction
        results = model.predict(source=image_path, conf=CONFIDENCE_THRESHOLD)
        # visualize the results
        result_image_with_boxes = results[0].plot()
        # save the visualized
        output_image_path = os.path.join(OUTPUT_VIS_DIR, image_file)
        cv2.imwrite(output_image_path, result_image_with_boxes)

    print("\nInference complete.")
    print(f"Visualizations saved to '{OUTPUT_VIS_DIR}'.")

    print("\nExample of accessing raw prediction data for the first image:")
    if image_files:
        first_image_path = os.path.join(TEST_IMAGE_DIR, image_files[0])
        results = model.predict(source=first_image_path, conf=CONFIDENCE_THRESHOLD)
        
        # get the boxes object
        boxes = results[0].boxes
        
        for box in boxes:
            # get coordinates in (x_min, y_min, x_max, y_max) format
            xyxy = box.xyxy[0].cpu().numpy() 
            
            # get confidence score and class id
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            
            print(f"  - Found '{class_name}' with confidence {confidence:.2f} at {xyxy}")


if __name__ == '__main__':
    main()
