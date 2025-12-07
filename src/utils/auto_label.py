import os
import cv2
import torch
from ultralytics import YOLO
from tqdm import tqdm
import argparse

def auto_label(
    input_dir: str, 
    output_dir: str, 
    model_path: str = "src/models/yolov8x.pt", 
    conf_threshold: float = 0.4,
    classes: list = None
):
    """
    Automatically labels images using a pre-trained YOLO model.
    
    Args:
        input_dir (str): Path to directory with images.
        output_dir (str): Path to directory to save labels and visualized images.
        model_path (str): Path to the YOLO model weights.
        conf_threshold (float): Confidence threshold for detections.
        classes (list): List of class IDs to filter (e.g., [0] for person, [2] for car). None for all.
                        COCO classes: 0: person, 2: car, 4: airplane, 5: bus, 7: truck
    """
    
    # Ensure output directories exist
    labels_dir = os.path.join(output_dir, "labels")
    vis_dir = os.path.join(output_dir, "visualized")
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback to download if not found locally, though we expect it in src/models
        print("Attempting to download/load base model name...")
        model = YOLO(os.path.basename(model_path))

    # Supported image extensions
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    # Get list of images
    images = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in valid_extensions]
    
    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(images)} images. Starting auto-labeling...")

    for img_name in tqdm(images):
        img_path = os.path.join(input_dir, img_name)
        
        # Run inference
        results = model(img_path, conf=conf_threshold, classes=classes, verbose=False)[0]
        
        # 1. Save Labels (YOLO format: class x_center y_center width height)
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")
        
        with open(label_path, "w") as f:
            for box in results.boxes:
                # YOLO format is normalized (0-1)
                cls_id = int(box.cls[0])
                x_center, y_center, width, height = box.xywhn[0].tolist()
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # 2. Save Visualization (for verification)
        # Plot draws boxes on the image (numpy array)
        res_plotted = results.plot()
        cv2.imwrite(os.path.join(vis_dir, img_name), res_plotted)

    print(f"Done! \nLabels saved to: {labels_dir}\nVisualizations saved to: {vis_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-label images using YOLOv8")
    parser.add_argument("--input", type=str, default="/app/input", help="Input directory containing images")
    parser.add_argument("--output", type=str, default="/app/output/auto_labels", help="Output directory")
    # Using 'l' (large) model by default for better accuracy during labeling, assuming we downloaded it.
    # If not, it will pull from ultralytics.
    parser.add_argument("--model", type=str, default="src/models/yolov8l.pt", help="Path to YOLO model")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # COCO classes relevant for aviation/ground handling (roughly):
    # 0: person, 2: car, 4: airplane, 5: bus, 7: truck
    RELEVANT_CLASSES = [0, 2, 4, 5, 7] 
    
    auto_label(args.input, args.output, args.model, args.conf, classes=RELEVANT_CLASSES)

