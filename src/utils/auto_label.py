import os
import cv2
import torch
import argparse
from tqdm import tqdm
import supervision as sv
import numpy as np

# Импорт моделей
from ultralytics import YOLO
try:
    from rfdetr import RFDETRSmall, RFDETRMedium, RFDETRLarge, RFDETRNano
    RF_DETR_AVAILABLE = True
except ImportError:
    RF_DETR_AVAILABLE = False
    print("Warning: RF-DETR not available.")

def get_rf_detr_model(model_size='s'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_size == 'n': return RFDETRNano(device=device)
    if model_size == 's': return RFDETRSmall(device=device)
    if model_size == 'm': return RFDETRMedium(device=device)
    if model_size == 'l': return RFDETRLarge(device=device)
    return RFDETRSmall(device=device)

def save_yolo_label(file_path, detections, img_width, img_height, class_map=None):
    """
    Сохраняет детекции в формате YOLO: class_id x_center y_center width height (normalized)
    """
    with open(file_path, "w") as f:
        # xyxy - это bounding boxes
        # class_id - это индексы классов
        for xyxy, class_id in zip(detections.xyxy, detections.class_id):
            x1, y1, x2, y2 = xyxy
            
            # Нормализация
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2
            
            cx_n = cx / img_width
            cy_n = cy / img_height
            w_n = w / img_width
            h_n = h / img_height
            
            # Если у нас есть маппинг классов (например, RF-DETR COCO -> наш COCO), можно применить.
            # Но обычно RF-DETR обучен на COCO, как и YOLO, поэтому ID совпадают.
            
            f.write(f"{int(class_id)} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}\n")

def auto_label(
    input_dir: str, 
    output_dir: str, 
    model_path: str = "src/models/yolov8x.pt", 
    model_type: str = "yolo", # "yolo" or "rf-detr"
    conf_threshold: float = 0.4,
    classes: list = None
):
    # Ensure output directories exist
    labels_dir = os.path.join(output_dir, "labels")
    vis_dir = os.path.join(output_dir, "visualized")
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # Load model
    model = None
    rf_detr_model = None
    
    if model_type == "rf-detr":
        if not RF_DETR_AVAILABLE:
            print("Error: RF-DETR library not installed. Exiting.")
            return
        # model_path в данном случае может служить индикатором размера (например, "s", "m")
        # Или мы можем добавить отдельный аргумент. 
        # Допустим, если model_path содержит "rf-detr-m", берем medium.
        size = 's'
        if '-m' in model_path: size = 'm'
        if '-l' in model_path: size = 'l'
        if '-n' in model_path: size = 'n'
        
        print(f"Loading RF-DETR ({size})...")
        rf_detr_model = get_rf_detr_model(size)
    else:
        print(f"Loading YOLO model from {model_path}...")
        try:
            model = YOLO(model_path)
        except Exception as e:
            print(f"Error loading YOLO: {e}")
            model = YOLO(os.path.basename(model_path))

    # Supported image extensions
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in valid_extensions]
    
    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(images)} images. Starting auto-labeling with {model_type}...")
    
    # Annotators for visualization
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    for img_name in tqdm(images):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        
        height, width = img.shape[:2]
        
        detections = None
        
        # --- INFERENCE ---
        if model_type == "rf-detr":
            # RF-DETR predict возвращает sv.Detections или свой формат, который нужно конвертировать
            # Используем .predict(..., conf=...)
            res = rf_detr_model.predict(img, conf=conf_threshold)
            # Если res уже sv.Detections:
            detections = res
        else:
            # YOLO
            results = model(img_path, conf=conf_threshold, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)

        # --- FILTER CLASSES ---
        if classes is not None and detections is not None:
            # supervision filter
            # detections.class_id - numpy array
            mask = np.isin(detections.class_id, classes)
            detections = detections[mask]

        if detections is None or len(detections) == 0:
            continue

        # --- SAVE LABELS ---
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")
        save_yolo_label(label_path, detections, width, height)
        
        # --- SAVE VISUALIZATION ---
        annotated_image = box_annotator.annotate(scene=img.copy(), detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
        cv2.imwrite(os.path.join(vis_dir, img_name), annotated_image)

    print(f"Done! \nLabels saved to: {labels_dir}\nVisualizations saved to: {vis_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-label images using YOLOv8 or RF-DETR")
    parser.add_argument("--input", type=str, default="/app/input", help="Input directory")
    parser.add_argument("--output", type=str, default="/app/output/auto_labels", help="Output directory")
    parser.add_argument("--type", type=str, default="yolo", choices=["yolo", "rf-detr"], help="Model type")
    parser.add_argument("--model", type=str, default="src/models/yolov8l.pt", help="Path to YOLO model or size hint for RF-DETR (e.g. 'rf-detr-l')")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # COCO classes relevant for aviation/ground handling (roughly):
    # 0: person, 2: car, 4: airplane, 5: bus, 7: truck
    RELEVANT_CLASSES = [0, 2, 4, 5, 7] 
    
    auto_label(args.input, args.output, args.model, args.type, args.conf, classes=RELEVANT_CLASSES)
