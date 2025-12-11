import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union

try:
    from rfdetr import RFDETRSmall, RFDETRMedium, RFDETRLarge, RFDETRNano
    import supervision as sv
except ImportError:
    RFDETRSmall = None
    print("Warning: rfdetr or supervision not installed.")

class RFDETRInference:
    """
    Обертка для инференса модели RF-DETR (используя библиотеку rfdetr).
    """
    
    def __init__(self, model_type: str = "s", conf_threshold: float = 0.5):
        """
        Инициализация модели.
        
        Args:
            model_type (str): Размер модели: 'n' (nano), 's' (small), 'm' (medium), 'l' (large).
            conf_threshold (float): Порог уверенности.
        """
        if RFDETRSmall is None:
            raise ImportError("rfdetr library is not installed.")
            
        self.conf_threshold = conf_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Loading RF-DETR model ({model_type}) on {self.device}...")
        
        # Выбор класса модели
        if model_type.lower() == 'n':
            self.model = RFDETRNano(device=self.device)
        elif model_type.lower() == 's':
            self.model = RFDETRSmall(device=self.device)
        elif model_type.lower() == 'm':
            self.model = RFDETRMedium(device=self.device)
        elif model_type.lower() == 'l':
            self.model = RFDETRLarge(device=self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'n', 's', 'm', or 'l'.")
            
        # Загрузка весов (обычно происходит автоматически при первом вызове или init)
        # self.model.get_model() # Можно вызвать явно, если нужно
        
    def predict(self, image: np.ndarray) -> sv.Detections:
        """
        Запуск детекции на изображении.
        
        Args:
            image (np.ndarray): Изображение (OpenCV BGR).
            
        Returns:
            sv.Detections: Результаты детекции в формате supervision.
        """
        # Инференс
        # Метод predict возвращает sv.Detections напрямую в последних версиях rfdetr?
        # Или требует конвертации. Проверим на практике.
        # Обычно: results = model.predict(source=image, conf=self.conf_threshold)
        
        results = self.model.predict(image, conf=self.conf_threshold)
        
        # Если results уже sv.Detections, возвращаем их. 
        # Если нет (например, список), то нужна конвертация.
        # Предполагаем, что библиотека rfdetr от Roboflow возвращает sv.Detections (они авторы supervision).
        
        return results

    def visualize(self, image: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """
        Визуализация результатов.
        """
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
        
        return annotated_image

if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        
        if not os.path.exists(img_path):
             print(f"Error: file {img_path} not found")
             sys.exit(1)
             
        # Инициализация (скачает веса при первом запуске)
        model = RFDETRInference(model_type='s')
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: could not read image {img_path}")
            sys.exit(1)
            
        print("Running inference...")
        dets = model.predict(img)
        print(f"Detected {len(dets.xyxy)} objects")
        
        vis = model.visualize(img, dets)
        out_path = "rf_detr_output.jpg"
        cv2.imwrite(out_path, vis)
        print(f"Saved visualization to {out_path}")
    else:
        print("Usage: python src/inference/rf_detr.py <image_path>")
