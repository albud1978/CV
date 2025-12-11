import cv2
import sys
import os
import argparse
import supervision as sv
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology

class TextSearch:
    def __init__(self, ontology: dict):
        """
        ontology: Словарь { "prompt": "class_name" }
        """
        print("Loading Grounding DINO model...")
        self.model = GroundingDINO(ontology=CaptionOntology(ontology))

    def predict(self, image_path: str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Инференс
        # autodistill возвращает sv.Detections
        detections = self.model.predict(image_path)
        return detections

    def visualize(self, image_path: str, detections: sv.Detections, output_path: str = "search_result.jpg"):
        image = cv2.imread(image_path)
        
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        # Фильтрация пустых детекций
        if len(detections) == 0:
            print("No objects found.")
            cv2.imwrite(output_path, image)
            return

        annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
        
        cv2.imwrite(output_path, annotated_image)
        print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search objects by text description using Grounding DINO")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt (what to find)")
    parser.add_argument("--class-name", type=str, default="object", help="Class name for visualization")
    parser.add_argument("--output", type=str, default="search_result.jpg", help="Output image path")
    
    args = parser.parse_args()
    
    # Формируем онтологию: { "prompt": "class_name" }
    ontology = {args.prompt: args.class_name}
    
    searcher = TextSearch(ontology)
    detections = searcher.predict(args.image)
    
    print(f"Found {len(detections)} objects matching '{args.prompt}'")
    searcher.visualize(args.image, detections, args.output)


