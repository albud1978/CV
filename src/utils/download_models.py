import os
import torch
from ultralytics import YOLO

def download_models():
    """
    Downloads standard YOLOv8 models and sets up directory for others.
    """
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Checking models in {models_dir}...")

    # 1. YOLOv8 / YOLO11
    # Ultralytics automatically downloads weights on first use, 
    # but we can force download them here to "warm up" the cache.
    yolo_models_to_fetch = [
        'yolov8n.pt',   # Nano (fastest)
        'yolov8s.pt',   # Small (balanced)
        'yolov8l.pt',   # Large (accurate)
        'yolo11n.pt',   # YOLO11 Nano (newest)
        'yolov8n-pose.pt', # Pose estimation
    ]

    for model_name in yolo_models_to_fetch:
        print(f"Downloading {model_name}...")
        try:
            model = YOLO(model_name)
            # Just init to trigger download
            print(f"✔ {model_name} ready.")
        except Exception as e:
            print(f"✘ Failed to download {model_name}: {e}")

    # 2. SAM (Segment Anything)
    # Ultralytics supports SAM models too
    sam_models = [
        'sam2_b.pt', # SAM2 Base
        # 'sam_b.pt' # SAM Base (original)
    ]
    
    for sam_model in sam_models:
        print(f"Downloading {sam_model} (via Ultralytics)...")
        try:
            # Ultralytics wrapper for SAM
            # Note: This might require specific ultralytics version support for SAM2
            if 'sam2' in sam_model:
                 print(f"Skipping {sam_model} auto-download via Ultralytics (check support manually).")
            else:
                model = YOLO(sam_model)
                print(f"✔ {sam_model} ready.")
        except Exception as e:
             print(f"⚠ Could not download {sam_model} via standard YOLO interface. Error: {e}")

    # 3. Molmo 2 (Transformers)
    # We only pre-cache the processor and a small model (4B) to save space
    molmo_models = [
        "allenai/Molmo2-4B",
    ]
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        for m_id in molmo_models:
            print(f"Pre-caching Molmo 2 model: {m_id}...")
            AutoProcessor.from_pretrained(m_id, trust_remote_code=True)
            # Мы не скачиваем веса модели полностью здесь, чтобы не забивать диск сразу,
            # но процессор скачается. Для полной загрузки:
            # AutoModelForCausalLM.from_pretrained(m_id, trust_remote_code=True)
            print(f"✔ {m_id} processor ready.")
    except ImportError:
        print("⚠ Transformers not installed, skipping Molmo pre-cache.")
    except Exception as e:
        print(f"⚠ Error pre-caching Molmo: {e}")

    print("\nModel setup complete. Weights are cached in standard location.")

if __name__ == "__main__":
    download_models()


