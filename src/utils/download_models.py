"""
Скрипт проверки и загрузки моделей.
Рекомендуется использовать scripts/download_models.sh для полной загрузки.
"""
import os

# Пути к моделям
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
YOLO_DIR = os.path.join(MODELS_DIR, 'yolo')
SAM2_DIR = os.path.join(MODELS_DIR, 'sam2')
MOLMO_DIR = os.path.join(MODELS_DIR, 'Molmo2-4B')


def check_models():
    """Проверяет наличие всех необходимых моделей."""
    print("=" * 50)
    print("Проверка моделей CV Pipeline")
    print("=" * 50)
    
    all_ok = True
    
    # YOLO
    print("\n📦 YOLO модели:")
    yolo_models = ['yolov8l.pt', 'yolov8l-seg.pt', 'yolo11l.pt']
    for m in yolo_models:
        path = os.path.join(YOLO_DIR, m)
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024 / 1024
            print(f"  ✓ {m} ({size:.1f} МБ)")
        else:
            print(f"  ✗ {m} — НЕ НАЙДЕН")
            all_ok = False
    
    # SAM2
    print("\n📦 SAM2 модели:")
    sam_models = ['sam2.1_hiera_small.pt']
    for m in sam_models:
        path = os.path.join(SAM2_DIR, m)
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024 / 1024
            print(f"  ✓ {m} ({size:.1f} МБ)")
        else:
            print(f"  ✗ {m} — НЕ НАЙДЕН")
            all_ok = False
    
    # Molmo2
    print("\n📦 Molmo2-4B:")
    molmo_weights = [f'model-0000{i}-of-00004.safetensors' for i in range(1, 5)]
    for m in molmo_weights:
        path = os.path.join(MOLMO_DIR, m)
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024 / 1024 / 1024
            print(f"  ✓ {m} ({size:.1f} ГБ)")
        else:
            print(f"  ✗ {m} — НЕ НАЙДЕН")
            all_ok = False
    
    print("\n" + "=" * 50)
    if all_ok:
        print("✅ Все модели на месте!")
    else:
        print("⚠️  Некоторые модели отсутствуют.")
        print("   Запустите: ./scripts/download_models.sh")
    print("=" * 50)
    
    return all_ok


def get_model_path(model_type: str, model_name: str = None) -> str:
    """
    Возвращает путь к модели.
    
    Args:
        model_type: 'yolo', 'sam2', 'molmo'
        model_name: имя файла модели (опционально)
    
    Returns:
        Полный путь к модели
    """
    paths = {
        'yolo': YOLO_DIR,
        'sam2': SAM2_DIR,
        'molmo': MOLMO_DIR,
    }
    
    defaults = {
        'yolo': 'yolo11l.pt',
        'sam2': 'sam2.1_hiera_small.pt',
        'molmo': None,  # Molmo загружается как папка
    }
    
    base_dir = paths.get(model_type)
    if not base_dir:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if model_type == 'molmo':
        return MOLMO_DIR
    
    name = model_name or defaults.get(model_type)
    return os.path.join(base_dir, name)


if __name__ == "__main__":
    check_models()
