import argparse
from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk import DetectionTask
from dds_cloudapi_sdk import TextPrompt
from dds_cloudapi_sdk import DetectionModel
import os
import cv2

def run_dino_1_5(image_path, prompt, api_token):
    """
    Пример использования Grounding DINO 1.5 через официальный API.
    Требует токен от IDEA-Research.
    """
    token = api_token or os.getenv("DINO_API_TOKEN")
    if not token:
        print("Error: API Token required. Set DINO_API_TOKEN env var or pass --token")
        return

    # Конфигурация клиента
    config = Config(token)
    client = Client(config)

    # URL изображения (API требует URL или загрузки, SDK умеет грузить)
    # В текущей версии SDK v0.2.1 лучше передавать URL, если есть. 
    # Но он умеет работать с локальными файлами через upload.
    
    print(f"Sending request for {image_path}...")
    
    task = DetectionTask(
        image_url=client.upload_file(image_path), # Загружаем файл на сервер
        prompts=[TextPrompt(text=prompt)],
        targets=[DetectionModel.GroundingDino1_5_Pro] # Или Edge
    )
    
    client.run_task(task)
    result = task.result
    
    print(f"Found {len(result.objects)} objects.")
    
    # Визуализация (простая)
    img = cv2.imread(image_path)
    for obj in result.objects:
        bbox = obj.bbox # [x, y, w, h] or similar? Need to check SDK docs.
        # Обычно API возвращает [x_min, y_min, x_max, y_max]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{obj.category}: {obj.score:.2f}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                   
    out_path = "dino_1_5_result.jpg"
    cv2.imwrite(out_path, img)
    print(f"Result saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--token", help="API Token")
    args = parser.parse_args()
    
    run_dino_1_5(args.image, args.prompt, args.token)

