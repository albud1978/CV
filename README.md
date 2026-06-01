# CV - Computer Vision для гражданской авиации

## Описание проекта

Проект по разработке моделей компьютерного зрения (Computer Vision) для задач гражданской авиации. Основные задачи включают детекцию спецтехники и персонала, контроль соблюдения регламентов и обеспечение безопасности на перроне.

Стек технологий: **PyTorch**, **YOLOv8/v11** (Ultralytics), **YOLO-World** (zero-shot), **SAM 2** (Meta), **RF-DETR** (Roboflow), **Molmo2-4B** (VLM).

## Архитектура
Целевая архитектура проекта (Target Architecture) описана в документе:
[Эталонная Архитектура Мультимодальной AI-системы](docs/REFERENCE_ARCHITECTURE.md).

Она включает в себя связку RF-DETR, SAM 2, SigLIP и других SOTA-компонентов.

## Структура проекта

```
CV/
├── src/                        # Исходный код
│   ├── configs/                # Конфигурационные файлы
│   ├── inference/              # Скрипты для запуска моделей (инференс)
│   ├── models/                 # Веса моделей (.pt) и архитектуры
│   ├── train/                  # Скрипты обучения
│   └── utils/                  # Вспомогательные утилиты (авторазметка, загрузка)
├── input/                      # Входные данные (симлинк на Nextcloud)
├── output/                     # Результаты работы (симлинк на Nextcloud)
│   ├── auto_labels/            # Результаты авторазметки (labels + visualized)
│   └── ...
├── docs/                       # Документация проекта
├── docker-compose.yml          # Конфигурация Docker Compose
├── Dockerfile                  # Описание Docker-образа
├── README.md                   # Описание проекта
├── CHANGELOG.md                # История изменений
└── requirements.txt            # Python-зависимости
```

## Требования для развертывания

Проект полностью контейнеризирован. Для запуска на любой машине (включая сервера с RTX 4080/5080) требуются только:
1.  **Docker Desktop** (или Docker Engine + Docker Compose Plugin).
2.  **NVIDIA Drivers**: Актуальные драйверы для вашей видеокарты (CUDA Toolkit на хосте ставить **не обязательно**, он уже есть внутри Docker-образа).
3.  **Доступ к данным**: Папки с датасетом (input) и результатами (output).

## Инструкция по запуску (Deployment Guide)

### 1. Подготовка окружения
Клонируйте репозиторий и создайте файл `.env` в корне проекта. В нем укажите абсолютные пути к папкам данных на **текущей машине**:

Пример `.env` для Windows (WSL2):
```bash
CV_INPUT_PATH="/mnt/c/Users/Admin/Nextcloud/CV/input"
CV_OUTPUT_PATH="/mnt/c/Users/Admin/Nextcloud/CV/output"
```

Пример `.env` для Linux Server:
```bash
CV_INPUT_PATH="/home/user/data/input"
CV_OUTPUT_PATH="/home/user/data/output"
```

### 2. Запуск контейнера
Соберите и запустите окружение одной командой. Docker скачает базовый образ PyTorch, установит все зависимости (YOLO, SAM 2, RF-DETR) и настроит GPU.

```bash
docker-compose up -d --build
```

### 3. Проверка работоспособности
Убедитесь, что контейнер видит видеокарту:
```bash
docker-compose exec cv-dev nvidia-smi
```
*Вы должны увидеть вашу видеокарту (например, RTX 5080) и версию драйвера.*

### 4. Загрузка весов моделей
**Обязательный шаг!** Перед первым запуском скачайте веса всех моделей:
```bash
./scripts/download_models.sh
```

Этот скрипт загрузит:
- **YOLOv8/11 + YOLO-World** (детекция, сегментация, zero-shot) — ~300 МБ
- **SAM 2** (сегментация по точкам) — ~176 МБ  
- **Molmo2-4B** (VLM для анализа изображений/видео) — ~19 ГБ

> ⚠️ Для Molmo2-4B требуется стабильное интернет-соединение. При проблемах с HuggingFace используйте VPN.

Веса сохранятся в `src/models/` и будут доступны при следующих запусках.

### 5. Запуск скриптов
Примеры команд (выполняются из корня проекта):

**Автоматическая разметка (Auto-Labeling):**
```bash
docker-compose exec cv-dev python3 src/utils/auto_label.py --input /app/input --output /app/output/labels --conf 0.4
```

## Работа с моделями

*   **YOLOv8/11**: Используется через библиотеку `ultralytics`. Веса: `src/models/yolo/*.pt`.
*   **YOLO-World**: Zero-shot детекция любых объектов по текстовому описанию. Веса: `yolov8l-worldv2.pt`.
*   **SAM 2**: Сегментация по точкам/маскам. Веса: `src/models/sam2/*.pt`.
*   **RF-DETR**: Установлен через `pip install rfdetr`.
*   **Molmo2-4B**: Vision-Language модель от Allen AI для анализа изображений и видео. Веса: `src/models/Molmo2-4B/`.

### Пример YOLO-World (zero-shot детекция касок)
```bash
docker-compose exec cv-dev python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8l-worldv2.pt')
model.set_classes(['person', 'helmet', 'hard hat'])
results = model.predict('input/your_image.jpg', conf=0.2)
results[0].save('output/result.jpg')
print(f'Найдено: {len(results[0].boxes)} объектов')
"
```

### Пример использования Molmo2-4B
```bash
docker-compose exec cv-dev python3 -c "
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

processor = AutoProcessor.from_pretrained('/app/src/models/Molmo2-4B', trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    '/app/src/models/Molmo2-4B',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

messages = [{'role': 'user', 'content': [
    {'type': 'text', 'text': 'Describe this image.'},
    {'type': 'image', 'image': '/app/input/your_image.jpg'},
]}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors='pt', return_dict=True)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=200)

print(processor.tokenizer.decode(output[0], skip_special_tokens=True))
"
```

## Решение проблем

*   **Ошибка `nvidia-container-cli: requirement error`**: Убедитесь, что Docker Desktop настроен на использование WSL2 backend (Windows) или установлен NVIDIA Container Toolkit (Linux).
*   **Ошибка памяти (`DataLoader worker exited`)**: В `docker-compose.yml` уже настроен `shm_size: '8gb'`, этого должно хватать.
*   **Смена видеокарты (например, на RTX 5080)**: Текущий образ использует CUDA 12.1. Если новая видеокарта потребует более новую версию CUDA (например, 12.4+), просто измените первую строку в `Dockerfile` на:
    `FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime` (или актуальную версию с Docker Hub).
