# CV - Computer Vision для гражданской авиации

## Описание проекта

Проект по разработке моделей компьютерного зрения (Computer Vision) для задач гражданской авиации. Основные задачи включают детекцию спецтехники и персонала, контроль соблюдения регламентов и обеспечение безопасности на перроне.

Стек технологий: **PyTorch**, **YOLOv8/v11** (Ultralytics), **SAM 2** (Meta), **RF-DETR** (Roboflow).

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
Перед первым запуском скачайте веса моделей (YOLOv8/11, SAM 2) внутрь проекта:
```bash
docker-compose exec cv-dev python3 src/utils/download_models.py
```
Веса сохранятся в `src/models/` и будут доступны при следующих запусках.

### 5. Запуск скриптов
Примеры команд (выполняются из корня проекта):

**Автоматическая разметка (Auto-Labeling):**
```bash
docker-compose exec cv-dev python3 src/utils/auto_label.py --input /app/input --output /app/output/labels --conf 0.4
```

## Работа с моделями

*   **YOLOv8/11**: Используется через библиотеку `ultralytics`. Веса: `src/models/yolov8*.pt`.
*   **SAM 2**: Установлен из официального репозитория Meta. Используется для точной сегментации.
*   **RF-DETR**: Установлен через `pip install rfdetr`.

## Решение проблем

*   **Ошибка `nvidia-container-cli: requirement error`**: Убедитесь, что Docker Desktop настроен на использование WSL2 backend (Windows) или установлен NVIDIA Container Toolkit (Linux).
*   **Ошибка памяти (`DataLoader worker exited`)**: В `docker-compose.yml` уже настроен `shm_size: '8gb'`, этого должно хватать.
*   **Смена видеокарты (например, на RTX 5080)**: Текущий образ использует CUDA 12.1. Если новая видеокарта потребует более новую версию CUDA (например, 12.4+), просто измените первую строку в `Dockerfile` на:
    `FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime` (или актуальную версию с Docker Hub).
