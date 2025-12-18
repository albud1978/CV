# Changelog

Все значимые изменения проекта документируются в этом файле.

Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.0.0/).

## [Unreleased]

### Добавлено
- Обновлена эталонная архитектура: добавлен **Molmo 2** (Allen Institute for AI) как VLM-слой для семантического понимания видео.
- Описаны сценарии интеграции Molmo 2: real-time алерты, пост-анализ, интерактивный поиск.
- Добавлен roadmap интеграции Molmo 2 и требования к железу.
- Интегрирована модель **RF-DETR** (Real-Time DEtection TRansformer) через официальную библиотеку Roboflow.
- Обновлен скрипт `auto_label.py`: добавлена поддержка `--type rf-detr` для авторазметки.
- Интегрирована модель **Grounding DINO** (v1.0 локально, v1.5 через API SDK).
- Созданы примеры инференса: `src/inference/text_search.py`, `src/inference/dino_1_5_api.py`, `src/inference/sam2_demo.py`.
- Создан модуль инференса `src/inference/rf_detr.py`.
- Docker образ обновлен до версии `devel` (содержит nvcc) для поддержки компиляции сложных моделей.

### Исправлено
- Устранены зависания при сборке Docker: оптимизировано потребление VRAM (выключены конфликтующие контейнеры Ollama/VLLM).
- Исправлена проблема с монтированием томов Docker: удалены битые симлинки `input`/`output`, созданы локальные директории.
- Сгенерирован файл `.env` для корректного запуска на текущем хосте.
- Исправлен потенциальный краш в `src/inference/sam2_demo.py` при отсутствии валидного промпта (проверка `masks is None`).
- Добавлена проверка на валидность изображения в `src/inference/dino_1_5_api.py` (защита от `AttributeError` если `imread` вернет `None`).
- Добавлена проверка на валидность изображения в `src/inference/text_search.py` (защита от `AttributeError` если `imread` вернет `None`).



### Добавлено
- Обновлен `requirements.txt`: добавлена установка SAM 2 (git), Roboflow Inference, RF-DETR
- Скрипт авторазметки данных `src/utils/auto_label.py` (использует YOLOv8x/l)
- Скрипт загрузки моделей `src/utils/download_models.py`
- Обновлен `requirements.txt` (добавлены `ultralytics`, `roboflow`, `supervision`)
- Обновлен `README.md` (инструкция по запуску через Docker)
- Создана структура подкаталогов в `src/` (models, utils, configs, train, inference)
- Добавлена Docker-среда (`Dockerfile`, `docker-compose.yml`) с поддержкой GPU
- Создана папка `docs` для документации
- Инициализация проекта
- Создана базовая структура папок
- Настроены симлинки на Nextcloud для input/output данных
- Добавлен README.md с описанием проекта
- Добавлен .gitignore
- Добавлены правила ведения проекта (.cursorrules)

---

## Шаблон записи

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Добавлено
- Новые функции

### Изменено
- Изменения в существующей функциональности

### Исправлено
- Исправления ошибок

### Удалено
- Удалённые функции

### Результаты
- Метрики и результаты экспериментов
```


