"""Конвейер подготовки данных и обучения CV-моделей.

Модули:
    ingest — извлечение и дедупликация кадров из видео, генерация frames_manifest.
    label  — авто-разметка учителем (Roboflow Auto Label / autodistill). TODO.
    train  — версионирование датасета и обучение student-модели. TODO.
    eval   — оценка качества и генерация model_card. TODO.

См. docs/PIPELINE_ARCHITECTURE.md.
"""
