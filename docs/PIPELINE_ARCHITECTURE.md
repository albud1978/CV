# Архитектура конвейера авто-разметки и обучения CV-моделей

> Документ описывает **пайплайн подготовки данных и обучения** (data → labels → model).
> Это отдельный контур от `REFERENCE_ARCHITECTURE.md`, который описывает **продакшн-инференс**
> (RF-DETR → SAM 2 → Molmo 2). Связь: данный конвейер производит обученный детектор (student),
> который затем используется в inference-стеке.

## 1. Цель

По языковому описанию объектов (`ontology.yaml`) и набору видео из `input/` получить:

1. Проверяемый версионированный датасет в Roboflow.
2. Обученную модель детекции (RF-DETR / YOLO).
3. Метрики качества (`metrics.json`) и карточку модели (`model_card.md`).
4. Контур активного дообучения на новых ошибках.

**Главный принцип** — Human-in-the-Loop, а не «полный автопилот»: модель-учитель быстро
создаёт псевдоразметку, человек удаляет/правит критичные ошибки.

## 2. Стек (по умолчанию)

| Слой | Инструмент по умолчанию | Лицензия | Альтернатива |
|---|---|---|---|
| Извлечение кадров | PySceneDetect (Adaptive) + pHash dedup | BSD-3 / MIT | MOG2 (`motion_detect.py`) |
| Учитель (auto-label) | Roboflow Auto Label через MCP (`autolabel_start`) | hosted | локально `autodistill-grounded-sam-2` (Florence-2 + SAM 2) |
| Human review | Roboflow Annotate UI | hosted | — |
| Версионирование | Roboflow `versions_generate` | hosted | — |
| Student (обучение) | RF-DETR-M (primary) | Apache 2.0 | YOLOv8-l (baseline), RF-DETR-N (smoke) |
| Оценка | Roboflow `model_evals_*` + SAHI | hosted / Apache 2.0 | `ultralytics val` |
| Оркестрация | Cursor + Roboflow MCP (`https://mcp.roboflow.com/mcp`) | — | — |

> SAM 3 как учитель — **опциональный апгрейд** после факт-чека доступности весов
> (`facebook/sam3`) и условий Meta SAM License. До тех пор дефолт — открытые SAM 2 / Florence-2.

## 3. Архитектура пайплайна (5 слоёв + гейты)

```
ВХОД:  task.yaml  +  ontology.yaml  +  ./input/<videos>
   |
   v
СЛОЙ 1 — INGEST                                  src/pipeline/ingest.py
   ffmpeg normalize -> PySceneDetect (Adaptive) -> pHash dedup
   -> frames_manifest.parquet (video_id, scene_id, ts, hash, split)
   -> upload в Roboflow (images_prepare_upload[_zip])
   [Gate G0] манифест валиден, доля дублей <= 5%
   |
   v
СЛОЙ 2 — AUTO-LABEL (один учитель)               src/pipeline/label.py
   Default:  Roboflow Auto Label (autolabel_start)
   Fallback: локально autodistill-grounded-sam-2
   per-class confidence из ontology.yaml
   -> pseudo_labels.coco.json + гистограмма confidence по классам
   [Gate G1] AP учителя на golden-set >= 0.70
   |
   v
СЛОЙ 3 — HUMAN REVIEW                             Roboflow Annotate UI
   фильтр по confidence, batch review, удаление "битых" кадров
   review_queue.csv = низкая уверенность U Cleanlab top-N
   [Gate G2] >= 95% approved, баланс классов в норме
   |
   v
СЛОЙ 4 — VERSION + TRAIN                          src/pipeline/train.py
   versions_generate: 70/20/10 stratified by video_id
   augment policy = configs/augment_conservative.yaml
   parallel:  RF-DETR-M (primary) + YOLOv8-l (baseline)
   |
   v
СЛОЙ 5 — EVALUATE + DEPLOY + LOOP                 src/pipeline/eval.py
   model_evals_* (mAP, confusion, per-class) + SAHI на golden test
   -> metrics.json + model_card.md (авто)
   Workflows: SAM verifier + student detector + sink
   |
   v
АКТИВНОЕ ДООБУЧЕНИЕ: production errors -> новые кадры -> назад в СЛОЙ 2
```

## 4. Гейты (критерии перехода)

| Гейт | Условие перехода | Что делаем, если не прошли |
|---|---|---|
| **G0** | манифест валиден, дублей ≤ 5%, кадры залиты в проект | ужесточить/ослабить порог pHash, перепроверить ffmpeg |
| **G1** | AP учителя на golden-set (200 кадров) ≥ 0.70 | сменить промпты в `ontology.yaml`, понизить per-class conf; крайняя мера — сменить учителя |
| **G2** | ≥ 95% кадров approved, нет класса с < 20 инстансами | целенаправленно добрать видео с редкими классами |

## 5. Контракты данных (артефакты)

| Артефакт | Назначение | Ключевые поля |
|---|---|---|
| `task.yaml` | Постановка задачи для агента и повторного запуска | goal, task_type, classes, deployment_target, acceptance_metrics |
| `ontology.yaml` | Промпты и правила классов | class_id, label, positive_prompts, negative_prompts, confidence |
| `frames_manifest.parquet` | Происхождение каждого кадра | video_id, scene_id, ts, hash, quality, split, batch |
| `pseudo_labels.coco.json` | Авто-разметка учителя | bbox/mask, class, confidence, teacher_id, prompt_id |
| `golden_set.coco.json` | 200 кадров ручной разметки для калибровки учителя | bbox, class (ground truth) |
| `review_queue.csv` | Очередь ручной проверки | reason, risk_score, suggested_action, decision |
| `dataset_version.json` | Слепок датасета | version_id, preprocessing, augmentation, split_rules |
| `metrics.json` | Сравнение моделей и порогов | mAP50, mAP50-95, per_class, slices, thresholds |
| `model_card.md` | Паспорт модели | назначение, классы, ограничения, метрики, версия |

## 6. Структура кода

```
src/
├── pipeline/                      # конвейер обучения (NEW)
│   ├── __init__.py
│   ├── ingest.py                  # ffmpeg + PySceneDetect + pHash -> frames_manifest
│   ├── label.py                   # MCP autolabel_start | autodistill (TODO)
│   ├── train.py                   # versions_generate + models_train (TODO)
│   └── eval.py                    # model_evals_* + SAHI golden test (TODO)
├── inference/                     # продакшн-инференс (EXISTING)
│   ├── rf_detr.py                 # student model
│   ├── sam2_demo.py               # tracking
│   └── molmo2.py                  # VLM QA
├── utils/
│   ├── motion_detect.py           # MOG2 motion-based нарезка (EXISTING)
│   └── auto_label.py              # локальная авто-разметка YOLO/RF-DETR (EXISTING)
└── configs/                       # см. configs/ в корне проекта
```

Конфиги-контракты лежат в `configs/` в корне репозитория (`task.example.yaml`,
`ontology.example.yaml`, `augment_conservative.yaml`).

## 7. Архитектура разработки: без мультиагента (для MVP)

Для текущего этапа мультиагент **не используется**. Один основной чат-агент Cursor +
Roboflow MCP закрывает весь путь. Декомпозиция на 6 ролей оправдана только при параллельном
переборе нескольких teacher/student-комбинаций неделями.

Вместо мультиагента:

- **Один основной агент** с MCP: `roboflow`, `github` (PR с экспериментами).
- **Плейбуки-режимы** в `.cursor/rules/` (правила подгружаются по контексту задачи).
- **Background Agents Cursor** — только на 2 сценария:
  1. Параллельный A/B учителей на golden-set (если первый не дал ≥ 0.70 AP).
  2. Долгое обучение (RF-DETR-L) — фоновый мониторинг `models_get_training_status`.
- **Roboflow Skills** (`data-management`, `training-and-evaluation`, `inference`, `universe`) —
  подгружаются как MCP-ресурсы по необходимости.

**Когда переходить на мультиагент:** ≥ 3 экспериментальных конфига параллельно неделями,
когда ручной мониторинг становится узким местом.

## 8. Карта инструментов Roboflow MCP по слоям

| Слой | MCP-инструменты |
|---|---|
| 1. Ingest | `projects_create`, `projects_get`, `images_prepare_upload`, `images_prepare_upload_zip`, `images_upload_zip_status` |
| 2. Auto-label | `autolabel_start`, `autolabel_job_get` |
| 3. Review | `annotation_jobs_create`, `annotation_batches_list`, `annotation_batches_get`, `annotations_save` |
| 4. Version + Train | `versions_generate`, `versions_get`, `models_train`, `models_get_training_status`, `trainings_get_results` |
| 5. Eval + Deploy | `model_evals_get_map_results`, `model_evals_get_confusion_matrix`, `model_evals_get_performance_by_class`, `workflows_create`, `workflows_run` |
| Bootstrap | `universe_search`, `projects_fork` (напр. `airport-gse/airport-ground-vehicles`) |

## 9. Фазы внедрения

### Фаза 0 — инфраструктура (выполнено / в процессе)
- [x] Roboflow workspace (`ban-nzbro`), API-ключ, MCP в Cursor.
- [ ] Установить пайплайн-зависимости: `scenedetect`, `imagehash`, `sahi`, `cleanlab`, `autodistill`.
- [ ] Факт-чек SAM 3 (доступ к весам + лицензия) — решает, апгрейдим ли учителя.

### Фаза 1 — первый рабочий цикл (smoke → MVP)
1. 1 видео из `input/` → ingest → 200-300 кадров.
2. Upload в новый Roboflow project через MCP.
3. `autolabel_start` с 3 классами (MVP-3).
4. Ручной обзор в Roboflow Annotate.
5. `versions_generate` (70/20/10 stratified by video_id).
6. `models_train` (RF-DETR-N для smoke, далее RF-DETR-M).
7. baseline mAP в `metrics.json` через `model_evals_get_map_results`.

**Критерий перехода:** mAP@50 ≥ 0.75 на golden test, per-class AP ≥ 0.5 для всех классов.

### Фаза 2 — итерация качества
1. Cleanlab loop: predict on train → top-N подозрительных → ревью.
2. A/B учителей (только если потолок качества).
3. Сравнение RF-DETR-M vs YOLOv8-l.
4. SAHI на полнокадровом инференсе для дальних объектов.

**Критерий перехода:** mAP@50 ≥ 0.90, mAP@50:95 ≥ 0.65.

### Фаза 3 — продакшн и универсализация
1. Экспорт лучшей модели в ONNX → TensorRT FP16.
2. Универсальный CLI `cv-pipeline new --classes "..." --videos ./input/`.
3. Continuous data flywheel: новые видео → дельта-дообучение.

## 10. Стандарты обучения (зафиксированы в `.cursor/rules/cv-pipeline.mdc`)

- `imgsz=1280` для сцен с мелкими/удалёнными объектами.
- `close_mosaic=10`, `mixup=0`, `erasing=0`, `copy_paste=0`.
- Split 70/20/10, **stratified by `video_id`** (не случайно по кадрам — иначе утечка).
- Per-class confidence в Auto Label: редкие классы — 0.20, частые — 0.40.
- SAHI только на инференсе (slice 512–832, overlap 0.2–0.25).

## 11. Открытые вопросы (факт-чек)

См. `docs/CAVEATS.md` (переносится из исследовательского отчёта). Критичные:
- SAM 3: реальная доступность весов + текст Meta SAM License (field-of-use).
- `autodistill-sam3` — наличие на PyPI.
- Бенчмарки из публикаций (mAP 0.987) — только ориентир, не KPI.
