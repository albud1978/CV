# Универсальный конвейер авто‑разметки и обучения CV‑моделей: архитектура на Roboflow MCP + Cursor

## TL;DR
- **Рекомендуемая базовая конфигурация (по состоянию на май 2026):** SAM 3 (Meta, релиз 19 ноября 2025, 848 М параметров, лицензия Meta SAM License) в качестве «учителя» через `autodistill-sam3`, RF‑DETR (Apache 2.0, ICLR 2026) или YOLOv8/YOLO11 в качестве «ученика», Roboflow как единая платформа разметки/версионирования/обучения, оркестрация через Roboflow MCP Server (`https://mcp.roboflow.com/mcp`, ~50+ tools в 12 категориях по `llms.txt`) из Cursor 2.x с фоновыми/параллельными агентами. Это даёт текстово‑промптируемую авто‑разметку с боксами **и** масками, человеческую фильтрацию в одном UI и обучение/деплой одной кнопкой.
- **Для самолётного use‑case (~10 классов GSE — Ground Power Unit, catering truck, fuel truck, baggage cart, pushback tractor, chocks, cones, ground crew, jet bridge, belt loader)** оптимально: дедупликация кадров по PySceneDetect + perceptual/CLIP‑embedding, авто‑разметка SAM 3, ручное удаление «грязных» кадров в Roboflow Annotate, обучение RF‑DETR‑M/L или YOLOv8‑l/x с `imgsz=1280`, `close_mosaic=10–15`, отключёнными агрессивными erasing/cutout, и SAHI tiled inference для дальних мелких объектов. По публикации AIMS Press 2024 «An airport apron ground service surveillance algorithm based on improved YOLO network» на схожих сценах достижим mAP@50 ≈ 0.987 при чистой разметке (YOLOv5 + SPD‑Conv на собственном корпусе авторов).
- **Главная архитектурная развилка — где «жить» конвейеру:** (a) полностью в Roboflow Cloud (MCP запускает Auto Label + Roboflow Train + Workflows), (b) гибридно (Roboflow для разметки/версий, локальные RTX 4080 / RTX Pro 6000 Blackwell для обучения учителей и студентов), (c) полностью локально (SAM 3 / Grounded‑SAM 2 / Autodistill + Ultralytics на своём железе). Для бюджета пользователя оптимален гибрид (b): SAM 3 при 848 М параметров требует ≥16 ГБ VRAM и масштабируется на RTX Pro 6000, а Roboflow MCP даёт оркестрацию без написания SDK‑клея.

## Key Findings

### 1. Состояние моделей‑учителей в мае 2026
- **SAM 3 (Meta, 19.11.2025).** Архитектура: общий Perception Encoder (5,4 млрд пар «изображение–текст» предобучения) + DETR‑детектор + SAM 2‑трекер с presence‑head. 848 М параметров (по официальному репо `facebookresearch/sam3`; сравнительная таблица Ultralytics приводит 473,6 М — расхождение скорее всего связано с подсчётом одного из двух подсетей, флагуем). Файл весов `sam3.pt` ≈ 3,45 ГБ. Скорость: **30 мс/кадр на H200** при 100+ объектах; на RTX Pro 6000 (Ultralytics‑бенч) ~2921 мс/im, что говорит о необходимости TensorRT/FP16. Главное новшество — **Promptable Concept Segmentation (PCS)**: короткая текстовая фраза («fuel truck», «ground power unit») сегментирует **все** инстансы в кадре. По key‑metrics‑таблице Ultralytics SAM 3 docs: «Achieves 88% of estimated lower bound on SA‑Co/Gold»; LVIS Zero‑Shot Mask AP **47.0 vs предыдущий лучший 38.5 (+22 %)** (SAM 3, arXiv 2511.16719, Carion et al., Meta, Nov 2025; подтверждено verbatim в Ultralytics SAM 3 docs). По SA‑Co box cgF1: **SAM 3 = 55.7 против DINO‑X 22.5 и OWLv2 24.5** (SAM 3 arXiv 2511.16719; MarkTechPost summary: «SAM 3 reports cgF1 of 55.7, while OWLv2 reaches 24.5, DINO‑X reaches 22.5 and Gemini 2.5 reaches 14.4»). Веса под gated‑доступом на HF `facebook/sam3`. Лицензия — custom **Meta SAM License**: разрешено коммерческое использование с ограничениями (запрет военного/ITAR применения, condition‑terminating patent clause).
- **Grounded‑SAM 2 (IDEA‑Research).** Это «pipeline‑модель»: Grounding DINO / Grounding DINO 1.5 / DINO‑X / Florence‑2 даёт боксы по тексту, SAM 2 превращает их в маски и треки в видео. Apache 2.0 на код. Уже встроен в Autodistill (`autodistill-grounded-sam-2`). Для нашего конвейера интересен как «второе мнение» против SAM 3, особенно для видео‑трекинга.
- **Grounding DINO.** Apache 2.0, март 2023, transformer‑детектор. По исследованию Ilyas et al. 2024 (arXiv:2408.11221, Aptiv / Univ. Wuppertal) — лучший baseline на out‑of‑distribution бенчмарках: «Grounding DINO achieves the best results on RoadObstacle21 and LostAndFound in our study with an AP₅₀..₉₅ of **48.3 % и 25.4 %** respectively». На сельхоз‑данных по Mullins et al. 2024 («Enhanced Image Annotation in Wild Blueberry … Fields Using Sequential Zero‑Shot Detection and Segmentation Models», PMC12694221): «optimal confidence thresholds were 0.00365 for YOLO‑World and 0.360 for Grounding DINO. Grounding DINO achieved a significantly higher mean IoU (**0.642**) compared to YOLO‑World (**0.503**)». Минус — вычислительно тяжелее.
- **YOLO‑World v2 (Tencent, февраль 2024).** CNN‑детектор, **GPL‑v3**. ~13 М параметров в S‑варианте. По Roboflow model page (`roboflow.com/model/yolo-world`): «On an NVIDIA V100, the small variant reaches **~74 FPS** at standard resolutions». На LVIS — 35,4 AP. **GPL‑v3 — критическое ограничение для коммерческого деплоя**: можно использовать как «учителя» внутри пайплайна, но не для прямого встраивания в закрытый продукт.
- **DINO‑X (IDEA Research, ноябрь 2024).** Текущий SOTA в open‑world detection: 56.0 AP на COCO, 59.8 LVIS‑minival, 52.4 LVIS‑val; на LVIS‑rare обгоняет Grounding DINO 1.6 Pro на 5,8/5,0 AP. Поддерживает текст‑, визуальные и кастомные промпты, prompt‑free detection. Раздаётся преимущественно через DeepDataSpace API (не полностью открытые веса), но интегрирован в Grounded‑SAM‑2.
- **Florence‑2 (Microsoft, июнь 2024).** Лицензия **MIT**, 0.23/0.77 B параметров. Универсальный VLM (детекция, сегментация, caption, OCR). В сочетании с SAM 2 даёт «caption → phrase grounding → segmentation» пайплайн (autodistill‑grounded‑sam‑2 использует именно его). Для авто‑разметки удобен тем, что лицензия максимально либеральная.
- **OWLv2 (Google, июнь 2023).** Apache 2.0, трансформер с CLIP‑бэкбоном, обучен по рецепту OWL‑ST на >1 млрд псевдо‑аннотаций. AP до 47,2 % на LVIS rare. Сильно проигрывает SAM 3 по cgF1 (24,5 vs 55,7) и не даёт масок. Хороший «второй мнение» бэйслайн для боксов.
- **Roboflow Auto Label.** Хостовый сервис, с ноября 2025 поддерживает **SAM 3‑powered Auto Label** прямо в UI: загрузил батч → выбрал «SAM 3» → ввёл текстовые промпты + confidence threshold по каждому классу → получил предразметку, готовую к ручному ревью.

### 2. Autodistill — клей между учителем и учеником
- Autodistill — open‑source‑фреймворк Roboflow с парадигмой «base model → ontology → target model». Ontology — это `CaptionOntology({"fuel truck": "fuel_truck", "ground power unit": "gpu", ...})`, где ключ идёт в промпт SAM 3 / Grounding DINO, а значение — это итоговое имя класса в YOLO‑датасете.
- Подтверждено существование пакета **`autodistill-sam3`** (GitHub `autodistill/autodistill-sam3`, PyPI v0.1.0, последний коммит 25 ноября 2025). README: «*The Autodistill SAM 3 package only works on a GPU*». Установка: `pip install autodistill-sam3` + `ROBOFLOW_API_KEY=...` (через который качаются веса). Возвращает одновременно боксы и маски — поддерживает обучение и детектора, и сегментатора.
- Target models: `autodistill-yolov8`, `autodistill-yolov11`, `autodistill-rfdetr`, `autodistill-detr` и др. Один и тот же скрипт может «перебрать» нескольких учеников.

### 3. Roboflow MCP Server — реальная поверхность API для агентов
- Endpoint: `https://mcp.roboflow.com/mcp`, транспорт — streamable HTTP, auth — заголовок `x-api-key`. Apache 2.0, конфигурация для Cursor — стандартный JSON в `~/.cursor/mcp.json` с типом `http`.
- На странице `roboflow.com/mcp` рекламируются **67 tools across 12 categories**; в `llms.txt` сервера на момент проверки перечислено ~50+ tools в 9 категориях. Маркетинговая FAQ ещё содержит старое число «30 tools». Канонический источник — `https://mcp.roboflow.com/llms.txt`.
- Категории и наиболее важные для нашего пайплайна tools:
  - **Projects:** `projects_list`, `projects_create`, `projects_fork`, `projects_get`, `projects_health`.
  - **Images:** `images_prepare_upload`, `images_prepare_upload_zip`, `images_upload_zip_status`, `images_search`.
  - **Auto‑label:** `autolabel_start` («*Start a hosted auto‑label job over a batch of images*»), `autolabel_job_get`.
  - **Annotations / Batches:** `annotations_save`, `annotation_batches_list/get`, `annotation_jobs_create`.
  - **Versions:** `versions_generate`, `versions_get`, `versions_export` — управление train/val/test split, аугментациями и snapshot’ами датасета.
  - **Models / Training:** `models_train`, `models_get_training_status`, `trainings_get_results`, `trainings_cancel/stop`, `models_infer`, `models_list/get`, `models_star_nas`. Поддерживаемый выбор архитектур: RF‑DETR, YOLO11, YOLOv8, YOLO‑NAS, instance segmentation; блог Roboflow про MCP прямо упоминает «full training menu, from standard RF‑DETR to complete Neural Architecture Search jobs».
  - **Model Evaluations:** `model_evals_get_map_results`, `model_evals_get_confusion_matrix`, `model_evals_get_performance_by_class`, `model_evals_get_image_predictions`, `model_evals_get_recommendations`.
  - **Workflows:** `workflows_create/update/run`, `workflow_blocks_list/get_schema`, `workflow_specs_validate/run` — позволяют агенту собирать инференс‑граф (SAM 3 → фильтр уверенности → постпроцесс → запись результата).
  - **Devices/Streams:** управление edge‑деплоем и стримами.
  - **Universe:** `universe_search`, `universe_dataset_images_search` — поиск готовых датасетов (например, найденный нами `airport-gse/airport-ground-vehicles` на 982 изображения с YOLOv8‑разметкой можно форкнуть и использовать для bootstrap’а классов).
- Кроме tools, MCP отдаёт **Skills** — markdown‑playbooks как ресурсы протокола, чтобы у агента в Cursor было «доменное знание» Roboflow: `api-reference`, `data-management`, `inference`, `training-and-evaluation`, `universe`, `product-navigation`, `plans-and-pricing`. Их можно установить локально через `npx @roboflow/skills install`.

### 4. Cursor 2.x как оркестратор
- Cursor поддерживает MCP «first‑class» (через `~/.cursor/mcp.json`). С версии 2.0 доступны **Background Agents** (параллельная работа в облаке) и **многопоточная Agent‑архитектура** с Git‑worktree‑изоляцией — до 8 параллельных агентов на одном репозитории. Это позволяет в одном проекте запустить, например, агента «датасет», агента «обучение» и агента «отчётность» одновременно.
- Полезные дополнительные MCP в проекте: `github` (PR с экспериментами), `playwright` (e2e‑проверка UI Roboflow), а также «agent‑to‑agent» MCP из cursor‑community для делегирования между специализированными агентами.
- Cursor CLI (`cursor --headless`) с мая 2026 стабилен на macOS/Linux/Windows и позволяет встроить агента в CI: команды «когда приходит новое видео — запусти `extract → autolabel → review → train`» становятся обычными pipeline‑шагами.

### 5. Кадры из видео: дедупликация без потери редких сцен
- **PySceneDetect** (BSD‑3) — индустриальный стандарт: `ContentDetector` для жёстких склеек, `AdaptiveDetector` для плавных движений камеры, `ThresholdDetector` для fade in/out. CLI: `scenedetect -i video.mp4 detect-adaptive save-images`. Удобно сохранять start/middle/end‑кадр каждой сцены.
- Поверх сцен — **perceptual hashing** (`imagehash.phash`) и **CLIP/DINOv2 embedding similarity** с DBSCAN/HDBSCAN кластеризацией для отсечения near‑duplicate. В практических пайплайнах это даёт до 45 % экономии хранения и снижает «утечку» между train/val при стандартном random split.
- Для аэродромных видео (часто статичная камера, медленное движение GSE) `AdaptiveDetector` + порог сходства pHash ≤ 6 битов + минимальный шаг 1 сек дают разумный baseline; на каждом «событии» (подъезд машины, открытие двери) сохраняем 3–5 представителей.

### 6. Тренировка маленьких/удалённых объектов
- **imgsz=1280 практически всегда лучше дефолтных 640** для GSE‑сцен, где техника часто занимает <5 % площади. По публичным экспериментам Ultralytics, для 1920×1080 c объектами ~10 px по ширине recall растёт с ~0.50 до ~0.66 при переходе 640→1280 (ценой ~2× latency).
- **SAHI (sliced inference)** — обязателен для инференса на полнокадровых 4К видео аэродромного перрона. Параметры: `slice_height/width=512–832`, `overlap=0.2–0.25`. Используется только на инференсе и/или на этапе авто‑разметки, в обучение тайлы заводить не обязательно, если imgsz уже 1280.
- **Аугментации.** Для мелких быстро движущихся объектов **избегать**: агрессивный `erasing`, `mixup`, двойного мозаика. Mosaic полезен только в первой половине обучения и должен быть **закрыт за 10–15 эпох до конца** (`close_mosaic=10`). На уровне Roboflow Augmentation: не включать mosaic, если он уже включён в YOLO; не выкручивать HSV‑shift > 0.05/0.5/0.5; perspective ≤ 0.001.
- **Confidence/NMS.** Низкий conf (0.001–0.01) полезен для оценки recall, но в продакшен‑деплой ставить 0.20–0.30 и калибровать через `model_evals_get_confidence_sweep`. Опыт пользователя (слишком низкий conf «топит» мелкие объекты ложными срабатываниями) подтверждается публикациями.
- **Split.** 70/20/10 train/val/test — разумный дефолт; в Roboflow задаётся в `versions_generate`. Для видео — обязательно делать **stratified split по видео‑ID**, а не случайно по кадрам, иначе будут утечки.
- **Класс‑баланс.** Для 10 GSE‑классов с типовым перекосом (ground crew/cones — десятки тысяч инстансов, jet bridge/belt loader — сотни) либо применять class‑weighted loss, либо целенаправленно набирать видео с редкими классами и контролировать через `model_evals_get_performance_by_class`.

### 7. Существующие данные/работы по аэродромному перрону (полезный bootstrap)
- **Roboflow Universe `airport-gse/airport-ground-vehicles`** — 982 размеченных YOLOv8‑кадра, можно форкнуть через MCP `projects_fork`.
- **MDPI Sensors 2024** «Detection and Control Framework for Unpiloted Ground Support Equipment within the Aircraft Stand» — собственный YOLO‑бэйслайн, F1 ≈ 0.845, 95 % accuracy на guiding markers.
- **AAD‑dataset (DT‑YOLO, MDPI 2024/2025)** — 8 643 изображения, 85 275 инстансов, 6 классов GSE/персонала. Архитектура: YOLOv5 + D‑CTR (transformer self‑attention в backbone).
- **AIMS Press 2024** «An airport apron ground service surveillance algorithm based on improved YOLO network» — YOLOv5 + SPD‑Conv для мелких объектов, mAP@50 = 0.987 на собственном корпусе. По прямой цитате: «the original model was converted to TensorRT and OpenVINO format models, which increased the inference efficiency of the GPU and CPU by **55.3 %** and **137.1 %**, respectively». Это benchmark и одновременно подсказка по deployment‑оптимизации.

### 8. Бенчмаркинг и сравнение учителей/учеников
- **Метрики:** mAP@50, mAP@50:95, per‑class AP, precision/recall, confusion matrix — все доступны через `model_evals_*` или локально через `ultralytics val`. Для авто‑разметки дополнительно — **agreement‑метрика** между двумя учителями (SAM 3 vs Grounded‑SAM 2) на одних и тех же кадрах: чем меньше расхождение, тем меньше нужно ручной чистки.
- **Качество псевдо‑меток** оценивается на «золотом» вручную размеченном подмножестве в 100–300 кадров; стандарт — посчитать AP учителя относительно человеческой разметки на этом сабсете перед дистилляцией.
- **Cleanlab + ActiveLab** — Python‑пакеты для нахождения label errors и приоритизации того, что переразметить. Для object detection cleanlab поддерживает per‑box scoring; в практическом цикле: train → predict on train → cleanlab → top‑N подозрительных боксов → в Roboflow на ручной ревью.

## Details

### Полная архитектура (универсальная, swap‑friendly)

```
                       ┌──────────────────────────────────────────────────────────────┐
                       │   Пользователь:  text-prompts + видео  (любая будущая задача) │
                       └──────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
   ┌──────────────────────────────────────────────────────────────────────────────────────┐
   │  СЛОЙ 1 — Ingest & Frames                                                            │
   │   - ffmpeg unify (codec/fps)                                                         │
   │   - PySceneDetect (AdaptiveDetector) → сцены                                         │
   │   - sample N кадров на сцену + fixed FPS-baseline (например 1 fps)                    │
   │   - pHash + CLIP-embedding DBSCAN → дедуп                                            │
   │   - Запись в S3/локальный том + манифест JSONL                                       │
   └──────────────────────────────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
   ┌──────────────────────────────────────────────────────────────────────────────────────┐
   │  СЛОЙ 2 — Auto-Labeling Teachers (swappable)                                          │
   │   Default A:  SAM 3  via autodistill-sam3  (текст + маски + боксы)                    │
   │   Default B:  Roboflow Auto Label (SAM 3) через MCP autolabel_start                  │
   │   Alt:        Grounded-SAM 2 (Florence-2 + SAM 2) — для видео-трекинга             │
   │   Alt:        Grounding DINO + SAM 2 (Apache 2.0, если SAM-лицензия проблема)        │
   │   Alt:        OWLv2 / YOLO-World / DINO-X — для A/B-сравнения                        │
   │   → COCO/YOLO json + masks RLE; запись в Roboflow project                            │
   └──────────────────────────────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
   ┌──────────────────────────────────────────────────────────────────────────────────────┐
   │  СЛОЙ 3 — Human-in-the-Loop                                                          │
   │   - Roboflow Annotate: фильтр по confidence, ревью батчами                           │
   │   - Удаление «битых» кадров одной кнопкой                                            │
   │   - Cleanlab pass (опционально) на train-предсказаниях → приоритизация ревью         │
   │   - Annotation jobs в Roboflow → распределение по аннотаторам                        │
   └──────────────────────────────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
   ┌──────────────────────────────────────────────────────────────────────────────────────┐
   │  СЛОЙ 4 — Dataset Versioning & Augmentation                                          │
   │   - versions_generate: train/val/test 70/20/10, stratified by video_id               │
   │   - Augmentation conservative: HSV±0.015/0.5/0.4, flip-lr 0.5, scale 0.5,             │
   │     mosaic ON только в YOLO (не в Roboflow одновременно), close_mosaic=10            │
   │   - Snapshot версии: «dataset@v1 SAM3», «dataset@v2 GroundedSAM2», «dataset@v3 mix»  │
   └──────────────────────────────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
   ┌──────────────────────────────────────────────────────────────────────────────────────┐
   │  СЛОЙ 5 — Student Training (swappable)                                               │
   │   Default A:  RF-DETR-M/L (Apache 2.0, ICLR 2026) — лучший accuracy/latency tradeoff │
   │   Default B:  YOLOv8-l/x (то, на чём пользователь уже работает) imgsz=1280           │
   │   Compute:    локально RTX 4080 (small/medium), RTX Pro 6000 Blackwell (large)       │
   │              + Roboflow Train в облаке для контрольного запуска                      │
   │   Логирование: TensorBoard + W&B + Roboflow Train results                            │
   └──────────────────────────────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
   ┌──────────────────────────────────────────────────────────────────────────────────────┐
   │  СЛОЙ 6 — Evaluation & Benchmarking                                                   │
   │   - mAP@50, mAP@50:95, per-class via model_evals_*                                    │
   │   - Confusion matrix, confidence sweep                                                │
   │   - Сравнение teacher↔teacher (agreement) и student↔teacher                          │
   │   - SAHI tiled inference на golden test set для дальних объектов                     │
   └──────────────────────────────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
   ┌──────────────────────────────────────────────────────────────────────────────────────┐
   │  СЛОЙ 7 — Deployment                                                                  │
   │   - Roboflow Inference (serverless / dedicated / self-hosted)                         │
   │   - TensorRT / ONNX-экспорт для edge (Jetson, RTX-станции на перроне)                │
   │   - Workflows: SAM 3 endpoint + student-модель в одном графе                          │
   └──────────────────────────────────────────────────────────────────────────────────────┘

  ОРКЕСТРАЦИЯ (cross-cutting):
    Cursor 2.x (Composer + Background Agents) ↔ Roboflow MCP (https://mcp.roboflow.com/mcp)
                                              ↔ GitHub MCP (PR с экспериментами/конфигами)
                                              ↔ локальные Python-скрипты (PySceneDetect, autodistill, ultralytics, sahi)
```

### Многоагентная схема в Cursor (опциональная, но эффективная)
Учитывая Background Agents и поддержку до 8 параллельных потоков, разумная декомпозиция на агентов‑специалистов:

1. **`agent/data`** — отвечает за ingest, дедуп, апдейт манифеста. Tools: локальный shell + `roboflow.images_prepare_upload*`.
2. **`agent/labeler`** — выбирает учителя (SAM 3 / Grounded‑SAM 2 / OWLv2), формирует ontology, запускает `autolabel_start`, мониторит `autolabel_job_get`. Принимает решение «достаточно ли качество» по «золотому» сабсету.
3. **`agent/reviewer`** — постит сводки в чат: «батч X готов к ревью, 1240 кадров, средний conf 0.78, классы с низким recall: belt_loader, jet_bridge». Тригерит человека.
4. **`agent/trainer`** — генерирует версию (`versions_generate`), запускает `models_train` в Roboflow и/или локальный `yolo train`/`rfdetr.train`, мониторит loss‑кривые, возвращает чекпойнты.
5. **`agent/evaluator`** — гоняет `model_evals_*`, считает SAHI metrics, сравнивает «teacher A vs teacher B vs student». Через GitHub MCP создаёт PR с обновлением `BENCHMARK.md`.
6. **`agent/deployer`** — собирает Workflow, экспортирует в ONNX/TensorRT, деплоит на dedicated‑эндпоинт или edge через `devices_*` tools.

Этот разрез разумен, но **не обязателен**: один универсальный агент с Cursor’s Composer + Roboflow MCP уже способен пройти весь путь. Многоагентность даёт выигрыш, когда параллельно сравнивается несколько teacher/student‑комбинаций.

### Сравнение «учителей» — сводная таблица (по состоянию на май 2026)

| Модель | Лицензия | Текст‑промпт | Маски | Видео‑трекинг | VRAM ориентир. | Сильная сторона | Слабая сторона |
|---|---|---|---|---|---|---|---|
| **SAM 3** | Meta SAM License (комм. ✓, не ITAR) | ✓ (PCS) | ✓ | ✓ (наследует SAM 2) | 16+ ГБ | Лучший cgF1 на SA‑Co (55.7), exhaustive instance detection | Gated‑веса, лицензия не OSI‑стандарт |
| **Grounded‑SAM 2** | Apache 2.0 (детектор) + SAM2 license | ✓ (через Florence‑2 / GD) | ✓ | ✓ | 12–16 ГБ | Гибкость, авто‑caption pipeline | Многокомпонентный, больше точек отказа |
| **Grounding DINO** | Apache 2.0 | ✓ | ✗ (только боксы) | ✗ | 8–12 ГБ | Чистый детектор, лучший IoU на OOD (Ilyas et al. 2024) | Без масок, медленнее CNN |
| **YOLO‑World v2** | **GPL‑v3** | ✓ | ✗ | ✗ | 4–8 ГБ | Real‑time, ~74 FPS на V100 (Roboflow model page) | GPL заражает продукт |
| **DINO‑X** | API через DeepDataSpace | ✓ + visual prompt | ✓ (с SAM 2) | ✓ | API | SOTA AP, long‑tail | Не полностью открытые веса |
| **Florence‑2** | **MIT** | ✓ (caption→ground) | ✓ (region) | ✗ | 4–8 ГБ | Идеальная лицензия, компактен | Менее точен на редких классах |
| **OWLv2** | Apache 2.0 | ✓ | ✗ | ✗ | 6–10 ГБ | Открытые веса, scalable | cgF1 24.5 — заметно ниже SAM 3 |

**Рекомендация по умолчанию: SAM 3 как основной учитель + Grounded‑SAM 2 как второй для cross‑check.** Для проектов, где Meta SAM License запрещена политикой клиента, fallback — Florence‑2 + SAM 2 (всё под MIT/Apache).

### Cursor + Roboflow MCP: конкретный setup‑чеклист
1. `npm install -g @cursor/cli` и `cursor auth`.
2. Получить ключ на `https://app.roboflow.com/settings/api`.
3. `~/.cursor/mcp.json`:
   ```json
   {
     "mcpServers": {
       "roboflow": {
         "type": "http",
         "url": "https://mcp.roboflow.com/mcp",
         "headers": {
           "x-api-key": "RF_xxx",
           "Accept": "application/json, text/event-stream"
         }
       },
       "github": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"], "env": {"GITHUB_TOKEN": "ghp_xxx"} }
     }
   }
   ```
4. `npx @roboflow/skills install` — подтянуть Skills (`data-management`, `training-and-evaluation`, `inference` и др.).
5. В корне проекта — `.cursor/rules/cv-pipeline.mdc` с правилами: «всегда использовать `imgsz=1280` для GSE‑датасета, train/val/test = 70/20/10 stratified by `video_id`, close_mosaic=10, никогда не включать `mixup` и `erasing>0.4` для классов <100 инстансов».

### Аэродромный use‑case: конкретный «нулевой запуск»
Промпты для SAM 3 (с короткими существительными — это то, как обучен PCS):
```
ground power unit
catering truck
fuel truck
baggage cart
pushback tractor
wheel chock
traffic cone
ground crew person
jet bridge
belt loader
```
Тонкости:
- «pushback tractor» лучше работает, чем «tug»; «wheel chock» лучше, чем «chocks» (PCS обучен на noun‑phrases в единственном/множественном без артикля).
- Класс **person** надо квалифицировать как «ground crew person in high‑visibility vest» — иначе SAM 3 будет ловить пассажиров на удалённом джет‑мосте.
- Confidence‑thresholds в Roboflow Auto Label выставить **по классам**: для редких (jet bridge, belt loader) — 0.20, для частых (cones) — 0.40, чтобы не утонуть в шуме.
- Использовать **Roboflow Universe** `universe_search("ground support equipment")` и форкнуть `airport-gse/airport-ground-vehicles` (982 размеченных кадра) как bootstrap, добавить к нему наши SAM 3‑аннотации.

### Train/val/test и аугментации — финальный YAML для YOLOv8
```yaml
# yolo train cfg
imgsz: 1280
epochs: 150
batch: 16            # RTX 4080 16GB: batch=8 при imgsz=1280; RTX Pro 6000: batch=16-24
optimizer: AdamW
lr0: 0.001
cos_lr: True
close_mosaic: 10
mosaic: 1.0
mixup: 0.0           # выключено для мелких объектов
copy_paste: 0.0
erasing: 0.0         # выключено
hsv_h: 0.015
hsv_s: 0.5
hsv_v: 0.4
fliplr: 0.5
flipud: 0.0
scale: 0.5
translate: 0.1
perspective: 0.0
label_smoothing: 0.05
amp: True
```
Для RF‑DETR Roboflow обычно сама подбирает гиперпараметры через NAS; вручную трогать не надо.

### Workflow для дальнейшей инференции (Roboflow Workflows + SAM 3 endpoint)
- Блок `roboflow_core/sam3@v3` принимает `text` (список классов), отдаёт masks (RLE) и боксы.
- В одном Workflow можно совместить: `SAM 3` (для верификации) + ваш `student` YOLO (для real‑time) + `Detection Event Log` + `S3 Sink` для логирования.
- Workflows тоже триггерятся через MCP (`workflows_run`).

## Recommendations

### Фаза 0 (1–3 дня): инфраструктура и доступы
1. Завести Roboflow workspace, получить API‑ключ, подключить MCP к Cursor.
2. Запросить gated‑доступ к `facebook/sam3` на HF и прочитать **Meta SAM License** (особенно field‑of‑use restrictions — нет военного/ITAR).
3. На RTX Pro 6000 поставить CUDA 12.4+, `ultralytics>=8.3.237`, `rfdetr`, `autodistill`, `autodistill-sam3`, `sahi`, `pyscenedetect[opencv]`, `imagehash`, `cleanlab`, `supervision`.
4. Прогнать smoke‑test: `from autodistill_sam3 import SegmentAnything3; base_model.label("./test_frames")` — убедиться, что SAM 3 поднимается и пишет COCO‑аннотации.

### Фаза 1 (1–2 недели): первый рабочий цикл на GSE
1. Собрать ≥10 видео (минимум 30 мин, разные часы суток и погода).
2. Прогнать `Слой 1` (PySceneDetect AdaptiveDetector + pHash dedup) — целиться в 3 000–6 000 уникальных кадров на этой стадии.
3. **Параллельный teacher‑run**: одна и та же папка → (a) SAM 3 локально, (b) Roboflow Auto Label с SAM 3, (c) Grounded‑SAM 2 (Florence‑2). Это создаст 3 версии датасета.
4. Создать **golden set** из 200 вручную размеченных кадров — оценить precision/recall каждого учителя.
5. Выбрать лучшего учителя, отревьюить выбранный датасет в Roboflow Annotate (≤4 часа на 3 000 кадров при хорошем учителе).
6. Сгенерировать `version v1`, обучить RF‑DETR‑M и YOLOv8‑l параллельно (RTX 4080 для YOLO, RTX Pro 6000 для RF‑DETR).
7. Зафиксировать baseline mAP в `BENCHMARK.md` через `model_evals_get_map_results`.

**Критерий перехода на Фазу 2:** mAP@50 ≥ 0.75 на golden test, per‑class AP ≥ 0.5 для всех 10 классов.

### Фаза 2 (2–4 недели): итерация качества и сравнение учителей
1. Включить Cleanlab/ActiveLab loop: предсказать на train → top‑500 подозрительных → ревью.
2. A/B‑сравнить teacher A (SAM 3) vs teacher B (Grounded‑SAM 2) на свежей пачке данных. Если разница в AP < 2 п.п., оставаться на SAM 3 ради простоты лицензии и масок.
3. Сравнить student‑архитектуры: RF‑DETR‑M, RF‑DETR‑L, YOLOv8‑l, YOLOv8‑x, YOLO11‑l. **Бенчмарк‑таблица должна включать**: mAP@50, mAP@50:95, latency на RTX Pro 6000 (TensorRT FP16), per‑class AP, GPL/Apache‑статус.
4. Включить SAHI на тестовом инференсе full‑HD/4K видео для дальних объектов; сравнить с прямым `imgsz=1920`.

**Критерий перехода на Фазу 3:** mAP@50 ≥ 0.90, mAP@50:95 ≥ 0.65 (целевые значения исходя из публикаций по AAD‑dataset и AIMS Press 2024).

### Фаза 3 (1–2 недели): продакшн‑деплой и универсализация
1. Экспорт лучшей модели в ONNX → TensorRT FP16; замер latency на edge‑hw. Ожидаемый прирост по AIMS Press 2024: «+55.3 % GPU throughput, +137.1 % CPU throughput» после TensorRT/OpenVINO конверсии.
2. Сделать **универсальный CLI/Cursor‑скрипт** `cv-pipeline new --classes "fuel truck, baggage cart, …" --videos ./input/`, который выполняет всё end‑to‑end. Этот скрипт — главный «универсальный продукт» проекта.
3. Документировать **playbook** (можно как локальный Skill) с конкретными значениями `imgsz`, `close_mosaic`, list of forbidden augmentations.
4. Настроить **continuous data flywheel**: новые видео автоматически → `agent/data` → дельта‑дообучение раз в неделю.

### Когда менять рекомендации
- **Если лицензия Meta SAM не подходит** (например, оборонные/госзаказы): переключиться на default‑B (Grounded‑SAM 2 с Grounding DINO + SAM 2 под Apache 2.0 + SAM2‑license, либо Florence‑2 + SAM 2).
- **Если объекты мельче 5 px по ширине**: добавить SAHI и в training (slicing dataset), перейти на YOLOv8‑P2 / YOLO26‑P2 варианты с дополнительным feature pyramid level.
- **Если бюджет ограничен только RTX 4080**: для обучения брать YOLOv8‑s/m, RF‑DETR‑N/S; для инференса SAM 3 использовать Roboflow Cloud (autolabel_start) вместо локального.
- **Если требуется ≥30 FPS real‑time на edge**: жертвовать DETR‑архитектурой в пользу YOLO11/YOLOv8 + TensorRT INT8.

## Caveats

- **SAM 3 параметры — разночтение.** Официальный README `facebookresearch/sam3` указывает **848 М параметров**, в то время как сравнительная таблица Ultralytics приводит **473.6 М**. Разница, вероятно, в том, что 848 М — это полная сумма detector + tracker + PE backbone, а 473.6 М — это активные при одном проходе. Перед production‑планированием VRAM лучше провести собственный замер на RTX Pro 6000.
- **Лицензия SAM 3.** Meta SAM License **не является OSI‑совместимой**: содержит field‑of‑use restrictions (запрет военного/ITAR применения), право Meta изменять условия и patent‑termination clause. Для коммерческого деплоя в чувствительных отраслях нужна юридическая экспертиза. RF‑DETR (Apache 2.0) и Florence‑2 (MIT) — безопаснее.
- **Расхождение в числе MCP‑tools.** Маркетинговая страница `roboflow.com/mcp` обещает 67 tools в 12 категориях; `mcp.roboflow.com/llms.txt` перечисляет ~50+ tools в 9 категориях; FAQ ещё содержит старое «30 tools». Канонический источник — `llms.txt`, который сервер обновляет автоматически.
- **YOLO‑World v2 — GPL‑v3.** Использование в любом цикле, где результирующая модель распространяется в составе закрытого продукта, требует юр.анализа. Использовать только как «учителя», не как «ученика».
- **DINO‑X доступен преимущественно через API DeepDataSpace.** Открытых весов аналогичных Grounding DINO нет, что усложняет on‑prem.
- **Roboflow Train SLA и стоимость.** В отчёте намеренно не приведены конкретные долларовые цифры — они меняются; для актуальных цен использовать `roboflow.com/pricing` или MCP‑скилл `plans-and-pricing`.
- **Frame dedup может «съесть» редкие события.** Если порог сходства слишком жёсткий, теряются единичные кадры с belt_loader/jet_bridge. Решение — после дедупа прогонять «sanity check» классификатором CLIP/Florence‑2 на наличие редких концептов и возвращать кадры обратно.
- **PCS у SAM 3 чувствителен к формулировке промпта.** Сложные фразы («worker in orange vest holding a hose») работают хуже, чем простые noun‑phrases. Для более сложных запросов есть SAM3‑I (arXiv 2512.04585, декабрь 2025) — instruction‑following extension — но пока это исследовательский проект, не production‑ready.
- **Roboflow MCP — single‑tenant API key.** Все агенты в проекте используют один и тот же ключ; для команд лучше создать workspace‑specific keys и не комитить их в репозиторий.
- **Цифры benchmarks (mAP 0.987 на airport apron из AIMS Press 2024 / F1 0.845 из MDPI Sensors 2024)** получены на собственных закрытых датасетах авторов и могут не воспроизвестись на вашем перроне с другими ракурсами и техникой. Использовать как ориентир, а не как контракт.
- **Bench Mullins et al. 2024 (Grounding DINO IoU 0.642 vs YOLO‑World 0.503)** относится к сельскохозяйственному домену (wild blueberry fields); абсолютные числа на аэродромных сценах будут другими, но **ранжирование** (Grounding DINO стабильно выше по IoU, чем YOLO‑World CNN‑семейство) корректно переносится.