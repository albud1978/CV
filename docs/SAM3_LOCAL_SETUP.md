# Локальная установка SAM 3 / SAM 3.1 (RTX 5090)

Инструкция по развёртыванию SAM 3.1 как локального «учителя» авто-разметки на домашней
машине с RTX 5090 (32 ГБ VRAM). Альтернатива облачному Roboflow Auto Label: данные не уходят
в облако, нет лимитов хостинга.

> Статус (июнь 2026): SAM 3 (Meta, 19.11.2025) и улучшённый **SAM 3.1** (03.2026) доступны.
> Веса gated на HuggingFace, но `autodistill-sam3` тянет их через `ROBOFLOW_API_KEY`.

## 1. Требования

| Компонент | Значение |
|---|---|
| GPU | RTX 5090, 32 ГБ VRAM (SAM 3 нужно ~16 ГБ; 5090 с запасом) |
| CUDA | 12.4+ (для Blackwell — драйвер ≥ 560, лучше CUDA 12.8) |
| Python | 3.10–3.11 |
| Диск | ~5 ГБ под веса SAM 3.1 |
| Ключ | `ROBOFLOW_API_KEY` (из app.roboflow.com/settings/api) |

## 2. Установка

```bash
# Желательно отдельное окружение
python -m venv .venv && source .venv/bin/activate

# PyTorch под Blackwell (CUDA 12.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Autodistill + SAM 3 + вспомогательные
pip install autodistill autodistill-sam3 supervision

# Ключ для скачивания весов SAM 3.1 (в обход HF-гейта)
export ROBOFLOW_API_KEY="ВАШ_КЛЮЧ"
```

## 3. Авто-разметка папки кадров (boxes + masks)

```python
from autodistill_sam3 import SegmentAnything3
from autodistill.detection import CaptionOntology

# ключ = текстовый промпт для SAM 3 (короткие noun-phrases!),
# значение = имя класса в итоговом датасете
ontology = CaptionOntology({
    "ground air conditioning unit": "asu",
    "aircraft ground heater": "heater",
    "ground crew person": "crew",
    "flexible hose": "hose",
})

base_model = SegmentAnything3(ontology=ontology)

# Разметить всю папку -> создаётся датасет (YOLO/COCO) с боксами и масками
base_model.label(
    input_folder="./frames",
    extension=".jpg",
    output_folder="./dataset_sam3",
)
```

Проверка на одном изображении:

```python
import supervision as sv
from autodistill.helpers import load_image

detections = base_model.predict("frame.jpg")
image = load_image("frame.jpg", return_format="cv2")
ann = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX).annotate(image.copy(), detections)
ann = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX).annotate(
    ann, detections, labels=[base_model.ontology.classes()[i] for i in detections.class_id]
)
sv.plot_image(ann)
```

## 4. Дистилляция в студента (RF-DETR / YOLO)

```python
from autodistill_rfdetr import RFDETRMedium   # pip install autodistill-rfdetr
# или: from autodistill_yolov8 import YOLOv8

target = RFDETRMedium()
target.train("./dataset_sam3/data.yaml", epochs=150, imgsz=1280)
```

Стандарты обучения — `configs/augment_conservative.yaml` и `.cursor/rules/cv-pipeline.mdc`
(imgsz=1280, close_mosaic=10, mixup=0, split 70/20/10 stratified by video_id).

## 5. Подсказки по промптам SAM 3 (PCS)

- Использовать **короткие noun-phrases без артиклей**: «ground air conditioning unit», а не
  «large yellow unit with a hose...» (длинные описания работают хуже).
- Класс person квалифицировать: «ground crew person in high-visibility vest» — иначе ловит
  пассажиров на телетрапе.
- Per-class confidence: редкие/мелкие — 0.20, частые — 0.40.
- Для мелких/удалённых объектов на 4K — связка с SAHI на инференсе.

## 6. Альтернатива через transformers (без autodistill)

```python
from transformers import Sam3Processor, Sam3Model   # transformers ≥ 4.57 + доступ к facebook/sam3.1
processor = Sam3Processor.from_pretrained("facebook/sam3.1")
model = Sam3Model.from_pretrained("facebook/sam3.1")
# требует HF-аутентификации (hf auth login) и принятия gated-лицензии
```

## 7. Лицензия (важно)

SAM License (Meta): коммерческое использование разрешено с ограничениями (нет military/ITAR;
при редистрибуции весов SAM они остаются под SAM License). В нашем сценарии это безопасно:
SAM 3 используется только как «учитель», в продукт идёт дистиллированный RF-DETR (Apache 2.0),
сами веса SAM не распространяются.

## 8. Типичные проблемы

- **OOM на больших кадрах**: уменьшить размер входа / FP16 / батч=1.
- **Веса не качаются**: проверить `ROBOFLOW_API_KEY` в env; для transformers-пути — `hf auth login`
  и принять лицензию на странице `facebook/sam3.1`.
- **Blackwell (5090)**: ставить torch именно под cu128, иначе CUDA-ядра не соберутся.
