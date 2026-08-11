#!/usr/bin/env bash
# Конвейер heater одной командой: слова -> маски -> датасет -> модель.
#
# Использование:
#   ./scripts/reboot_heater.sh                      # полный прогон (нужен GPU и веса учителя)
#   BACKEND=stub ./scripts/reboot_heater.sh         # сухой прогон конвейера без весов
#   TASK=detect ./scripts/reboot_heater.sh          # быстрая detect-ветка вместо segment
#   SKIP_TRAIN=1 ./scripts/reboot_heater.sh         # только разметка и датасет
#
# Приёмка по событиям (шаг 6) запускается отдельно — ей нужны прогоны по видео:
#   python -m src.pipeline.run_video --weights ... --video ... --out data/events/<id>
#   python -m src.pipeline.eval_events --events data/events

set -euo pipefail

FRAMES="${FRAMES:-data/dataset_heater}"
LABELS="${LABELS:-data/labels_v1}"
YOLO_DIR="${YOLO_DIR:-data/yolo_heater}"
RUNS="${RUNS:-runs/heater_v1}"
ONTOLOGY="${ONTOLOGY:-configs/ontology.gse_heater.yaml}"
BACKEND="${BACKEND:-local}"
TASK="${TASK:-segment}"
ARCH="${ARCH:-}"
QA_LIMIT="${QA_LIMIT:-60}"

echo "=== Конвейер heater ==="
echo "кадры:     $FRAMES"
echo "онтология: $ONTOLOGY"
echo "учитель:   $BACKEND | задача: $TASK"
echo

if [ ! -d "$FRAMES" ]; then
  echo "Нет папки кадров '$FRAMES'. Сначала нарежьте датасет:"
  echo "  python -m src.pipeline.build_dataset --config configs/dataset_split.yaml \\"
  echo "      --output $FRAMES --repair"
  exit 1
fi

echo "--- 1/4 учитель: слова -> маски ---"
python -m src.pipeline.teacher \
  --frames "$FRAMES" --ontology "$ONTOLOGY" \
  --out "$LABELS/passes" --backend "$BACKEND"

echo
echo "--- 2/4 слияние + авто-QA + треки ---"
python -m src.pipeline.fuse \
  --passes-dir "$LABELS/passes" --ontology "$ONTOLOGY" \
  --out "$LABELS" --frames-dir "$FRAMES" --qa-limit "$QA_LIMIT"

echo
echo "--- 3/4 YOLO-датасет ($TASK) ---"
python -m src.pipeline.to_yolo \
  --coco "$LABELS/annotations_coco.json" --frames "$FRAMES" \
  --ontology "$ONTOLOGY" --out "$YOLO_DIR" --task "$TASK"

if [ "${SKIP_TRAIN:-0}" = "1" ]; then
  echo
  echo "Обучение пропущено (SKIP_TRAIN=1). Датасет готов: $YOLO_DIR/data.yaml"
  exit 0
fi

echo
echo "--- 4/4 обучение студента ---"
TRAIN_ARGS=(--data "$YOLO_DIR/data.yaml" --ontology "$ONTOLOGY" --task "$TASK" --out "$RUNS")
[ -n "$ARCH" ] && TRAIN_ARGS+=(--arch "$ARCH")
python -m src.pipeline.train "${TRAIN_ARGS[@]}"

echo
echo "Готово."
echo "  разметка:  $LABELS/annotations_coco.json"
echo "  отбраковка: $LABELS/rejections.jsonl (проверьте, что срезали)"
echo "  веса:      $RUNS/train/weights/best.pt"
echo "  карточка:  $RUNS/train/model_card.md"
echo
echo "Дальше — экспорт на телефон и события:"
echo "  python -m src.pipeline.export_edge --weights $RUNS/train/weights/best.pt \\"
echo "      --formats onnx ncnn tflite --imgsz 640 --data $YOLO_DIR/data.yaml --int8"
