"""СЛОЙ 3 — COCO -> датасет Ultralytics YOLO (detect или segment) без утечки split.

Одна и та же разметка отдаёт обе ветки:
    * ``--task detect``  — bbox выводится из маски. Быстрый первый прогон
      («сначала детекция»), не требует отдельной разметки.
    * ``--task segment`` — полигоны как есть. Нужен для геометрии подключения
      (концы рукава, толщина) и для продовой логики.

Split берётся ИЗ КАДРА (поле `split`, проставленное `build_dataset.py` по видео),
а не случайным перемешиванием: кадры одного видео скоррелированы, случайное
разбиение даёт завышенные метрики и модель, которая не работает на новой съёмке.

Кадры без объектов сохраняются с ПУСТЫМ файлом меток — это осознанные негативы
(пустой перрон), они снижают ложные срабатывания.

Использование:
    python -m src.pipeline.to_yolo --coco data/labels_v1/annotations_coco.json \\
        --frames data/dataset_heater --out data/yolo_heater --task segment
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from src.pipeline.manifest import find_frames
from src.pipeline.ontology import Ontology

SPLITS = ("train", "val", "test")


def _norm_polygon(points: list[list[int]], width: int, height: int) -> list[float]:
    """Нормирует полигон в координаты [0, 1] с обрезкой по границам кадра."""
    out: list[float] = []
    for x, y in points:
        out.append(min(max(float(x) / width, 0.0), 1.0))
        out.append(min(max(float(y) / height, 0.0), 1.0))
    return out


def _bbox_line(bbox: list[float], width: int, height: int) -> list[float]:
    """COCO bbox (x, y, w, h) -> YOLO (xc, yc, w, h), нормированный."""
    x, y, w, h = bbox
    return [
        min(max((x + w / 2) / width, 0.0), 1.0),
        min(max((y + h / 2) / height, 0.0), 1.0),
        min(max(w / width, 0.0), 1.0),
        min(max(h / height, 0.0), 1.0),
    ]


def convert(
    coco_path: str | Path,
    frames_root: str | Path,
    out_root: str | Path,
    onto: Ontology,
    task: str = "segment",
    copy_images: bool = False,
    default_split: str = "train",
) -> dict[str, Any]:
    """Раскладывает YOLO-датасет и пишет data.yaml.

    Args:
        coco_path: Файл COCO из `fuse.py`.
        frames_root: Корень с исходными кадрами (для поиска файлов по имени).
        out_root: Куда положить датасет.
        onto: Онтология (порядок классов = индексы YOLO).
        task: ``detect`` или ``segment``.
        copy_images: Копировать кадры вместо символических ссылок.
        default_split: Куда класть кадры без проставленного split.

    Returns:
        Сводка ``{split: {images, instances}}`` плюс путь к data.yaml.
    """
    coco = json.loads(Path(coco_path).read_text(encoding="utf-8"))
    labels = onto.trainable_labels()
    index = {label: i for i, label in enumerate(labels)}
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}

    # Поиск файла кадра по имени: раскладка `<root>/<split>/<video>/name.jpg`.
    by_name = {p.name: p for p in find_frames(frames_root)}

    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    out = Path(out_root)
    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {s: {"images": 0, "instances": 0} for s in SPLITS}
    missing = 0

    for img in coco["images"]:
        split = img.get("split") or default_split
        if split not in SPLITS:
            split = default_split
        src = by_name.get(img["file_name"])
        if src is None:
            missing += 1
            continue

        dst = out / "images" / split / img["file_name"]
        if not dst.exists():
            if copy_images:
                shutil.copy2(src, dst)
            else:
                # Относительная ссылка: датасет остаётся переносимым внутри диска.
                os.symlink(os.path.relpath(src.resolve(), dst.parent), dst)

        lines: list[str] = []
        for ann in anns_by_image.get(img["id"], []):
            name = cat_name.get(ann["category_id"])
            if name not in index:
                continue
            cls = index[name]
            if task == "segment" and ann.get("segmentation"):
                flat = ann["segmentation"][0]
                pts = [[flat[i], flat[i + 1]] for i in range(0, len(flat) - 1, 2)]
                coords = _norm_polygon(pts, img["width"], img["height"])
            else:
                coords = _bbox_line(ann["bbox"], img["width"], img["height"])
            lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in coords))

        (out / "labels" / split / f"{Path(img['file_name']).stem}.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        stats[split]["images"] += 1
        stats[split]["instances"] += len(lines)

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        "# Сгенерировано src/pipeline/to_yolo.py — не править вручную\n"
        f"path: {out.resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(labels)}\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(labels)),
        encoding="utf-8",
    )

    return {"task": task, "splits": stats, "missing_frames": missing, "data_yaml": str(data_yaml)}


def main() -> None:
    """CLI конвертации COCO -> YOLO."""
    ap = argparse.ArgumentParser(description="СЛОЙ 3 — COCO -> YOLO датасет")
    ap.add_argument("--coco", required=True)
    ap.add_argument("--frames", required=True, help="Корень кадров")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ontology", default="configs/ontology.gse_heater.yaml")
    ap.add_argument("--task", choices=["detect", "segment"], default="segment")
    ap.add_argument("--copy", action="store_true", help="Копировать кадры вместо симлинков")
    args = ap.parse_args()

    summary = convert(args.coco, args.frames, args.out, Ontology.load(args.ontology),
                      args.task, args.copy)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["missing_frames"]:
        print(f"ВНИМАНИЕ: не найдено кадров: {summary['missing_frames']}")
    empty = [s for s, v in summary["splits"].items() if v["images"] == 0]
    if empty:
        print(f"ВНИМАНИЕ: пустые split: {empty} — проверьте configs/dataset_split.yaml")


if __name__ == "__main__":
    main()
