"""СЛОЙ 2b — офлайн-слияние проходов учителя в чистый датасет.

Здесь одноклассовые проходы превращаются в согласованную разметку. Всё, что
учитель не умеет (развести персонал и пассажиров, отличить рукав от кабеля,
вспомнить, что было в соседнем кадре), делается тут проверяемыми правилами:

    1. склейка проходов по кадрам;
    2. person -> staff/passenger по доле hi-vis пикселей в маске;
    3. авто-QA (`autoqa.py`) — отбраковка с указанием причины;
    4. временная консистентность (`link.py`) — внутри каждого видео;
    5. запись COCO instance-segmentation + rejections.jsonl + QA-картинки.

Использование:
    python -m src.pipeline.fuse \\
        --passes-dir data/labels_v1/passes \\
        --ontology configs/ontology.gse_heater.yaml \\
        --out data/labels_v1 --qa-limit 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.pipeline import autoqa, link
from src.pipeline.manifest import group_by_video
from src.pipeline.ontology import Ontology

# Цвета классов для QA-отрисовки (BGR).
QA_COLORS = {
    "unit": (0, 0, 255),
    "wide_hose": (255, 80, 0),
    "cable": (200, 0, 200),
    "aircraft": (120, 120, 120),
    "staff": (0, 220, 0),
    "passenger": (0, 140, 255),
}


def load_passes(passes_dir: str | Path) -> list[dict[str, Any]]:
    """Читает per-pass JSON и склеивает их в список кадров с общими инстансами.

    Поддерживает как формат `teacher.py` (с метаданными и score), так и старый
    формат `autolabel_rf.py fetch` (только image_name/width/height/masks).
    """
    by_name: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(passes_dir).glob("*.json")):
        if path.name.startswith("_"):
            continue
        for rec in json.loads(path.read_text(encoding="utf-8")):
            name = rec["image_name"]
            frame = by_name.setdefault(name, {
                "image_name": name,
                "image_path": rec.get("image_path"),
                "width": rec.get("width"),
                "height": rec.get("height"),
                "video_id": rec.get("video_id", "unknown"),
                "split": rec.get("split", "unassigned"),
                "timestamp": float(rec.get("timestamp", 0.0) or 0.0),
                "instances": [],
            })
            # Метаданные могут быть только в одном из проходов — добираем.
            for key in ("image_path", "width", "height"):
                if not frame.get(key) and rec.get(key):
                    frame[key] = rec[key]
            for m in rec.get("masks", []):
                frame["instances"].append({
                    "label": m["label"],
                    "points": m["points"],
                    "score": float(m.get("score", 1.0)),
                    "pass": path.stem,
                })
    return list(by_name.values())


def split_person(frames: list[dict[str, Any]], onto: Ontology, frames_dir: str | Path | None) -> int:
    """Делит инстансы `person` на staff/passenger по цвету жилета.

    SAM 3 не различает наземный персонал и пассажиров (промпты со «vest» дают 0),
    зато светоотражающий жилет отлично виден в HSV. Кадр читается с диска — если
    файла нет, инстанс помечается `staff` (консервативно: персонал важнее для
    сцен обслуживания и не портит событие подключения).

    Returns:
        Количество разведённых инстансов.
    """
    ranges = [(np.array(lo, np.uint8), np.array(hi, np.uint8)) for lo, hi in onto.vest_ranges()]
    threshold = float(onto.person_split.get("vest_fraction_threshold", 0.06))
    changed = 0
    for frame in frames:
        persons = [i for i in frame["instances"] if i["label"] == "person"]
        if not persons:
            continue
        path = frame.get("image_path")
        if not path and frames_dir:
            path = str(Path(frames_dir) / frame["image_name"])
        image = cv2.imread(str(path)) if path else None
        for inst in persons:
            if image is None:
                inst["label"] = "staff"
            else:
                frac = autoqa.vest_fraction(image, inst["points"], ranges)
                inst["label"] = "staff" if frac >= threshold else "passenger"
                inst["vest_fraction"] = round(frac, 4)
            changed += 1
    return changed


def to_coco(frames: list[dict[str, Any]], onto: Ontology) -> dict[str, Any]:
    """Собирает COCO instance-segmentation из отфильтрованных кадров."""
    labels = onto.trainable_labels()
    cat_ids = {label: i + 1 for i, label in enumerate(labels)}   # COCO нумерует с 1
    coco: dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": [{"id": cid, "name": name} for name, cid in cat_ids.items()],
    }
    ann_id = 1
    for img_id, frame in enumerate(sorted(frames, key=lambda f: f["image_name"]), 1):
        coco["images"].append({
            "id": img_id,
            "file_name": frame["image_name"],
            "width": int(frame["width"]),
            "height": int(frame["height"]),
            "video_id": frame.get("video_id"),
            "split": frame.get("split"),
            "timestamp": frame.get("timestamp"),
        })
        for inst in frame.get("instances", []):
            if inst["label"] not in cat_ids:
                continue
            pts = np.asarray(inst["points"], np.int32).reshape(-1, 2)
            x, y, w, h = cv2.boundingRect(pts)
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_ids[inst["label"]],
                "segmentation": [pts.flatten().tolist()],
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": float(cv2.contourArea(pts)),
                "iscrowd": 0,
                "score": float(inst.get("score", 1.0)),
                "synthetic": bool(inst.get("synthetic", False)),
            })
            ann_id += 1
    return coco


def render_qa(frame: dict[str, Any], out_path: Path, frames_dir: str | Path | None) -> bool:
    """Рисует принятые маски поверх кадра (визуальная проверка выборкой)."""
    path = frame.get("image_path")
    if not path and frames_dir:
        path = str(Path(frames_dir) / frame["image_name"])
    image = cv2.imread(str(path)) if path else None
    if image is None:
        return False
    overlay = image.copy()
    for inst in frame.get("instances", []):
        pts = np.asarray(inst["points"], np.int32).reshape(-1, 2)
        color = QA_COLORS.get(inst["label"], (200, 200, 200))
        cv2.fillPoly(overlay, [pts], color)
        x, y, _, _ = cv2.boundingRect(pts)
        cv2.putText(image, inst["label"], (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", cv2.addWeighted(overlay, 0.4, image, 0.6, 0))
    if not ok:
        return False
    out_path.write_bytes(buf.tobytes())   # imwrite молча падает на drvfs/юникоде
    return True


def fuse(
    passes_dir: str | Path,
    onto: Ontology,
    out_dir: str | Path,
    frames_dir: str | Path | None = None,
    qa_limit: int = 0,
) -> dict[str, Any]:
    """Полный офлайн-этап: склейка -> person split -> авто-QA -> треки -> COCO.

    Returns:
        Сводка по классам, отбраковке и трекам.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames = load_passes(passes_dir)
    if not frames:
        raise SystemExit(f"Проходы не найдены в {passes_dir}")
    raw_total = sum(len(f["instances"]) for f in frames)

    split_person(frames, onto, frames_dir)

    rejections: list[dict] = []
    for frame in frames:
        kept, rejected = autoqa.filter_frame(frame, onto)
        frame["instances"] = kept
        for r in rejected:
            r["image_name"] = frame["image_name"]
        rejections.extend(rejected)

    link_stats: dict[str, int] = {"tracks": 0, "dropped": 0, "filled": 0}
    for _, group in group_by_video(frames).items():
        dropped, stats = link.apply(group, onto)
        rejections.extend(dropped)
        for key in link_stats:
            link_stats[key] += stats.get(key, 0)

    coco = to_coco(frames, onto)
    (out / "annotations_coco.json").write_text(
        json.dumps(coco, ensure_ascii=False), encoding="utf-8"
    )
    with (out / "rejections.jsonl").open("w", encoding="utf-8") as f:
        for r in rejections:
            r.pop("points", None)          # полигоны не нужны — важна причина и кадр
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    rendered = 0
    if qa_limit:
        for frame in frames[:qa_limit]:
            rendered += int(render_qa(frame, out / "qa" / frame["image_name"], frames_dir))

    per_class: dict[str, int] = {}
    for ann in coco["annotations"]:
        name = coco["categories"][ann["category_id"] - 1]["name"]
        per_class[name] = per_class.get(name, 0) + 1
    per_split: dict[str, int] = {}
    for img in coco["images"]:
        per_split[img.get("split", "unassigned")] = per_split.get(img.get("split", "unassigned"), 0) + 1

    summary = {
        "frames": len(frames),
        "instances_raw": raw_total,
        "instances_kept": len(coco["annotations"]),
        "kept_ratio": round(len(coco["annotations"]) / max(raw_total, 1), 3),
        "per_class": per_class,
        "frames_per_split": per_split,
        "link": link_stats,
        "rejections_by_reason": autoqa.summarize(rejections),
        "qa_rendered": rendered,
    }
    (out / "fuse_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    """CLI слияния проходов."""
    ap = argparse.ArgumentParser(description="СЛОЙ 2b — слияние проходов учителя")
    ap.add_argument("--passes-dir", required=True)
    ap.add_argument("--ontology", default="configs/ontology.gse_heater.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames-dir", default=None,
                    help="Папка кадров, если в проходах нет image_path")
    ap.add_argument("--qa-limit", type=int, default=0, help="Сколько QA-картинок отрисовать")
    args = ap.parse_args()

    summary = fuse(args.passes_dir, Ontology.load(args.ontology), args.out,
                   args.frames_dir, args.qa_limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nCOCO: {args.out}/annotations_coco.json")
    print(f"Отбраковка с причинами: {args.out}/rejections.jsonl")


if __name__ == "__main__":
    main()
