"""Продовый прогон: видео -> детекции -> состояние подключения -> тайм-коды событий.

Замыкает конвейер: обученный локальный YOLO-seg + те же правила, что чистили
разметку (`autoqa`), + геометрия связи (`relation`) + антидребезг (`events`).
Один и тот же код фильтров на обучении и на инференсе — это осознанно: пороги
калибруются один раз и ведут себя одинаково в обеих точках.

Видео анализируется НЕ покадрово, а с шагом `--fps` (по умолчанию 1 кадр/с):
событие длится минуты, 25 FPS для него избыточны, а экономия на порядок.

Использование:
    python -m src.pipeline.run_video --weights runs/heater/train/weights/best.pt \\
        --video "input/.../01_Ст 1_камера 50 ....mkv" --out data/events/v01 \\
        --fps 1 --save-video
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from src.pipeline import autoqa, events as ev, relation
from src.pipeline.ontology import Ontology
from src.pipeline.teacher import mask_to_polygon

STATE_COLORS = {
    relation.STATE_CONNECTED: (0, 200, 0),
    relation.STATE_DISCONNECTED: (0, 0, 220),
    relation.STATE_UNKNOWN: (120, 120, 120),
}


def predict_frame(model: Any, image: np.ndarray, onto: Ontology, conf: float) -> list[dict]:
    """Прогоняет модель по кадру и возвращает инстансы в формате конвейера."""
    labels = onto.trainable_labels()
    result = model.predict(image, conf=conf, verbose=False)[0]
    instances: list[dict] = []

    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return instances

    classes = boxes.cls.tolist()
    scores = boxes.conf.tolist()
    for i, (cls, score) in enumerate(zip(classes, scores)):
        label = labels[int(cls)] if int(cls) < len(labels) else str(int(cls))
        if masks is not None and masks.xy is not None and i < len(masks.xy):
            points = np.asarray(masks.xy[i], np.int32).tolist()
            if len(points) < 3:
                continue
        else:  # detect-модель: рамка как полигон
            x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
            points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        instances.append({"label": label, "points": points, "score": float(score)})
    return instances


def draw_overlay(image: np.ndarray, instances: list[dict], verdict: relation.FrameVerdict) -> np.ndarray:
    """Рисует маски и плашку состояния (для визуальной приёмки заказчиком)."""
    from src.pipeline.fuse import QA_COLORS

    overlay = image.copy()
    for inst in instances:
        pts = np.asarray(inst["points"], np.int32).reshape(-1, 2)
        cv2.fillPoly(overlay, [pts], QA_COLORS.get(inst["label"], (200, 200, 200)))
    blended = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)

    color = STATE_COLORS.get(verdict.state, (200, 200, 200))
    text = f"{ev.hms(verdict.timestamp)}  {verdict.state.upper()}  p={verdict.score:.2f}  {verdict.reason}"
    cv2.rectangle(blended, (0, 0), (blended.shape[1], 44), (20, 20, 20), -1)
    cv2.putText(blended, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return blended


def process_video(
    model: Any,
    video_path: str,
    onto: Ontology,
    out_dir: str | Path,
    sample_fps: float = 1.0,
    conf: float = 0.25,
    save_video: bool = False,
    aircraft_zone: Optional[list] = None,
    max_seconds: Optional[float] = None,
) -> dict[str, Any]:
    """Гоняет модель по видео и собирает отчёт о событиях подключения.

    Returns:
        JSON-отчёт (`events.report`) с добавленной статистикой по кадрам.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Не удалось открыть видео: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(round(fps / max(sample_fps, 1e-6))), 1)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    writer = None

    verdicts: list[relation.FrameVerdict] = []
    per_frame: list[dict[str, Any]] = []
    idx, processed, corrupt = 0, 0, 0

    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step:
            idx += 1
            continue
        ok, frame = cap.retrieve()
        idx += 1
        if not ok or frame is None:
            corrupt += 1
            if corrupt > 30:      # битый GOP в хвосте — дальше смысла нет
                break
            continue
        corrupt = 0
        timestamp = (idx - 1) / fps
        if max_seconds and timestamp > max_seconds:
            break

        h, w = frame.shape[:2]
        instances = predict_frame(model, frame, onto, conf)
        kept, _ = autoqa.filter_frame(
            {"width": w, "height": h, "instances": instances}, onto
        )
        verdict = relation.analyze_frame(kept, w, h, onto, timestamp, aircraft_zone)
        verdicts.append(verdict)
        per_frame.append({
            "timestamp": round(timestamp, 2),
            "state": verdict.state,
            "score": round(verdict.score, 3),
            "reason": verdict.reason,
            "labels": sorted({i["label"] for i in kept}),
        })
        processed += 1

        if save_video:
            if writer is None:
                writer = cv2.VideoWriter(
                    str(out / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                    max(sample_fps, 1.0), (w, h),
                )
            writer.write(draw_overlay(frame, kept, verdict))

        if processed % 200 == 0:
            print(f"  {ev.hms(timestamp)} обработано {processed} кадров")

    cap.release()
    if writer is not None:
        writer.release()

    detected, episodes = ev.detect_events(verdicts, onto)
    report = ev.report(detected, episodes, video_id=Path(video_path).stem)
    report["frames_analyzed"] = processed
    report["sample_fps"] = sample_fps
    report["duration_sec"] = round(verdicts[-1].timestamp, 1) if verdicts else 0.0

    (out / "events.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "frame_states.jsonl").open("w", encoding="utf-8") as f:
        for row in per_frame:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return report


def main() -> None:
    """CLI продового прогона по видео."""
    ap = argparse.ArgumentParser(description="Видео -> события подключения обогревателя")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--ontology", default="configs/ontology.gse_heater.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=1.0, help="Частота анализа, кадр/с")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--save-video", action="store_true", help="Записать overlay.mp4")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--aircraft-zone", default=None,
                    help="JSON-полигон зоны стоянки ВС для этой камеры (резерв)")
    args = ap.parse_args()

    from ultralytics import YOLO   # тяжёлый импорт — только при реальном запуске

    zone = json.loads(Path(args.aircraft_zone).read_text(encoding="utf-8")) if args.aircraft_zone else None
    report = process_video(
        YOLO(args.weights), args.video, Ontology.load(args.ontology), args.out,
        args.fps, args.conf, args.save_video, zone, args.max_seconds,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nСобытия: {args.out}/events.json")


if __name__ == "__main__":
    main()
