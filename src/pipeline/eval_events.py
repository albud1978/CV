"""Оценка СОБЫТИЙ, а не только mAP: попали ли мы в тайм-коды подключения.

Заказчику не нужен mAP — ему нужны верные времена «подключили»/«отключили».
Модель с mAP 0.9 и дрожащим состоянием хуже модели с mAP 0.8 и устойчивым.
Поэтому приёмка считается здесь.

Разметка правды берётся из `configs/dataset_split.yaml`: заказчик уже задал по
каждому видео `positive_window_sec` — интервалы присутствия обогревателя. Это
готовый эталон уровня события, ручной разметки для него не требуется.

Метрики на видео:
    temporal_iou      — пересечение/объединение предсказанных и эталонных интервалов;
    boundary_error    — ошибка тайм-кода каждого перехода (секунды);
    false_events      — предсказанные переходы, которым нет пары в эталоне;
    missed_events     — эталонные переходы, которых модель не нашла.

Использование:
    python -m src.pipeline.eval_events --events data/events \\
        --split-config configs/dataset_split.yaml --out data/events/report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import yaml

from src.pipeline.events import hms
from src.pipeline.ontology import Ontology

Interval = tuple[float, float]


def normalize_windows(windows: list, duration: float) -> list[Interval]:
    """Приводит `positive_window_sec` к списку интервалов, подставляя длительность вместо null."""
    out: list[Interval] = []
    for w in windows or []:
        start = float(w[0] or 0.0)
        end = float(w[1]) if w[1] is not None else duration
        if end > start:
            out.append((start, min(end, duration) if duration else end))
    return out


def intervals_overlap(a: list[Interval], b: list[Interval]) -> float:
    """Суммарная длительность пересечения двух наборов интервалов."""
    total = 0.0
    for a_start, a_end in a:
        for b_start, b_end in b:
            total += max(0.0, min(a_end, b_end) - max(a_start, b_start))
    return total


def intervals_duration(intervals: list[Interval]) -> float:
    """Суммарная длительность интервалов."""
    return sum(max(0.0, end - start) for start, end in intervals)


def temporal_iou(pred: list[Interval], gt: list[Interval]) -> float:
    """IoU по времени: насколько предсказанное состояние совпадает с эталонным."""
    inter = intervals_overlap(pred, gt)
    union = intervals_duration(pred) + intervals_duration(gt) - inter
    return round(inter / union, 4) if union > 0 else 0.0


def boundaries(intervals: list[Interval], duration: float) -> list[tuple[str, float]]:
    """Переходы, соответствующие набору интервалов.

    Границы, совпадающие с началом/концом записи, событиями НЕ считаются: перехода
    мы не видели (видео 01 начинается с уже подключённым рукавом).
    """
    out: list[tuple[str, float]] = []
    for start, end in intervals:
        if start > 1e-6:
            out.append(("hose_connected", start))
        if duration and end < duration - 1e-6:
            out.append(("hose_disconnected", end))
    return out


def match_boundaries(
    predicted: list[tuple[str, float]],
    expected: list[tuple[str, float]],
    tolerance: float,
) -> dict[str, Any]:
    """Сопоставляет предсказанные переходы эталонным (жадно, по близости времени)."""
    remaining = list(predicted)
    matched: list[dict[str, Any]] = []
    missed: list[dict[str, Any]] = []

    for name, ts in expected:
        candidates = [(abs(p_ts - ts), i) for i, (p_name, p_ts) in enumerate(remaining)
                      if p_name == name]
        if not candidates:
            missed.append({"event": name, "expected": round(ts, 1), "timecode": hms(ts)})
            continue
        error, idx = min(candidates)
        _, p_ts = remaining.pop(idx)
        matched.append({
            "event": name,
            "expected": round(ts, 1),
            "predicted": round(p_ts, 1),
            "error_sec": round(error, 1),
            "within_tolerance": bool(error <= tolerance),
        })

    return {
        "matched": matched,
        "missed": missed,
        "false_events": [{"event": n, "predicted": round(t, 1), "timecode": hms(t)}
                         for n, t in remaining],
    }


def evaluate_video(
    report: dict[str, Any], windows: list, onto: Ontology, duration: Optional[float] = None
) -> dict[str, Any]:
    """Считает событийные метрики по одному видео."""
    duration = float(duration or report.get("duration_sec") or 0.0)
    gt = normalize_windows(windows, duration)
    pred = [(float(e["start"]), float(e["end"])) for e in report.get("episodes", [])]
    tolerance = float(onto.acceptance.get("event_onset_tolerance_sec", 120.0))

    match = match_boundaries(
        [(e["name"], float(e["timestamp"])) for e in report.get("events", [])],
        boundaries(gt, duration),
        tolerance,
    )
    max_false = int(onto.acceptance.get("max_false_events_per_video", 1))
    passed = (
        not match["missed"]
        and len(match["false_events"]) <= max_false
        and all(m["within_tolerance"] for m in match["matched"])
    )

    return {
        "video_id": report.get("video_id"),
        "duration_sec": round(duration, 1),
        "temporal_iou": temporal_iou(pred, gt),
        "gt_connected_sec": round(intervals_duration(gt), 1),
        "pred_connected_sec": round(intervals_duration(pred), 1),
        **match,
        "passed": passed,
    }


def main() -> None:
    """CLI оценки событий по всем прогнанным видео."""
    ap = argparse.ArgumentParser(description="Событийные метрики против окон присутствия")
    ap.add_argument("--events", required=True, help="Папка с <видео>/events.json")
    ap.add_argument("--split-config", default="configs/dataset_split.yaml")
    ap.add_argument("--ontology", default="configs/ontology.gse_heater.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    onto = Ontology.load(args.ontology)
    cfg = yaml.safe_load(Path(args.split_config).read_text(encoding="utf-8"))
    windows_by_stem = {Path(v["file"]).stem: v for v in cfg.get("videos", [])}

    results = []
    for path in sorted(Path(args.events).glob("*/events.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        stem = report.get("video_id") or path.parent.name
        entry = windows_by_stem.get(stem)
        if entry is None:
            print(f"ПРОПУСК: нет эталонных окон для {stem}")
            continue
        res = evaluate_video(report, entry.get("positive_window_sec", []), onto)
        res["split"] = entry.get("split")
        res["condition"] = entry.get("condition")
        results.append(res)

    if not results:
        raise SystemExit("Не найдено ни одного events.json с эталоном")

    summary = {
        "videos": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "mean_temporal_iou": round(sum(r["temporal_iou"] for r in results) / len(results), 4),
        "total_false_events": sum(len(r["false_events"]) for r in results),
        "total_missed_events": sum(len(r["missed"]) for r in results),
        "per_video": results,
    }
    out = Path(args.out or Path(args.events) / "event_metrics.json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nОтчёт: {out}")


if __name__ == "__main__":
    main()
