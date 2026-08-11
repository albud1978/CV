"""Временная консистентность псевдоразметки: треки, голосование, добор пропусков.

Главный множитель качества и объёма разметки на СТАТИЧНЫХ камерах. Учитель
(SAM 3) работает покадрово и ничего не знает о времени, поэтому:

    * мигание — маска появилась в одном кадре и исчезла — почти всегда ложная
      сработка. Трек длиной 1 отбрасывается (`vote`);
    * пропуск — юнит стоит на месте, но в одном кадре не распознан. Полигон
      соседнего кадра переносится в пропущенный (`fill_gaps`) — бесплатные метки.

Обе операции применяются ТОЛЬКО к статичным классам (`stationary: true` в
онтологии). Люди действительно появляются на один кадр, их резать нельзя.
"""

from __future__ import annotations

from typing import Any

from src.pipeline import geometry as g
from src.pipeline.ontology import Ontology


def _match(track_points, cand_points, iou_threshold: float) -> float:
    """IoU между последней маской трека и кандидатом (0 — не связывать)."""
    iou = g.polygon_iou(track_points, cand_points)
    return iou if iou >= iou_threshold else 0.0


def link_frames(frames: list[dict[str, Any]], onto: Ontology) -> list[dict[str, Any]]:
    """Присваивает ``track_id`` инстансам последовательности кадров одного видео.

    Кадры должны идти по возрастанию времени. Связывание жадное, покласcовое:
    каждый инстанс сопоставляется активному треку того же класса с максимальным
    IoU выше порога; иначе открывается новый трек.

    Args:
        frames: Кадры вида ``{"timestamp": float, "instances": [...]}`` одного видео.
        onto: Онтология (пороги секции ``link``).

    Returns:
        Те же кадры (модифицированные на месте) с полем ``track_id`` у инстансов.
    """
    cfg = onto.link
    iou_threshold = float(cfg.get("iou_threshold", 0.25))
    max_gap = int(cfg.get("max_gap_frames", 2))

    # active[label] = список треков {id, points, last_index, length}
    active: dict[str, list[dict]] = {}
    next_id = 1

    for idx, frame in enumerate(frames):
        by_label: dict[str, list[dict]] = {}
        for inst in frame.get("instances", []):
            by_label.setdefault(inst["label"], []).append(inst)

        for label, group in by_label.items():
            tracks = [t for t in active.get(label, []) if idx - t["last_index"] <= max_gap]
            used: set[int] = set()
            for inst in group:
                pts = g.as_points(inst["points"])
                best_track, best_iou = None, 0.0
                for t in tracks:
                    if id(t) in used:
                        continue
                    iou = _match(t["points"], pts, iou_threshold)
                    if iou > best_iou:
                        best_track, best_iou = t, iou
                if best_track is None:
                    best_track = {"id": next_id, "length": 0}
                    next_id += 1
                    tracks.append(best_track)
                used.add(id(best_track))
                best_track["points"] = pts
                best_track["last_index"] = idx
                best_track["length"] += 1
                inst["track_id"] = best_track["id"]
            active[label] = tracks

    return frames


def track_lengths(frames: list[dict[str, Any]]) -> dict[int, int]:
    """Длина каждого трека в кадрах."""
    lengths: dict[int, int] = {}
    for frame in frames:
        for inst in frame.get("instances", []):
            tid = inst.get("track_id")
            if tid is not None:
                lengths[tid] = lengths.get(tid, 0) + 1
    return lengths


def vote(
    frames: list[dict[str, Any]], onto: Ontology
) -> tuple[list[dict[str, Any]], list[dict]]:
    """Отбрасывает короткие треки статичных классов (мигание = ложная сработка).

    Returns:
        Кортеж (кадры без коротких треков, отброшенные инстансы с причиной).
    """
    cfg = onto.link
    min_len = int(cfg.get("min_track_len", 2))
    stationary_only = bool(cfg.get("apply_to_stationary_only", True))
    stationary = onto.stationary_labels()

    lengths = track_lengths(frames)
    dropped: list[dict] = []
    for frame in frames:
        survivors = []
        for inst in frame.get("instances", []):
            if stationary_only and inst["label"] not in stationary:
                survivors.append(inst)
                continue
            tid = inst.get("track_id")
            if tid is not None and lengths.get(tid, 0) < min_len:
                rec = dict(inst)
                rec["reject_reason"] = "short_track"
                rec["reject_detail"] = f"len={lengths.get(tid, 0)}<{min_len}"
                rec["image_name"] = frame.get("image_name")
                dropped.append(rec)
                continue
            survivors.append(inst)
        frame["instances"] = survivors
    return frames, dropped


def fill_gaps(frames: list[dict[str, Any]], onto: Ontology) -> int:
    """Переносит маску статичного объекта в кадры, где учитель её пропустил.

    Работает только для статичных классов и только внутри уже связанного трека
    (между двумя реальными наблюдениями), поэтому не «выдумывает» объект там,
    где его не было ни до, ни после. Добавленные инстансы помечаются
    ``synthetic=True`` — их можно исключить из golden-set и из метрик учителя.

    Returns:
        Количество добавленных инстансов.
    """
    cfg = onto.link
    if not bool(cfg.get("fill_gaps", False)):
        return 0
    max_gap = int(cfg.get("max_gap_frames", 2))
    stationary = onto.stationary_labels()

    # Для каждого трека: индексы кадров, где он наблюдался.
    seen: dict[int, list[int]] = {}
    polygons: dict[tuple[int, int], dict] = {}
    for idx, frame in enumerate(frames):
        for inst in frame.get("instances", []):
            tid = inst.get("track_id")
            if tid is None or inst["label"] not in stationary:
                continue
            seen.setdefault(tid, []).append(idx)
            polygons[(tid, idx)] = inst

    added = 0
    for tid, indices in seen.items():
        for prev_idx, next_idx in zip(indices, indices[1:]):
            gap = next_idx - prev_idx - 1
            if gap <= 0 or gap > max_gap:
                continue
            donor = polygons[(tid, prev_idx)]
            for missing in range(prev_idx + 1, next_idx):
                clone = {
                    "label": donor["label"],
                    "points": list(donor["points"]),
                    "score": float(donor.get("score", 1.0)) * 0.9,
                    "track_id": tid,
                    "synthetic": True,
                }
                frames[missing].setdefault("instances", []).append(clone)
                added += 1
    return added


def apply(frames: list[dict[str, Any]], onto: Ontology) -> tuple[list[dict], dict[str, int]]:
    """Полный проход временной консистентности: link -> vote -> fill.

    Args:
        frames: Кадры ОДНОГО видео по возрастанию времени.
        onto: Онтология.

    Returns:
        Кортеж (отброшенные инстансы, сводка ``{tracks, dropped, filled}``).
    """
    if not bool(onto.link.get("enabled", True)):
        return [], {"tracks": 0, "dropped": 0, "filled": 0}
    link_frames(frames, onto)
    total_tracks = len(track_lengths(frames))
    _, dropped = vote(frames, onto)
    filled = fill_gaps(frames, onto)
    return dropped, {"tracks": total_tracks, "dropped": len(dropped), "filled": filled}
