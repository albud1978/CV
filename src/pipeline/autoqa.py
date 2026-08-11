"""Автоматический контроль качества псевдоразметки (замена сплошного ручного ревью).

Установка заказчика: «даже если придётся выкинуть множество некорректных разметок,
это не страшно». Отсюда стратегия — агрессивно резать по проверяемым признакам и
логировать ПРИЧИНУ каждого отброса, чтобы по `rejections.jsonl` было видно, не
срезали ли заодно полезное (и подкрутить порог, а не гадать).

Проверки идут от дешёвых к дорогим:
    1. вырожденный полигон;                4. толщина/вытянутость (шланг vs кабель);
    2. уверенность ниже порога класса;     5. NMS внутри класса;
    3. площадь вне разумных границ;        6. пространственные правила (шланг у юнита);
                                           7. лимит инстансов на кадр.

Все пороги — из секции `autoqa` онтологии, в px для опорной ширины кадра.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from src.pipeline import geometry as g
from src.pipeline.ontology import Ontology

# Причины отбраковки — фиксированный словарь, чтобы сводку можно было агрегировать.
REASON_DEGENERATE = "degenerate_polygon"
REASON_LOW_SCORE = "low_score"
REASON_TOO_SMALL = "area_too_small"
REASON_TOO_LARGE = "area_too_large"
REASON_THIN = "hose_too_thin"
REASON_THICK = "hose_too_thick"
REASON_NOT_ELONGATED = "hose_not_elongated"
REASON_CABLE_TOO_THICK = "cable_too_thick"
REASON_CABLE_IS_HOSE = "cable_overlaps_hose"
REASON_NMS = "nms_duplicate"
REASON_FAR_FROM_UNIT = "hose_far_from_unit"
REASON_OVER_LIMIT = "over_per_image_limit"


def _reject(rejected: list[dict], inst: dict, reason: str, detail: Optional[str] = None) -> None:
    """Складывает инстанс в отбраковку с указанием причины."""
    rec = dict(inst)
    rec["reject_reason"] = reason
    if detail:
        rec["reject_detail"] = detail
    rejected.append(rec)


def filter_frame(frame: dict[str, Any], onto: Ontology) -> tuple[list[dict], list[dict]]:
    """Применяет все правила авто-QA к инстансам одного кадра.

    Args:
        frame: Кадр вида ``{"width", "height", "instances": [{label, points, score}]}``.
        onto: Онтология задачи.

    Returns:
        Кортеж (принятые инстансы, отбракованные с полем ``reject_reason``).
    """
    cfg = onto.autoqa
    width = int(frame["width"])
    height = int(frame["height"])
    frame_area = float(max(width * height, 1))
    ref = float(cfg.get("reference_width", onto.reference_width))

    def px(value: float) -> float:
        return g.scale_px(value, width, ref)

    kept: list[dict] = []
    rejected: list[dict] = []

    # --- 1-4: покадровые проверки отдельного инстанса ---
    min_points = int(cfg.get("min_polygon_points", 3))
    area_frac = cfg.get("area_frac", {}) or {}
    hose_cfg = cfg.get("wide_hose", {}) or {}
    cable_cfg = cfg.get("cable", {}) or {}

    for inst in frame.get("instances", []):
        label = inst["label"]
        pts = g.as_points(inst["points"])
        if len(pts) < min_points or g.polygon_area(pts) <= 0:
            _reject(rejected, inst, REASON_DEGENERATE)
            continue

        score = float(inst.get("score", 1.0))
        if score < onto.confidence_of(label):
            _reject(rejected, inst, REASON_LOW_SCORE, f"score={score:.2f}")
            continue

        bounds = area_frac.get(label)
        if bounds:
            frac = g.polygon_area(pts) / frame_area
            if frac < float(bounds[0]):
                _reject(rejected, inst, REASON_TOO_SMALL, f"frac={frac:.6f}")
                continue
            if frac > float(bounds[1]):
                _reject(rejected, inst, REASON_TOO_LARGE, f"frac={frac:.6f}")
                continue

        if label == "wide_hose":
            thickness = g.mean_thickness(pts)
            if thickness < px(float(hose_cfg.get("min_thickness_px", 8))):
                _reject(rejected, inst, REASON_THIN, f"thickness={thickness:.1f}px")
                continue
            if thickness > px(float(hose_cfg.get("max_thickness_px", 140))):
                _reject(rejected, inst, REASON_THICK, f"thickness={thickness:.1f}px")
                continue
            elong = g.elongation(pts)
            if elong < float(hose_cfg.get("min_elongation", 2.0)):
                _reject(rejected, inst, REASON_NOT_ELONGATED, f"elongation={elong:.2f}")
                continue

        if label == "cable":
            thickness = g.mean_thickness(pts)
            if thickness > px(float(cable_cfg.get("max_thickness_px", 10))):
                _reject(rejected, inst, REASON_CABLE_TOO_THICK, f"thickness={thickness:.1f}px")
                continue

        enriched = dict(inst)
        enriched["points"] = pts.tolist()
        kept.append(enriched)

    # --- 5: NMS внутри класса ---
    nms_iou = float(cfg.get("nms_iou", 0.5))
    by_label: dict[str, list[dict]] = {}
    for inst in kept:
        by_label.setdefault(inst["label"], []).append(inst)
    kept = []
    for label, group in by_label.items():
        survivors, dropped = g.nms_polygons(group, iou_threshold=nms_iou)
        kept.extend(survivors)
        for d in dropped:
            _reject(rejected, d, REASON_NMS)

    # --- 6: пространственные правила ---
    prox = cfg.get("proximity", {}) or {}
    units = [g.as_points(i["points"]) for i in kept if i["label"] == "unit"]
    hoses = [i for i in kept if i["label"] == "wide_hose"]
    hose_limit = px(float(prox.get("hose_to_unit_px", onto.proximity_px)))
    keep_without_unit = bool(prox.get("keep_hose_without_unit", True))

    if hoses and (units or not keep_without_unit):
        survivors = []
        for hose in hoses:
            pts = g.as_points(hose["points"])
            _, dist = g.closest_polygon(pts, units) if units else (None, float("inf"))
            if dist <= hose_limit:
                hose["dist_to_unit_px"] = round(float(dist), 1)
                survivors.append(hose)
            else:
                _reject(rejected, hose, REASON_FAR_FROM_UNIT, f"dist={dist:.0f}px")
        kept = [i for i in kept if i["label"] != "wide_hose"] + survivors

    # Кабель, лежащий поверх принятого шланга, — это тот же рукав, увиденный
    # «жадным» промптом cable. Убираем, чтобы не учить противоречивым меткам.
    cable_iou = float(prox.get("cable_suppressed_by_hose_iou", 0.3))
    hose_polys = [g.as_points(i["points"]) for i in kept if i["label"] == "wide_hose"]
    if hose_polys:
        survivors = []
        for inst in kept:
            if inst["label"] != "cable":
                survivors.append(inst)
                continue
            pts = g.as_points(inst["points"])
            if any(g.polygon_iou(pts, h) >= cable_iou for h in hose_polys):
                _reject(rejected, inst, REASON_CABLE_IS_HOSE)
            else:
                survivors.append(inst)
        kept = survivors

    # --- 7: лимит инстансов класса на кадр (оставляем самые уверенные) ---
    limits = cfg.get("max_per_image", {}) or {}
    if limits:
        by_label = {}
        for inst in kept:
            by_label.setdefault(inst["label"], []).append(inst)
        kept = []
        for label, group in by_label.items():
            limit = limits.get(label)
            if limit is None or len(group) <= int(limit):
                kept.extend(group)
                continue
            group.sort(key=lambda d: float(d.get("score", 1.0)), reverse=True)
            kept.extend(group[: int(limit)])
            for d in group[int(limit):]:
                _reject(rejected, d, REASON_OVER_LIMIT, f"limit={limit}")

    return kept, rejected


def summarize(rejections: list[dict]) -> dict[str, int]:
    """Считает отбраковку по причинам (для отчёта и подбора порогов)."""
    out: dict[str, int] = {}
    for r in rejections:
        key = f"{r.get('label', '?')}:{r.get('reject_reason', '?')}"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def vest_fraction(image_bgr: np.ndarray, points: np.ndarray, ranges: list[tuple]) -> float:
    """Доля hi-vis (жилетных) пикселей внутри полигональной маски.

    Args:
        image_bgr: Изображение BGR.
        points: Вершины полигона (N, 2).
        ranges: Список HSV-диапазонов [(lower, upper), ...] — объединяются по OR.

    Returns:
        Отношение площади hi-vis-пикселей к площади маски в [0, 1].
    """
    import cv2  # локальный импорт: функция нужна только при наличии кадров на диске

    h, w = image_bgr.shape[:2]
    poly = np.zeros((h, w), np.uint8)
    cv2.fillPoly(poly, [g.as_points(points)], 255)
    total = int((poly > 0).sum())
    if total == 0:
        return 0.0
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    vest = np.zeros((h, w), np.uint8)
    for lower, upper in ranges:
        vest = cv2.bitwise_or(
            vest, cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
        )
    return float((cv2.bitwise_and(vest, poly) > 0).sum()) / float(total)
