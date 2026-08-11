"""Логика «обогреватель подключён к ВС»: из детекций кадра — в оценку состояния.

Модель заказчика: heater = тележка (`unit`) + ШИРОКИЙ гофрошланг (`wide_hose`),
второй конец которого заведён под борт (`aircraft`). Тонкий кабель от такой же
тележки означает GPU, а не обогреватель.

Что делает модуль на ОДНОМ кадре:
    1. отбирает шланги достаточной длины;
    2. у каждого шланга находит два конца (наиболее удалённая пара точек контура);
    3. ближний к тележке конец объявляет «юнитным», второй — «бортовым»;
    4. меряет расстояние бортового конца до маски ВС -> P(подключён).

Три уровня уверенности, вместо булева ответа:
    * геометрическое подтверждение — конец касается/перекрывает ВС;
    * близость — конец в пределах `connection.gap_px` от борта;
    * резерв — ВС в кадре не найдено (ночь, ракурс), но широкий рукав раскатан
      рядом с юнитом. Даёт пониженный score, чтобы конечный автомат требовал
      более длительного подтверждения.

Покадровый score сглаживается по времени в `events.py` — одиночный кадр никогда
не порождает событие.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from src.pipeline import geometry as g
from src.pipeline.ontology import Ontology

STATE_CONNECTED = "connected"
STATE_DISCONNECTED = "disconnected"
STATE_UNKNOWN = "unknown"


@dataclass
class FrameVerdict:
    """Оценка состояния подключения по одному кадру."""

    timestamp: float
    state: str
    score: float                      # P(подключён) в [0, 1]
    reason: str
    heater_complex: bool = False      # найдена связка unit + wide_hose
    gpu_suspect: bool = False         # unit + тонкий кабель без широкого шланга
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _by_label(instances: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for inst in instances:
        out.setdefault(inst["label"], []).append(inst)
    return out


def _hose_length(points) -> float:
    """Длина рукава — расстояние между его концами."""
    (x1, y1), (x2, y2) = g.polygon_endpoints(points)
    return float(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)


def analyze_frame(
    instances: list[dict],
    width: int,
    height: int,
    onto: Ontology,
    timestamp: float = 0.0,
    aircraft_zone: Optional[list] = None,
) -> FrameVerdict:
    """Оценивает состояние подключения по инстансам одного кадра.

    Args:
        instances: Инстансы кадра ``[{label, points, score}]`` (после авто-QA).
        width: Ширина кадра (для масштабирования порогов).
        height: Высота кадра.
        onto: Онтология задачи.
        timestamp: Время кадра в секундах от начала видео.
        aircraft_zone: Необязательный полигон зоны стоянки ВС для этой камеры.
            Используется, когда сам самолёт не задетектирован (ночь/ракурс).

    Returns:
        Вердикт кадра с вероятностью подключения и разбором.
    """
    cfg = onto.connection
    scores = cfg.get("scores", {}) or {}
    ref = onto.reference_width
    gap_px = g.scale_px(float(cfg.get("gap_px", 60)), width, ref)
    min_len = g.scale_px(float(cfg.get("min_hose_length_px", 60)), width, ref)
    proximity = g.scale_px(onto.proximity_px, width, ref)
    fallback_cfg = cfg.get("fallback", {}) or {}

    groups = _by_label(instances)
    units = [g.as_points(i["points"]) for i in groups.get("unit", [])]
    aircrafts = [g.as_points(i["points"]) for i in groups.get("aircraft", [])]
    if not aircrafts and aircraft_zone:
        aircrafts = [g.as_points(aircraft_zone)]
    cables = [g.as_points(i["points"]) for i in groups.get("cable", [])]

    hoses = []
    for inst in groups.get("wide_hose", []):
        pts = g.as_points(inst["points"])
        if _hose_length(pts) >= min_len:
            hoses.append((inst, pts))

    # Связка «обогреватель»: рукав рядом с тележкой.
    heater_complex = False
    for _, pts in hoses:
        if not units:
            heater_complex = True
            break
        _, dist = g.closest_polygon(pts, units)
        if dist <= proximity:
            heater_complex = True
            break

    # Подозрение на GPU: тележка с тонким кабелем и без широкого рукава.
    gpu_suspect = bool(units and cables and not hoses)

    # --- сцена не информативна: судить не о чем ---
    if not hoses and not units and not aircrafts:
        return FrameVerdict(
            timestamp=timestamp,
            state=STATE_UNKNOWN,
            score=0.0,
            reason="scene_empty",
            details={"instances": len(instances)},
        )

    # --- рукава нет, но техника/борт видны: уверенное «не подключён» ---
    if not hoses:
        return FrameVerdict(
            timestamp=timestamp,
            state=STATE_DISCONNECTED,
            score=float(scores.get("disconnected_hose_absent", 0.02)),
            reason="hose_absent",
            gpu_suspect=gpu_suspect,
            details={"units": len(units), "aircrafts": len(aircrafts), "cables": len(cables)},
        )

    best_score = 0.0
    best_reason = "hose_far_from_aircraft"
    best_details: dict[str, Any] = {}

    for inst, pts in hoses:
        end_a, end_b = g.polygon_endpoints(pts)
        # Ближний к тележке конец считаем «юнитным», второй уходит к борту.
        if units:
            d_a = min(g.point_to_polygon_distance(end_a, u) for u in units)
            d_b = min(g.point_to_polygon_distance(end_b, u) for u in units)
            unit_end, aircraft_end = (end_a, end_b) if d_a <= d_b else (end_b, end_a)
            d_unit = min(d_a, d_b)
        else:
            unit_end, aircraft_end, d_unit = None, None, float("inf")

        if aircrafts:
            if aircraft_end is not None:
                d_air = min(g.point_to_polygon_distance(aircraft_end, a) for a in aircrafts)
            else:
                # Тележка не найдена — проверяем оба конца, берём лучший.
                d_air = min(
                    min(g.point_to_polygon_distance(e, a) for a in aircrafts)
                    for e in (end_a, end_b)
                )
            if d_air <= 0.0:
                score = float(scores.get("connected_geometric", 0.95))
                reason = "hose_end_touches_aircraft"
            elif d_air <= gap_px:
                score = float(scores.get("connected_near", 0.75))
                reason = "hose_end_near_aircraft"
            else:
                score = float(scores.get("disconnected_hose_far", 0.15))
                reason = "hose_far_from_aircraft"
            details = {
                "d_aircraft_px": round(float(d_air), 1),
                "d_unit_px": None if d_unit == float("inf") else round(float(d_unit), 1),
                "gap_px": round(gap_px, 1),
                "unit_end": unit_end,
                "aircraft_end": aircraft_end,
            }
        elif bool(fallback_cfg.get("enabled", True)):
            # ВС не видно: раскатанный широкий рукав у тележки сам по себе —
            # признак работы обогревателя. Уверенность понижена намеренно.
            near_unit = (not units) or d_unit <= proximity
            score = float(fallback_cfg.get("score", 0.55)) if near_unit else 0.2
            reason = "fallback_hose_presence" if near_unit else "hose_far_from_unit"
            details = {"d_unit_px": None if d_unit == float("inf") else round(float(d_unit), 1)}
        else:
            return FrameVerdict(
                timestamp=timestamp,
                state=STATE_UNKNOWN,
                score=0.0,
                reason="aircraft_not_detected",
                heater_complex=heater_complex,
                details={"hoses": len(hoses)},
            )

        if score > best_score:
            best_score, best_reason, best_details = score, reason, details

    return FrameVerdict(
        timestamp=timestamp,
        state=STATE_CONNECTED if best_score >= 0.5 else STATE_DISCONNECTED,
        score=best_score,
        reason=best_reason,
        heater_complex=heater_complex,
        gpu_suspect=gpu_suspect,
        details=best_details,
    )


def analyze_sequence(
    frames: list[dict[str, Any]],
    onto: Ontology,
    aircraft_zone: Optional[list] = None,
) -> list[FrameVerdict]:
    """Прогоняет :func:`analyze_frame` по последовательности кадров видео.

    Args:
        frames: ``[{"timestamp", "width", "height", "instances"}]`` по возрастанию времени.
        onto: Онтология.
        aircraft_zone: Полигон зоны стоянки для камеры (необязательно).

    Returns:
        Список покадровых вердиктов в том же порядке.
    """
    return [
        analyze_frame(
            f.get("instances", []),
            int(f["width"]),
            int(f["height"]),
            onto,
            timestamp=float(f.get("timestamp", 0.0)),
            aircraft_zone=aircraft_zone,
        )
        for f in frames
    ]
