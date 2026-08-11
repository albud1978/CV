"""Конечный автомат событий: покадровые вердикты -> тайм-коды подключения/отключения.

Заказчику нужен не «процент кадров», а две строки в отчёте: во сколько рукав
подключили и во сколько отключили. Между покадровой оценкой и этим ответом стоят
три механизма:

    * **скользящее окно** `hold_seconds` — решение принимается по среднему score,
      одиночный кадр никогда не переключает состояние;
    * **гистерезис** (`enter_threshold` > `exit_threshold`) — состояние не дрожит
      у порога, когда персонал ходит перед камерой;
    * **пост-обработка эпизодов** — короткие «дырки» внутри подключения
      склеиваются, короткие эпизоды выбрасываются как шум.

Кадры со статусом `unknown` (сцену не видно) не голосуют: политика `hold`
удерживает последнее состояние, а не роняет его в «отключено».
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable

from src.pipeline.ontology import Ontology
from src.pipeline.relation import STATE_CONNECTED, STATE_DISCONNECTED, STATE_UNKNOWN, FrameVerdict


@dataclass
class Episode:
    """Непрерывный отрезок времени в состоянии `connected`."""

    start: float
    end: float
    mean_score: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Event:
    """Переход состояния — то, что уходит в отчёт."""

    name: str
    timestamp: float
    timecode: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def hms(seconds: float) -> str:
    """Секунды от начала видео -> ``ЧЧ:ММ:СС``."""
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _rolling_state(
    verdicts: list[FrameVerdict], onto: Ontology
) -> list[tuple[float, str, float]]:
    """Прогоняет окно с гистерезисом. Возвращает [(время_начала_окна, состояние, score)]."""
    cfg = onto.events
    hold = float(cfg.get("hold_seconds", 30.0))
    enter_thr = float(cfg.get("enter_threshold", 0.6))
    exit_thr = float(cfg.get("exit_threshold", 0.35))
    unknown_policy = str(cfg.get("unknown_policy", "hold"))
    state = str(cfg.get("initial_state", STATE_DISCONNECTED))

    switches: list[tuple[float, str, float]] = []
    window: list[FrameVerdict] = []

    for v in verdicts:
        window.append(v)
        while window and v.timestamp - window[0].timestamp > hold:
            window.pop(0)

        if unknown_policy == "hold":
            known = [w for w in window if w.state != STATE_UNKNOWN]
        else:  # unknown трактуется как «не подключён»
            known = window
        if not known:
            continue

        mean = sum(w.score for w in known) / len(known)
        if state != STATE_CONNECTED and mean >= enter_thr:
            state = STATE_CONNECTED
            switches.append((window[0].timestamp, state, mean))
        elif state == STATE_CONNECTED and mean <= exit_thr:
            state = STATE_DISCONNECTED
            switches.append((window[0].timestamp, state, mean))

    return switches


def _episodes_from_switches(
    switches: list[tuple[float, str, float]],
    t_start: float,
    t_end: float,
    initial_state: str,
) -> list[Episode]:
    """Строит отрезки `connected` из последовательности переключений."""
    episodes: list[Episode] = []
    state = initial_state
    open_at = t_start if initial_state == STATE_CONNECTED else None
    open_score = 0.0
    for ts, new_state, score in switches:
        if new_state == STATE_CONNECTED and state != STATE_CONNECTED:
            open_at, open_score = ts, score
        elif new_state == STATE_DISCONNECTED and state == STATE_CONNECTED and open_at is not None:
            episodes.append(Episode(open_at, ts, (open_score + score) / 2))
            open_at = None
        state = new_state
    if open_at is not None:
        episodes.append(Episode(open_at, t_end, open_score))
    return episodes


def _postprocess(episodes: list[Episode], min_seconds: float) -> list[Episode]:
    """Склеивает короткие разрывы, затем выбрасывает короткие эпизоды."""
    if not episodes:
        return []
    merged = [episodes[0]]
    for ep in episodes[1:]:
        prev = merged[-1]
        if ep.start - prev.end < min_seconds:
            merged[-1] = Episode(
                prev.start, ep.end, (prev.mean_score + ep.mean_score) / 2
            )
        else:
            merged.append(ep)
    return [ep for ep in merged if ep.duration >= min_seconds]


def detect_events(
    verdicts: Iterable[FrameVerdict], onto: Ontology
) -> tuple[list[Event], list[Episode]]:
    """Превращает покадровые вердикты в события и эпизоды подключения.

    Args:
        verdicts: Покадровые вердикты (будут отсортированы по времени).
        onto: Онтология (секция ``events``).

    Returns:
        Кортеж (события-переходы, эпизоды состояния `connected`).
    """
    seq = sorted(verdicts, key=lambda v: v.timestamp)
    if not seq:
        return [], []

    cfg = onto.events
    initial_state = str(cfg.get("initial_state", STATE_DISCONNECTED))
    min_seconds = float(cfg.get("min_episode_seconds", 0.0))
    t_start, t_end = seq[0].timestamp, seq[-1].timestamp

    switches = _rolling_state(seq, onto)
    episodes = _postprocess(
        _episodes_from_switches(switches, t_start, t_end, initial_state), min_seconds
    )

    events: list[Event] = []
    for ep in episodes:
        # Подключение в самом начале записи событием не считаем: мы не видели перехода.
        if ep.start > t_start:
            events.append(
                Event("hose_connected", ep.start, hms(ep.start), round(ep.mean_score, 3))
            )
        # Аналогично, если к концу записи рукав всё ещё подключён — отключения не было.
        if ep.end < t_end:
            events.append(
                Event("hose_disconnected", ep.end, hms(ep.end), round(ep.mean_score, 3))
            )
    return events, episodes


def report(events: list[Event], episodes: list[Episode], video_id: str = "") -> dict[str, Any]:
    """Собирает итоговый JSON-отчёт по видео."""
    return {
        "video_id": video_id,
        "events": [e.to_dict() for e in events],
        "episodes": [
            {
                "start": round(ep.start, 2),
                "end": round(ep.end, 2),
                "start_timecode": hms(ep.start),
                "end_timecode": hms(ep.end),
                "duration_sec": round(ep.duration, 2),
                "mean_score": round(ep.mean_score, 3),
            }
            for ep in episodes
        ],
        "connected_seconds": round(sum(ep.duration for ep in episodes), 2),
    }
