"""Тесты конвейера на синтетических сценах (без данных, весов и GPU).

Запуск:
    python3 -m tests.test_pipeline      # или: pytest tests/test_pipeline.py

Проверяем ту часть, которая принимает решения: геометрию масок, авто-QA,
временную консистентность, логику подключения и конечный автомат событий.
Синтетическая сцена повторяет реальную компоновку перрона: тележка слева,
рукав вправо, борт справа.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import autoqa, eval_events, events, geometry as g, link, relation  # noqa: E402
from src.pipeline.ontology import Ontology  # noqa: E402

W, H = 1280, 720
ONTO = Ontology.load(Path(__file__).resolve().parents[1] / "configs/ontology.gse_heater.yaml")


def rect(x1: int, y1: int, x2: int, y2: int) -> list[list[int]]:
    """Прямоугольник как полигон."""
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def strip(x1: int, x2: int, y: int, thickness: int) -> list[list[int]]:
    """Горизонтальная лента заданной толщины — модель рукава/кабеля."""
    half = thickness // 2
    return rect(x1, y - half, x2, y + half)


UNIT = rect(100, 400, 200, 500)
HOSE = strip(200, 600, 450, 16)
AIRCRAFT_NEAR = rect(620, 300, 1200, 600)     # конец рукава в 20 px от борта
AIRCRAFT_FAR = rect(900, 300, 1200, 600)      # конец рукава в 300 px от борта


def frame(instances: list[dict], timestamp: float = 0.0) -> dict:
    return {"width": W, "height": H, "timestamp": timestamp, "instances": instances}


def inst(label: str, points: list, score: float = 0.9) -> dict:
    return {"label": label, "points": points, "score": score}


# --------------------------------------------------------------------------- geometry


def test_thickness_and_elongation():
    assert 8.0 < g.mean_thickness(HOSE) < 16.0, "2S/P должна быть близка к ширине ленты"
    assert g.mean_thickness(strip(200, 600, 450, 4)) < 4.5
    assert g.elongation(HOSE) > 10
    assert 0.9 < g.elongation(UNIT) < 1.1


def test_distances_and_endpoints():
    assert g.polygon_distance(UNIT, HOSE) == 0.0, "лента начинается на границе тележки"
    assert abs(g.polygon_distance(HOSE, AIRCRAFT_NEAR) - 20.0) < 1.5
    e1, e2 = g.polygon_endpoints(HOSE)
    xs = sorted([e1[0], e2[0]])
    assert xs[0] <= 205 and xs[1] >= 595, "концы рукава должны лежать на его торцах"


def test_scale_px():
    assert g.scale_px(60, 1280) == 60
    assert g.scale_px(60, 1920) == 90, "порог обязан масштабироваться под разрешение"


def test_polygon_iou():
    assert g.polygon_iou(UNIT, UNIT) > 0.99
    assert g.polygon_iou(UNIT, AIRCRAFT_FAR) == 0.0


# --------------------------------------------------------------------------- autoqa


def test_autoqa_rejects_thin_hose():
    kept, rejected = autoqa.filter_frame(
        frame([inst("unit", UNIT), inst("wide_hose", strip(200, 600, 450, 4))]), ONTO
    )
    assert [i["label"] for i in kept] == ["unit"]
    assert rejected[0]["reject_reason"] == autoqa.REASON_THIN


def test_autoqa_rejects_hose_far_from_unit():
    kept, rejected = autoqa.filter_frame(
        frame([inst("unit", UNIT), inst("wide_hose", strip(800, 1200, 650, 16))]), ONTO
    )
    assert [i["label"] for i in kept] == ["unit"]
    assert rejected[0]["reject_reason"] == autoqa.REASON_FAR_FROM_UNIT


def test_autoqa_keeps_valid_scene_and_dedups():
    kept, rejected = autoqa.filter_frame(
        frame([
            inst("unit", UNIT),
            inst("wide_hose", HOSE),
            inst("wide_hose", strip(202, 598, 451, 16)),   # дубль того же рукава
            inst("aircraft", AIRCRAFT_NEAR),
        ]),
        ONTO,
    )
    labels = sorted(i["label"] for i in kept)
    assert labels == ["aircraft", "unit", "wide_hose"]
    assert rejected[0]["reject_reason"] == autoqa.REASON_NMS


def test_autoqa_drops_cable_lying_on_hose():
    kept, rejected = autoqa.filter_frame(
        frame([
            inst("unit", UNIT),
            inst("wide_hose", HOSE),
            inst("cable", strip(210, 590, 450, 8)),        # тот же рукав «жадным» промптом
        ]),
        ONTO,
    )
    assert "cable" not in [i["label"] for i in kept]
    assert rejected[0]["reject_reason"] == autoqa.REASON_CABLE_IS_HOSE


def test_autoqa_low_score_filtered():
    kept, rejected = autoqa.filter_frame(frame([inst("unit", UNIT, score=0.05)]), ONTO)
    assert kept == [] and rejected[0]["reject_reason"] == autoqa.REASON_LOW_SCORE


# --------------------------------------------------------------------------- link


def test_link_drops_flicker_keeps_persistent():
    frames = [
        frame([inst("unit", UNIT)], 0.0),
        frame([inst("unit", UNIT), inst("unit", rect(900, 100, 950, 150))], 5.0),
        frame([inst("unit", UNIT)], 10.0),
    ]
    dropped, stats = link.apply(frames, ONTO)
    assert stats["tracks"] == 2
    assert len(dropped) == 1 and dropped[0]["reject_reason"] == "short_track"
    assert all(len(f["instances"]) == 1 for f in frames), "устойчивая тележка сохранена"


def test_link_does_not_cut_people():
    frames = [
        frame([inst("staff", rect(300, 300, 320, 360))], 0.0),
        frame([], 5.0),
    ]
    dropped, _ = link.apply(frames, ONTO)
    assert dropped == [], "люди появляются на один кадр — резать их нельзя"


def test_link_fill_gaps_when_enabled():
    frames = [
        frame([inst("unit", UNIT)], 0.0),
        frame([], 5.0),
        frame([inst("unit", UNIT)], 10.0),
    ]
    onto = Ontology(dict(ONTO.raw))
    onto.raw["link"] = dict(onto.raw["link"], fill_gaps=True)
    link.apply(frames, onto)
    assert len(frames[1]["instances"]) == 1
    assert frames[1]["instances"][0]["synthetic"] is True


# --------------------------------------------------------------------------- relation


def test_connected_when_hose_reaches_aircraft():
    v = relation.analyze_frame(
        [inst("unit", UNIT), inst("wide_hose", HOSE), inst("aircraft", AIRCRAFT_NEAR)],
        W, H, ONTO,
    )
    assert v.state == relation.STATE_CONNECTED
    assert v.score >= 0.75 and v.heater_complex
    assert v.reason in ("hose_end_near_aircraft", "hose_end_touches_aircraft")


def test_disconnected_when_hose_far_from_aircraft():
    v = relation.analyze_frame(
        [inst("unit", UNIT), inst("wide_hose", HOSE), inst("aircraft", AIRCRAFT_FAR)],
        W, H, ONTO,
    )
    assert v.state == relation.STATE_DISCONNECTED and v.reason == "hose_far_from_aircraft"


def test_fallback_without_aircraft():
    v = relation.analyze_frame([inst("unit", UNIT), inst("wide_hose", HOSE)], W, H, ONTO)
    assert v.state == relation.STATE_CONNECTED and v.reason == "fallback_hose_presence"
    assert v.score < 0.75, "резервное правило обязано быть менее уверенным"


def test_gpu_is_not_heater():
    v = relation.analyze_frame(
        [inst("unit", UNIT), inst("cable", strip(200, 600, 450, 6)), inst("aircraft", AIRCRAFT_NEAR)],
        W, H, ONTO,
    )
    assert v.state == relation.STATE_DISCONNECTED and v.gpu_suspect


def test_empty_scene_is_unknown():
    v = relation.analyze_frame([], W, H, ONTO)
    assert v.state == relation.STATE_UNKNOWN


def test_thresholds_follow_resolution():
    """Та же сцена в 1920px: порог gap_px масштабируется, вердикт не меняется."""
    scale = 1.5
    scaled = lambda poly: [[int(x * scale), int(y * scale)] for x, y in poly]  # noqa: E731
    v = relation.analyze_frame(
        [inst("unit", scaled(UNIT)), inst("wide_hose", scaled(HOSE)),
         inst("aircraft", scaled(AIRCRAFT_NEAR))],
        int(W * scale), int(H * scale), ONTO,
    )
    assert v.state == relation.STATE_CONNECTED


# --------------------------------------------------------------------------- events


def _timeline(pattern: list[tuple[float, float, float]], step: float = 5.0):
    """Строит вердикты: pattern = [(начало, конец, score)]."""
    out = []
    for start, end, score in pattern:
        t = start
        while t < end:
            state = relation.STATE_CONNECTED if score >= 0.5 else relation.STATE_DISCONNECTED
            out.append(relation.FrameVerdict(t, state, score, "synthetic"))
            t += step
    return out


def test_single_connect_disconnect_cycle():
    verdicts = _timeline([(0, 600, 0.02), (600, 2400, 0.9), (2400, 3600, 0.02)])
    evs, eps = events.detect_events(verdicts, ONTO)
    names = [e.name for e in evs]
    assert names == ["hose_connected", "hose_disconnected"], names
    assert abs(evs[0].timestamp - 600) <= 60, evs[0].timestamp
    assert abs(evs[1].timestamp - 2400) <= 60, evs[1].timestamp
    assert len(eps) == 1 and abs(eps[0].duration - 1800) <= 120


def test_recording_starts_already_connected():
    """Видео 01: рукав уже подключён на первом кадре — событие подключения не выдумываем."""
    verdicts = _timeline([(0, 2520, 0.9), (2520, 4200, 0.02)])
    evs, eps = events.detect_events(verdicts, ONTO)
    assert [e.name for e in evs] == ["hose_disconnected"]
    assert abs(evs[0].timestamp - 2520) <= 60


def test_short_noise_produces_no_events():
    verdicts = _timeline([(0, 1200, 0.02), (1200, 1215, 0.95), (1215, 2400, 0.02)])
    evs, eps = events.detect_events(verdicts, ONTO)
    assert evs == [] and eps == [], "всплеск в 15 c не должен рождать событие"


def test_unknown_frames_hold_state():
    """Персонал перекрыл камеру: сцена не видна — состояние удерживается, а не падает."""
    verdicts = _timeline([(0, 1800, 0.9)])
    blind = [relation.FrameVerdict(t, relation.STATE_UNKNOWN, 0.0, "scene_empty")
             for t in range(1800, 2100, 5)]
    verdicts += blind + _timeline([(2100, 3600, 0.9)])
    evs, _ = events.detect_events(verdicts, ONTO)
    assert evs == [], "слепой участок не должен порождать отключение и повторное подключение"


def test_report_shape():
    verdicts = _timeline([(0, 600, 0.02), (600, 2400, 0.9), (2400, 3000, 0.02)])
    evs, eps = events.detect_events(verdicts, ONTO)
    rep = events.report(evs, eps, video_id="v01")
    assert rep["video_id"] == "v01" and rep["connected_seconds"] > 1500
    assert rep["events"][0]["timecode"].count(":") == 2


# --------------------------------------------------------------------------- eval_events


def test_normalize_windows_open_end():
    assert eval_events.normalize_windows([[300, None]], 3600) == [(300.0, 3600.0)]
    assert eval_events.normalize_windows([[0, 2580]], 4200) == [(0.0, 2580.0)]


def test_temporal_iou():
    assert eval_events.temporal_iou([(0, 100)], [(0, 100)]) == 1.0
    assert eval_events.temporal_iou([(0, 100)], [(50, 150)]) == 0.3333
    assert eval_events.temporal_iou([(0, 100)], [(200, 300)]) == 0.0


def test_boundaries_ignore_recording_edges():
    """Окно от 0 до конца записи не содержит наблюдаемых переходов."""
    assert eval_events.boundaries([(0.0, 3600.0)], 3600.0) == []
    assert eval_events.boundaries([(600.0, 2400.0)], 3600.0) == [
        ("hose_connected", 600.0), ("hose_disconnected", 2400.0)
    ]


def test_evaluate_video_pass_and_fail():
    report = {
        "video_id": "v01",
        "duration_sec": 3600,
        "episodes": [{"start": 620, "end": 2380}],
        "events": [
            {"name": "hose_connected", "timestamp": 620},
            {"name": "hose_disconnected", "timestamp": 2380},
        ],
    }
    res = eval_events.evaluate_video(report, [[600, 2400]], ONTO)
    assert res["passed"] and res["temporal_iou"] > 0.95
    assert all(m["within_tolerance"] for m in res["matched"])

    noisy = dict(report, events=report["events"] + [
        {"name": "hose_connected", "timestamp": 3000},
        {"name": "hose_disconnected", "timestamp": 3100},
    ])
    res_noisy = eval_events.evaluate_video(noisy, [[600, 2400]], ONTO)
    assert not res_noisy["passed"] and len(res_noisy["false_events"]) == 2


def test_evaluate_video_missed_event():
    report = {"video_id": "v10", "duration_sec": 3200, "episodes": [], "events": []}
    res = eval_events.evaluate_video(report, [[300, 2000]], ONTO)
    assert not res["passed"] and len(res["missed"]) == 2 and res["temporal_iou"] == 0.0


# --------------------------------------------------------------------------- runner


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok   {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL {name}: {exc}")
        except Exception as exc:  # pragma: no cover — диагностика
            failed += 1
            print(f"  ERR  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} тестов пройдено")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
