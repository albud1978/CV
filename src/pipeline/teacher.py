"""СЛОЙ 2 — учитель: из СЛОВ в маски. Точка входа «SpeechToCV».

Вход — только текст (`ontology.yaml`) и папка кадров. Выход — псевдоразметка,
готовая к слиянию. Никакого ручного ведения по кнопкам UI.

Ключевое правило, добытое эмпирикой (см. шапку онтологии):
**один концепт на проход**. В многопромптовом проходе SAM 3 при касании объектов
(человек у шланга в момент подключения) сливает маски, путает класс или подавляет
одну из них. Поэтому проходы идут последовательно и независимо, а склейка —
офлайн в `fuse.py`, где работают проверяемые правила, а не NMS внутри учителя.

Бэкенды:
    local  — учитель крутится на своей машине (autodistill: SAM 3 / Grounded-SAM-2 /
             Grounding DINO). ДЕФОЛТ: нет round-trip'ов, нет дедупликации кадров
             на стороне сервиса, полный контроль и воспроизводимость.
    stub   — детерминированная заглушка для сухого прогона конвейера без весов
             (проверить раскладку файлов и слияние без GPU).

Roboflow в перезагруженной схеме отвечает за ревью, версии и хостовое обучение,
а не за покадровую генерацию масок: его дедупликация кадров по контенту
ПЕРЕЗАПИСЫВАЕТ аннотации при повторных проходах — прямой конфликт с многопроходной
схемой (проверено, см. CHANGELOG). Загрузка готовой разметки — `upload_rf.py`.

Использование:
    python -m src.pipeline.teacher \\
        --frames data/dataset_heater \\
        --ontology configs/ontology.gse_heater.yaml \\
        --out data/labels_v1/passes \\
        --backend local --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from src.pipeline.manifest import find_frames, frame_meta, load_manifest
from src.pipeline.ontology import Ontology, PassSpec

# Минимальная площадь маски в пикселях — отсекает пыль ещё до авто-QA.
MIN_MASK_AREA_PX = 24


def mask_to_polygon(mask: np.ndarray, epsilon_ratio: float = 0.002) -> Optional[list[list[int]]]:
    """Переводит бинарную маску в упрощённый полигон (крупнейший контур).

    Args:
        mask: Бинарная маска (H, W).
        epsilon_ratio: Доля периметра для аппроксимации Дугласа-Пекера.

    Returns:
        Список вершин ``[[x, y], ...]`` или None для вырожденной маски.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < MIN_MASK_AREA_PX:
        return None
    eps = epsilon_ratio * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
    if len(approx) < 3:
        return None
    return approx.astype(int).tolist()


def box_to_polygon(box: list[float]) -> list[list[int]]:
    """Прямоугольник (x1, y1, x2, y2) в полигон — для учителей без масок."""
    x1, y1, x2, y2 = [int(v) for v in box]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


# --------------------------------------------------------------------- бэкенды


def _load_local_model(prompt: str, target_label: str, device: str) -> Callable:
    """Готовит локального open-vocabulary учителя под ОДИН концепт.

    Порядок предпочтения: SAM 3 (маски, лучшее качество на перроне) ->
    Grounded-SAM-2 (маски, Apache-2.0 связка) -> Grounding DINO (только боксы).
    Импорты ленивые: модуль обязан импортироваться на машине без весов и GPU.

    Returns:
        Функция ``predict(image_path) -> [{points, score}]``.
    """
    from autodistill.detection import CaptionOntology  # noqa: WPS433 — ленивый импорт

    caption = CaptionOntology({prompt: target_label})
    model = None
    errors: list[str] = []

    for loader, name in (
        (lambda: __import__("autodistill_sam3", fromlist=["SAM3"]).SAM3(ontology=caption), "SAM 3"),
        (lambda: __import__("autodistill_grounded_sam_2", fromlist=["GroundedSAM2"]).GroundedSAM2(
            ontology=caption), "Grounded-SAM-2"),
        (lambda: __import__("autodistill_grounding_dino", fromlist=["GroundingDINO"]).GroundingDINO(
            ontology=caption), "Grounding DINO"),
    ):
        try:
            model = loader()
            print(f"    учитель: {name}")
            break
        except Exception as exc:  # noqa: BLE001 — перебираем доступные бэкенды
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    if model is None:
        raise RuntimeError(
            "Не удалось поднять локального учителя. Установите один из пакетов "
            "autodistill-sam3 / autodistill-grounded-sam-2 / autodistill-grounding-dino.\n"
            + "\n".join(errors)
        )

    def predict(image_path: str) -> list[dict[str, Any]]:
        detections = model.predict(image_path)
        out: list[dict[str, Any]] = []
        masks = getattr(detections, "mask", None)
        boxes = getattr(detections, "xyxy", None)
        scores = getattr(detections, "confidence", None)
        count = len(masks) if masks is not None else (len(boxes) if boxes is not None else 0)
        for i in range(count):
            points = mask_to_polygon(masks[i]) if masks is not None else box_to_polygon(boxes[i])
            if points is None:
                continue
            out.append({
                "points": points,
                "score": float(scores[i]) if scores is not None else 1.0,
            })
        return out

    return predict


def _stub_predictor(prompt: str, target_label: str) -> Callable:
    """Детерминированная заглушка: позволяет прогнать конвейер без весов и GPU.

    Кладёт правдоподобную по форме фигуру в зависящее от имени файла место —
    достаточно, чтобы проверить раскладку артефактов, слияние и экспорт.
    """
    def predict(image_path: str) -> list[dict[str, Any]]:
        image = cv2.imread(image_path)
        if image is None:
            return []
        h, w = image.shape[:2]
        seed = sum(ord(c) for c in Path(image_path).name)
        if seed % 3 == 0:                       # часть кадров пустая — это нормально
            return []
        if target_label == "wide_hose":
            y = int(h * 0.6)
            pts = [[int(w * 0.2), y - 8], [int(w * 0.5), y - 8],
                   [int(w * 0.5), y + 8], [int(w * 0.2), y + 8]]
        elif target_label == "aircraft":
            pts = [[int(w * 0.52), int(h * 0.3)], [int(w * 0.95), int(h * 0.3)],
                   [int(w * 0.95), int(h * 0.8)], [int(w * 0.52), int(h * 0.8)]]
        else:
            pts = [[int(w * 0.14), int(h * 0.55)], [int(w * 0.21), int(h * 0.55)],
                   [int(w * 0.21), int(h * 0.68)], [int(w * 0.14), int(h * 0.68)]]
        return [{"points": pts, "score": 0.8}]

    return predict


# --------------------------------------------------------------------- проходы


def run_pass(
    spec: PassSpec,
    frames: list[Path],
    manifest: dict[str, dict[str, Any]],
    predictor: Callable,
) -> list[dict[str, Any]]:
    """Гоняет один одноклассовый проход учителя по всем кадрам.

    Returns:
        Записи вида ``{image_name, image_path, width, height, video_id, timestamp,
        split, masks: [{label, points, score}]}``.
    """
    records: list[dict[str, Any]] = []
    for i, path in enumerate(frames, 1):
        image = cv2.imread(str(path))
        if image is None:
            continue
        h, w = image.shape[:2]
        raw = predictor(str(path))
        masks = [
            {"label": spec.maps_to, "points": d["points"], "score": float(d.get("score", 1.0))}
            for d in raw
            if float(d.get("score", 1.0)) >= spec.confidence
        ]
        meta = frame_meta(path.name, manifest)
        records.append({
            "image_name": path.name,
            "image_path": str(path),
            "width": w,
            "height": h,
            "masks": masks,
            **meta,
        })
        if i % 100 == 0:
            print(f"    {i}/{len(frames)} кадров")
    return records


def main() -> None:
    """CLI: прогон всех проходов онтологии и запись per-pass JSON."""
    ap = argparse.ArgumentParser(description="СЛОЙ 2 — учитель: слова -> маски")
    ap.add_argument("--frames", required=True, help="Корень датасета кадров")
    ap.add_argument("--ontology", default="configs/ontology.gse_heater.yaml")
    ap.add_argument("--out", required=True, help="Папка для per-pass JSON")
    ap.add_argument("--backend", choices=["local", "stub"], default="local")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", default=None, help="Ограничить split (train/val/test)")
    ap.add_argument("--limit", type=int, default=0, help="Взять только N кадров (быстрый прогон)")
    ap.add_argument("--only-pass", default=None, help="Прогнать один проход по имени")
    args = ap.parse_args()

    onto = Ontology.load(args.ontology)
    manifest = load_manifest(args.frames)
    frames = find_frames(args.frames, args.split)
    if args.limit:
        frames = frames[: args.limit]
    if not frames:
        raise SystemExit(f"Кадры не найдены в {args.frames}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Кадров: {len(frames)} | проходов: {len(onto.passes)} | бэкенд: {args.backend}")

    summary: dict[str, int] = {}
    for spec in onto.passes:
        if args.only_pass and spec.name != args.only_pass:
            continue
        print(f"\n[проход {spec.name}] промпт '{spec.prompt}' -> {spec.maps_to} "
              f"(conf {spec.confidence})")
        started = time.time()
        predictor = (
            _stub_predictor(spec.prompt, spec.maps_to)
            if args.backend == "stub"
            else _load_local_model(spec.prompt, spec.maps_to, args.device)
        )
        records = run_pass(spec, frames, manifest, predictor)
        total = sum(len(r["masks"]) for r in records)
        summary[spec.name] = total
        path = out_dir / f"{spec.name}.json"
        path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        print(f"    масок: {total} за {time.time() - started:.1f} c -> {path}")

    (out_dir / "_summary.json").write_text(
        json.dumps({"frames": len(frames), "masks_per_pass": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nИтого масок по проходам: {summary}")


if __name__ == "__main__":
    main()
