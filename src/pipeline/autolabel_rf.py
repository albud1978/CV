"""СЛОЙ 2 — Auto-Label через Roboflow SAM 3: многопроходная одноклассовая разметка.

Зачем многопроходно:
    В одном многопромптовом проходе SAM 3 (PCS) при касании объектов разных классов
    (человек у шланга в момент подключения) сливает маски в один объект, путает класс
    (шланг помечается как person) или подавляет одну из масок. Один концепт на проход
    устраняет межклассовую конкуренцию -> чистые маски, которые затем объединяются офлайн.

Особенность Roboflow:
    Кадры дедуплицируются по контенту (общий id на воркспейс), повторный autolabel
    ПЕРЕЗАПИСЫВАЕТ аннотации. Поэтому проходы делаем последовательно и ВЫГРУЖАЕМ
    результат каждого прохода (`fetch`) до запуска следующего.

Подкоманды:
    fetch  — выгрузить маски одного прохода из RF annotation job в per-pass JSON.
    merge  — объединить проходы, разделить person -> staff/passenger по цвету жилета,
             отфильтровать тонкие "hose" по толщине маски, отрисовать QA и записать COCO.

Триггер самого autolabel (autolabel_start) выполняется через Roboflow MCP — см.
configs/ontology.gse_heater.yaml -> autolabel.passes.

Использование:
    # 1) после каждого одноклассового autolabel-прохода:
    python -m src.pipeline.autolabel_rf fetch \\
        --project gse-heater-seg --images-json data/rf_test_v2/image_ids.json \\
        --label hose --out data/rf_test_v2/passes/hose.json
    # 2) когда все проходы выгружены:
    python -m src.pipeline.autolabel_rf merge \\
        --passes-dir data/rf_test_v2/passes --frames-dir data/rf_test_v2/frames \\
        --config configs/ontology.gse_heater.yaml --out-dir data/rf_test_v2/merged
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROBOFLOW_API = "https://api.roboflow.com"


def _api_key() -> str:
    """Возвращает ключ Roboflow из переменной окружения ROBOFLOW_API_KEY."""
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise RuntimeError("Не задан ROBOFLOW_API_KEY в окружении")
    return key


def _get_json(url: str) -> dict:
    """GET с разбором JSON-ответа."""
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_pass(
    project: str,
    images: dict[str, str],
    label: Optional[str],
    workspace: str = "ban-nzbro",
) -> list[dict]:
    """Выгружает маски одного прохода из Roboflow.

    Args:
        project: slug проекта.
        images: соответствие {имя_файла: image_id}.
        label: оставить только маски этого класса (None — все).
        workspace: slug воркспейса.

    Returns:
        Список записей {image_name, width, height, masks:[{label, points}]}.
    """
    key = _api_key()
    out: list[dict] = []
    for name, iid in images.items():
        meta = _get_json(f"{ROBOFLOW_API}/{workspace}/{project}/images/{iid}?api_key={key}")
        ann = meta["image"]["annotation"]
        masks = []
        for b in ann.get("boxes", []):
            if not b.get("points"):
                continue
            if label is not None and b.get("label") != label:
                continue
            masks.append({"label": b["label"], "points": [[int(p[0]), int(p[1])] for p in b["points"]]})
        out.append({
            "image_name": name,
            "width": ann.get("width"),
            "height": ann.get("height"),
            "masks": masks,
        })
    return out


def vest_fraction(image_bgr: np.ndarray, points: np.ndarray, ranges: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """Доля hi-vis (жилетных) пикселей внутри полигональной маски.

    Args:
        image_bgr: изображение BGR.
        points: вершины полигона (N,2) int32.
        ranges: список диапазонов HSV [(lower, upper), ...] — объединяются (OR).

    Returns:
        Отношение площади hi-vis-пикселей к площади маски в [0,1].
    """
    h, w = image_bgr.shape[:2]
    poly = np.zeros((h, w), np.uint8)
    cv2.fillPoly(poly, [points], 255)
    total = int((poly > 0).sum())
    if total == 0:
        return 0.0
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    vest = np.zeros((h, w), np.uint8)
    for lower, upper in ranges:
        vest = cv2.bitwise_or(vest, cv2.inRange(hsv, lower, upper))
    inside = cv2.bitwise_and(vest, poly)
    return float((inside > 0).sum()) / float(total)


def mask_thickness(points: np.ndarray) -> float:
    """Средняя толщина полигона (px): 2*площадь/периметр."""
    area = cv2.contourArea(points)
    per = cv2.arcLength(points, True)
    return 2.0 * area / max(per, 1.0)


def _bbox(points: np.ndarray) -> tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(points)
    return x, y, x + w, y + h


def _poly_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Минимальное расстояние между полигонами (px); 0 если bbox пересекаются.

    Грубая, но быстрая оценка близости: пересечение рамок -> касание (0),
    иначе минимальная попарная дистанция между вершинами.
    """
    ax1, ay1, ax2, ay2 = _bbox(a)
    bx1, by1, bx2, by2 = _bbox(b)
    if ax1 <= bx2 and bx1 <= ax2 and ay1 <= by2 and by1 <= ay2:
        return 0.0
    pa = a.reshape(-1, 2).astype(np.float32)
    pb = b.reshape(-1, 2).astype(np.float32)
    d = np.sqrt(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(axis=2))
    return float(d.min())


def _load_yaml(path: str) -> dict:
    import yaml  # локальный импорт — yaml нужен только для merge
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Цвета классов для QA (BGR).
QA_COLORS = {
    "staff": (0, 220, 0),
    "passenger": (0, 140, 255),
    "wide_hose": (255, 80, 0),
    "hose": (255, 80, 0),
    "unit": (0, 0, 255),
    "cable": (200, 0, 200),
}


def merge_passes(
    passes_dir: str,
    frames_dir: str,
    config_path: str,
    out_dir: str,
) -> dict:
    """Объединяет проходы, делит person->staff/passenger, фильтрует hose, пишет QA+COCO.

    Args:
        passes_dir: папка с per-pass JSON (hose.json, person.json, ...).
        frames_dir: папка с исходными кадрами (для цвета жилета и QA).
        config_path: configs/ontology.gse_heater.yaml.
        out_dir: куда писать QA-картинки и annotations_coco.json.

    Returns:
        Сводка {класс: количество масок}.
    """
    cfg = _load_yaml(config_path)
    split = cfg.get("person_split", {})
    if split.get("hsv_ranges"):
        ranges = [(np.array(r["lower"], np.uint8), np.array(r["upper"], np.uint8)) for r in split["hsv_ranges"]]
    else:  # обратная совместимость со старым форматом одного диапазона
        ranges = [(np.array(split.get("hsv_lower", [30, 70, 80]), np.uint8),
                   np.array(split.get("hsv_upper", [75, 255, 255]), np.uint8))]
    vest_thr = float(split.get("vest_fraction_threshold", 0.10))
    # порог толщины для wide_hose
    hose_min_thick = 8.0
    for c in cfg.get("classes", []):
        if c.get("label") == "wide_hose":
            hose_min_thick = float(c.get("mask_thickness_filter", {}).get("min_thickness_px", 8))
    # пространственный фильтр: шланг должен быть рядом с unit (см. derived.heater_complex)
    proximity_px = float(cfg.get("derived", {}).get("heater_complex", {}).get("proximity_px", 250))
    keep_hose_without_unit = True  # если unit в кадре не найден — не теряем шланг (recall)

    out = Path(out_dir)
    (out / "qa").mkdir(parents=True, exist_ok=True)

    # собрать маски по изображениям из всех проходов
    by_image: dict[str, list[dict]] = {}
    for pj in sorted(Path(passes_dir).glob("*.json")):
        for rec in json.loads(pj.read_text(encoding="utf-8")):
            by_image.setdefault(rec["image_name"], []).append(rec)

    coco = {"images": [], "annotations": [], "categories": []}
    cat_ids = {"staff": 1, "passenger": 2, "wide_hose": 3, "unit": 4, "cable": 5}
    coco["categories"] = [{"id": i, "name": n} for n, i in cat_ids.items()]
    counts: dict[str, int] = {k: 0 for k in cat_ids}
    ann_id = 1

    for img_idx, (name, recs) in enumerate(sorted(by_image.items()), 1):
        fp = Path(frames_dir) / name
        image = cv2.imread(str(fp))
        if image is None:
            continue
        H, W = image.shape[:2]
        coco["images"].append({"id": img_idx, "file_name": name, "width": W, "height": H})

        # 1-й проход по кадру: разнести маски, собрать unit-полигоны, отложить hose
        finals: list[tuple[str, np.ndarray]] = []     # (финальный класс, полигон)
        unit_polys: list[np.ndarray] = []
        hose_cands: list[np.ndarray] = []
        for rec in recs:
            for m in rec["masks"]:
                pts = np.array(m["points"], np.int32)
                if len(pts) < 3:
                    continue
                raw = m["label"]
                if raw == "person":
                    frac = vest_fraction(image, pts, ranges)
                    finals.append(("staff" if frac >= vest_thr else "passenger", pts))
                elif raw == "hose":
                    if mask_thickness(pts) >= hose_min_thick:
                        hose_cands.append(pts)  # решение позже — по близости к unit
                elif raw in ("cart", "machine", "vehicle", "generator", "trailer", "truck"):
                    finals.append(("unit", pts))
                    unit_polys.append(pts)
                elif raw == "cable":
                    finals.append(("cable", pts))

        # 2-й проход: оставить только шланги рядом с unit (если unit найден)
        for pts in hose_cands:
            if unit_polys:
                near = any(_poly_distance(pts, u) <= proximity_px for u in unit_polys)
                if not near:
                    continue  # далеко от юнита -> наземный кабель, отбрасываем
            elif not keep_hose_without_unit:
                continue
            finals.append(("wide_hose", pts))

        # запись в COCO + QA
        overlay = image.copy()
        for final, pts in finals:
            if final not in cat_ids:
                continue
            counts[final] += 1
            x, y, w, h = cv2.boundingRect(pts)
            coco["annotations"].append({
                "id": ann_id, "image_id": img_idx, "category_id": cat_ids[final],
                "segmentation": [pts.flatten().tolist()],
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": float(cv2.contourArea(pts)), "iscrowd": 0,
            })
            ann_id += 1
            color = QA_COLORS.get(final, (200, 200, 200))
            cv2.fillPoly(overlay, [pts], color)
            cv2.putText(image, final, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        blended = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
        cv2.imwrite(str(out / "qa" / name), blended)

    (out / "annotations_coco.json").write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    return counts


def main() -> None:
    """CLI: подкоманды fetch / merge."""
    ap = argparse.ArgumentParser(description="Многопроходный одноклассовый Auto-Label Roboflow SAM 3")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="выгрузить маски одного прохода")
    f.add_argument("--project", required=True)
    f.add_argument("--workspace", default="ban-nzbro")
    f.add_argument("--images-json", required=True, help="JSON {имя_файла: image_id}")
    f.add_argument("--label", default=None, help="оставить только этот класс")
    f.add_argument("--out", required=True)

    m = sub.add_parser("merge", help="объединить проходы + staff/passenger + COCO + QA")
    m.add_argument("--passes-dir", required=True)
    m.add_argument("--frames-dir", required=True)
    m.add_argument("--config", required=True)
    m.add_argument("--out-dir", required=True)

    args = ap.parse_args()
    if args.cmd == "fetch":
        images = json.loads(Path(args.images_json).read_text(encoding="utf-8"))
        recs = fetch_pass(args.project, images, args.label, args.workspace)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
        total = sum(len(r["masks"]) for r in recs)
        print(f"fetch: {len(recs)} кадров, {total} масок (label={args.label}) -> {args.out}")
    elif args.cmd == "merge":
        counts = merge_passes(args.passes_dir, args.frames_dir, args.config, args.out_dir)
        print("merge: маски по классам ->", counts)
        print(f"COCO: {args.out_dir}/annotations_coco.json, QA: {args.out_dir}/qa/")


if __name__ == "__main__":
    main()
