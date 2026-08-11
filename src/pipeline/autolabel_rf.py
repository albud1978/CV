"""Мост к Roboflow: выгрузка результата облачной авто-разметки в формат проходов.

После перезагрузки конвейера учитель по умолчанию локальный (`teacher.py`), а
Roboflow отвечает за ревью, версии и хостовое обучение. Этот модуль остаётся
для сценария «разметили в облаке — забрали к себе»:

    fetch — выгрузить маски одного одноклассового прохода из Roboflow в
            `passes/<имя>.json` (тот же формат, что пишет `teacher.py`).

Вся офлайн-логика (person -> staff/passenger, фильтры, треки, COCO) живёт в
`fuse.py` — здесь она НЕ дублируется, подкоманда `merge` просто вызывает её.

ВАЖНО (проверено): Roboflow дедуплицирует кадры по контенту на весь воркспейс,
поэтому повторный autolabel ПЕРЕЗАПИСЫВАЕТ аннотации общей записи. При облачных
проходах выгружайте результат ПОСЛЕ КАЖДОГО прохода, до запуска следующего.

Использование:
    python -m src.pipeline.autolabel_rf fetch \\
        --project gse-heater-seg --images-json data/labels_v1/image_ids.json \\
        --label hose --maps-to wide_hose --out data/labels_v1/passes/hose.json
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Optional

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
    maps_to: Optional[str] = None,
) -> list[dict]:
    """Выгружает маски одного прохода из Roboflow.

    Args:
        project: slug проекта.
        images: соответствие {имя_файла: image_id}.
        label: оставить только маски этого класса учителя (None — все).
        workspace: slug воркспейса.
        maps_to: во что переименовать класс (например, hose -> wide_hose).

    Returns:
        Список записей ``{image_name, width, height, masks:[{label, points, score}]}``.
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
            masks.append({
                "label": maps_to or b["label"],
                "points": [[int(p[0]), int(p[1])] for p in b["points"]],
                "score": float(b.get("confidence", 1.0)),
            })
        out.append({
            "image_name": name,
            "width": ann.get("width"),
            "height": ann.get("height"),
            "masks": masks,
        })
    return out


def main() -> None:
    """CLI: подкоманды fetch / merge."""
    ap = argparse.ArgumentParser(description="Мост к Roboflow Auto Label (SAM 3)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="выгрузить маски одного прохода")
    f.add_argument("--project", required=True)
    f.add_argument("--workspace", default="ban-nzbro")
    f.add_argument("--images-json", required=True, help="JSON {имя_файла: image_id}")
    f.add_argument("--label", default=None, help="оставить только этот класс учителя")
    f.add_argument("--maps-to", default=None, help="переименовать класс (hose -> wide_hose)")
    f.add_argument("--out", required=True)

    m = sub.add_parser("merge", help="слияние проходов (обёртка над src.pipeline.fuse)")
    m.add_argument("--passes-dir", required=True)
    m.add_argument("--frames-dir", required=True)
    m.add_argument("--config", default="configs/ontology.gse_heater.yaml")
    m.add_argument("--out-dir", required=True)
    m.add_argument("--qa-limit", type=int, default=40)

    args = ap.parse_args()
    if args.cmd == "fetch":
        images = json.loads(Path(args.images_json).read_text(encoding="utf-8"))
        recs = fetch_pass(args.project, images, args.label, args.workspace, args.maps_to)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
        total = sum(len(r["masks"]) for r in recs)
        print(f"fetch: {len(recs)} кадров, {total} масок (label={args.label}) -> {args.out}")
    elif args.cmd == "merge":
        from src.pipeline.fuse import fuse                     # локальный импорт: тяжёлые зависимости
        from src.pipeline.ontology import Ontology

        summary = fuse(args.passes_dir, Ontology.load(args.config), args.out_dir,
                       args.frames_dir, args.qa_limit)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
