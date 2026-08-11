"""СЛОЙ 4 — обучение студента (Ultralytics YOLO) + метрики и карточка модели.

Студент — YOLO11-seg, а не RF-DETR, по требованию продукта: на выходе нужна
ЛОКАЛЬНАЯ модель, экспортируемая на телефон (ONNX -> NCNN/TFLite). Маски нужны
логике подключения (концы рукава, толщина), поэтому дефолт — `segment`; ветка
`detect` остаётся для быстрого первого прогона.

Гиперпараметры берутся из `configs/augment_conservative.yaml` (imgsz=1280,
close_mosaic=10, mixup/erasing/copy_paste=0) — политика зафиксирована правилами
проекта под мелкие удалённые объекты на перроне.

После обучения считается val/test и пишутся:
    metrics.json  — mAP50, mAP50-95, per-class, сравнение с порогами приёмки;
    model_card.md — паспорт модели (данные, ограничения, лицензии, метрики).

Лицензия: Ultralytics YOLO — AGPL-3.0. Для встраивания в продукт нужна
коммерческая лицензия Ultralytics либо замена студента на RF-DETR (Apache-2.0).

Использование:
    python -m src.pipeline.train --data data/yolo_heater/data.yaml \\
        --task segment --arch yolo11s-seg --out runs/heater_v1
    python -m src.pipeline.train --data data/yolo_heater/data.yaml --edge   # под телефон
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.pipeline.ontology import Ontology

# Ключи augment-конфига, которые Ultralytics принимает как есть.
PASSTHROUGH_KEYS = {
    "epochs", "batch", "optimizer", "lr0", "cos_lr", "mosaic", "close_mosaic",
    "mixup", "copy_paste", "erasing", "hsv_h", "hsv_s", "hsv_v", "fliplr",
    "flipud", "scale", "translate", "perspective", "label_smoothing", "amp",
}


def load_train_cfg(path: str | Path) -> dict[str, Any]:
    """Читает политику аугментаций и отфильтровывает неизвестные ключи."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {k: v for k, v in raw.items() if k in PASSTHROUGH_KEYS}


def _metrics_dict(results: Any, names: dict[int, str]) -> dict[str, Any]:
    """Вытаскивает mAP и per-class AP из результата ultralytics val."""
    box = getattr(results, "box", None)
    seg = getattr(results, "seg", None)
    out: dict[str, Any] = {}
    for prefix, metric in (("box", box), ("mask", seg)):
        if metric is None:
            continue
        out[prefix] = {
            "map50": round(float(metric.map50), 4),
            "map50_95": round(float(metric.map), 4),
            "precision": round(float(metric.mp), 4),
            "recall": round(float(metric.mr), 4),
            "per_class_ap50": {
                names.get(int(c), str(c)): round(float(v), 4)
                for c, v in zip(getattr(metric, "ap_class_index", []), metric.ap50)
            },
        }
    return out


def write_model_card(
    path: Path,
    onto: Ontology,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    dataset_summary: dict[str, Any],
    weights: str,
) -> None:
    """Пишет паспорт модели рядом с весами."""
    primary = metrics.get("test", {}).get("mask") or metrics.get("test", {}).get("box") or {}
    acc = onto.acceptance
    map50 = primary.get("map50")
    verdict = "—"
    if map50 is not None:
        threshold = float(acc.get("map50_min", 0.75))
        verdict = "ПРОЙДЕН" if map50 >= threshold else f"НЕ ПРОЙДЕН (нужно ≥ {threshold})"

    lines = [
        f"# Карточка модели: {args.arch} ({args.task})",
        "",
        f"- **Дата обучения:** {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- **Веса:** `{weights}`",
        f"- **Данные:** `{args.data}`",
        f"- **Разрешение:** {args.imgsz}, эпох: {args.epochs}",
        f"- **Платформа:** {platform.platform()}",
        "",
        "## Назначение",
        "",
        "Детекция наземного обогревателя ВС и его широкого гофрошланга на перроне; "
        "выход модели используется логикой `relation.py` + `events.py` для получения "
        "тайм-кодов подключения/отключения рукава.",
        "",
        "## Классы",
        "",
        *[f"- `{i}` — {label}" for i, label in enumerate(onto.trainable_labels())],
        "",
        "## Метрики (test split — отдельные съёмки, новые условия)",
        "",
        "```json",
        json.dumps(metrics.get("test", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        f"**Гейт приёмки mAP@50:** {verdict}",
        "",
        "## Данные",
        "",
        "```json",
        json.dumps(dataset_summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Ограничения",
        "",
        "- Разметка получена авто-разметкой (учитель SAM 3) с автоматической отбраковкой; "
        "часть меток не проверялась человеком — см. `rejections.jsonl` и golden-set.",
        "- Split сделан по видео: метрики отражают перенос на НОВЫЕ съёмки тех же камер. "
        "Перенос на другие аэропорты/камеры не проверялся.",
        "- Ночь и дождь представлены одним-двумя видео — статистика по ним слабая.",
        "- Состояние «подключён» при отсутствии детекции ВС определяется резервным "
        "правилом по присутствию рукава (пониженная уверенность).",
        "",
        "## Лицензии",
        "",
        f"{onto.student.get('license_note', '').strip()}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """CLI обучения студента."""
    ap = argparse.ArgumentParser(description="СЛОЙ 4 — обучение студента YOLO")
    ap.add_argument("--data", required=True, help="data.yaml из to_yolo.py")
    ap.add_argument("--ontology", default="configs/ontology.gse_heater.yaml")
    ap.add_argument("--augment-config", default="configs/augment_conservative.yaml")
    ap.add_argument("--task", choices=["detect", "segment"], default="segment")
    ap.add_argument("--arch", default=None, help="Переопределить архитектуру онтологии")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="runs/heater", help="project-директория ultralytics")
    ap.add_argument("--name", default="train")
    ap.add_argument("--edge", action="store_true",
                    help="Пресет под телефон: лёгкая архитектура и малое разрешение")
    args = ap.parse_args()

    onto = Ontology.load(args.ontology)
    student = onto.student
    args.arch = args.arch or (student.get("edge_arch") if args.edge else student.get("arch"))
    args.imgsz = args.imgsz or int(student.get("edge_imgsz" if args.edge else "imgsz", 640))

    cfg = load_train_cfg(args.augment_config)
    cfg["imgsz"] = args.imgsz
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.batch:
        cfg["batch"] = args.batch
    args.epochs = cfg.get("epochs", 100)

    from ultralytics import YOLO   # тяжёлый импорт — только при реальном запуске

    weights = args.arch if args.arch.endswith(".pt") else f"{args.arch}.pt"
    print(f"Студент: {weights} | задача: {args.task} | imgsz={args.imgsz} | эпох={args.epochs}")
    model = YOLO(weights)
    model.train(data=args.data, project=args.out, name=args.name, device=args.device,
                exist_ok=True, **cfg)

    names = {i: n for i, n in enumerate(onto.trainable_labels())}
    metrics = {
        "val": _metrics_dict(model.val(data=args.data, split="val", imgsz=args.imgsz), names),
        "test": _metrics_dict(model.val(data=args.data, split="test", imgsz=args.imgsz), names),
    }

    run_dir = Path(args.out) / args.name
    best = run_dir / "weights" / "best.pt"
    data_cfg = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    dataset_summary = {"data_yaml": args.data, "classes": data_cfg.get("names")}

    (run_dir / "metrics.json").write_text(
        json.dumps({"arch": args.arch, "task": args.task, "imgsz": args.imgsz,
                    "acceptance": onto.acceptance, **metrics},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_model_card(run_dir / "model_card.md", onto, args, metrics, dataset_summary, str(best))

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nВеса: {best}\nМетрики: {run_dir}/metrics.json\nКарточка: {run_dir}/model_card.md")


if __name__ == "__main__":
    main()
