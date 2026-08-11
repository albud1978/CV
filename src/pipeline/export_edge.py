"""СЛОЙ 5 — экспорт студента под on-edge: ONNX / NCNN / TFLite (+ замер задержки).

Требование продукта: на выходе ЛОКАЛЬНАЯ модель, способная работать на телефоне.
Практика по целям:

    * **NCNN** — лучший вариант для Android-CPU: маленький рантайм, стабильные
      int8/fp16, не тянет за собой TF. Ultralytics экспортирует напрямую.
    * **TFLite int8** — если приложение уже на TF Lite / нужен NNAPI-делегат.
      INT8 требует калибровочной выборки: берём кадры СВОЕГО датасета, иначе
      диапазоны активаций уедут и точность просядет сильнее необходимого.
    * **CoreML** — iOS.
    * **ONNX** — эталон для проверки паритета и для десктопного инференса.

После экспорта прогоняется замер задержки на CPU (onnxruntime, если он есть):
цифра нужна, чтобы решить n/s и разрешение, а не гадать.

Использование:
    python -m src.pipeline.export_edge --weights runs/heater/train/weights/best.pt \\
        --formats ncnn tflite onnx --imgsz 640 --data data/yolo_heater/data.yaml --int8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def export(
    weights: str,
    formats: list[str],
    imgsz: int,
    data: str | None = None,
    int8: bool = False,
    half: bool = False,
) -> dict[str, str]:
    """Экспортирует веса в перечисленные форматы.

    Args:
        weights: Путь к `best.pt`.
        formats: Список форматов (`onnx`, `ncnn`, `tflite`, `coreml`, `engine`).
        imgsz: Разрешение экспорта (должно совпадать с рантаймом!).
        data: `data.yaml` — источник калибровочных кадров для int8.
        int8: Квантование в int8 (требует `data` для tflite).
        half: FP16 (для onnx/engine на GPU).

    Returns:
        ``{формат: путь}`` по успешно экспортированным моделям.
    """
    from ultralytics import YOLO   # тяжёлый импорт — только при реальном запуске

    model = YOLO(weights)
    produced: dict[str, str] = {}
    for fmt in formats:
        kwargs: dict[str, Any] = {"format": fmt, "imgsz": imgsz}
        if fmt in {"tflite", "engine"} and int8:
            kwargs["int8"] = True
            if data:
                kwargs["data"] = data     # калибровка на своих кадрах, не на COCO
        if fmt in {"onnx", "engine"} and half:
            kwargs["half"] = True
        if fmt == "onnx":
            kwargs["opset"] = 12          # широкая совместимость мобильных рантаймов
            kwargs["simplify"] = True
        print(f"[экспорт] {fmt} imgsz={imgsz} int8={kwargs.get('int8', False)}")
        try:
            produced[fmt] = str(model.export(**kwargs))
        except Exception as exc:  # noqa: BLE001 — один формат не должен рушить остальные
            print(f"    ОШИБКА {fmt}: {type(exc).__name__}: {exc}")
    return produced


def benchmark_onnx(onnx_path: str, imgsz: int, runs: int = 30) -> dict[str, float] | None:
    """Замеряет задержку ONNX-модели на CPU (грубый прокси мобильной производительности).

    Returns:
        ``{mean_ms, p95_ms, fps}`` или None, если onnxruntime недоступен.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime не установлен — замер пропущен")
        return None

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    dummy = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
    session.run(None, {name: dummy})       # прогрев

    times = []
    for _ in range(runs):
        started = time.perf_counter()
        session.run(None, {name: dummy})
        times.append((time.perf_counter() - started) * 1000)
    times.sort()
    mean = sum(times) / len(times)
    return {
        "mean_ms": round(mean, 2),
        "p95_ms": round(times[int(len(times) * 0.95) - 1], 2),
        "fps": round(1000 / mean, 1),
    }


def main() -> None:
    """CLI экспорта и замера."""
    ap = argparse.ArgumentParser(description="СЛОЙ 5 — экспорт студента на edge")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--formats", nargs="+", default=["onnx", "ncnn", "tflite"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--data", default=None, help="data.yaml для int8-калибровки")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--bench-runs", type=int, default=30)
    args = ap.parse_args()

    if args.int8 and "tflite" in args.formats and not args.data:
        print("ВНИМАНИЕ: int8 без --data калибруется на случайных данных — точность просядет")

    produced = export(args.weights, args.formats, args.imgsz, args.data, args.int8, args.half)
    report: dict[str, Any] = {"weights": args.weights, "imgsz": args.imgsz, "exports": produced}

    if "onnx" in produced:
        bench = benchmark_onnx(produced["onnx"], args.imgsz, args.bench_runs)
        if bench:
            report["cpu_benchmark_onnx"] = bench
            print(f"CPU ONNX: {bench['mean_ms']} мс (p95 {bench['p95_ms']}), ~{bench['fps']} FPS")

    out = Path(args.weights).parent / "export_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nОтчёт: {out}")


if __name__ == "__main__":
    main()
