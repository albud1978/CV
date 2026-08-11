"""Чтение манифеста кадров (`frames_manifest.parquet|jsonl`) и сборка кадров-записей.

Манифест — контракт происхождения кадра (видео, split, время). Он связывает
нарезку (`build_dataset.py`) с разметкой, обучением и событийной логикой:
без `video_id` split течёт, без `timestamp` невозможно построить события.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def load_manifest(dataset_root: str | Path) -> dict[str, dict[str, Any]]:
    """Загружает манифест и индексирует его по имени файла кадра.

    Args:
        dataset_root: Корень датасета, где лежит ``frames_manifest.*``.

    Returns:
        ``{имя_файла: {video_id, split, timestamp, image_path, ...}}``.
        Пустой словарь, если манифеста нет (кадры всё равно можно размечать —
        просто без времени и без гарантий split).
    """
    root = Path(dataset_root)
    rows: list[dict[str, Any]] = []

    parquet = root / "frames_manifest.parquet"
    jsonl = root / "frames_manifest.jsonl"
    if parquet.exists():
        try:
            import pandas as pd

            rows = pd.read_parquet(parquet).to_dict("records")
        except Exception:
            rows = []
    if not rows and jsonl.exists():
        rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line]

    return {Path(r["image_path"]).name: r for r in rows}


def find_frames(dataset_root: str | Path, split: Optional[str] = None) -> list[Path]:
    """Возвращает пути ко всем кадрам датасета (опционально — одного split).

    Раскладка, которую делает ``build_dataset.py``: ``<root>/<split>/<video>/*.jpg``.
    Плоская папка кадров тоже поддерживается.
    """
    root = Path(dataset_root)
    pattern = f"{split}/**/*.jpg" if split else "**/*.jpg"
    frames = sorted(p for p in root.glob(pattern) if p.is_file())
    if not frames and not split:
        frames = sorted(root.glob("*.jpg"))
    return frames


def frame_meta(name: str, manifest: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Метаданные кадра из манифеста с безопасными значениями по умолчанию."""
    row = manifest.get(name, {})
    return {
        "video_id": row.get("video_id", "unknown"),
        "split": row.get("split", "unassigned"),
        "timestamp": float(row.get("timestamp", 0.0) or 0.0),
        "frame_idx": int(row.get("frame_idx", 0) or 0),
        "source": row.get("source", "unknown"),
    }


def group_by_video(frames: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Группирует кадры по ``video_id`` и сортирует каждую группу по времени.

    Временная логика (треки, события) обязана работать внутри одного видео:
    склейка кадров разных камер в одну ленту даёт бессмысленные переходы.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for f in frames:
        out.setdefault(f.get("video_id", "unknown"), []).append(f)
    for video_id in out:
        out[video_id].sort(key=lambda f: (float(f.get("timestamp", 0.0)), f.get("image_name", "")))
    return out
