"""СЛОЙ 1 — Ingest: извлечение и дедупликация кадров из видео.

Пайплайн кадра:
    видео -> PySceneDetect (AdaptiveDetector) -> кадры-представители сцен
          -> perceptual hash (pHash) дедупликация
          -> сохранение кадров в output/ + frames_manifest.parquet

Манифест фиксирует происхождение каждого кадра (video_id, scene_id, timestamp, hash,
split-заготовка), что нужно для воспроизводимости и stratified-split по video_id.

Зависимости (опциональные, ставятся отдельно):
    scenedetect, imagehash, pandas, pyarrow

Использование:
    python -m src.pipeline.ingest \\
        --input "input/Видео Рощино 23.04.25/04_Ст 1_камера 50 2025-04-15T06.52.01-7.08.15.mkv" \\
        --output output/frames/rwith_04 \\
        --max-frames 300

См. docs/PIPELINE_ARCHITECTURE.md, СЛОЙ 1.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2

# --- Опциональные зависимости: не валим импорт модуля, если их нет ---
try:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import AdaptiveDetector, ContentDetector
    _SCENEDETECT_AVAILABLE = True
except ImportError:
    _SCENEDETECT_AVAILABLE = False

try:
    import imagehash
    from PIL import Image
    _IMAGEHASH_AVAILABLE = True
except ImportError:
    _IMAGEHASH_AVAILABLE = False

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}


@dataclass
class FrameRecord:
    """Запись о кадре в манифесте.

    Attributes:
        video_id: Идентификатор исходного видео (имя файла без расширения).
        scene_id: Порядковый номер сцены в видео.
        frame_idx: Индекс кадра в исходном видео.
        timestamp: Временная метка кадра в секундах.
        phash: Perceptual hash кадра (hex-строка) или None, если imagehash недоступен.
        image_path: Путь к сохранённому кадру.
        split: Заготовка под train/val/test (заполняется на этапе версионирования).
    """

    video_id: str
    scene_id: int
    frame_idx: int
    timestamp: float
    phash: Optional[str]
    image_path: str
    split: str = "unassigned"


def _detect_scene_frames(video_path: str, min_scene_len: int = 15) -> list[tuple[int, float]]:
    """Находит кадры-представители сцен через PySceneDetect.

    Args:
        video_path: Путь к видеофайлу.
        min_scene_len: Минимальная длина сцены в кадрах.

    Returns:
        Список кортежей (frame_idx, timestamp_sec) — середина каждой сцены.
        Если PySceneDetect недоступен, список пуст (вызывающий код использует fallback).
    """
    if not _SCENEDETECT_AVAILABLE:
        return []

    video = open_video(video_path)
    scene_manager = SceneManager()
    # AdaptiveDetector устойчив к плавному движению камеры (типично для перрона).
    scene_manager.add_detector(AdaptiveDetector(min_scene_len=min_scene_len))
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()

    fps = video.frame_rate
    representatives: list[tuple[int, float]] = []
    for start, end in scene_list:
        start_f = start.get_frames()
        end_f = end.get_frames()
        mid_f = (start_f + end_f) // 2
        representatives.append((mid_f, mid_f / fps if fps else 0.0))
    return representatives


def _sample_fixed_fps(video_path: str, sample_fps: float = 1.0) -> list[tuple[int, float]]:
    """Fallback-сэмплер: равномерная выборка кадров с заданным FPS.

    Используется, если PySceneDetect не установлен.

    Args:
        video_path: Путь к видеофайлу.
        sample_fps: Сколько кадров в секунду извлекать.

    Returns:
        Список кортежей (frame_idx, timestamp_sec).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    step = max(1, int(round(fps / sample_fps)))
    return [(idx, idx / fps) for idx in range(0, total, step)]


def _compute_phash(frame_bgr) -> Optional[str]:
    """Считает perceptual hash кадра.

    Args:
        frame_bgr: Кадр в формате BGR (OpenCV).

    Returns:
        Hex-строка pHash или None, если imagehash недоступен.
    """
    if not _IMAGEHASH_AVAILABLE:
        return None
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return str(imagehash.phash(Image.fromarray(rgb)))


def _is_duplicate(phash_hex: str, seen: list[str], threshold: int) -> bool:
    """Проверяет, является ли кадр near-duplicate по расстоянию Хэмминга.

    Args:
        phash_hex: pHash нового кадра.
        seen: Список уже принятых pHash.
        threshold: Максимальное расстояние Хэмминга для признания дубликатом.

    Returns:
        True, если найден достаточно похожий кадр.
    """
    if not _IMAGEHASH_AVAILABLE:
        return False
    h = imagehash.hex_to_hash(phash_hex)
    for prev in seen:
        if (h - imagehash.hex_to_hash(prev)) <= threshold:
            return True
    return False


def ingest_video(
    video_path: str,
    output_dir: str,
    max_frames: int = 300,
    phash_threshold: int = 6,
    sample_fps: float = 1.0,
    min_scene_len: int = 15,
) -> list[FrameRecord]:
    """Извлекает и дедуплицирует кадры из одного видео.

    Args:
        video_path: Путь к видеофайлу.
        output_dir: Директория для сохранения кадров и манифеста.
        max_frames: Верхний предел числа сохраняемых кадров.
        phash_threshold: Порог Хэмминга для дедупликации (меньше -> строже).
        sample_fps: FPS для fallback-сэмплера (если нет PySceneDetect).
        min_scene_len: Минимальная длина сцены для AdaptiveDetector.

    Returns:
        Список FrameRecord по сохранённым кадрам.
    """
    video_path_obj = Path(video_path)
    video_id = video_path_obj.stem
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    candidates = _detect_scene_frames(video_path, min_scene_len=min_scene_len)
    if not candidates:
        if _SCENEDETECT_AVAILABLE:
            print("PySceneDetect не нашёл сцен — fallback на равномерный сэмплинг.")
        else:
            print("PySceneDetect не установлен — fallback на равномерный сэмплинг по FPS.")
        candidates = _sample_fixed_fps(video_path, sample_fps=sample_fps)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    records: list[FrameRecord] = []
    seen_hashes: list[str] = []
    dup_count = 0

    for scene_id, (frame_idx, ts) in enumerate(candidates):
        if len(records) >= max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        phash_hex = _compute_phash(frame)
        if phash_hex is not None and _is_duplicate(phash_hex, seen_hashes, phash_threshold):
            dup_count += 1
            continue
        if phash_hex is not None:
            seen_hashes.append(phash_hex)

        image_name = f"{video_id}_f{frame_idx:08d}.jpg"
        image_path = out / image_name
        cv2.imwrite(str(image_path), frame)

        records.append(
            FrameRecord(
                video_id=video_id,
                scene_id=scene_id,
                frame_idx=frame_idx,
                timestamp=round(ts, 3),
                phash=phash_hex,
                image_path=str(image_path),
            )
        )

    cap.release()

    total_candidates = len(candidates)
    dup_ratio = dup_count / total_candidates if total_candidates else 0.0
    print(
        f"[{video_id}] кандидатов: {total_candidates}, сохранено: {len(records)}, "
        f"дублей отсеяно: {dup_count} ({dup_ratio:.1%})"
    )
    if dup_ratio > 0.05 and _IMAGEHASH_AVAILABLE:
        print("  Gate G0: доля дублей > 5% среди кандидатов — это нормально для статичной камеры.")

    return records


def save_manifest(records: list[FrameRecord], output_dir: str) -> str:
    """Сохраняет манифест кадров (parquet при наличии pandas, иначе JSONL).

    Args:
        records: Список записей о кадрах.
        output_dir: Директория для манифеста.

    Returns:
        Путь к сохранённому файлу манифеста.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in records]

    if _PANDAS_AVAILABLE:
        try:
            path = out / "frames_manifest.parquet"
            pd.DataFrame(rows).to_parquet(path, index=False)
            return str(path)
        except Exception as exc:  # pyarrow может отсутствовать
            print(f"Parquet недоступен ({exc}); пишу JSONL.")

    path = out / "frames_manifest.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(path)


def process(
    input_path: str,
    output_dir: str,
    max_frames: int = 300,
    phash_threshold: int = 6,
    sample_fps: float = 1.0,
) -> str:
    """Обрабатывает видеофайл или директорию с видео.

    Args:
        input_path: Путь к видеофайлу или директории.
        output_dir: Директория для кадров и манифеста.
        max_frames: Лимит кадров на ОДНО видео.
        phash_threshold: Порог дедупликации.
        sample_fps: FPS для fallback-сэмплера.

    Returns:
        Путь к сохранённому манифесту.
    """
    in_path = Path(input_path)
    if in_path.is_file():
        videos = [in_path] if in_path.suffix.lower() in VIDEO_EXTENSIONS else []
    else:
        videos = sorted(
            f for f in in_path.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        )

    if not videos:
        raise ValueError(f"Видео не найдены: {input_path}")

    all_records: list[FrameRecord] = []
    for video in videos:
        print(f"\nОбработка: {video.name}")
        all_records.extend(
            ingest_video(
                str(video),
                output_dir,
                max_frames=max_frames,
                phash_threshold=phash_threshold,
                sample_fps=sample_fps,
            )
        )

    manifest_path = save_manifest(all_records, output_dir)
    print(f"\nИтого кадров: {len(all_records)}. Манифест: {manifest_path}")
    return manifest_path


def main() -> None:
    """CLI-точка входа для СЛОЯ 1 (ingest)."""
    parser = argparse.ArgumentParser(
        description="Ingest: извлечение и дедупликация кадров из видео (СЛОЙ 1)."
    )
    parser.add_argument("--input", "-i", required=True, help="Видеофайл или директория с видео")
    parser.add_argument("--output", "-o", required=True, help="Директория для кадров и манифеста")
    parser.add_argument("--max-frames", "-n", type=int, default=300, help="Лимит кадров на видео")
    parser.add_argument(
        "--phash-threshold", "-p", type=int, default=6,
        help="Порог Хэмминга для дедупликации (меньше -> строже)"
    )
    parser.add_argument(
        "--sample-fps", "-f", type=float, default=1.0,
        help="FPS для fallback-сэмплера, если PySceneDetect не установлен"
    )
    args = parser.parse_args()

    if not _SCENEDETECT_AVAILABLE or not _IMAGEHASH_AVAILABLE:
        print(
            "ВНИМАНИЕ: для полного пайплайна установите зависимости:\n"
            "  pip install scenedetect[opencv] imagehash pandas pyarrow\n"
            "Сейчас часть функций работает в fallback-режиме.\n"
        )

    process(
        input_path=args.input,
        output_dir=args.output,
        max_frames=args.max_frames,
        phash_threshold=args.phash_threshold,
        sample_fps=args.sample_fps,
    )


if __name__ == "__main__":
    main()
