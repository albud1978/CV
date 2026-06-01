"""СЛОЙ 1 — Ingest: автоматическая нарезка видео на кадры для обучения.

Стратегия для статичных камер наблюдения (перрон):
    1. Начальная сцена (initial): N кадров из первых секунд видео — фон/establishing,
       полезны как negatives и для понимания пустой сцены.
    2. Наборы по движению (motion): MOG2 (src/utils/motion_detect.py) находит интервалы
       реальной активности; внутри каждого события кадры сэмплируются с motion_fps.
    3. Дедупликация по perceptual hash (pHash) с мягким порогом — внутри набора убираем
       почти идентичные кадры, но сохраняем разнообразие поз/ракурсов одного события.
    4. Манифест frames_manifest.parquet фиксирует происхождение каждого кадра
       (source, event_id, motion_score, timestamp) для воспроизводимости.

Зависимости (опциональные): scenedetect, imagehash, pandas, pyarrow.

Использование:
    python -m src.pipeline.ingest \\
        --input "/app/input/Видео Рощино 23.04.25/01_Ст 1_камера 50 2025-04-14T05.57.01-7.07.01.mkv" \\
        --output /app/output/frames/r01 \\
        --mode motion --max-frames 400

См. docs/PIPELINE_ARCHITECTURE.md, СЛОЙ 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2

# Импорт MotionDetector с подстраховкой пути (на случай запуска не через -m).
try:
    from src.utils.motion_detect import MotionDetector
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.motion_detect import MotionDetector

# --- Опциональные зависимости: не валим импорт модуля, если их нет ---
try:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import AdaptiveDetector
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
class FrameCandidate:
    """Кадр-кандидат до сохранения и дедупликации."""

    frame_idx: int
    timestamp: float
    source: str          # 'initial' | 'motion' | 'scene' | 'fps'
    event_id: int        # id события движения (-1 для не-motion)
    motion_score: float  # средний % движения в событии (0.0 для не-motion)


@dataclass
class FrameRecord:
    """Запись о сохранённом кадре в манифесте.

    Attributes:
        video_id: Идентификатор исходного видео (имя файла без расширения).
        source: Источник кадра ('initial' | 'motion' | 'scene' | 'fps').
        event_id: Id события движения (-1, если кадр не из motion-набора).
        motion_score: Средний % движения в событии.
        frame_idx: Индекс кадра в исходном видео.
        timestamp: Временная метка кадра в секундах.
        phash: Perceptual hash (hex) или None, если imagehash недоступен.
        image_path: Путь к сохранённому кадру.
        split: Заготовка под train/val/test (заполняется на версионировании).
    """

    video_id: str
    source: str
    event_id: int
    motion_score: float
    frame_idx: int
    timestamp: float
    phash: Optional[str]
    image_path: str
    split: str = "unassigned"


def _video_meta(video_path: str) -> tuple[float, int]:
    """Возвращает (fps, total_frames) видео."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total


def sample_initial_scene(
    video_path: str, n_frames: int = 10, seconds: float = 30.0
) -> list[FrameCandidate]:
    """Равномерно сэмплирует кадры из начала видео (establishing-сцена).

    Args:
        video_path: Путь к видео.
        n_frames: Сколько кадров взять из начала.
        seconds: Длина начального окна в секундах.

    Returns:
        Список кандидатов с source='initial'.
    """
    fps, total = _video_meta(video_path)
    window = min(int(seconds * fps), total)
    if window <= 0 or n_frames <= 0:
        return []
    step = max(1, window // n_frames)
    return [
        FrameCandidate(idx, idx / fps, "initial", -1, 0.0)
        for idx in range(0, window, step)
    ][:n_frames]


def sample_motion_segments(
    video_path: str,
    motion_fps: float = 2.0,
    motion_threshold: float = 0.3,
    min_duration: float = 1.0,
    buffer_seconds: float = 3.0,
    gap_threshold: float = 5.0,
    max_frame: int | None = None,
) -> list[FrameCandidate]:
    """Находит интервалы движения (MOG2) и сэмплирует кадры внутри них.

    Args:
        video_path: Путь к видео.
        motion_fps: Сколько кадров в секунду извлекать ВНУТРИ события движения.
        motion_threshold: Порог движения (% площади кадра) для MOG2.
        min_duration: Минимальная длительность события (сек).
        buffer_seconds: Буфер после прекращения движения (сек).
        gap_threshold: Слияние событий, если пауза между ними меньше (сек).

    Returns:
        Список кандидатов с source='motion', проставленными event_id и motion_score.
    """
    detector = MotionDetector(
        motion_threshold=motion_threshold,
        min_duration=min_duration,
        buffer_seconds=buffer_seconds,
        gap_threshold=gap_threshold,
    )
    segments, meta = detector.analyze_video(video_path, max_frame=max_frame)
    fps = meta["fps"] or 25.0
    step = max(1, int(round(fps / motion_fps)))

    candidates: list[FrameCandidate] = []
    for event_id, seg in enumerate(segments):
        for idx in range(seg.start_frame, seg.end_frame + 1, step):
            candidates.append(
                FrameCandidate(idx, idx / fps, "motion", event_id, float(seg.avg_motion))
            )
    print(f"  Найдено событий движения: {len(segments)}, кадров-кандидатов: {len(candidates)}")
    return candidates


def _sample_fixed_fps(video_path: str, sample_fps: float = 1.0) -> list[FrameCandidate]:
    """Fallback-сэмплер: равномерная выборка кадров с заданным FPS."""
    fps, total = _video_meta(video_path)
    step = max(1, int(round(fps / sample_fps)))
    return [FrameCandidate(idx, idx / fps, "fps", -1, 0.0) for idx in range(0, total, step)]


def _compute_phash(frame_bgr) -> Optional[str]:
    """Считает perceptual hash кадра (hex) или None, если imagehash недоступен."""
    if not _IMAGEHASH_AVAILABLE:
        return None
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return str(imagehash.phash(Image.fromarray(rgb)))


def _is_duplicate(phash_hex: str, seen: list[str], threshold: int) -> bool:
    """True, если кадр near-duplicate по расстоянию Хэмминга к уже принятым."""
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
    mode: str = "motion",
    max_frames: int = 400,
    phash_threshold: int = 3,
    initial_frames: int = 10,
    initial_seconds: float = 30.0,
    motion_fps: float = 2.0,
    motion_threshold: float = 0.3,
    sample_fps: float = 1.0,
) -> list[FrameRecord]:
    """Извлекает и дедуплицирует кадры из одного видео.

    Args:
        video_path: Путь к видеофайлу.
        output_dir: Директория для кадров и манифеста.
        mode: 'motion' (initial + motion, по умолчанию) | 'fps' (равномерно).
        max_frames: Верхний предел числа сохраняемых кадров.
        phash_threshold: Порог Хэмминга для дедупликации (меньше -> мягче, больше кадров).
        initial_frames: Сколько кадров взять из начальной сцены.
        initial_seconds: Длина начального окна (сек).
        motion_fps: FPS сэмплинга внутри событий движения.
        motion_threshold: Порог MOG2 (% площади кадра).
        sample_fps: FPS для режима 'fps'.

    Returns:
        Список FrameRecord по сохранённым кадрам.
    """
    video_id = Path(video_path).stem
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Сбор кандидатов согласно режиму.
    candidates: list[FrameCandidate] = []
    if mode == "motion":
        print("Начальная сцена (initial)...")
        candidates += sample_initial_scene(video_path, initial_frames, initial_seconds)
        print("Анализ движения (MOG2)...")
        candidates += sample_motion_segments(
            video_path, motion_fps=motion_fps, motion_threshold=motion_threshold
        )
        if len(candidates) <= initial_frames:
            print("Движение не обнаружено — добавляю равномерный fps-сэмплинг как fallback.")
            candidates += _sample_fixed_fps(video_path, sample_fps=sample_fps)
    else:
        candidates = _sample_fixed_fps(video_path, sample_fps=sample_fps)

    # Сортируем по индексу кадра для последовательного чтения.
    candidates.sort(key=lambda c: c.frame_idx)

    # 2. Чтение, дедуп, сохранение.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    records: list[FrameRecord] = []
    seen_hashes: list[str] = []
    dup_count = 0
    processed_idx = set()

    for cand in candidates:
        if len(records) >= max_frames:
            break
        if cand.frame_idx in processed_idx:
            continue
        processed_idx.add(cand.frame_idx)

        cap.set(cv2.CAP_PROP_POS_FRAMES, cand.frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        phash_hex = _compute_phash(frame)
        if phash_hex is not None and _is_duplicate(phash_hex, seen_hashes, phash_threshold):
            dup_count += 1
            continue
        if phash_hex is not None:
            seen_hashes.append(phash_hex)

        image_name = f"{video_id}_{cand.source}_f{cand.frame_idx:08d}.jpg"
        image_path = out / image_name
        cv2.imwrite(str(image_path), frame)

        records.append(
            FrameRecord(
                video_id=video_id,
                source=cand.source,
                event_id=cand.event_id,
                motion_score=round(cand.motion_score, 4),
                frame_idx=cand.frame_idx,
                timestamp=round(cand.timestamp, 3),
                phash=phash_hex,
                image_path=str(image_path),
            )
        )

    cap.release()

    n_initial = sum(1 for r in records if r.source == "initial")
    n_motion = sum(1 for r in records if r.source == "motion")
    print(
        f"[{video_id}] сохранено: {len(records)} "
        f"(initial={n_initial}, motion={n_motion}), дублей отсеяно: {dup_count}"
    )
    return records


def save_manifest(records: list[FrameRecord], output_dir: str) -> str:
    """Сохраняет манифест кадров (parquet при наличии pandas/pyarrow, иначе JSONL)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in records]

    if _PANDAS_AVAILABLE:
        try:
            path = out / "frames_manifest.parquet"
            pd.DataFrame(rows).to_parquet(path, index=False)
            return str(path)
        except Exception as exc:
            print(f"Parquet недоступен ({exc}); пишу JSONL.")

    path = out / "frames_manifest.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(path)


def process(
    input_path: str,
    output_dir: str,
    mode: str = "motion",
    max_frames: int = 400,
    phash_threshold: int = 3,
    initial_frames: int = 10,
    initial_seconds: float = 30.0,
    motion_fps: float = 2.0,
    motion_threshold: float = 0.3,
    sample_fps: float = 1.0,
) -> str:
    """Обрабатывает видеофайл или директорию с видео; возвращает путь к манифесту."""
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
                mode=mode,
                max_frames=max_frames,
                phash_threshold=phash_threshold,
                initial_frames=initial_frames,
                initial_seconds=initial_seconds,
                motion_fps=motion_fps,
                motion_threshold=motion_threshold,
                sample_fps=sample_fps,
            )
        )

    manifest_path = save_manifest(all_records, output_dir)
    print(f"\nИтого кадров: {len(all_records)}. Манифест: {manifest_path}")
    return manifest_path


def main() -> None:
    """CLI-точка входа для СЛОЯ 1 (ingest)."""
    parser = argparse.ArgumentParser(
        description="Ingest: нарезка видео на кадры (начальная сцена + движение)."
    )
    parser.add_argument("--input", "-i", required=True, help="Видеофайл или директория с видео")
    parser.add_argument("--output", "-o", required=True, help="Директория для кадров и манифеста")
    parser.add_argument(
        "--mode", choices=["motion", "fps"], default="motion",
        help="motion = начальная сцена + наборы по движению (default); fps = равномерно"
    )
    parser.add_argument("--max-frames", "-n", type=int, default=400, help="Лимит кадров на видео")
    parser.add_argument(
        "--phash-threshold", "-p", type=int, default=3,
        help="Порог Хэмминга дедупликации (меньше -> мягче, сохраняем больше кадров)"
    )
    parser.add_argument("--initial-frames", type=int, default=10, help="Кадров из начальной сцены")
    parser.add_argument("--initial-seconds", type=float, default=30.0, help="Окно начальной сцены, сек")
    parser.add_argument("--motion-fps", type=float, default=2.0, help="FPS сэмплинга внутри движения")
    parser.add_argument("--motion-threshold", type=float, default=0.3, help="Порог MOG2, %% площади")
    parser.add_argument("--sample-fps", "-f", type=float, default=1.0, help="FPS для режима 'fps'")
    args = parser.parse_args()

    if not _IMAGEHASH_AVAILABLE:
        print("ВНИМАНИЕ: imagehash не установлен — дедупликация отключена.")
    if not _PANDAS_AVAILABLE:
        print("ВНИМАНИЕ: pandas не установлен — манифест будет в JSONL.")

    process(
        input_path=args.input,
        output_dir=args.output,
        mode=args.mode,
        max_frames=args.max_frames,
        phash_threshold=args.phash_threshold,
        initial_frames=args.initial_frames,
        initial_seconds=args.initial_seconds,
        motion_fps=args.motion_fps,
        motion_threshold=args.motion_threshold,
        sample_fps=args.sample_fps,
    )


if __name__ == "__main__":
    main()
