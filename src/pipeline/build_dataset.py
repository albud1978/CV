"""Драйвер сборки датасета по configs/dataset_split.yaml.

Связывает разбиение train/val/test (по ВИДЕО, без утечки) с нарезкой кадров:
    - позитивы берём ВНУТРИ positive_window_sec (где присутствует обогреватель)
      стратегией initial+motion (см. src.pipeline.ingest);
    - негативы добираем равномерно ВНЕ окна (пустой перрон / без шланга);
    - каждый кадр штампуется split и video_id -> единый манифест без утечки.

Кадры раскладываются по split-подпапкам: <output>/<split>/<video_id>/...
Манифест: <output>/frames_manifest.(parquet|jsonl).

Использование:
    python -m src.pipeline.build_dataset \\
        --config configs/dataset_split.yaml \\
        --output data/dataset_heater \\
        --max-positive 80 --max-negative 15

ВАЖНО: вывод пишем на ext4 (data/), НЕ в output/ (Nextcloud-монтирование /mnt/c) —
там cv2-запись из долгого процесса нестабильна (молчаливый False).
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import yaml

from src.pipeline.ingest import (
    FrameCandidate,
    FrameRecord,
    sample_initial_scene,
    sample_motion_segments,
    _sample_fixed_fps,
    _compute_phash,
    _is_duplicate,
    _video_meta,
    save_manifest,
)
import cv2


def _repair_video(video_path: str, max_sec: float, out_root: str = "data/_repaired") -> str:
    """Перекодирует видео в чистый mp4, отбрасывая повреждённые пакеты (ffmpeg).

    OpenCV `cap.read()` на битом GOP не возвращает False, а БЛОКИРУЕТСЯ в декодере
    (зависание навсегда). ffmpeg с `+discardcorrupt`/`ignore_err` дропает такие пакеты
    и продолжает. Кодируем только до max_sec (+буфер) — нужный кусок.

    Returns:
        Путь к восстановленному файлу (или исходный, если ffmpeg недоступен/ошибка).
    """
    src = Path(video_path)
    out_dir = Path(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / (src.stem + ".mp4")
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  repair: используем готовую копию {dst.name}")
        return str(dst)
    dur = int(max_sec + 30)
    cmd = [
        "ffmpeg", "-y", "-fflags", "+discardcorrupt", "-err_detect", "ignore_err",
        "-i", str(src), "-t", str(dur),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-an",
        str(dst),
    ]
    print(f"  repair: ffmpeg чистит поток (до {dur}c) -> {dst.name} ...")
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        print(f"  repair: НЕ удалось ({r.returncode}); используем исходный файл")
        return video_path
    return str(dst)


def _save_image(path: Path, frame) -> bool:
    """Надёжная запись кадра: imencode + бинарная запись (юникод-пути, проверка результата).

    cv2.imwrite на drvfs/Nextcloud-монтировании и с не-ASCII путями молча возвращает
    False. Поэтому кодируем в память и пишем сами; затем проверяем, что файл создан.
    """
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return False
    path.write_bytes(buf.tobytes())
    return path.exists() and path.stat().st_size > 0


def _frame_idx_from_name(name: str) -> int:
    """Извлекает индекс кадра из имени вида ``<video>_<source>_f00012345.jpg``.

    Нужно для «резюме»: кадры уже на диске, а исходные метаданные потеряны —
    индекс в имени остаётся единственным источником времени кадра.
    """
    m = re.search(r"_f(\d+)\.jpg$", name)
    return int(m.group(1)) if m else 0


def _in_windows(ts: float, windows: list[list]) -> bool:
    """True, если timestamp попадает в любое из окон [start, end] (end=None -> до конца)."""
    for start, end in windows:
        if ts >= (start or 0) and (end is None or ts <= end):
            return True
    return False


def _sample_negatives(video_path: str, windows: list[list], n: int) -> list[FrameCandidate]:
    """Равномерно сэмплирует n кадров ВНЕ позитивных окон (кандидаты в негативы)."""
    fps, total = _video_meta(video_path)
    if n <= 0 or total <= 0:
        return []
    step = max(1, total // (n * 3))  # с запасом — часть отсеется окнами/дедупом
    out: list[FrameCandidate] = []
    for idx in range(0, total, step):
        ts = idx / fps
        if not _in_windows(ts, windows):
            out.append(FrameCandidate(idx, ts, "negative", -1, 0.0))
    return out[:: max(1, len(out) // n)][:n] if out else []


def build_video(
    video_path: str,
    output_dir: str,
    split: str,
    windows: list[list],
    max_positive: int,
    max_negative: int,
    phash_threshold: int,
    motion_fps: float,
    motion_threshold: float,
    repair: bool = False,
) -> list[FrameRecord]:
    """Нарезает один видеофайл: позитивы внутри окон + негативы вне, штампует split."""
    video_id = Path(video_path).stem
    out = Path(output_dir) / split / video_id
    out.mkdir(parents=True, exist_ok=True)

    # Граница анализа: конец последнего окна присутствия (open-ended -> старт+45мин).
    ends = [w[1] if w[1] is not None else (w[0] or 0) + 45 * 60 for w in windows]
    max_sec = max(ends) if ends else 0

    # При --repair заранее чистим поток через ffmpeg (обход зависаний на битых GOP).
    if repair:
        video_path = _repair_video(video_path, max_sec)

    # Граница анализа движения в кадрах: ускоряет MOG2 и не декодит ненужный хвост.
    fps, total = _video_meta(video_path)
    max_frame = min(total, int((max_sec + 30) * fps)) if ends else total

    # Позитивы: initial + motion, отфильтрованные по окнам присутствия.
    print(f"  [{split}/{video_id}] позитивы (initial+motion, до кадра {max_frame})...")
    pos = sample_initial_scene(video_path, n_frames=10, seconds=30.0)
    pos += sample_motion_segments(
        video_path, motion_fps=motion_fps, motion_threshold=motion_threshold,
        max_frame=max_frame,
    )
    pos = [c for c in pos if _in_windows(c.timestamp, windows)]
    if len(pos) <= 1:  # движение не нашлось внутри окна -> равномерный добор
        fb = [c for c in _sample_fixed_fps(video_path, sample_fps=0.5)
              if _in_windows(c.timestamp, windows)]
        pos += fb

    # Негативы вне окон.
    neg = _sample_negatives(video_path, windows, max_negative)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    records: list[FrameRecord] = []
    seen: list[str] = []
    for group, limit in ((sorted(pos, key=lambda c: c.frame_idx), max_positive),
                         (neg, max_negative)):
        saved_in_group = 0
        processed: set[int] = set()
        for cand in group:
            if saved_in_group >= limit:
                break
            if cand.frame_idx in processed:
                continue
            processed.add(cand.frame_idx)
            cap.set(cv2.CAP_PROP_POS_FRAMES, cand.frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            ph = _compute_phash(frame)
            if ph is not None and _is_duplicate(ph, seen, phash_threshold):
                continue
            name = f"{video_id}_{cand.source}_f{cand.frame_idx:08d}.jpg"
            if not _save_image(out / name, frame):
                print(f"    ВНИМАНИЕ: не удалось записать {name}")
                continue
            if ph is not None:
                seen.append(ph)
            records.append(FrameRecord(
                video_id=video_id, source=cand.source, event_id=cand.event_id,
                motion_score=round(cand.motion_score, 4), frame_idx=cand.frame_idx,
                timestamp=round(cand.timestamp, 3), phash=ph,
                image_path=str(out / name), split=split,
            ))
            saved_in_group += 1

    cap.release()
    n_pos = sum(1 for r in records if r.source != "negative")
    n_neg = sum(1 for r in records if r.source == "negative")
    print(f"  [{split}/{video_id}] сохранено: позитивов={n_pos}, негативов={n_neg}")
    return records


def main() -> None:
    """CLI: собрать датасет по dataset_split.yaml."""
    ap = argparse.ArgumentParser(description="Сборка датасета по dataset_split.yaml")
    ap.add_argument("--config", "-c", default="configs/dataset_split.yaml")
    ap.add_argument("--output", "-o", required=True, help="Корень датасета")
    ap.add_argument("--max-positive", type=int, default=80, help="Лимит позитивов на видео")
    ap.add_argument("--max-negative", type=int, default=15, help="Лимит негативов на видео")
    ap.add_argument("--phash-threshold", "-p", type=int, default=3)
    ap.add_argument("--motion-fps", type=float, default=1.0)
    ap.add_argument("--motion-threshold", type=float, default=0.3)
    ap.add_argument("--only", help="Обработать только видео с этим id (например v06)")
    ap.add_argument("--force", action="store_true",
                    help="Пере-нарезать даже если кадры уже есть (по умолчанию пропускаем готовые)")
    ap.add_argument("--repair", action="store_true",
                    help="Чинить битые видео через ffmpeg перед нарезкой (обход зависаний декодера)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    folder = Path(cfg["folder"])
    smp = cfg.get("sampling", {})
    neg_default = smp.get("negatives_per_video", args.max_negative)

    all_records: list[FrameRecord] = []
    for v in cfg["videos"]:
        if args.only and v["id"] != args.only:
            continue
        vp = folder / v["file"]
        if not vp.exists():
            print(f"ПРОПУСК: нет файла {vp}")
            continue
        out_dir = Path(args.output) / v["split"] / Path(v["file"]).stem
        existing = list(out_dir.glob("*.jpg")) if out_dir.exists() else []
        if existing and not args.force:
            print(f"\n=== {v['id']} [{v['split']}] УЖЕ ГОТОВ ({len(existing)} кадров) — пропуск ===")
            # Восстанавливаем frame_idx из имени файла и время по fps видео: без
            # временных меток манифест бесполезен для событийной логики (events.py).
            try:
                fps, _ = _video_meta(str(vp))
            except Exception:
                fps = 25.0
            all_records += [FrameRecord(
                video_id=out_dir.name, source=("negative" if "negative" in p.name else "motion"),
                event_id=-1, motion_score=0.0,
                frame_idx=_frame_idx_from_name(p.name),
                timestamp=_frame_idx_from_name(p.name) / max(fps, 1e-6),
                phash=None, image_path=str(p), split=v["split"],
            ) for p in existing]
            continue
        print(f"\n=== {v['id']} [{v['split']}] {v['camera']} | {v['condition']} ===")
        all_records += build_video(
            video_path=str(vp),
            output_dir=args.output,
            split=v["split"],
            windows=v["positive_window_sec"],
            max_positive=args.max_positive,
            max_negative=neg_default,
            phash_threshold=args.phash_threshold,
            motion_fps=args.motion_fps,
            motion_threshold=args.motion_threshold,
            repair=args.repair,
        )

    save_manifest(all_records, args.output)
    by_split: dict[str, int] = {}
    for r in all_records:
        by_split[r.split] = by_split.get(r.split, 0) + 1
    print(f"\nИтого кадров: {len(all_records)} | по split: {by_split}")


if __name__ == "__main__":
    main()
