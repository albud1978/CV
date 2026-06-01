"""
Скрипт для детекции движения в видео и нарезки сегментов с активностью.

Использует OpenCV Background Subtraction (MOG2) для обнаружения движения.
Подходит для видео с фиксированных камер наблюдения.

Использование:
    python src/utils/motion_detect.py --input input/Video --output output/motion_clips
    
    # С настройкой чувствительности
    python src/utils/motion_detect.py --input input/Video --output output/motion_clips --threshold 0.5 --min-duration 2
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class MotionSegment:
    """Сегмент видео с движением."""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    avg_motion: float  # Средний % движения в сегменте


class MotionDetector:
    """
    Детектор движения на основе Background Subtraction (MOG2).
    """
    
    def __init__(
        self,
        motion_threshold: float = 0.3,
        min_duration: float = 1.0,
        buffer_seconds: float = 60.0,
        gap_threshold: float = 60.0,
        history: int = 500,
        var_threshold: int = 16,
        detect_shadows: bool = False
    ):
        """
        Инициализация детектора.
        
        Args:
            motion_threshold: Порог движения (0-100, % площади кадра с движением)
            min_duration: Минимальная длительность сегмента (секунды)
            buffer_seconds: Буфер после прекращения движения (секунды)
            gap_threshold: Максимальный разрыв для объединения сегментов (секунды)
            history: Количество кадров для обучения фона
            var_threshold: Порог вариации для MOG2
            detect_shadows: Детектировать тени (замедляет работу)
        """
        self.motion_threshold = motion_threshold
        self.min_duration = min_duration
        self.buffer_seconds = buffer_seconds
        self.gap_threshold = gap_threshold
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        
    def analyze_video(
        self, video_path: str, max_frame: int | None = None
    ) -> Tuple[List[MotionSegment], dict]:
        """
        Анализ видео на наличие движения.
        
        Args:
            video_path: Путь к видеофайлу
            max_frame: Если задан — декодируем только первые max_frame кадров
                (ускоряет анализ и обходит битые GOP в хвосте файла).
            
        Returns:
            Tuple[List[MotionSegment], dict]: Список сегментов с движением и метаданные
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
        
        # Метаданные видео
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        metadata = {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration": duration
        }
        
        # Background Subtractor
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows
        )
        
        # Ограничение анализа (окно присутствия / обход битого хвоста)
        limit = total_frames if not max_frame else min(total_frames, max_frame)

        # Анализ кадров
        motion_frames = []  # (frame_idx, motion_percent)
        
        frame_idx = 0
        fail_streak = 0
        pbar = tqdm(total=limit, desc="Анализ движения", unit="кадр")
        
        while frame_idx < limit:
            ret, frame = cap.read()
            if not ret:
                # Один сбой декодирования (битый кадр) не должен ронять весь анализ:
                # пропускаем до 30 подряд, дальше считаем, что поток закончился.
                fail_streak += 1
                if fail_streak > 30:
                    break
                frame_idx += 1
                pbar.update(1)
                continue
            fail_streak = 0
            
            # Применяем background subtraction
            fg_mask = bg_subtractor.apply(frame)
            
            # Морфологические операции для удаления шума
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # Вычисляем процент движения
            motion_pixels = np.count_nonzero(fg_mask)
            total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
            motion_percent = (motion_pixels / total_pixels) * 100
            
            motion_frames.append((frame_idx, motion_percent))
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        # Находим сегменты с движением
        segments = self._find_motion_segments(motion_frames, fps)
        
        return segments, metadata
    
    def _find_motion_segments(
        self, 
        motion_frames: List[Tuple[int, float]], 
        fps: float
    ) -> List[MotionSegment]:
        """
        Находит непрерывные сегменты с движением.
        """
        if not motion_frames:
            return []
        
        segments = []
        in_motion = False
        start_frame = 0
        motion_values = []
        
        buffer_frames = int(self.buffer_seconds * fps)
        min_frames = int(self.min_duration * fps)
        
        for frame_idx, motion_percent in motion_frames:
            if motion_percent >= self.motion_threshold:
                if not in_motion:
                    # Начало движения — БЕЗ буфера назад, сразу с момента обнаружения
                    start_frame = frame_idx
                    in_motion = True
                    motion_values = []
                motion_values.append(motion_percent)
            else:
                if in_motion:
                    # Конец движения (с буфером вперёд)
                    end_frame = min(len(motion_frames) - 1, frame_idx + buffer_frames)
                    
                    # Проверяем минимальную длительность
                    if end_frame - start_frame >= min_frames:
                        segments.append(MotionSegment(
                            start_frame=start_frame,
                            end_frame=end_frame,
                            start_time=start_frame / fps,
                            end_time=end_frame / fps,
                            avg_motion=np.mean(motion_values) if motion_values else 0
                        ))
                    
                    in_motion = False
        
        # Если видео закончилось во время движения
        if in_motion:
            end_frame = len(motion_frames) - 1
            if end_frame - start_frame >= min_frames:
                segments.append(MotionSegment(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_time=start_frame / fps,
                    end_time=end_frame / fps,
                    avg_motion=np.mean(motion_values) if motion_values else 0
                ))
        
        # Объединяем близкие сегменты
        segments = self._merge_close_segments(segments, fps)
        
        return segments
    
    def _merge_close_segments(
        self, 
        segments: List[MotionSegment], 
        fps: float,
        gap_threshold: float = None  # секунды — объединяем если пауза меньше этого значения
    ) -> List[MotionSegment]:
        # Используем значение из конструктора, если не передано явно
        if gap_threshold is None:
            gap_threshold = getattr(self, 'gap_threshold', 60.0)
        """
        Объединяет близко расположенные сегменты.
        """
        if len(segments) < 2:
            return segments
        
        merged = [segments[0]]
        gap_frames = int(gap_threshold * fps)
        
        for seg in segments[1:]:
            last = merged[-1]
            if seg.start_frame - last.end_frame <= gap_frames:
                # Объединяем
                merged[-1] = MotionSegment(
                    start_frame=last.start_frame,
                    end_frame=seg.end_frame,
                    start_time=last.start_time,
                    end_time=seg.end_time,
                    avg_motion=(last.avg_motion + seg.avg_motion) / 2
                )
            else:
                merged.append(seg)
        
        return merged
    
    def extract_segment(
        self,
        video_path: str,
        segment: MotionSegment,
        output_path: str
    ) -> bool:
        """
        Извлекает сегмент видео.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Кодек для записи
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, segment.start_frame)
        
        for _ in range(segment.end_frame - segment.start_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()
        out.release()
        
        return True


def process_videos(
    input_path_str: str,
    output_dir: str,
    motion_threshold: float = 0.3,
    min_duration: float = 1.0,
    buffer_seconds: float = 60.0,
    gap_threshold: float = 60.0
):
    """
    Обрабатывает одно видео или все видео в директории.
    """
    input_path = Path(input_path_str)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Поддерживаемые форматы
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'}
    
    # Если указан файл — обрабатываем только его
    if input_path.is_file():
        if input_path.suffix.lower() in video_extensions:
            video_files = [input_path]
        else:
            print(f"Файл {input_path} не является видео")
            return
    else:
        # Если директория — ищем все видео
        video_files = [
            f for f in input_path.iterdir() 
            if f.is_file() and f.suffix.lower() in video_extensions
        ]
    
    if not video_files:
        print(f"Видео не найдены в {input_path_str}")
        return
    
    print(f"Найдено {len(video_files)} видео")
    print(f"Порог движения: {motion_threshold}%")
    print(f"Мин. длительность: {min_duration}с")
    print(f"Буфер после движения: {buffer_seconds}с")
    print(f"Объединение сегментов (gap): {gap_threshold}с")
    print("-" * 50)
    
    detector = MotionDetector(
        motion_threshold=motion_threshold,
        min_duration=min_duration,
        buffer_seconds=buffer_seconds,
        gap_threshold=gap_threshold
    )
    
    total_segments = 0
    
    for video_file in video_files:
        print(f"\n📹 Обработка: {video_file.name}")
        
        try:
            segments, metadata = detector.analyze_video(str(video_file))
            
            print(f"   Длительность: {metadata['duration']:.1f}с")
            print(f"   Найдено сегментов с движением: {len(segments)}")
            
            if not segments:
                print("   ⚠️  Движение не обнаружено")
                continue
            
            # Извлекаем сегменты
            for i, seg in enumerate(segments):
                output_name = f"{video_file.stem}_motion_{i+1:03d}.mp4"
                output_file = output_path / output_name
                
                success = detector.extract_segment(str(video_file), seg, str(output_file))
                
                if success:
                    duration = seg.end_time - seg.start_time
                    print(f"   ✅ {output_name} ({seg.start_time:.1f}s - {seg.end_time:.1f}s, {duration:.1f}s)")
                    total_segments += 1
                else:
                    print(f"   ❌ Ошибка извлечения: {output_name}")
                    
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
    
    print("\n" + "=" * 50)
    print(f"✅ Готово! Извлечено {total_segments} сегментов")
    print(f"📁 Результаты: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Детекция движения в видео и извлечение активных сегментов"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="/app/input/Video",
        help="Путь к видеофайлу или директории с видео"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="/app/output/motion_clips",
        help="Директория для результатов"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.3,
        help="Порог движения в %% площади кадра (default: 0.3)"
    )
    parser.add_argument(
        "--min-duration", "-m",
        type=float,
        default=5.0,
        help="Минимальная длительность сегмента в секундах (default: 5.0)"
    )
    parser.add_argument(
        "--buffer", "-b",
        type=float,
        default=60.0,
        help="Буфер после прекращения движения в секундах (default: 60.0)"
    )
    parser.add_argument(
        "--gap", "-g",
        type=float,
        default=60.0,
        help="Макс. разрыв для объединения сегментов в секундах (default: 60.0)"
    )
    
    args = parser.parse_args()
    
    process_videos(
        input_path_str=args.input,
        output_dir=args.output,
        motion_threshold=args.threshold,
        min_duration=args.min_duration,
        buffer_seconds=args.buffer,
        gap_threshold=args.gap
    )


if __name__ == "__main__":
    main()

