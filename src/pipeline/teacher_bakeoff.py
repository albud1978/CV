"""СЛОЙ 2 (бейк-офф) — сравнение моделей-учителей open-vocabulary на наборе кадров.

Прогоняет один и тот же текстовый промпт через несколько учителей и сохраняет
аннотированные изображения для визуального сравнения человеком.

Поддерживаемые учителя (через transformers, без тяжёлых сборок):
    grounding_dino — IDEA-Research/grounding-dino-base (Apache 2.0), боксы по фразе.
    florence2      — microsoft/Florence-2-large (MIT), open-vocabulary detection.

Использование:
    python -m src.pipeline.teacher_bakeoff \\
        --frames /app/data/bakeoff/frames \\
        --out /app/data/bakeoff/out \\
        --teacher grounding_dino \\
        --prompt "ground air conditioning unit . flexible hose . ground crew person"

См. docs/PIPELINE_ARCHITECTURE.md, СЛОЙ 2.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch


def _annotate(image_bgr, boxes_xyxy, labels, scores) -> np.ndarray:
    """Рисует боксы с подписями на изображении."""
    out = image_bgr.copy()
    for (x1, y1, x2, y2), label, score in zip(boxes_xyxy, labels, scores):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, (0, 200, 0), 2)
        text = f"{label} {score:.2f}"
        cv2.rectangle(out, (p1[0], p1[1] - 18), (p1[0] + 9 * len(text), p1[1]), (0, 200, 0), -1)
        cv2.putText(out, text, (p1[0] + 2, p1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def run_grounding_dino(frames, out_dir, prompt, box_threshold=0.25, text_threshold=0.20):
    """Запускает Grounding DINO (transformers) на наборе кадров."""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "IDEA-Research/grounding-dino-base"
    print(f"Загрузка {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    summary = []
    for frame_path in frames:
        image = cv2.imread(frame_path)
        if image is None:
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = processor(images=rgb, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[rgb.shape[:2]],
        )[0]
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results.get("text_labels", results.get("labels"))
        labels = [str(l) for l in labels]

        annotated = _annotate(image, boxes, labels, scores)
        name = Path(frame_path).name
        cv2.imwrite(str(Path(out_dir) / name), annotated)
        summary.append({"frame": name, "n_det": len(boxes),
                        "max_score": float(scores.max()) if len(scores) else 0.0})
    return summary


def run_florence2(frames, out_dir, prompt):
    """Запускает Florence-2 open-vocabulary detection (transformers)."""
    from transformers import AutoProcessor, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model_id = "microsoft/Florence-2-large"
    print(f"Загрузка {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True, attn_implementation="eager"
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    task = "<OPEN_VOCABULARY_DETECTION>"
    summary = []
    for frame_path in frames:
        image = cv2.imread(frame_path)
        if image is None:
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil = Image.fromarray(rgb)
        inputs = processor(text=task + prompt, images=pil, return_tensors="pt").to(device, dtype)
        with torch.no_grad():
            gen = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=1,
                do_sample=False,
                use_cache=False,
            )
        text = processor.batch_decode(gen, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            text, task=task, image_size=(pil.width, pil.height)
        )
        det = parsed.get(task, {})
        boxes = det.get("bboxes", [])
        labels = det.get("bboxes_labels", det.get("labels", ["obj"] * len(boxes)))
        scores = [1.0] * len(boxes)  # Florence-2 не возвращает score

        annotated = _annotate(image, boxes, labels, scores)
        name = Path(frame_path).name
        cv2.imwrite(str(Path(out_dir) / name), annotated)
        summary.append({"frame": name, "n_det": len(boxes)})
    return summary


TEACHERS = {
    "grounding_dino": run_grounding_dino,
    "florence2": run_florence2,
}


def main() -> None:
    """CLI: запуск выбранного учителя на наборе кадров."""
    parser = argparse.ArgumentParser(description="Бейк-офф учителей open-vocabulary (СЛОЙ 2).")
    parser.add_argument("--frames", required=True, help="Директория с кадрами")
    parser.add_argument("--out", required=True, help="Базовая директория для аннотаций")
    parser.add_argument("--teacher", required=True, choices=list(TEACHERS.keys()))
    parser.add_argument("--prompt", required=True, help="Текстовый промпт (классы через ' . ')")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    args = parser.parse_args()

    frames = sorted(glob.glob(os.path.join(args.frames, "*.jpg")))
    if not frames:
        raise ValueError(f"Кадры не найдены: {args.frames}")

    out_dir = Path(args.out) / args.teacher
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Учитель: {args.teacher} | кадров: {len(frames)} | промпт: {args.prompt!r}")
    fn = TEACHERS[args.teacher]
    if args.teacher == "grounding_dino":
        summary = fn(frames, out_dir, args.prompt,
                     box_threshold=args.box_threshold, text_threshold=args.text_threshold)
    else:
        summary = fn(frames, out_dir, args.prompt)

    total_det = sum(s["n_det"] for s in summary)
    frames_with_det = sum(1 for s in summary if s["n_det"] > 0)
    print(f"\nИтог [{args.teacher}]: кадров с детекцией {frames_with_det}/{len(frames)}, "
          f"всего детекций {total_det}")
    with open(out_dir / "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
