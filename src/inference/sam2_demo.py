import os
import torch
import cv2
import numpy as np
import argparse
import sys

# Проверка импорта SAM 2
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("Error: SAM 2 not installed. Please check requirements.")
    sys.exit(1)

def download_sam2_weights(model_cfg, output_dir="src/models"):
    """
    Заглушка или реальная загрузка. Веса SAM 2 нужно качать отдельно.
    URL зависит от конфига.
    """
    # Large model
    url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
    name = "sam2_hiera_large.pt"
    dest = os.path.join(output_dir, name)
    
    if not os.path.exists(dest):
        print(f"Downloading {name}...")
        os.system(f"curl -L {url} -o {dest}")
    return dest

def run_sam2(image_path, prompt_type="box", prompt_data=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Config and Checkpoint
    # В репо SAM 2 конфиги лежат в sam2/configs/sam2
    # Нам нужно найти путь к конфигу внутри установленного пакета или использовать имя
    model_cfg = "sam2_hiera_l.yaml" 
    checkpoint = download_sam2_weights(model_cfg)
    
    # Load model
    print("Loading SAM 2...")
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    
    # Read Image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    predictor.set_image(image_rgb)
    
    masks = None
    scores = None
    logits = None
    
    if prompt_type == "box":
        # prompt_data = [x1, y1, x2, y2]
        box = np.array(prompt_data)
        masks, scores, logits = predictor.predict(
            box=box,
            multimask_output=False
        )
    elif prompt_type == "point":
        # prompt_data = [x, y]
        point_coords = np.array([prompt_data])
        point_labels = np.array([1]) # 1 = foreground
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False
        )
    
    if masks is None:
        print("Error: No valid prompt type provided (use 'box' or 'point') or prediction failed.")
        return

    # Visualize
    # masks shape: (1, H, W)
    mask = masks[0]
    
    # Накладываем маску
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = [0, 255, 0] # Green
    
    result = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
    
    if prompt_type == "box":
        x1, y1, x2, y2 = prompt_data
        cv2.rectangle(result, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        
    out_path = "sam2_result.jpg"
    cv2.imwrite(out_path, result)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--box", nargs=4, type=int, help="Box prompt: x1 y1 x2 y2")
    parser.add_argument("--point", nargs=2, type=int, help="Point prompt: x y")
    args = parser.parse_args()
    
    if args.box:
        run_sam2(args.image, "box", args.box)
    elif args.point:
        run_sam2(args.image, "point", args.point)
    else:
        print("Please provide --box or --point")

