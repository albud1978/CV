import argparse
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import os
import sys

def run_molmo_demo(image_path, prompt, model_id="allenai/Molmo-7B-D-0924"):
    """
    Запуск инференса Molmo 2 (или совместимой версии).
    По умолчанию используется Molmo-7B-D-0924, так как Molmo-2 может требовать специфичной версии.
    Если пользователь укажет явно Molmo-2, будет использован он.
    """
    print(f"Loading Molmo model: {model_id}...")
    
    # Определяем устройство
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        # Загрузка процессора и модели
        # trust_remote_code=True часто нужен для новых моделей AllenAI
        processor = AutoProcessor.from_pretrained(
            model_id, 
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map='auto'
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map='auto'
        )
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        print("Ensure you have transformers>=4.46.0 and accelerate installed.")
        return

    # Загрузка изображения
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
        
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Подготовка инпутов
    # Molmo process methods might vary, standard is similar to Qwen/Llava
    inputs = processor.process(
        images=[image],
        text=prompt
    )
    
    # Перенос на устройство
    inputs = {k: v.to(device).unsqueeze(0) for k, v in inputs.items()}
    
    # Генерация
    print("Generating response...")
    with torch.no_grad():
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )

    # Декодирование
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    response = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    print("-" * 20)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print("-" * 20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Molmo VLM inference.")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.", help="Text prompt")
    parser.add_argument("--model", type=str, default="allenai/Molmo-7B-D-0924", help="Hugging Face model ID (e.g. allenai/Molmo-2-4B)")
    
    args = parser.parse_args()
    
    run_molmo_demo(args.image, args.prompt, args.model)

