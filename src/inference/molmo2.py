"""
Molmo 2 — Vision-Language Model для понимания изображений и видео.

Модели от Allen Institute for AI (Ai2):
- allenai/Molmo-7B-D-0924 (7B параметров, ~14 ГБ VRAM)
- allenai/Molmo-72B-0924 (72B параметров, требует несколько GPU)

Возможности:
- Ответы на вопросы по изображениям/видео
- Pointing (указание координат объектов)
- Подсчёт объектов
- OCR (чтение текста)

Использование:
    python src/inference/molmo2.py --image test.jpg --prompt "Опиши что на изображении" --quant 4bit
"""

import os
import sys
import argparse
import torch
from PIL import Image
from typing import Optional, List, Union

# Проверка импортов
try:
    from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Run: pip install transformers>=4.45.0")


class Molmo2Inference:
    """
    Обёртка для инференса модели Molmo 2.
    """
    
    # Доступные модели Molmo 2 (декабрь 2025)
    MODELS = {
        # Molmo 2 — новые модели
        "4b": "allenai/Molmo2-4B",
        "8b": "allenai/Molmo2-8B", 
        "7b": "allenai/Molmo2-O-7B",
        "video": "allenai/Molmo2-VideoPoint-4B",
        # Molmo 1 — старые модели (для совместимости)
        "molmo1-7b": "allenai/Molmo-7B-D-0924",
    }
    
    def __init__(
        self, 
        model_name: str = "7b",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
        quantization: str = "none"
    ):
        """
        Инициализация модели Molmo 2.
        
        Args:
            model_name: Размер модели ("7b" или "72b")
            device: Устройство ("cuda", "cpu", или None для автовыбора)
            torch_dtype: Тип данных (bfloat16 для экономии памяти)
            trust_remote_code: Доверять коду модели с HuggingFace
            quantization: Режим квантования ("none", "8bit", "4bit")
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required. Install with: pip install transformers>=4.45.0")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        
        # Получаем имя модели
        if model_name in self.MODELS:
            self.model_id = self.MODELS[model_name]
        else:
            self.model_id = model_name  # Позволяет указать полный путь
            
        print(f"Загрузка модели {self.model_id} (Quantization: {quantization})...")
        print(f"Устройство: {self.device}, dtype: {torch_dtype}")
        
        # Настройки квантования
        quantization_config = None
        if quantization == "8bit":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == "4bit":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype
            )

        # Загрузка процессора (токенизатор + image processor)
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map="auto"
        )
        
        # Подготовка аргументов для загрузки модели
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch_dtype,
            "device_map": "auto"
        }
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        # Загрузка модели
        print(f"Попытка загрузки через AutoModelForCausalLM...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs
            )
        except Exception as e:
            print(f"Предупреждение: AutoModelForCausalLM не сработал: {e}")
            print("Попытка загрузки через базовый AutoModel...")
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(
                self.model_id,
                **model_kwargs
            )
        
        print(f"✓ Модель загружена")
        
    def predict(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """
        Выполнить инференс на изображении(ях).
        
        Args:
            images: Путь к изображению, PIL Image, или список изображений
            prompt: Текстовый запрос
            max_new_tokens: Максимальное количество токенов в ответе
            temperature: Температура генерации (0.0 = детерминированный)
            
        Returns:
            str: Ответ модели
        """
        # Нормализация входных изображений
        if isinstance(images, str):
            images = [Image.open(images)]
        elif isinstance(images, Image.Image):
            images = [images]
        else:
            images = [Image.open(img) if isinstance(img, str) else img for img in images]
        
        # Конвертация в RGB
        images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]
        
        # Подготовка входных данных
        inputs = self.processor.process(
            images=images,
            text=prompt
        )
        
        # Перенос на устройство (self.model.device используется как база, но при map="auto" тензоры раскиданы)
        # При device_map="auto" обычно достаточно to(device) для inputs, если модель сама управляет.
        # Но для inputs надо просто to(device) первого слоя или основного устройства.
        # Безопаснее просто to("cuda") если мы на GPU.
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {k: v.to(target_device).unsqueeze(0) for k, v in inputs.items()}
        
        # Генерация
        with torch.no_grad():
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    stop_strings=["<|endoftext|>"],
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                ),
                tokenizer=self.processor.tokenizer
            )
        
        # Декодирование
        generated_tokens = output[0, inputs["input_ids"].size(1):]
        response = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def point(self, image: Union[str, Image.Image], object_description: str) -> dict:
        """
        Найти объект на изображении и вернуть его координаты.
        
        Args:
            image: Изображение
            object_description: Описание объекта ("человек в жилете", "топливозаправщик")
            
        Returns:
            dict: {"found": bool, "coordinates": (x, y) или None, "response": str}
        """
        prompt = f"Point to the {object_description} in the image."
        response = self.predict(image, prompt)
        
        # Парсинг координат из ответа (Molmo возвращает координаты в формате <point x="..." y="...">)
        result = {
            "found": False,
            "coordinates": None,
            "response": response
        }
        
        # Простой парсинг (может потребоваться доработка под формат Molmo 2)
        if "x=" in response and "y=" in response:
            try:
                import re
                x_match = re.search(r'x="?(\d+\.?\d*)"?', response)
                y_match = re.search(r'y="?(\d+\.?\d*)"?', response)
                if x_match and y_match:
                    result["found"] = True
                    result["coordinates"] = (float(x_match.group(1)), float(y_match.group(1)))
            except Exception:
                pass
                
        return result
    
    def describe(self, image: Union[str, Image.Image]) -> str:
        """
        Получить детальное описание изображения.
        """
        return self.predict(image, "Describe this image in detail.")
    
    def count(self, image: Union[str, Image.Image], object_type: str) -> dict:
        """
        Подсчитать количество объектов определённого типа.
        
        Args:
            image: Изображение
            object_type: Тип объекта ("people", "cars", "trucks")
            
        Returns:
            dict: {"count": int или None, "response": str}
        """
        prompt = f"How many {object_type} are in this image? Answer with just the number."
        response = self.predict(image, prompt)
        
        result = {"count": None, "response": response}
        
        # Парсинг числа
        import re
        numbers = re.findall(r'\d+', response)
        if numbers:
            result["count"] = int(numbers[0])
            
        return result
    
    def ocr(self, image: Union[str, Image.Image]) -> str:
        """
        Прочитать текст на изображении.
        """
        return self.predict(image, "Read all text visible in this image.")
    
    def answer(self, image: Union[str, Image.Image], question: str) -> str:
        """
        Ответить на вопрос по изображению.
        """
        return self.predict(image, question)


def main():
    parser = argparse.ArgumentParser(description="Molmo 2 Inference")
    parser.add_argument("--image", type=str, required=True, help="Путь к изображению")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.", 
                        help="Текстовый запрос")
    parser.add_argument("--model", type=str, default="4b", 
                        choices=["4b", "8b", "7b", "video", "molmo1-7b"],
                        help="Модель: 4b, 8b (Molmo2), 7b (Molmo2-O), video (VideoPoint)")
    parser.add_argument("--task", type=str, default="qa", 
                        choices=["qa", "describe", "point", "count", "ocr"],
                        help="Тип задачи")
    parser.add_argument("--object", type=str, default="person",
                        help="Объект для point/count задач")
    parser.add_argument("--quant", type=str, default="none", 
                        choices=["none", "8bit", "4bit"], 
                        help="Режим квантования (экономия VRAM)")
    
    args = parser.parse_args()
    
    # Проверка файла
    if not os.path.exists(args.image):
        print(f"Error: файл {args.image} не найден")
        sys.exit(1)
    
    # Инициализация модели
    model = Molmo2Inference(model_name=args.model, quantization=args.quant)
    
    # Выполнение задачи
    if args.task == "describe":
        result = model.describe(args.image)
        print(f"\n📝 Описание:\n{result}")
        
    elif args.task == "point":
        result = model.point(args.image, args.object)
        print(f"\n📍 Pointing:")
        print(f"   Найден: {result['found']}")
        print(f"   Координаты: {result['coordinates']}")
        print(f"   Ответ: {result['response']}")
        
    elif args.task == "count":
        result = model.count(args.image, args.object)
        print(f"\n🔢 Подсчёт:")
        print(f"   Количество: {result['count']}")
        print(f"   Ответ: {result['response']}")
        
    elif args.task == "ocr":
        result = model.ocr(args.image)
        print(f"\n📖 OCR:\n{result}")
        
    else:  # qa
        result = model.answer(args.image, args.prompt)
        print(f"\n💬 Ответ:\n{result}")


if __name__ == "__main__":
    main()
