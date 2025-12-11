FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# Установка системных зависимостей для OpenCV и других библиотек
# libgl1-mesa-glx и libglib2.0-0 обязательны для cv2
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Копируем requirements и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код (для продакшн сборки, в dev перекрывается volume-ом)
COPY src/ ./src/

# Настройка PYTHONPATH
# Используем явное определение, чтобы избежать предупреждений линтера об undefined variable
ENV PYTHONPATH="/app/src"

# Команда по умолчанию (можно переопределить в docker-compose)
CMD ["python3"]


