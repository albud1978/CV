#!/bin/bash
# ============================================
# Скрипт загрузки всех моделей для CV проекта
# Запуск: ./scripts/download_models.sh
# ============================================

set -e

echo "╔════════════════════════════════════════╗"
echo "║  Загрузка моделей CV Pipeline          ║"
echo "╚════════════════════════════════════════╝"
echo ""

# Определяем корень проекта
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ============================================
# 1. YOLO модели (Ultralytics)
# ============================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 1/3: YOLO модели (~220 МБ)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mkdir -p src/models/yolo

YOLO_MODELS=(
    "yolov8l.pt|https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l.pt"
    "yolov8l-seg.pt|https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-seg.pt"
    "yolo11l.pt|https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt"
    "yolov8l-worldv2.pt|https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-worldv2.pt"
)

for entry in "${YOLO_MODELS[@]}"; do
    file="${entry%%|*}"
    url="${entry##*|}"
    if [ -f "src/models/yolo/$file" ]; then
        echo "  ✓ $file уже существует"
    else
        echo "  → Загружаю $file..."
        wget -q --show-progress -c "$url" -O "src/models/yolo/$file"
    fi
done

# ============================================
# 2. SAM 2 модели
# ============================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 2/3: SAM 2 модели (~176 МБ)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mkdir -p src/models/sam2

SAM_MODELS=(
    "sam2.1_hiera_small.pt|https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
)

for entry in "${SAM_MODELS[@]}"; do
    file="${entry%%|*}"
    url="${entry##*|}"
    if [ -f "src/models/sam2/$file" ]; then
        echo "  ✓ $file уже существует"
    else
        echo "  → Загружаю $file..."
        wget -q --show-progress -c "$url" -O "src/models/sam2/$file"
    fi
done

# ============================================
# 3. Molmo2-4B (Vision-Language Model)
# ============================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 3/3: Molmo2-4B (~19 ГБ)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚠️  Большая модель! Загрузка займёт время."
echo ""

MOLMO_DIR="src/models/Molmo2-4B"
MOLMO_URL="https://huggingface.co/allenai/Molmo2-4B/resolve/main"

mkdir -p "$MOLMO_DIR"

# Проверяем наличие основных весов
if [ -f "$MOLMO_DIR/model-00001-of-00004.safetensors" ] && \
   [ -f "$MOLMO_DIR/model-00004-of-00004.safetensors" ]; then
    echo "  ✓ Molmo2-4B уже загружена"
else
    echo "  → Загружаю Molmo2-4B..."
    
    MOLMO_FILES=(
        "config.json"
        "configuration_molmo2.py"
        "generation_config.json"
        "image_processing_molmo2.py"
        "modeling_molmo2.py"
        "model.safetensors.index.json"
        "model-00001-of-00004.safetensors"
        "model-00002-of-00004.safetensors"
        "model-00003-of-00004.safetensors"
        "model-00004-of-00004.safetensors"
        "preprocessor_config.json"
        "processing_molmo2.py"
        "special_tokens_map.json"
        "tokenizer_config.json"
        "tokenizer.json"
        "video_processing_molmo2.py"
        "vocab.json"
        "merges.txt"
        "added_tokens.json"
        "chat_template.jinja"
    )
    
    cd "$MOLMO_DIR"
    for file in "${MOLMO_FILES[@]}"; do
        if [ -f "$file" ]; then
            echo "    ✓ $file"
        else
            echo "    → $file..."
            wget -q --show-progress -c "${MOLMO_URL}/${file}" -O "$file" || echo "    ⚠ Ошибка: $file"
        fi
    done
    cd "$PROJECT_ROOT"
fi

# ============================================
# Итог
# ============================================
echo ""
echo "╔════════════════════════════════════════╗"
echo "║  ✅ Загрузка завершена!                ║"
echo "╚════════════════════════════════════════╝"
echo ""
echo "Структура моделей:"
echo "  src/models/"
echo "  ├── yolo/      $(ls -1 src/models/yolo/*.pt 2>/dev/null | wc -l) файлов"
echo "  ├── sam2/      $(ls -1 src/models/sam2/*.pt 2>/dev/null | wc -l) файлов"
echo "  └── Molmo2-4B/ $(ls -1 src/models/Molmo2-4B/*.safetensors 2>/dev/null | wc -l)/4 файлов"
echo ""
echo "Запуск проекта:"
echo "  docker compose up -d"
echo "  docker compose exec cv-dev python3 -c \"from ultralytics import YOLO; print(YOLO('src/models/yolo/yolo11l.pt'))\""
