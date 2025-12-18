#!/bin/bash
# Скрипт установки Docker + NVIDIA Container Toolkit на Ubuntu 24.04
# Запуск: chmod +x install_docker.sh && ./install_docker.sh

set -e  # Остановка при ошибке

echo "=== 1. Удаление старых версий Docker ==="
sudo apt remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

echo "=== 2. Установка зависимостей ==="
sudo apt update
sudo apt install -y ca-certificates curl gnupg

echo "=== 3. Добавление GPG-ключа Docker ==="
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "=== 4. Добавление репозитория Docker ==="
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

echo "=== 5. Установка Docker Engine ==="
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "=== 6. Добавление пользователя в группу docker ==="
sudo usermod -aG docker $USER

echo "=== 7. Установка NVIDIA Container Toolkit ==="
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit

echo "=== 8. Настройка Docker для GPU ==="
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo ""
echo "============================================"
echo "✅ Docker успешно установлен!"
echo "============================================"
echo ""
echo "⚠️  ВАЖНО: Перелогиньтесь или выполните:"
echo "    newgrp docker"
echo ""
echo "Затем проверьте установку:"
echo "    docker --version"
echo "    docker compose version"
echo "    docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi"
echo ""






