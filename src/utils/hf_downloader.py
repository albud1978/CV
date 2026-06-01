import argparse
import os
import sys
import time
from typing import List, Optional

from huggingface_hub import list_repo_files, snapshot_download


def _default_cache_dir() -> str:
    # Prefer HF_HOME if set (we mount it as a persistent volume in docker-compose)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return os.path.join(hf_home, "hub")
    return os.path.expanduser("~/.cache/huggingface/hub")


def _sleep_backoff(attempt: int) -> None:
    # 5, 10, 20, 40, 60...
    delay = min(60, 5 * (2 ** max(0, attempt - 1)))
    print(f"⏳ Жду {delay}с перед повтором...", flush=True)
    time.sleep(delay)


def download_repo(
    repo_id: str,
    local_dir: str,
    cache_dir: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
    max_workers: int = 2,
    retries: int = 8,
    etag_timeout: int = 60,
) -> str:
    """
    Устойчивое скачивание репозитория с HuggingFace с поддержкой resume/retry.

    По умолчанию качаем в локальную директорию (в идеале — примонтированную),
    а кеш — в персистентную папку HF_HOME/.hf чтобы переживать пересоздание контейнера.
    """
    cache_dir = cache_dir or _default_cache_dir()
    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Repo: {repo_id}")
    print(f"Local dir: {local_dir}")
    print(f"Cache dir: {cache_dir}")
    if allow_patterns:
        print(f"Allow patterns: {allow_patterns}")

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            return snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                allow_patterns=allow_patterns,
                max_workers=max_workers,
                etag_timeout=etag_timeout,
            )
        except Exception as e:
            last_err = e
            print(f"✘ Попытка {attempt}/{retries} не удалась: {e}", flush=True)
            _sleep_backoff(attempt)

    raise RuntimeError(f"Не удалось скачать {repo_id} после {retries} попыток: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Устойчивое скачивание моделей с HuggingFace")
    parser.add_argument("--repo", type=str, default="allenai/Molmo2-4B", help="HF repo_id (model)")
    parser.add_argument(
        "--local-dir",
        type=str,
        default="/app/src/models/molmo2-4b",
        help="Куда складывать файлы модели (лучше примонтированная папка)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Куда складывать кеш HF Hub (по умолчанию HF_HOME/hub или ~/.cache/huggingface/hub)",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Скачать только веса (model-*.safetensors) и индекс",
    )
    parser.add_argument("--max-workers", type=int, default=2, help="Параллелизм скачивания")
    parser.add_argument("--retries", type=int, default=8, help="Количество попыток при ошибках сети")
    parser.add_argument("--etag-timeout", type=int, default=60, help="Таймаут для HEAD/etag")

    args = parser.parse_args()

    # Важно: чтобы избежать cas-bridge/xet, можно задать HF_HUB_DISABLE_XET=1
    # (в docker-compose мы включаем HF_HUB_ENABLE_HF_TRANSFER=1 для ускорения).
    if os.environ.get("HF_HUB_DISABLE_XET") == "1":
        print("HF_HUB_DISABLE_XET=1 (Xet отключен)", flush=True)

    allow_patterns = None
    if args.weights_only:
        allow_patterns = ["model-*.safetensors", "model.safetensors.index.json"]
        # заранее покажем, что именно хотим качать
        try:
            files = [f for f in list_repo_files(args.repo) if not f.startswith(".")]
            weights = [f for f in files if f.startswith("model-") and f.endswith(".safetensors")]
            print(f"Найдено весов: {len(weights)} ({', '.join(weights[:4])}{'...' if len(weights) > 4 else ''})")
        except Exception as e:
            print(f"Не удалось получить список файлов (не критично): {e}")

    download_repo(
        repo_id=args.repo,
        local_dir=args.local_dir,
        cache_dir=args.cache_dir,
        allow_patterns=allow_patterns,
        max_workers=args.max_workers,
        retries=args.retries,
        etag_timeout=args.etag_timeout,
    )
    print("✓ Готово", flush=True)


if __name__ == "__main__":
    main()

