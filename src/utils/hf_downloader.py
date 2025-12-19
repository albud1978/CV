import os
import sys
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

def download_model(repo_id):
    print(f"Fetching file list for {repo_id}...")
    files = list_repo_files(repo_id)
    
    # Sort files to download smaller ones first to show progress quickly
    # But for models, it's better to just go through them.
    # We filter out .git files if any.
    files = [f for f in files if not f.startswith(".")]
    
    print(f"Found {len(files)} files to download.")
    
    for filename in files:
        print(f"\n--- Downloading {filename} ---")
        try:
            # hf_hub_download automatically uses tqdm if it's not disabled
            # and it will show the progress bar if the environment allows it.
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                resume_download=True
            )
            print(f"✓ Saved to {path}")
        except Exception as e:
            print(f"✘ Error downloading {filename}: {e}")

if __name__ == "__main__":
    repo = "allenai/Molmo2-4B"
    if len(sys.argv) > 1:
        repo = sys.argv[1]
    download_model(repo)

