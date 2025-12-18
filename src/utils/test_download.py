import os
import requests
import sys
from tqdm import tqdm

def download_file(url, dest):
    print(f"Downloading {url} to {dest}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1MB
    
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    with open(dest, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(dest)) as pbar:
            for data in response.iter_content(block_size):
                f.write(data)
                pbar.update(len(data))
    print(f"\nFinished downloading {dest}")

if __name__ == "__main__":
    # URL for a small file to test
    test_url = "https://huggingface.co/allenai/Molmo2-4B/resolve/main/config.json"
    test_dest = "src/models/molmo2_4b/config.json"
    download_file(test_url, test_dest)

