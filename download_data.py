import os
import urllib.request

# 1. Ensure the destination directory folder path exists
os.makedirs("data/raw", exist_ok=True)
target_path = "data/raw/candidates.jsonl"

# 2. PASTE YOUR CONVERTED DIRECT DOWNLOAD LINK HERE
download_url = "https://drive.google.com/file/d/1tB7Otd2EGldaDRu62cVnIoW6_G8A_HaT/view?usp=share_link"

print("Downloading candidates.jsonl dataset (487MB) from mirror storage...")
try:
    # This downloads the file directly into your data path
    urllib.request.urlretrieve(download_url, target_path)
    print("✨ Download complete! File successfully saved to data/raw/candidates.jsonl")
except Exception as e:
    print(f"❌ Error downloading file: {e}")
