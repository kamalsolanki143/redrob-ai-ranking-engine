import os
import gdown

FILE_ID = "1tB7Otd2EGldaDRu62cVnIoW6_G8A_HaT"
DEST_PATH = "data/raw/candidates.jsonl"

os.makedirs(os.path.dirname(DEST_PATH), exist_ok=True)

url = f"https://drive.google.com/uc?id={FILE_ID}"
print("Downloading candidates.jsonl dataset (487MB) from Google Drive...")
gdown.download(url, DEST_PATH, quiet=False)
print(f"Download complete! Saved to {DEST_PATH}")
