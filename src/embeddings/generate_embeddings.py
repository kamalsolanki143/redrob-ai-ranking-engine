from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from loguru import logger

from src.preprocessing.build_text_blob import build_candidate_text
from src.utils.config import (
    CANDIDATES_PATH,
    EMBEDDINGS_PATH,
    IDS_PATH,
    MODEL_NAME,
    BATCH_SIZE,
    GOOGLE_DRIVE_FILE_ID,
)


def _download_from_drive(file_id: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    session = requests.Session()
    response = session.get(url, stream=True)
    response.raise_for_status()

    confirm = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            confirm = value
            break

    if confirm:
        url = f"https://drive.google.com/uc?export=download&confirm={confirm}&id={file_id}"
        response = session.get(url, stream=True)
        response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    desc = f"Downloading candidates.jsonl ({total // (1024*1024)}MB)"

    with open(dest, "wb") as f, tqdm(
        desc=desc, total=total, unit="B", unit_scale=True, unit_divisor=1024
    ) as pbar:
        for chunk in response.iter_content(chunk_size=65536):
            f.write(chunk)
            pbar.update(len(chunk))

    logger.info(f"Downloaded to {dest}")


def _ensure_candidates_file(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        logger.info(f"Found existing candidates file at {path}")
        return
    logger.warning(f"Candidates file not found at {path}, downloading from Google Drive...")
    _download_from_drive(GOOGLE_DRIVE_FILE_ID, path)


def load_candidates(jsonl_path: str | Path) -> list[dict]:
    path = Path(jsonl_path)
    _ensure_candidates_file(path)
    candidates = []
    logger.info(f"Loading candidates from {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading candidates"):
            line = line.strip()
            if not line:
                continue
            candidate = json.loads(line)
            candidates.append(candidate)
    logger.info(f"Loaded {len(candidates)} candidates")
    return candidates


def generate_embeddings(
    candidates: list[dict],
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
    save_path: str | Path = EMBEDDINGS_PATH,
) -> np.ndarray:
    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, device="cpu")

    texts = []
    candidate_ids = []
    for c in candidates:
        text = build_candidate_text(c)
        texts.append(text)
        candidate_ids.append(c.get("candidate_id", ""))

    logger.info(f"Generating embeddings for {len(texts)} candidates (batch_size={batch_size})")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(save_path), embeddings)
    logger.info(f"Saved embeddings to {save_path} (shape: {embeddings.shape})")

    ids_save_path = save_path.parent / "candidate_ids.pkl"
    with open(ids_save_path, "wb") as f:
        pickle.dump(candidate_ids, f)
    logger.info(f"Saved candidate IDs to {ids_save_path} ({len(candidate_ids)} ids)")

    return embeddings


def load_embeddings(
    embeddings_path: str | Path = EMBEDDINGS_PATH,
    ids_path: str | Path = IDS_PATH,
) -> tuple[np.ndarray, list]:
    embeddings = np.load(str(embeddings_path))
    with open(ids_path, "rb") as f:
        candidate_ids = pickle.load(f)
    logger.info(f"Loaded {len(candidate_ids)} embeddings with shape {embeddings.shape}")
    return embeddings, candidate_ids


if __name__ == "__main__":
    candidates = load_candidates(CANDIDATES_PATH)
    generate_embeddings(candidates)