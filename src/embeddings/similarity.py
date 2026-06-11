from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from loguru import logger

from src.preprocessing.build_text_blob import build_jd_text
from src.embeddings.generate_embeddings import load_embeddings
from src.embeddings.faiss_index import load_faiss_index, search_index
from src.utils.config import (
    JD_PATH,
    EMBEDDINGS_PATH,
    IDS_PATH,
    FAISS_INDEX_PATH,
    MODEL_NAME,
    TOP_K,
)


def _read_jd(jd_path: str | Path) -> dict:
    path = Path(jd_path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif suffix == ".docx":
        try:
            from docx import Document
        except ImportError:
            raise ImportError(
                "python-docx is required to read .docx files. Install it with: pip install python-docx"
            )
        doc = Document(str(path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return {"title": path.stem, "description": text}
    elif suffix == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return {"title": path.stem, "description": text}
    else:
        raise ValueError(f"Unsupported JD file format: {suffix}. Use .json, .docx, or .txt.")


def embed_jd(jd_text: str, model: SentenceTransformer) -> np.ndarray:
    embedding = model.encode(
        jd_text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding.reshape(1, -1)


def retrieve_top_candidates(
    jd_path: str | Path = JD_PATH,
    embeddings_path: str | Path = EMBEDDINGS_PATH,
    ids_path: str | Path = IDS_PATH,
    index_path: str | Path = FAISS_INDEX_PATH,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    logger.info("Loading JD...")
    jd_data = _read_jd(jd_path)
    jd_text = build_jd_text(jd_data)
    logger.info(f"JD text length: {len(jd_text)} chars")

    logger.info(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    logger.info("Embedding JD...")
    query_emb = embed_jd(jd_text, model)

    logger.info("Loading embeddings and FAISS index...")
    embeddings, candidate_ids = load_embeddings(embeddings_path, ids_path)
    index = load_faiss_index(index_path)

    logger.info(f"Searching for top {top_k} candidates...")
    results = search_index(index, query_emb, candidate_ids, top_k=top_k)

    df = pd.DataFrame(results, columns=["candidate_id", "semantic_score"])
    df["semantic_score"] = df["semantic_score"].clip(0.0, 1.0)
    logger.info(f"Returning {len(df)} candidates")

    top_1000_df = df.copy()

    return top_1000_df


if __name__ == "__main__":
    df = retrieve_top_candidates(top_k=10)
    print(df.to_string(index=False))
