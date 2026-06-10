from __future__ import annotations

from pathlib import Path

import numpy as np
import faiss
from loguru import logger

from src.utils.config import EMBEDDINGS_PATH, IDS_PATH, FAISS_INDEX_PATH
from src.embeddings.generate_embeddings import load_embeddings


def build_faiss_index(
    embeddings: np.ndarray,
    save_path: str | Path = FAISS_INDEX_PATH,
) -> faiss.Index:
    logger.info(f"Building FAISS index with {embeddings.shape[0]} vectors (dim={embeddings.shape[1]})")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    index.add(embeddings.astype(np.float32))
    logger.info(f"Added {index.ntotal} vectors to index")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(save_path))
    logger.info(f"Saved FAISS index to {save_path}")

    return index


def load_faiss_index(index_path: str | Path = FAISS_INDEX_PATH) -> faiss.Index:
    index_path = Path(index_path)
    logger.info(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(str(index_path))
    logger.info(f"Loaded index with {index.ntotal} vectors (dim={index.d})")
    return index


def search_index(
    index: faiss.Index,
    query_embedding: np.ndarray,
    candidate_ids: list,
    top_k: int = 1000,
) -> list[tuple[str, float]]:
    query = query_embedding.astype(np.float32)
    if query.ndim == 1:
        query = query.reshape(1, -1)

    faiss.normalize_L2(query)
    scores, indices = index.search(query, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        cid = candidate_ids[idx] if idx < len(candidate_ids) else f"UNKNOWN_{idx}"
        score = float(scores[0][i])
        results.append((cid, score))

    logger.info(f"Search returned {len(results)} results")
    return results


if __name__ == "__main__":
    embeddings, candidate_ids = load_embeddings(EMBEDDINGS_PATH, IDS_PATH)
    build_faiss_index(embeddings, FAISS_INDEX_PATH)