from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 512
EMBEDDING_DIM = 384
TOP_K = 1000

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

CANDIDATES_PATH = RAW_DIR / "candidates.jsonl"
GOOGLE_DRIVE_FILE_ID = "1tB7Otd2EGldaDRu62cVnIoW6_G8A_HaT"
JD_PATH = RAW_DIR / "job_description.docx"
EMBEDDINGS_PATH = PROCESSED_DIR / "candidate_embeddings.npy"
IDS_PATH = PROCESSED_DIR / "candidate_ids.pkl"
FAISS_INDEX_PATH = MODELS_DIR / "faiss_index.bin"