from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env")

# Source documents (.doc/.docx/.txt) by domain.
RAW_DOCS_DIR = BASE_DIR / "Multi-Agent"

# Offline processed artifacts.
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
FINETUNE_DIR = DATA_DIR / "finetune"
MODELS_DIR = DATA_DIR / "models"
CHAT_DB_PATH = Path(os.getenv("CHAT_DB_PATH", str(DATA_DIR / "chat_history.sqlite3")))

# Local embedding model directory (offline first).
LOCAL_MODEL_DIR = BASE_DIR / "model"

# FAISS artifacts.
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss.index"
EMBEDDINGS_NPY_PATH = VECTOR_STORE_DIR / "embeddings.npy"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.jsonl"
MANIFEST_PATH = VECTOR_STORE_DIR / "manifest.json"
LEGAL_DOCUMENT_METADATA_PATH = BASE_DIR / "metadata" / "legal_documents_metadata.csv"

# File handling.
ALLOWED_SOURCE_EXTENSIONS = {".doc", ".docx", ".txt"}
TEXT_OUTPUT_EXTENSION = ".txt"

# Chunking.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "60"))

# Embedding (offline local model, no API).
DEFAULT_EMBEDDING_MODEL_FALLBACK = "dangvantuan/vietnamese-embedding"


def _is_sentence_transformer_dir(path: Path) -> bool:
    # Minimal files expected in a local Sentence-Transformers export.
    required = [
        path / "modules.json",
        path / "config_sentence_transformers.json",
    ]
    return all(item.exists() for item in required)


EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    str(LOCAL_MODEL_DIR) if _is_sentence_transformer_dir(LOCAL_MODEL_DIR) else DEFAULT_EMBEDDING_MODEL_FALLBACK,
)
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# Retrieval defaults.
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_CANDIDATE_MULTIPLIER = int(os.getenv("MAX_CANDIDATE_MULTIPLIER", "6"))
MIN_RETRIEVAL_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.45"))

# Gemini is a fallback only. Local RAG must always be attempted first.
GEMINI_FALLBACK_ENABLED = os.getenv("GEMINI_FALLBACK_ENABLED", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CHAT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
