from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    EMBEDDINGS_NPY_PATH,
    FAISS_INDEX_PATH,
    MANIFEST_PATH,
    METADATA_PATH,
    MIN_CHUNK_CHARS,
    PROCESSED_DIR,
    VECTOR_STORE_DIR,
    ensure_directories,
)


def setup_stdout_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _format_install_hint(package: str) -> str:
    return (
        f"Try installing with the same interpreter:\n"
        f"  \"{sys.executable}\" -m pip install {package}"
    )


@dataclass
class ChunkItem:
    chunk_id: str
    domain: str
    source_file: str
    text: str
    chunk_index: int


def validate_local_model_dir(model_name: str) -> None:
    model_path = Path(model_name)
    if not model_path.exists() or not model_path.is_dir():
        return
    required = [
        model_path / "modules.json",
        model_path / "config_sentence_transformers.json",
    ]
    missing = [str(item.name) for item in required if not item.exists()]
    if missing:
        raise RuntimeError(
            f"Local model directory is incomplete: {model_path}\n"
            f"Missing required files: {', '.join(missing)}\n"
            "This usually means fine-tuning did not finish, or the directory was created "
            "before the model was saved.\n"
            "Fix it by running scripts/train_embedding.py first, then rebuild the vector store. "
            "If you only want to use the base model, pass "
            "--embedding-model dangvantuan/vietnamese-embedding."
        )


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    clean = "\n".join(lines)
    while "\n\n\n" in clean:
        clean = clean.replace("\n\n\n", "\n\n")
    return clean.strip()


def split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")

    clean = normalize_text(text)
    if not clean:
        return []
    if len(clean) <= chunk_size:
        return [clean] if len(clean) >= MIN_CHUNK_CHARS else []

    chunks: list[str] = []
    start = 0
    total = len(clean)
    min_boundary = int(chunk_size * 0.6)

    while start < total:
        end = min(start + chunk_size, total)
        if end < total:
            boundary = clean.rfind("\n", start + min_boundary, end)
            if boundary == -1:
                boundary = clean.rfind(" ", start + min_boundary, end)
            if boundary > start:
                end = boundary

        piece = clean[start:end].strip()
        if len(piece) >= MIN_CHUNK_CHARS:
            chunks.append(piece)

        if end >= total:
            break

        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks


def iter_processed_files(processed_dir: Path, domains: set[str] | None) -> list[Path]:
    files: list[Path] = []
    for path in processed_dir.rglob("*.txt"):
        if not path.is_file():
            continue
        if domains:
            relative = path.relative_to(processed_dir)
            if not relative.parts or relative.parts[0] not in domains:
                continue
        files.append(path)
    files.sort()
    return files


def collect_chunks(
    processed_dir: Path,
    domains: set[str] | None,
    chunk_size: int,
    overlap: int,
) -> list[ChunkItem]:
    items: list[ChunkItem] = []
    files = iter_processed_files(processed_dir=processed_dir, domains=domains)
    for file_path in files:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        chunks = split_text_into_chunks(text=text, chunk_size=chunk_size, overlap=overlap)
        relative = file_path.relative_to(processed_dir)
        domain = relative.parts[0] if relative.parts else "Unknown"
        source_file = str(relative).replace("\\", "/")
        for idx, chunk_text in enumerate(chunks):
            chunk_id = f"{source_file}::chunk_{idx}"
            items.append(
                ChunkItem(
                    chunk_id=chunk_id,
                    domain=domain,
                    source_file=source_file,
                    text=chunk_text,
                    chunk_index=idx,
                )
            )
    return items


def embed_chunks(texts: list[str], model_name: str, batch_size: int) -> np.ndarray:
    validate_local_model_dir(model_name)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: sentence-transformers.\n"
            f"Python executable: {sys.executable}\n"
            f"{_format_install_hint('sentence-transformers')}"
        ) from exc

    model = SentenceTransformer(model_name)

    tokenizer_limit = getattr(getattr(model, "tokenizer", None), "model_max_length", None)
    config_limit = getattr(getattr(model[0], "auto_model", None), "config", None)
    if config_limit is not None:
        config_limit = getattr(config_limit, "max_position_embeddings", None)
    limits: list[int] = []
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 100000:
        limits.append(int(tokenizer_limit))
    if isinstance(config_limit, int) and 0 < config_limit < 100000:
        limits.append(max(8, int(config_limit) - 2))
    if limits:
        safe_max_seq_length = min(limits)
        if model.max_seq_length != safe_max_seq_length:
            print(
                f"[WARN] Adjusting model.max_seq_length from {model.max_seq_length} "
                f"to safe limit {safe_max_seq_length} for model: {model_name}"
            )
            model.max_seq_length = safe_max_seq_length

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def try_build_faiss_index(vectors: np.ndarray):
    try:
        import faiss
    except ImportError as exc:
        print(
            "[WARN] faiss is not available. "
            "Falling back to numpy index (brute-force cosine search).\n"
            "To use FAISS for faster retrieval, install faiss-cpu on a compatible Python "
            "(typically Python 3.10/3.11 on Windows).\n"
            f"Python executable: {sys.executable}\n"
            f"{_format_install_hint('faiss-cpu')}"
        )
        return None

    if vectors.ndim != 2:
        raise ValueError("Embedding vectors must be a 2D array")

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def write_metadata_jsonl(items: list[ChunkItem], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in items:
            row = {
                "chunk_id": item.chunk_id,
                "domain": item.domain,
                "source_file": item.source_file,
                "chunk_index": item.chunk_index,
                "text": item.text,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest(
    manifest_path: Path,
    total_chunks: int,
    model_name: str,
    chunk_size: int,
    overlap: int,
    domains: list[str],
    index_backend: str,
) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_chunks": total_chunks,
        "embedding_model": model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
        "domains": sorted(domains),
        "index_backend": index_backend,
        "faiss_index_path": str(FAISS_INDEX_PATH),
        "embeddings_npy_path": str(EMBEDDINGS_NPY_PATH),
        "metadata_path": str(METADATA_PATH),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_faiss_index(index, index_path: Path) -> None:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "Missing faiss-cpu. Install requirements first. "
            "If you are on Windows/Python 3.12, use Python 3.10/3.11 for FAISS compatibility.\n"
            f"Python executable: {sys.executable}\n"
            f"{_format_install_hint('faiss-cpu')}"
        ) from exc

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))


def save_numpy_index(vectors: np.ndarray, embeddings_path: Path) -> None:
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, vectors)


def build_vector_store(
    processed_dir: Path,
    domains: set[str] | None,
    chunk_size: int,
    overlap: int,
    embedding_model: str,
    batch_size: int,
) -> dict[str, int | str]:
    ensure_directories()
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    chunks = collect_chunks(
        processed_dir=processed_dir,
        domains=domains,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    if not chunks:
        raise RuntimeError("No chunks found. Ensure data/processed has normalized .txt files.")

    texts = [item.text for item in chunks]
    vectors = embed_chunks(texts=texts, model_name=embedding_model, batch_size=batch_size)
    index = try_build_faiss_index(vectors=vectors)

    if index is not None:
        index_backend = "faiss"
        save_faiss_index(index=index, index_path=FAISS_INDEX_PATH)
        if EMBEDDINGS_NPY_PATH.exists():
            EMBEDDINGS_NPY_PATH.unlink()
    else:
        index_backend = "numpy"
        save_numpy_index(vectors=vectors, embeddings_path=EMBEDDINGS_NPY_PATH)
        if FAISS_INDEX_PATH.exists():
            FAISS_INDEX_PATH.unlink()

    write_metadata_jsonl(items=chunks, output_path=METADATA_PATH)
    write_manifest(
        manifest_path=MANIFEST_PATH,
        total_chunks=len(chunks),
        model_name=embedding_model,
        chunk_size=chunk_size,
        overlap=overlap,
        domains=sorted({item.domain for item in chunks}),
        index_backend=index_backend,
    )

    return {
        "index_backend": index_backend,
        "total_chunks": len(chunks),
        "embedding_dim": int(vectors.shape[1]),
        "index_path": str(FAISS_INDEX_PATH),
        "embeddings_path": str(EMBEDDINGS_NPY_PATH),
        "metadata_path": str(METADATA_PATH),
    }


def main() -> None:
    setup_stdout_utf8()
    parser = argparse.ArgumentParser(description="Build local FAISS vector store from processed .txt files.")
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--domain", action="append", default=[])
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=EMBEDDING_BATCH_SIZE)
    args = parser.parse_args()

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.stdout.strip():
            print(f"[ENV] {completed.stdout.strip()}")
    except Exception:
        pass

    processed_dir = Path(args.processed_dir).resolve()
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")

    domains = set(args.domain) if args.domain else None
    result = build_vector_store(
        processed_dir=processed_dir,
        domains=domains,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
    )

    print("\n=== BUILD SUMMARY ===")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
