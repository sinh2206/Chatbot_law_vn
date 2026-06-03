from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (  # noqa: E402
    EMBEDDING_MODEL_NAME,
    EMBEDDINGS_NPY_PATH,
    FAISS_INDEX_PATH,
    MANIFEST_PATH,
    MAX_CANDIDATE_MULTIPLIER,
    METADATA_PATH,
    TOP_K,
)


def setup_stdout_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


@dataclass
class RetrievedItem:
    score: float
    domain: str
    source_file: str
    chunk_index: int
    text: str


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
            f"Local model directory is missing required files: {', '.join(missing)}\n"
            f"Model directory: {model_path}"
        )


def load_faiss_index(index_path: Path):
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "Missing faiss-cpu. Install requirements first. "
            "If you are on Windows/Python 3.12, use Python 3.10/3.11 for FAISS compatibility.\n"
            f"Try: \"{sys.executable}\" -m pip install faiss-cpu"
        ) from exc

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    return faiss.read_index(str(index_path))


def load_numpy_embeddings(embeddings_path: Path) -> np.ndarray:
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Numpy embeddings not found: {embeddings_path}")
    vectors = np.load(embeddings_path)
    if vectors.ndim != 2:
        raise ValueError("Numpy embeddings must be a 2D array")
    return np.asarray(vectors, dtype=np.float32)


def load_metadata(metadata_path: Path) -> list[dict[str, object]]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    rows: list[dict[str, object]] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_embedder(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Missing sentence-transformers. Install requirements first.\n"
            f"Try: \"{sys.executable}\" -m pip install sentence-transformers"
        ) from exc
    validate_local_model_dir(model_name)
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
    return model


def search(
    backend: str,
    index,
    metadata: list[dict[str, object]],
    embedder,
    query: str,
    top_k: int,
    domain: str | None,
    candidate_multiplier: int,
) -> list[RetrievedItem]:
    if not query.strip():
        return []

    query_vec = embedder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    query_vec = np.asarray(query_vec, dtype=np.float32)

    max_candidates = max(top_k, top_k * max(1, candidate_multiplier))
    if backend == "faiss":
        scores, indices = index.search(query_vec, max_candidates)
        score_list = scores[0]
        index_list = indices[0]
    else:
        all_scores = np.dot(index, query_vec[0])
        sorted_indices = np.argsort(-all_scores)[:max_candidates]
        score_list = all_scores[sorted_indices]
        index_list = sorted_indices

    results: list[RetrievedItem] = []
    for score, idx in zip(score_list, index_list):
        if idx < 0 or idx >= len(metadata):
            continue

        row = metadata[idx]
        row_domain = str(row.get("domain", ""))
        if domain and row_domain != domain:
            continue

        results.append(
            RetrievedItem(
                score=float(score),
                domain=row_domain,
                source_file=str(row.get("source_file", "")),
                chunk_index=int(row.get("chunk_index", -1)),
                text=str(row.get("text", "")),
            )
        )
        if len(results) >= top_k:
            break
    return results


def print_results(results: list[RetrievedItem], show_full: bool, snippet_chars: int) -> None:
    if not results:
        print("Khong tim thay ket qua phu hop.")
        return

    for i, item in enumerate(results, start=1):
        print("\n" + "=" * 90)
        print(
            f"[{i}] score={item.score:.4f} | domain={item.domain} | "
            f"source={item.source_file} | chunk={item.chunk_index}"
        )
        if show_full:
            print(item.text)
        else:
            snippet = item.text[:snippet_chars].strip()
            if len(item.text) > snippet_chars:
                snippet += " ..."
            print(snippet)


def read_manifest(manifest_path: Path) -> dict[str, object]:
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> None:
    setup_stdout_utf8()
    parser = argparse.ArgumentParser(description="Offline RAG query CLI using local vector index.")
    parser.add_argument("--query", default="")
    parser.add_argument("--domain", default="")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--embedding-model",
        default="",
        help="Embedding model path/name. Empty = use manifest embedding_model, then config default.",
    )
    parser.add_argument("--index-path", default=str(FAISS_INDEX_PATH))
    parser.add_argument("--metadata-path", default=str(METADATA_PATH))
    parser.add_argument("--manifest-path", default=str(MANIFEST_PATH))
    parser.add_argument("--show-full", action="store_true")
    parser.add_argument("--snippet-chars", type=int, default=350)
    parser.add_argument("--candidate-multiplier", type=int, default=MAX_CANDIDATE_MULTIPLIER)
    args = parser.parse_args()

    manifest = read_manifest(Path(args.manifest_path).resolve())
    backend = str(manifest.get("index_backend", "faiss")).lower()

    if backend == "numpy":
        embeddings_path = Path(
            str(manifest.get("embeddings_npy_path", EMBEDDINGS_NPY_PATH))
        ).resolve()
        index = load_numpy_embeddings(embeddings_path)
        index_display = str(embeddings_path)
    else:
        backend = "faiss"
        index_path = Path(args.index_path).resolve()
        index = load_faiss_index(index_path)
        index_display = str(index_path)

    metadata = load_metadata(Path(args.metadata_path).resolve())
    manifest_model = str(manifest.get("embedding_model", "")).strip()
    embedding_model = args.embedding_model.strip() or manifest_model or EMBEDDING_MODEL_NAME
    embedder = load_embedder(embedding_model)

    print("=== OFFLINE QUERY CLI (NO API) ===")
    if manifest:
        print(f"Chunks: {manifest.get('total_chunks', 'N/A')}")
        print(f"Embedding model in manifest: {manifest.get('embedding_model', 'N/A')}")
    print(f"Embedding model in use: {embedding_model}")
    print(f"Index backend: {backend}")
    print(f"Index data: {index_display}")
    print(f"Metadata: {args.metadata_path}")

    domain = args.domain.strip() or None

    if args.query.strip():
        results = search(
            backend=backend,
            index=index,
            metadata=metadata,
            embedder=embedder,
            query=args.query,
            top_k=max(1, args.top_k),
            domain=domain,
            candidate_multiplier=max(1, args.candidate_multiplier),
        )
        print_results(results, show_full=args.show_full, snippet_chars=max(50, args.snippet_chars))
        return

    print("\nNhap cau hoi (go 'exit' de thoat).")
    while True:
        query = input("\nBan: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("Ket thuc.")
            return

        results = search(
            backend=backend,
            index=index,
            metadata=metadata,
            embedder=embedder,
            query=query,
            top_k=max(1, args.top_k),
            domain=domain,
            candidate_multiplier=max(1, args.candidate_multiplier),
        )
        print_results(results, show_full=args.show_full, snippet_chars=max(50, args.snippet_chars))


if __name__ == "__main__":
    main()
