from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import chromadb

from config import (
    ALLOWED_EXTENSIONS,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    DOMAIN_LABELS,
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    MAX_CONTEXT_CHUNKS,
    SENTENCE_TRANSFORMER_MODEL,
    TOP_K,
    VECTOR_STORE_DIR,
    ensure_directories,
)
from expiry import LegalMetadataRegistry, infer_document_number

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    text: str
    metadata: dict[str, str]


def list_document_files(root_dir: Path | None = None) -> list[Path]:
    root = Path(root_dir or DATA_DIR)
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    ]
    files.sort()
    return files


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]

    normalized_lines: list[str] = []
    previous_blank = False
    for line in lines:
        if line:
            normalized_lines.append(line)
            previous_blank = False
        elif not previous_blank:
            normalized_lines.append("")
            previous_blank = True

    return "\n".join(normalized_lines).strip()


def split_text_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    normalized = _normalize_text(text)
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []
    start = 0
    total_length = len(normalized)
    min_boundary = int(chunk_size * 0.6)

    while start < total_length:
        end = min(start + chunk_size, total_length)

        if end < total_length:
            boundary = normalized.rfind("\n", start + min_boundary, end)
            if boundary == -1:
                boundary = normalized.rfind(" ", start + min_boundary, end)
            if boundary > start:
                end = boundary

        piece = normalized[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= total_length:
            break

        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks


def _read_text_file(path: Path) -> str:
    encodings = ["utf-8-sig", "utf-8", "cp1258", "cp1252", "latin-1"]
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_docx_file(path: Path) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError(
            "python-docx is required to read .docx files. Install dependencies first."
        ) from exc

    doc = Document(str(path))
    paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)


def _extract_doc_with_textract(path: Path) -> str | None:
    try:
        import textract  # type: ignore
    except ImportError:
        return None

    try:
        raw = textract.process(str(path))
    except Exception:
        return None

    for encoding in ("utf-8", "cp1258", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _extract_doc_with_command(path: Path, command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:
        return None

    raw = completed.stdout
    for encoding in ("utf-8", "cp1258", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _extract_doc_with_word_com(path: Path) -> str | None:
    if os.name != "nt":
        return None

    try:
        import win32com.client  # type: ignore
    except ImportError:
        return None

    temp_txt = Path(tempfile.gettempdir()) / f"{path.stem}_{os.getpid()}_tmp.txt"
    word = None
    doc = None

    try:
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        doc = word.Documents.Open(str(path.resolve()))
        doc.SaveAs(str(temp_txt), FileFormat=2)
    except Exception:
        return None
    finally:
        if doc is not None:
            try:
                doc.Close(False)
            except Exception:
                pass
        if word is not None:
            try:
                word.Quit()
            except Exception:
                pass

    if not temp_txt.exists():
        return None

    try:
        return _read_text_file(temp_txt)
    finally:
        try:
            temp_txt.unlink(missing_ok=True)
        except OSError:
            pass


def _read_doc_file(path: Path) -> str:
    strategies = [
        _extract_doc_with_textract,
        lambda p: _extract_doc_with_command(p, ["antiword", str(p)]),
        lambda p: _extract_doc_with_command(p, ["catdoc", str(p)]),
        _extract_doc_with_word_com,
    ]

    for strategy in strategies:
        text = strategy(path)
        if text and text.strip():
            return text

    raise RuntimeError(
        "Cannot read .doc file. Install one of these options: "
        "(1) textract + antiword, (2) antiword/catdoc command, or (3) pywin32 + Microsoft Word."
    )


def extract_text(path: Path) -> str:
    extension = path.suffix.lower()

    if extension == ".txt":
        return _read_text_file(path)
    if extension == ".docx":
        return _read_docx_file(path)
    if extension == ".doc":
        return _read_doc_file(path)

    raise ValueError(f"Unsupported file extension: {extension}")


def _extract_article_hint(chunk_text: str) -> str:
    matches = re.findall(r"\bĐiều\s+\d+[A-Za-z]?", chunk_text, flags=re.IGNORECASE)
    if not matches:
        return ""

    deduped: list[str] = []
    seen: set[str] = set()
    for item in matches:
        key = item.lower()
        if key not in seen:
            deduped.append(item)
            seen.add(key)
        if len(deduped) >= 3:
            break

    return "; ".join(deduped)


def _build_chunk_id(path: Path, chunk_index: int) -> str:
    raw = f"{path.as_posix()}::{chunk_index}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def build_metadata(
    path: Path,
    chunk_index: int,
    chunk_text: str,
    metadata_registry: LegalMetadataRegistry | None = None,
) -> dict[str, str]:
    domain_code = path.parent.name
    base_metadata = {
        "domain": domain_code,
        "domain_label": DOMAIN_LABELS.get(domain_code, domain_code),
        "file_name": path.name,
        "file_stem": path.stem,
        "source_path": str(path),
        "document_type": path.stem.split("_")[0] if "_" in path.stem else "Unknown",
        "document_number": infer_document_number(path.stem),
        "article_hint": _extract_article_hint(chunk_text),
        "chunk_index": str(chunk_index),
    }

    if metadata_registry is None:
        return base_metadata

    return metadata_registry.enrich_chunk_metadata(base_metadata)


def chunk_documents(
    root_dir: Path | None = None,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    metadata_registry: LegalMetadataRegistry | None = None,
) -> list[ChunkRecord]:
    files = list_document_files(root_dir=root_dir)
    chunk_records: list[ChunkRecord] = []
    registry = metadata_registry or LegalMetadataRegistry(autocreate_template=True)

    logger.info("Found %d documents in %s", len(files), root_dir or DATA_DIR)

    for file_path in files:
        try:
            text = extract_text(file_path)
        except Exception as exc:
            logger.warning("Skip %s due to read error: %s", file_path, exc)
            continue

        chunks = split_text_into_chunks(text=text, chunk_size=chunk_size, overlap=overlap)
        for chunk_index, chunk_text in enumerate(chunks):
            chunk_records.append(
                ChunkRecord(
                    chunk_id=_build_chunk_id(file_path, chunk_index),
                    text=chunk_text,
                    metadata=build_metadata(
                        file_path,
                        chunk_index,
                        chunk_text,
                        metadata_registry=registry,
                    ),
                )
            )

    logger.info("Generated %d chunks", len(chunk_records))
    return chunk_records


def group_chunks_by_domain(chunks: Iterable[ChunkRecord]) -> dict[str, list[ChunkRecord]]:
    grouped: dict[str, list[ChunkRecord]] = {}
    for chunk in chunks:
        domain = chunk.metadata.get("domain", "Unknown")
        grouped.setdefault(domain, []).append(chunk)
    return grouped


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str = EMBEDDING_MODEL) -> None:
        if not api_key:
            raise RuntimeError(
                "Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY in .env"
            )

        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError(
                "google-generativeai is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        genai.configure(api_key=api_key)
        self._genai = genai
        self.model = model

    def _embed(self, text: str, task_type: str) -> list[float]:
        response = self._genai.embed_content(
            model=self.model,
            content=text,
            task_type=task_type,
        )
        if isinstance(response, dict):
            embedding = response.get("embedding", [])
        else:
            embedding = getattr(response, "embedding", [])
        return [float(value) for value in embedding]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text, "retrieval_document") for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, "retrieval_query")


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = SENTENCE_TRANSFORMER_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode([text], normalize_embeddings=True)[0]
        return vector.tolist()


@dataclass
class SearchResult:
    text: str
    metadata: dict[str, str]
    distance: float


class ChromaVectorStore:
    def __init__(
        self,
        embedder: Embedder,
        persist_dir: Path = VECTOR_STORE_DIR,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        ensure_directories()
        self.embedder = embedder
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset_collection(self) -> None:
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()

    def upsert_chunks(self, chunks: list[ChunkRecord], batch_size: int = 32) -> int:
        if not chunks:
            return 0

        inserted = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            ids = [item.chunk_id for item in batch]
            documents = [item.text for item in batch]
            metadatas = [self._sanitize_metadata(item.metadata) for item in batch]
            embeddings = self.embedder.embed_documents(documents)

            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            inserted += len(batch)

        return inserted

    def similarity_search(
        self,
        query: str,
        top_k: int = TOP_K,
        domain_filter: str | None = None,
    ) -> list[SearchResult]:
        if self.count() == 0:
            return []

        query_embedding = self.embedder.embed_query(query)
        where = {"domain": domain_filter} if domain_filter else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max(1, min(top_k, MAX_CONTEXT_CHUNKS)),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        parsed: list[SearchResult] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            safe_metadata = self._sanitize_metadata(metadata or {})
            parsed.append(
                SearchResult(
                    text=text,
                    metadata=safe_metadata,
                    distance=float(distance),
                )
            )

        return parsed

    def list_domains(self) -> list[str]:
        payload = self.collection.get(include=["metadatas"])
        domains = {
            metadata.get("domain", "")
            for metadata in payload.get("metadatas", [])
            if metadata and metadata.get("domain")
        }
        return sorted(domains)

    def delete_by_domain(self, domain: str) -> int:
        payload = self.collection.get(where={"domain": domain}, include=[])
        ids = payload.get("ids", []) or []
        if not ids:
            return 0
        self.collection.delete(ids=ids)
        return len(ids)

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, object]) -> dict[str, str]:
        sanitized: dict[str, str] = {}
        for key, value in metadata.items():
            sanitized[key] = "" if value is None else str(value)
        return sanitized


def create_embedder(backend: str = EMBEDDING_BACKEND) -> Embedder:
    if backend == "gemini":
        return GeminiEmbedder(api_key=GEMINI_API_KEY or "", model=EMBEDDING_MODEL)
    if backend in {"sentence-transformers", "sentence_transformers", "sbert"}:
        return SentenceTransformerEmbedder(model_name=SENTENCE_TRANSFORMER_MODEL)

    raise ValueError(f"Unsupported EMBEDDING_BACKEND: {backend}")


def create_vector_store(backend: str = EMBEDDING_BACKEND) -> ChromaVectorStore:
    embedder = create_embedder(backend=backend)
    return ChromaVectorStore(embedder=embedder)


def build_vector_store(
    root_dir: Path | None = None,
    reset: bool = True,
    domain: str | None = None,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> dict[str, int | str]:
    registry = LegalMetadataRegistry(autocreate_template=True)
    vector_store = create_vector_store()

    effective_root = root_dir
    if domain:
        if domain not in DOMAIN_LABELS:
            raise ValueError(f"Invalid domain: {domain}")
        effective_root = DATA_DIR / domain

    chunks = chunk_documents(
        root_dir=effective_root,
        chunk_size=chunk_size,
        overlap=overlap,
        metadata_registry=registry,
    )

    if reset:
        vector_store.reset_collection()

    inserted = vector_store.upsert_chunks(chunks)
    return {
        "inserted": inserted,
        "total": vector_store.count(),
        "scope": domain or "all",
    }


def update_vector_store(
    domain: str | None = None,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> dict[str, int | str]:
    registry = LegalMetadataRegistry(autocreate_template=True)
    vector_store = create_vector_store()

    if domain:
        domain = domain.strip()
        if domain not in DOMAIN_LABELS:
            raise ValueError(f"Invalid domain: {domain}")

        domain_path = DATA_DIR / domain
        chunks = chunk_documents(
            root_dir=domain_path,
            chunk_size=chunk_size,
            overlap=overlap,
            metadata_registry=registry,
        )
        removed = vector_store.delete_by_domain(domain)
        inserted = vector_store.upsert_chunks(chunks)
        return {
            "removed": removed,
            "inserted": inserted,
            "total": vector_store.count(),
            "scope": domain,
        }

    chunks = chunk_documents(
        chunk_size=chunk_size,
        overlap=overlap,
        metadata_registry=registry,
    )
    vector_store.reset_collection()
    inserted = vector_store.upsert_chunks(chunks)
    return {
        "removed": 0,
        "inserted": inserted,
        "total": vector_store.count(),
        "scope": "all",
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Build or update legal vector store.")
    parser.add_argument("--build", action="store_true", help="Build vector store from documents.")
    parser.add_argument("--update", action="store_true", help="Update vector store (domain or full).")
    parser.add_argument("--domain", type=str, default="", help="Only process one domain.")
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="When --build, do not clear existing collection first.",
    )
    parser.add_argument(
        "--bootstrap-metadata",
        action="store_true",
        help="Generate metadata template and exit.",
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    registry = LegalMetadataRegistry(autocreate_template=True)
    if args.bootstrap_metadata:
        output = registry.bootstrap_template_from_data(overwrite=True)
        logger.info("Generated metadata template: %s", output)
        return

    domain = args.domain.strip() or None

    if args.build:
        if domain:
            result = update_vector_store(
                domain=domain,
                chunk_size=args.chunk_size,
                overlap=args.chunk_overlap,
            )
            logger.info("Built domain scope via upsert: %s", result)
        else:
            result = build_vector_store(
                root_dir=None,
                reset=not args.no_reset,
                chunk_size=args.chunk_size,
                overlap=args.chunk_overlap,
            )
            logger.info("Build completed: %s", result)

        report = registry.scan_expired_documents()
        logger.info(
            "Legal metadata scan: %d/%d documents marked expired",
            len(report.expired_documents),
            report.total_documents,
        )
        return

    if args.update:
        result = update_vector_store(
            domain=domain,
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
        )
        logger.info("Update completed: %s", result)
        return

    parser.print_help()


if __name__ == "__main__":
    _cli()
