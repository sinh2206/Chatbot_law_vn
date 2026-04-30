from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import chromadb

from config import (
    COLLECTION_NAME,
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    MAX_CONTEXT_CHUNKS,
    SENTENCE_TRANSFORMER_MODEL,
    TOP_K,
    VECTOR_STORE_DIR,
    ensure_directories,
)
from load_and_chunk import ChunkRecord

logger = logging.getLogger(__name__)


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
            if value is None:
                sanitized[key] = ""
            else:
                sanitized[key] = str(value)
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
