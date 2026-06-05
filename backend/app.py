from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from threading import RLock
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDINGS_NPY_PATH,
    FAISS_INDEX_PATH,
    GEMINI_API_KEY,
    GEMINI_FALLBACK_ENABLED,
    GEMINI_MODEL,
    LEGAL_DOCUMENT_METADATA_PATH,
    MANIFEST_PATH,
    MAX_CANDIDATE_MULTIPLIER,
    METADATA_PATH,
    MIN_RETRIEVAL_SCORE,
    TOP_K,
)
from scripts.gemini_fallback import (
    GeminiFallbackRequest,
    generate_gemini_fallback_answer,
)
from scripts.query_cli import (
    RetrievedItem,
    attach_document_status,
    decide_retrieval,
    format_source,
    load_embedder,
    load_faiss_index,
    load_legal_document_status,
    load_metadata,
    load_numpy_embeddings,
    read_manifest,
    search,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"


class ChatRequest(BaseModel):
    message: str | None = Field(default=None, description="User question.")
    query: str | None = Field(default=None, description="Alias for message.")
    domain: str | None = Field(default=None, description="Optional legal domain.")
    top_k: int = Field(default=TOP_K, ge=1, le=20)
    min_score: float = Field(default=MIN_RETRIEVAL_SCORE, ge=-1.0, le=1.0)
    candidate_multiplier: int = Field(default=MAX_CANDIDATE_MULTIPLIER, ge=1, le=50)
    gemini_fallback: bool | None = Field(
        default=None,
        description="If true, call Gemini only after local RAG fails.",
    )
    gemini_model: str | None = None
    snippet_chars: int = Field(default=700, ge=100, le=4000)


class ChatResponse(BaseModel):
    mode: str
    answer: str
    reason: str
    local_used: bool
    gemini_used: bool
    sources: list[dict[str, Any]]
    expired_sources: list[dict[str, Any]]
    low_confidence_sources: list[dict[str, Any]]
    metadata: dict[str, Any]


class RagEngine:
    def __init__(self) -> None:
        self._lock = RLock()
        self._loaded = False
        self.backend = "faiss"
        self.index: Any = None
        self.metadata: list[dict[str, object]] = []
        self.document_statuses: dict[str, Any] = {}
        self.embedder: Any = None
        self.manifest: dict[str, object] = {}
        self.embedding_model = EMBEDDING_MODEL_NAME
        self.index_display = ""

    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        with self._lock:
            self._loaded = False
            self.index = None
            self.metadata = []
            self.document_statuses = {}
            self.embedder = None
            self.manifest = {}
            self.embedding_model = EMBEDDING_MODEL_NAME
            self.index_display = ""

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return

            manifest_path = MANIFEST_PATH.resolve()
            manifest = read_manifest(manifest_path)
            backend = str(manifest.get("index_backend", "faiss")).lower()

            if backend == "numpy":
                embeddings_path = self._resolve_path(
                    str(manifest.get("embeddings_npy_path", "")),
                    EMBEDDINGS_NPY_PATH,
                )
                index = load_numpy_embeddings(embeddings_path)
                index_display = str(embeddings_path)
            else:
                backend = "faiss"
                index_path = self._resolve_path(
                    str(manifest.get("faiss_index_path", "")),
                    FAISS_INDEX_PATH,
                )
                index = load_faiss_index(index_path)
                index_display = str(index_path)

            metadata_path = self._resolve_path(
                str(manifest.get("metadata_path", "")),
                METADATA_PATH,
            )
            metadata = load_metadata(metadata_path)
            document_statuses = load_legal_document_status(
                LEGAL_DOCUMENT_METADATA_PATH.resolve()
            )
            manifest_model = str(manifest.get("embedding_model", "")).strip()
            embedding_model = manifest_model or EMBEDDING_MODEL_NAME
            embedder = load_embedder(embedding_model)

            self.backend = backend
            self.index = index
            self.metadata = metadata
            self.document_statuses = document_statuses
            self.embedder = embedder
            self.manifest = manifest
            self.embedding_model = embedding_model
            self.index_display = index_display
            self._loaded = True

    def query(self, request: ChatRequest) -> ChatResponse:
        self.load()
        question = (request.message or request.query or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="message/query is required")

        domain = request.domain.strip() if request.domain else None
        results = search(
            backend=self.backend,
            index=self.index,
            metadata=self.metadata,
            embedder=self.embedder,
            query=question,
            top_k=request.top_k,
            domain=domain,
            candidate_multiplier=request.candidate_multiplier,
        )
        results = attach_document_status(results, self.document_statuses)
        decision = decide_retrieval(
            results=results,
            metadata=self.metadata,
            domain=domain,
            min_score=request.min_score,
        )

        fallback_enabled = (
            GEMINI_FALLBACK_ENABLED
            if request.gemini_fallback is None
            else request.gemini_fallback
        )
        gemini_model = request.gemini_model or GEMINI_MODEL

        if decision.use_internal:
            return ChatResponse(
                mode="local_rag",
                answer=self._format_local_answer(
                    decision.usable_results,
                    snippet_chars=request.snippet_chars,
                ),
                reason=decision.reason,
                local_used=True,
                gemini_used=False,
                sources=[self._source_payload(item) for item in decision.usable_results],
                expired_sources=[
                    self._source_payload(item) for item in decision.expired_results
                ],
                low_confidence_sources=[
                    self._source_payload(item)
                    for item in decision.low_confidence_results
                ],
                metadata=self._response_metadata(),
            )

        if not fallback_enabled:
            return ChatResponse(
                mode="fallback_required",
                answer=(
                    "Khong co can cu noi bo du tin cay de tra loi. "
                    "Gemini fallback dang tat."
                ),
                reason=decision.reason,
                local_used=False,
                gemini_used=False,
                sources=[],
                expired_sources=[
                    self._source_payload(item) for item in decision.expired_results
                ],
                low_confidence_sources=[
                    self._source_payload(item)
                    for item in decision.low_confidence_results
                ],
                metadata=self._response_metadata(),
            )

        try:
            answer = generate_gemini_fallback_answer(
                request=GeminiFallbackRequest(
                    question=question,
                    reason=decision.reason,
                    domain=domain,
                    expired_sources=[
                        format_source(item) for item in decision.expired_results
                    ],
                    low_confidence_sources=[
                        format_source(item)
                        for item in decision.low_confidence_results[:5]
                    ],
                ),
                api_key=GEMINI_API_KEY,
                model_name=gemini_model,
            )
        except Exception as exc:
            return ChatResponse(
                mode="gemini_error",
                answer=f"Khong the goi Gemini fallback: {exc}",
                reason=decision.reason,
                local_used=False,
                gemini_used=False,
                sources=[],
                expired_sources=[
                    self._source_payload(item) for item in decision.expired_results
                ],
                low_confidence_sources=[
                    self._source_payload(item)
                    for item in decision.low_confidence_results
                ],
                metadata=self._response_metadata(),
            )

        return ChatResponse(
            mode="gemini_fallback",
            answer=answer,
            reason=decision.reason,
            local_used=False,
            gemini_used=True,
            sources=[],
            expired_sources=[
                self._source_payload(item) for item in decision.expired_results
            ],
            low_confidence_sources=[
                self._source_payload(item) for item in decision.low_confidence_results
            ],
            metadata=self._response_metadata(),
        )

    def _resolve_path(self, value: str, default_path: Path) -> Path:
        if value:
            candidate = Path(value)
            if candidate.exists():
                return candidate.resolve()
            if not candidate.is_absolute():
                rooted = ROOT_DIR / candidate
                if rooted.exists():
                    return rooted.resolve()
        return default_path.resolve()

    def _format_local_answer(
        self,
        results: list[RetrievedItem],
        snippet_chars: int,
    ) -> str:
        lines = ["Tim thay can cu trong kho noi bo. Gemini khong duoc goi."]
        for idx, item in enumerate(results, start=1):
            snippet = item.text[:snippet_chars].strip()
            if len(item.text) > snippet_chars:
                snippet += " ..."
            lines.append(
                f"\\n[{idx}] {item.source_file} | chunk={item.chunk_index} | "
                f"score={item.score:.4f}\\n{snippet}"
            )
        return "\\n".join(lines)

    def _source_payload(self, item: RetrievedItem) -> dict[str, Any]:
        return asdict(item)

    def _response_metadata(self) -> dict[str, Any]:
        return {
            "index_backend": self.backend,
            "embedding_model": self.embedding_model,
            "index": self.index_display,
            "total_chunks": self.manifest.get("total_chunks"),
            "domains": self.manifest.get("domains", []),
        }


engine = RagEngine()

app = FastAPI(
    title="Chatbot Law VN API",
    version="1.0.0",
    description="Local legal RAG API with optional Gemini fallback.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "loaded": engine.is_loaded(),
        "embedding_model": engine.embedding_model,
        "gemini_fallback_enabled": GEMINI_FALLBACK_ENABLED,
        "gemini_model": GEMINI_MODEL,
    }


@app.post("/reload")
def reload_engine() -> dict[str, Any]:
    engine.unload()
    engine.load()
    return {
        "ok": True,
        "embedding_model": engine.embedding_model,
        "index_backend": engine.backend,
        "total_chunks": engine.manifest.get("total_chunks"),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return engine.query(request)


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
