from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from threading import RLock
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import (
    CHAT_DB_PATH,
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

DOMAIN_LABELS = {
    "CCCD": "CCCD",
    "DatDai": "Đất đai",
    "DoanhNghiep": "Doanh nghiệp",
    "HoTich": "Hộ tịch",
    "Thue": "Thuế",
}

DOCUMENT_TYPE_LABELS = {
    "Luat": "Luật",
    "NghiDinh": "Nghị định",
    "ThongTu": "Thông tư",
    "QuyetDinh": "Quyết định",
}

ARTICLE_RE = re.compile(r"(Điều|điều)\s+(\d+[a-zA-Z]?)")
ARTICLE_HEADING_RE = re.compile(r"(?m)^\s*Điều\s+\d+[a-zA-Z]?\s*[\\.:]")
ARTICLE_HEADING_CAPTURE_RE = re.compile(r"(?m)^\s*Điều\s+(\d+[a-zA-Z]?)\s*[\\.:]")
CLAUSE_RE = re.compile(r"(?:^|\n)\s*(\d+[a-zA-Z]?)\.\s+")
SECTION_RE = re.compile(r"(?:(?:khoản|Khoản)\s+(\d+[a-zA-Z]?))?\s*(?:Điều|điều)\s+(\d+[a-zA-Z]?)")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?;:])\s+(?=[A-ZĐ0-9])")
LEADING_LEGAL_BOUNDARY_RE = re.compile(
    r"(?m)(^Điều\s+\d+[a-zA-Z]?\s*[\\.:]|^\d+[a-zA-Z]?\.\s+|^[a-z]\)\s+)"
)


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
    snippet_chars: int = Field(default=1400, ge=100, le=4000)


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


class ChatStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    question TEXT NOT NULL,
                    normalized_question TEXT NOT NULL,
                    domain TEXT,
                    mode TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    reason TEXT,
                    local_used INTEGER NOT NULL,
                    gemini_used INTEGER NOT NULL,
                    sources_json TEXT NOT NULL,
                    expired_sources_json TEXT NOT NULL,
                    low_confidence_sources_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_messages_cache
                ON chat_messages(cache_key, id DESC)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_messages_created
                ON chat_messages(created_at DESC)
                """
            )

    def cache_key(self, question: str, domain: str | None) -> tuple[str, str]:
        normalized = " ".join(question.strip().lower().split())
        raw_key = json.dumps(
            {"domain": domain or "", "question": normalized},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest(), normalized

    def get_cached_fallback(self, cache_key: str) -> ChatResponse | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM chat_messages
                WHERE cache_key = ? AND mode = 'gemini_fallback' AND gemini_used = 1
                ORDER BY id DESC
                LIMIT 1
                """,
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        metadata = json.loads(row["metadata_json"] or "{}")
        metadata["cache_hit"] = True
        metadata["cache_source"] = "sqlite"
        return ChatResponse(
            mode=row["mode"],
            answer=row["answer"],
            reason=row["reason"] or "cached_gemini_fallback",
            local_used=bool(row["local_used"]),
            gemini_used=False,
            sources=json.loads(row["sources_json"] or "[]"),
            expired_sources=json.loads(row["expired_sources_json"] or "[]"),
            low_confidence_sources=json.loads(row["low_confidence_sources_json"] or "[]"),
            metadata=metadata,
        )

    def save(
        self,
        *,
        cache_key: str,
        normalized_question: str,
        question: str,
        domain: str | None,
        response: ChatResponse,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO chat_messages (
                    created_at,
                    cache_key,
                    question,
                    normalized_question,
                    domain,
                    mode,
                    answer,
                    reason,
                    local_used,
                    gemini_used,
                    sources_json,
                    expired_sources_json,
                    low_confidence_sources_json,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    cache_key,
                    question,
                    normalized_question,
                    domain,
                    response.mode,
                    response.answer,
                    response.reason,
                    int(response.local_used),
                    int(response.gemini_used),
                    json.dumps(response.sources, ensure_ascii=False),
                    json.dumps(response.expired_sources, ensure_ascii=False),
                    json.dumps(response.low_confidence_sources, ensure_ascii=False),
                    json.dumps(response.metadata, ensure_ascii=False),
                ),
            )

    def recent(self, limit: int = 30) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 100))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, created_at, question, domain, mode, gemini_used
                FROM chat_messages
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]


class RagEngine:
    def __init__(self, store: ChatStore) -> None:
        self._lock = RLock()
        self.store = store
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
        cache_key, normalized_question = self.store.cache_key(question, domain)
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
            response = ChatResponse(
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
            self.store.save(
                cache_key=cache_key,
                normalized_question=normalized_question,
                question=question,
                domain=domain,
                response=response,
            )
            return response

        if not fallback_enabled:
            fallback_notice = self._fallback_notice(decision)
            response = ChatResponse(
                mode="fallback_required",
                answer=(
                    f"{fallback_notice}\n\n"
                    "Không có căn cứ nội bộ đủ tin cậy để trả lời. "
                    "Gemini fallback đang tắt."
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
            self.store.save(
                cache_key=cache_key,
                normalized_question=normalized_question,
                question=question,
                domain=domain,
                response=response,
            )
            return response

        cached_response = self.store.get_cached_fallback(cache_key)
        if cached_response is not None:
            self.store.save(
                cache_key=cache_key,
                normalized_question=normalized_question,
                question=question,
                domain=domain,
                response=cached_response,
            )
            return cached_response

        try:
            fallback_notice = self._fallback_notice(decision)
            fallback_result = generate_gemini_fallback_answer(
                request=GeminiFallbackRequest(
                    question=question,
                    reason=decision.reason,
                    domain=domain,
                    fallback_notice=fallback_notice,
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
            fallback_notice = self._fallback_notice(decision)
            response = ChatResponse(
                mode="gemini_error",
                answer=f"{fallback_notice}\n\nKhông thể gọi Gemini fallback: {exc}",
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
            self.store.save(
                cache_key=cache_key,
                normalized_question=normalized_question,
                question=question,
                domain=domain,
                response=response,
            )
            return response

        response = ChatResponse(
            mode="gemini_fallback",
            answer=fallback_result.answer,
            reason=decision.reason,
            local_used=False,
            gemini_used=True,
            sources=fallback_result.sources,
            expired_sources=[
                self._source_payload(item) for item in decision.expired_results
            ],
            low_confidence_sources=[
                self._source_payload(item) for item in decision.low_confidence_results
            ],
            metadata=self._response_metadata(),
        )
        self.store.save(
            cache_key=cache_key,
            normalized_question=normalized_question,
            question=question,
            domain=domain,
            response=response,
        )
        return response

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
        snippets: list[str] = []
        seen_snippets: set[str] = set()
        per_snippet_chars = max(500, min(snippet_chars, 1100))
        for item in results[:3]:
            snippet = self._clean_snippet(item.text, snippet_chars=per_snippet_chars)
            key = re.sub(r"\W+", "", snippet[:140].lower())
            if not snippet or key in seen_snippets:
                continue
            seen_snippets.add(key)
            snippets.append(snippet)
        answer_text = "\n\n".join(snippets)
        citations = [self._citation_payload(item) for item in results]
        seen_citations: set[str] = set()
        deduped_citations: list[dict[str, str]] = []
        for citation in citations:
            key = citation["citation"]
            if key in seen_citations:
                continue
            seen_citations.add(key)
            deduped_citations.append(citation)

        lines = [answer_text]
        if deduped_citations:
            lines.append("Căn cứ pháp lý:")
            for citation in deduped_citations[:5]:
                lines.append(f"- {citation['citation']}")
        return "\n".join(line for line in lines if line.strip())

    def _source_payload(self, item: RetrievedItem) -> dict[str, Any]:
        payload = asdict(item)
        payload.update(self._citation_payload(item))
        return payload

    def _clean_snippet(self, text: str, snippet_chars: int) -> str:
        text = text.replace("\ufeff", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        article_match = ARTICLE_HEADING_RE.search(text)
        if article_match:
            start = max(0, article_match.start())
            text = text[start:]
        else:
            text = self._drop_leading_fragment(text)
        snippet = self._truncate_to_complete_sentence(text, snippet_chars).strip()
        return re.sub(r"\n{3,}", "\n\n", snippet)

    def _drop_leading_fragment(self, text: str) -> str:
        if not text:
            return text
        lines = text.splitlines()
        if len(lines) > 1:
            first_line = lines[0].strip()
            second_line = lines[1].strip()
            if (
                first_line
                and second_line
                and first_line[:1].islower()
                and (second_line[:1].isupper() or second_line.startswith(("Điều ", "Mục ", "Chương ")))
                and len(first_line) < 180
            ):
                return "\n".join(lines[1:]).lstrip()
        if text[:1].isupper() or text[:1].isdigit() or text.startswith(("Điều ", "Mục ", "Chương ")):
            return text
        legal_boundary = LEADING_LEGAL_BOUNDARY_RE.search(text)
        sentence_boundary = SENTENCE_BOUNDARY_RE.search(text)
        candidates = [
            match.start()
            for match in (legal_boundary, sentence_boundary)
            if match is not None and 0 < match.start() < 260
        ]
        if not candidates:
            return text
        start = min(candidates)
        return text[start:].lstrip()

    def _truncate_to_complete_sentence(self, text: str, snippet_chars: int) -> str:
        if len(text) <= snippet_chars:
            return self._trim_trailing_fragment(text)
        window = text[:snippet_chars].rstrip()
        min_boundary = max(120, int(snippet_chars * 0.55))
        boundaries = []
        for legal_heading in ("\nĐiều ", "\nMục ", "\nChương "):
            index = window.rfind(legal_heading, min_boundary)
            if index != -1:
                boundaries.append(index)
        for match in SENTENCE_BOUNDARY_RE.finditer(window):
            if match.start() >= min_boundary:
                boundaries.append(match.start())
        end = max(boundaries) if boundaries else -1
        if end <= min_boundary:
            end = max(window.rfind("\n", min_boundary), window.rfind(". ", min_boundary))
        if end > min_boundary:
            return window[:end].rstrip()
        return self._trim_trailing_fragment(window.rstrip(" ,;:"))

    def _trim_trailing_fragment(self, text: str) -> str:
        text = text.rstrip()
        if not text or text[-1] in ".;:!?":
            return text
        matches = list(re.finditer(r"[.!?;:](?=\s|$)", text))
        if not matches:
            return text
        end = matches[-1].end()
        if end < max(80, int(len(text) * 0.45)):
            return text
        return text[:end].rstrip()

    def _citation_label(self, item: RetrievedItem) -> str:
        return self._citation_payload(item)["citation"]

    def _citation_payload(self, item: RetrievedItem) -> dict[str, str]:
        source_path = Path(item.source_file)
        stem = source_path.stem
        domain = DOMAIN_LABELS.get(item.domain, item.domain or source_path.parent.name)
        document_label = self._document_label(item, stem)
        clause, article = self._section_label(item.text)
        year = self._document_year(stem, item.document_number)
        document_number = self._document_number(item, stem)

        parts: list[str] = []
        if clause:
            parts.append(f"khoản {clause}")
        if article:
            parts.append(f"Điều {article}")
        parts.append(document_label)
        if domain:
            parts.append(f"ngành {domain}")
        if year:
            parts.append(f"năm {year}")
        return {
            "citation": " ".join(parts),
            "document_number": document_number,
        }

    def _document_label(self, item: RetrievedItem, stem: str) -> str:
        pieces = stem.split("_")
        doc_type = self._document_type_label(stem, item.document_number)
        if item.document_number:
            return f"{doc_type} {item.document_number}"
        number_match = re.search(r"(\d+)[_/.-](\d{4})", stem)
        if number_match:
            number = f"{number_match.group(1)}/{number_match.group(2)}"
            suffix = ""
            if "ND-CP" in stem or "ND_CP" in stem:
                suffix = "/NĐ-CP"
            elif "QH" in stem:
                qh_match = re.search(r"QH(\d+)", stem)
                suffix = f"/QH{qh_match.group(1)}" if qh_match else "/QH"
            elif "TT" in stem:
                suffix = "/TT"
            return f"{doc_type} {number}{suffix}"
        if item.document_title:
            return item.document_title
        return stem.replace("_", " ")

    def _document_number(self, item: RetrievedItem, stem: str) -> str:
        if item.document_number:
            return item.document_number
        number_match = re.search(r"(\d+)[_/.-](\d{4})", stem)
        if not number_match:
            return ""
        suffix = ""
        if "ND-CP" in stem or "ND_CP" in stem:
            suffix = "/NĐ-CP"
        elif "QH" in stem:
            qh_match = re.search(r"QH(\d+)", stem)
            suffix = f"/QH{qh_match.group(1)}" if qh_match else "/QH"
        elif "TT" in stem:
            suffix = "/TT"
        return f"{number_match.group(1)}/{number_match.group(2)}{suffix}"

    def _document_type_label(self, stem: str, document_number: str) -> str:
        normalized = f"{stem}_{document_number}".upper()
        if "QH" in normalized:
            return "Luật"
        if "ND-CP" in normalized or "ND_CP" in normalized or "NĐ-CP" in normalized:
            return "Nghị định"
        if "TT" in normalized:
            return "Thông tư"
        pieces = stem.split("_")
        return DOCUMENT_TYPE_LABELS.get(pieces[0], pieces[0] if pieces else "Tài liệu")

    def _document_year(self, stem: str, document_number: str) -> str:
        match = re.search(r"(20\d{2}|19\d{2})", document_number or stem)
        return match.group(1) if match else ""

    def _section_label(self, text: str) -> tuple[str, str]:
        heading_match = ARTICLE_HEADING_CAPTURE_RE.search(text)
        if heading_match:
            article = heading_match.group(1)
            clause_match = CLAUSE_RE.search(text[heading_match.end() :])
            clause = clause_match.group(1) if clause_match else ""
            return clause, article

        section_match = SECTION_RE.search(text)
        if section_match:
            clause = section_match.group(1) or ""
            article = section_match.group(2) or ""
        else:
            clause = ""
            article_match = ARTICLE_RE.search(text)
            article = article_match.group(2) if article_match else ""

        if not clause:
            clause_match = CLAUSE_RE.search(text)
            if clause_match:
                clause = clause_match.group(1)
        return clause, article

    def _fallback_notice(self, decision: Any) -> str:
        if decision.expired_results:
            item = decision.expired_results[0]
            label = item.document_title or item.document_number or item.source_file
            suffix = f" ({item.expiry_date})" if item.expiry_date else ""
            return f"tài liệu {label}{suffix} hết hạn, cần cập nhật"
        return "không tìm thấy tài liệu làm căn cứ cho câu hỏi trong kho nội bộ"

    def _response_metadata(self) -> dict[str, Any]:
        return {
            "index_backend": self.backend,
            "embedding_model": self.embedding_model,
            "index": self.index_display,
            "total_chunks": self.manifest.get("total_chunks"),
            "domains": self.manifest.get("domains", []),
            "chat_db": str(CHAT_DB_PATH),
        }


chat_store = ChatStore(CHAT_DB_PATH)
engine = RagEngine(chat_store)

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
        "chat_db": str(CHAT_DB_PATH),
    }


@app.get("/history")
def history(limit: int = 30) -> dict[str, Any]:
    return {
        "items": chat_store.recent(limit=limit),
        "db_path": str(CHAT_DB_PATH),
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
