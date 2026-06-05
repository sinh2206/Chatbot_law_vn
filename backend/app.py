from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime, time, timedelta, timezone
from email.message import EmailMessage
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import smtplib
from threading import RLock
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import (
    CHAT_DB_PATH,
    EMBEDDING_BATCH_SIZE,
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
    REPORT_DAILY_TIME,
    REPORT_SCHEDULER_ENABLED,
    REPORT_TIMEZONE,
    REPORT_WATCHLIST_TOPICS,
    REPORTS_DIR,
    SMTP_FROM_EMAIL,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_USERNAME,
    SMTP_USE_TLS,
    TOP_K,
)
from scripts.OCR import extract_pdf_text as extract_pdf_text_file
from scripts.OCR import ocr_file_to_text
from scripts.build_vector_store import build_legal_chunks_for_file, clean_legal_text
from scripts.build_vector_store import split_text_into_chunks
from scripts.gemini_fallback import (
    GeminiFallbackRequest,
    generate_gemini_fallback_answer,
)
from scripts.query_cli import (
    RetrievedItem,
    attach_document_status,
    decide_retrieval,
    detect_requested_documents,
    document_labels,
    format_source,
    load_embedder,
    load_faiss_index,
    load_legal_document_status,
    load_metadata,
    load_numpy_embeddings,
    read_manifest,
    recover_internal_decision,
    search,
    search_requested_local_documents,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"
UPLOADS_DIR = ROOT_DIR / "data" / "uploads"

DOMAIN_LABELS = {
    "CCCD": "CCCD",
    "DatDai": "Đất đai",
    "DoanhNghiep": "Doanh nghiệp",
    "HoTich": "Hộ tịch",
    "Thue": "Thuế",
    "Uploaded": "Tài liệu tải lên",
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
LIST_MARKER_RE = re.compile(r"^\s*(?:\d+[a-zA-Z]?\.|[a-zà-ỹđ]\))\s+")
SECTION_RE = re.compile(r"(?:(?:khoản|Khoản)\s+(\d+[a-zA-Z]?))?\s*(?:Điều|điều)\s+(\d+[a-zA-Z]?)")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?;:])\s+(?=[A-ZĐ0-9])")
LEADING_LEGAL_BOUNDARY_RE = re.compile(
    r"(?m)(^Điều\s+\d+[a-zA-Z]?\s*[\\.:]|^\d+[a-zA-Z]?\.\s+|^[a-z]\)\s+)"
)
WORD_RE = re.compile(r"[0-9a-zA-ZÀ-ỹĐđ]+")
LEGAL_HEADING_LINE_RE = re.compile(r"^\s*(?:Điều\s+\d+[a-zA-Z]?\s*[\\.:]|Mục\s+|Chương\s+)", re.IGNORECASE)
PDF_TEXT_MIN_CHARS = 300
PDF_TEMP_CHUNK_SIZE = 1600
PDF_TEMP_CHUNK_OVERLAP = 220


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


class PdfChatResponse(ChatResponse):
    uploaded_file: dict[str, Any]


class ReportSubscriberRequest(BaseModel):
    email: str = Field(..., min_length=3)
    name: str | None = None
    active: bool = True


class ReportSendRequest(BaseModel):
    recipients: list[str] | None = None
    force: bool = False


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
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS report_subscribers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE,
                    name TEXT,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
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

    def get_cached_fallback(
        self,
        cache_key: str,
        modes: tuple[str, ...] = ("gemini_fallback",),
    ) -> ChatResponse | None:
        placeholders = ",".join("?" for _ in modes)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT *
                FROM chat_messages
                WHERE cache_key = ? AND mode IN ({placeholders}) AND gemini_used = 1
                ORDER BY id DESC
                LIMIT 1
                """,
                (cache_key, *modes),
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

    def recent_messages(self, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT created_at, question, domain, mode, local_used, gemini_used, reason
                FROM chat_messages
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_subscriber(
        self,
        *,
        email: str,
        name: str | None,
        active: bool,
    ) -> dict[str, Any]:
        normalized_email = email.strip().lower()
        if "@" not in normalized_email:
            raise ValueError("Email không hợp lệ.")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO report_subscribers(email, name, active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(email) DO UPDATE SET
                    name = excluded.name,
                    active = excluded.active,
                    updated_at = excluded.updated_at
                """,
                (
                    normalized_email,
                    (name or "").strip() or None,
                    int(active),
                    now,
                    now,
                ),
            )
            row = connection.execute(
                """
                SELECT id, email, name, active, created_at, updated_at
                FROM report_subscribers
                WHERE email = ?
                """,
                (normalized_email,),
            ).fetchone()
        return dict(row)

    def subscribers(self, active_only: bool = True) -> list[dict[str, Any]]:
        where = "WHERE active = 1" if active_only else ""
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT id, email, name, active, created_at, updated_at
                FROM report_subscribers
                {where}
                ORDER BY email
                """
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

    def query_pdf(
        self,
        *,
        pdf_path: Path,
        original_filename: str,
        request: ChatRequest,
    ) -> PdfChatResponse:
        self.load()
        question = (request.message or request.query or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="message/query is required")

        pdf_text, extraction_method, page_count = self._extract_pdf_text(pdf_path)
        pdf_chunks = self._chunk_uploaded_pdf_text(
            text=pdf_text,
            original_filename=original_filename,
        )
        if not pdf_chunks:
            raise RuntimeError("Không trích xuất được nội dung đủ dài từ PDF.")

        domain = request.domain.strip() if request.domain else None
        internal_domain = None if domain == "Uploaded" else domain
        internal_decision = self._retrieve_internal_decision(
            question=question,
            request=request,
            domain=internal_domain,
        )
        pdf_results = self._search_temporary_pdf_chunks(
            chunks=pdf_chunks,
            query=question,
            top_k=request.top_k,
        )
        pdf_metadata = [
            {
                "domain": item.domain,
                "source_file": item.source_file,
                "chunk_id": item.chunk_id,
            }
            for item in pdf_chunks
        ]
        pdf_decision = decide_retrieval(
            results=pdf_results,
            metadata=pdf_metadata,
            domain="Uploaded",
            min_score=request.min_score * 0.45,
        )

        metadata = self._response_metadata()
        metadata["uploaded_pdf"] = {
            "original_filename": original_filename,
            "saved_pdf": str(pdf_path),
            "extraction_method": extraction_method,
            "page_count": page_count,
            "text_chars": len(pdf_text),
            "temporary_chunks": len(pdf_chunks),
            "temporary_search": True,
            "role": "primary_uploaded_context",
        }
        metadata["internal_rag"] = {
            "used": internal_decision.use_internal,
            "reason": internal_decision.reason,
            "domain": internal_domain,
        }

        uploaded_file = {
            "original_filename": original_filename,
            "saved_pdf": str(pdf_path),
            "extraction_method": extraction_method,
            "page_count": page_count,
            "text_chars": len(pdf_text),
            "temporary_chunks": len(pdf_chunks),
        }

        chosen_pdf_results = self._primary_pdf_results(
            pdf_results=pdf_results,
            usable_pdf_results=pdf_decision.usable_results,
            min_score=request.min_score,
        )
        if chosen_pdf_results:
            chosen_internal_results = (
                internal_decision.usable_results if internal_decision.use_internal else []
            )
            combined_results = chosen_pdf_results + chosen_internal_results
            answer = self._format_answer_with_priority_pdf(
                pdf_results=chosen_pdf_results,
                internal_results=chosen_internal_results,
                question=question,
                snippet_chars=request.snippet_chars,
            )
            metadata["pdf_primary_used"] = True
            metadata["internal_supplement_used"] = bool(chosen_internal_results)
            response = PdfChatResponse(
                mode=(
                    "pdf_primary_with_local_rag"
                    if chosen_internal_results
                    else "pdf_primary_rag"
                ),
                answer=answer,
                reason=(
                    "Đã ưu tiên căn cứ từ PDF người dùng tải lên, sau đó bổ sung "
                    "các ý có căn cứ trong data/processed."
                    if chosen_internal_results
                    else "Đã dùng căn cứ từ PDF người dùng tải lên theo mức ưu tiên cao nhất."
                ),
                local_used=True,
                gemini_used=False,
                sources=[self._source_payload(item) for item in combined_results],
                expired_sources=[
                    self._source_payload(item)
                    for item in internal_decision.expired_results
                ],
                low_confidence_sources=[
                    self._source_payload(item)
                    for item in (
                        internal_decision.low_confidence_results
                        + pdf_decision.low_confidence_results
                    )
                ],
                metadata=metadata,
                uploaded_file=uploaded_file,
            )
            return response

        if internal_decision.use_internal:
            answer = self._format_local_answer(
                internal_decision.usable_results,
                question=question,
                snippet_chars=request.snippet_chars,
            )
            metadata["pdf_primary_used"] = False
            metadata["internal_supplement_used"] = True
            return PdfChatResponse(
                mode="local_rag_pdf_not_matched",
                answer=answer,
                reason=(
                    "PDF đã được xử lý nhưng không tìm thấy đoạn đủ tin cậy; "
                    "đã dùng căn cứ có sẵn trong data/processed."
                ),
                local_used=True,
                gemini_used=False,
                sources=[
                    self._source_payload(item) for item in internal_decision.usable_results
                ],
                expired_sources=[
                    self._source_payload(item)
                    for item in internal_decision.expired_results
                ],
                low_confidence_sources=[
                    self._source_payload(item)
                    for item in (
                        internal_decision.low_confidence_results
                        + pdf_decision.low_confidence_results
                    )
                ],
                metadata=metadata,
                uploaded_file=uploaded_file,
            )

        fallback_enabled = (
            GEMINI_FALLBACK_ENABLED
            if request.gemini_fallback is None
            else request.gemini_fallback
        )
        if not fallback_enabled:
            return PdfChatResponse(
                mode="pdf_no_match",
                answer=(
                    "Không tìm thấy đoạn phù hợp trong PDF đã tải lên để làm căn cứ "
                    "trả lời câu hỏi. Gemini fallback đang tắt."
                ),
                reason="Các chunk tạm từ PDF có điểm truy hồi thấp.",
                local_used=False,
                gemini_used=False,
                sources=[],
                expired_sources=[],
                low_confidence_sources=[
                    self._source_payload(item)
                    for item in (
                        internal_decision.low_confidence_results
                        + pdf_decision.low_confidence_results
                    )
                ],
                metadata=metadata,
                uploaded_file=uploaded_file,
            )

        context = "\n\n".join(
            self._clean_snippet(item.text, snippet_chars=900)
            for item in pdf_results[:3]
            if item.text.strip()
        )
        try:
            fallback_result = generate_gemini_fallback_answer(
                request=GeminiFallbackRequest(
                    question=(
                        f"{question}\n\n"
                        "Ngữ cảnh trích từ PDF người dùng tải lên, chỉ dùng để tham khảo "
                        "nếu phù hợp:\n"
                        f"{context}"
                    ),
                    reason=(
                        "PDF da duoc trich xuat va chunk tam, nhung diem truy hoi "
                        "chua du nguong tin cay."
                    ),
                    domain=request.domain,
                    fallback_notice=(
                        "không tìm thấy tài liệu làm căn cứ cho câu hỏi trong PDF tải lên"
                    ),
                    low_confidence_sources=[
                        format_source(item) for item in pdf_results[:5]
                    ],
                ),
                api_key=GEMINI_API_KEY,
                model_name=request.gemini_model or GEMINI_MODEL,
            )
        except Exception as exc:
            return PdfChatResponse(
                mode="pdf_gemini_error",
                answer=(
                    "Không tìm thấy đoạn đủ tin cậy trong PDF đã tải lên.\n"
                    f"Không thể gọi Gemini fallback: {exc}"
                ),
                reason="Các chunk tạm từ PDF có điểm truy hồi thấp.",
                local_used=False,
                gemini_used=False,
                sources=[],
                expired_sources=[],
                low_confidence_sources=[
                    self._source_payload(item)
                    for item in (
                        internal_decision.low_confidence_results
                        + pdf_decision.low_confidence_results
                    )
                ],
                metadata=metadata,
                uploaded_file=uploaded_file,
            )

        return PdfChatResponse(
            mode="pdf_gemini_fallback",
            answer=fallback_result.answer,
            reason="Các chunk tạm từ PDF có điểm truy hồi thấp, đã gọi Gemini fallback.",
            local_used=False,
            gemini_used=True,
            sources=fallback_result.sources,
            expired_sources=[],
            low_confidence_sources=[
                self._source_payload(item)
                for item in (
                    internal_decision.low_confidence_results
                    + pdf_decision.low_confidence_results
                )
            ],
            metadata=metadata,
            uploaded_file=uploaded_file,
        )

    def _retrieve_internal_decision(
        self,
        *,
        question: str,
        request: ChatRequest,
        domain: str | None,
    ):
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
        decision = recover_internal_decision(
            decision=decision,
            query=question,
            metadata=self.metadata,
            min_score=request.min_score,
        )
        if decision.use_internal:
            return decision

        requested_local_results = search_requested_local_documents(
            backend=self.backend,
            index=self.index,
            metadata=self.metadata,
            embedder=self.embedder,
            query=question,
            top_k=request.top_k,
        )
        requested_local_results = attach_document_status(
            requested_local_results,
            self.document_statuses,
        )
        requested_local_decision = decide_retrieval(
            results=requested_local_results,
            metadata=self.metadata,
            domain=domain,
            min_score=request.min_score * 0.35,
        )
        if requested_local_decision.use_internal:
            return type(decision)(
                use_internal=True,
                reason=(
                    "Tìm thấy căn cứ một phần trong kho nội bộ từ văn bản pháp luật "
                    "được nhận diện theo nội dung câu hỏi."
                ),
                usable_results=requested_local_decision.usable_results,
                expired_results=decision.expired_results
                + requested_local_decision.expired_results,
                low_confidence_results=requested_local_decision.low_confidence_results,
            )
        return decision

    def _primary_pdf_results(
        self,
        *,
        pdf_results: list[RetrievedItem],
        usable_pdf_results: list[RetrievedItem],
        min_score: float,
    ) -> list[RetrievedItem]:
        if not pdf_results:
            return []
        if usable_pdf_results:
            return usable_pdf_results[:3]
        relaxed_min_score = min_score * 0.35
        return [item for item in pdf_results if item.score >= relaxed_min_score][:3]

    def _format_answer_with_priority_pdf(
        self,
        *,
        pdf_results: list[RetrievedItem],
        internal_results: list[RetrievedItem],
        question: str,
        snippet_chars: int,
    ) -> str:
        pdf_answer = self._format_local_answer(
            pdf_results,
            question=question,
            snippet_chars=snippet_chars,
        )
        pdf_answer = self._strip_citation_block(pdf_answer)
        if not internal_results:
            citations = [self._citation_payload(item)["citation"] for item in pdf_results]
            return self._append_citations(pdf_answer, citations)

        internal_answer = self._format_local_answer(
            internal_results,
            question=question,
            snippet_chars=snippet_chars,
        )
        internal_answer = self._strip_citation_block(internal_answer)
        citations = [
            self._citation_payload(item)["citation"]
            for item in pdf_results + internal_results
        ]

        lines = [
            pdf_answer,
            internal_answer,
        ]
        return self._append_citations(
            self._dedupe_answer_lines("\n".join(line for line in lines if line.strip())),
            citations,
        )

    def _append_citations(self, answer: str, citations: list[str]) -> str:
        return answer.strip()

    def _dedupe_answer_lines(self, answer: str) -> str:
        lines: list[str] = []
        seen: set[str] = set()
        for raw_line in answer.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            key = re.sub(r"\W+", "", line.lower())
            if key in seen:
                continue
            seen.add(key)
            lines.append(line)
        return "\n".join(lines)

    def _strip_citation_block(self, answer: str) -> str:
        return re.sub(r"(?is)\n?Căn cứ pháp lý:\n(?:- .+\n?)+", "", answer).strip()

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
        decision = recover_internal_decision(
            decision=decision,
            query=question,
            metadata=self.metadata,
            min_score=request.min_score,
        )
        if not decision.use_internal:
            requested_local_results = search_requested_local_documents(
                backend=self.backend,
                index=self.index,
                metadata=self.metadata,
                embedder=self.embedder,
                query=question,
                top_k=request.top_k,
            )
            requested_local_results = attach_document_status(
                requested_local_results,
                self.document_statuses,
            )
            requested_local_decision = decide_retrieval(
                results=requested_local_results,
                metadata=self.metadata,
                domain=domain,
                min_score=request.min_score * 0.35,
            )
            if requested_local_decision.use_internal:
                decision = type(decision)(
                    use_internal=True,
                    reason=(
                        "Tìm thấy căn cứ một phần trong kho nội bộ từ văn bản pháp luật "
                        "được nhận diện theo nội dung câu hỏi."
                    ),
                    usable_results=requested_local_decision.usable_results,
                    expired_results=decision.expired_results
                    + requested_local_decision.expired_results,
                    low_confidence_results=requested_local_decision.low_confidence_results,
                )

        fallback_enabled = (
            GEMINI_FALLBACK_ENABLED
            if request.gemini_fallback is None
            else request.gemini_fallback
        )
        gemini_model = request.gemini_model or GEMINI_MODEL

        if decision.use_internal:
            local_answer = self._format_local_answer(
                decision.usable_results,
                question=question,
                snippet_chars=request.snippet_chars,
            )
            local_sources = [self._source_payload(item) for item in decision.usable_results]
            expired_sources = [
                self._source_payload(item) for item in decision.expired_results
            ]
            low_confidence_sources = [
                self._source_payload(item) for item in decision.low_confidence_results
            ]
            requested_documents = detect_requested_documents(question, self.metadata)
            available_documents = document_labels(requested_documents, available=True)
            missing_documents = document_labels(requested_documents, available=False)
            metadata = self._response_metadata()
            if requested_documents:
                metadata["requested_documents"] = [
                    asdict(document) for document in requested_documents
                ]

            if missing_documents and fallback_enabled:
                cached_response = self.store.get_cached_fallback(
                    cache_key,
                    modes=("local_rag_with_api_supplement",),
                )
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
                    supplement_result = generate_gemini_fallback_answer(
                        request=GeminiFallbackRequest(
                            question=question,
                            reason=(
                                "RAG noi bo da co can cu cho cac van ban co san; "
                                "chi can tra cuu bo sung cac van ban con thieu."
                            ),
                            domain=domain,
                            available_local_documents=available_documents,
                            missing_documents=missing_documents,
                            local_answer=local_answer,
                        ),
                        api_key=GEMINI_API_KEY,
                        model_name=gemini_model,
                    )
                    metadata["api_supplement_documents"] = missing_documents
                    response = ChatResponse(
                        mode="local_rag_with_api_supplement",
                        answer=self._merge_api_supplement(
                            local_answer=local_answer,
                            supplement_answer=supplement_result.answer,
                            missing_documents=missing_documents,
                        ),
                        reason=(
                            f"{decision.reason} Bo sung API chi cho tai lieu con thieu."
                        ),
                        local_used=True,
                        gemini_used=True,
                        sources=local_sources + supplement_result.sources,
                        expired_sources=expired_sources,
                        low_confidence_sources=low_confidence_sources,
                        metadata=metadata,
                    )
                    self.store.save(
                        cache_key=cache_key,
                        normalized_question=normalized_question,
                        question=question,
                        domain=domain,
                        response=response,
                    )
                    return response
                except Exception as exc:
                    metadata["api_supplement_error"] = str(exc)
                    local_answer = (
                        f"{local_answer}\n\n"
                        "Tài liệu chưa có trong kho nội bộ cần bổ sung: "
                        f"{', '.join(missing_documents)}.\n"
                        f"Không thể gọi Gemini để bổ sung: {exc}"
                    )

            elif missing_documents:
                metadata["missing_documents"] = missing_documents
                local_answer = (
                    f"{local_answer}\n\n"
                    "Tài liệu chưa có trong kho nội bộ cần bổ sung: "
                    f"{', '.join(missing_documents)}."
                )

            response = ChatResponse(
                mode="local_rag",
                answer=local_answer,
                reason=decision.reason,
                local_used=True,
                gemini_used=False,
                sources=local_sources,
                expired_sources=expired_sources,
                low_confidence_sources=low_confidence_sources,
                metadata=metadata,
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

    def _extract_pdf_text(self, pdf_path: Path) -> tuple[str, str, int]:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required to read PDF uploads.") from exc

        with fitz.open(pdf_path) as document:
            page_count = document.page_count

        extracted_text, extraction_method = extract_pdf_text_file(
            path=pdf_path,
            min_chars=PDF_TEXT_MIN_CHARS,
        )
        if len(extracted_text) >= PDF_TEXT_MIN_CHARS:
            return extracted_text, extraction_method, page_count

        ocr_text = ocr_file_to_text(
            source_path=pdf_path,
            lang="vie+eng",
            pdf_dpi=220,
        )
        if len(ocr_text) < 30:
            raise RuntimeError("PDF không có text thật và OCR trích xuất quá ít nội dung.")
        return clean_legal_text(ocr_text), "ocr_tesseract", page_count

    def _chunk_uploaded_pdf_text(
        self,
        *,
        text: str,
        original_filename: str,
    ) -> list[RetrievedItem]:
        safe_stem = self._safe_file_stem(original_filename)
        source_file = f"Uploaded/{safe_stem}.pdf"
        legal_chunks = build_legal_chunks_for_file(
            text=clean_legal_text(text),
            source_file=source_file,
            domain="Uploaded",
            chunk_size=PDF_TEMP_CHUNK_SIZE,
        )
        if any(chunk.article_id for chunk in legal_chunks):
            legal_chunks = [chunk for chunk in legal_chunks if chunk.article_id]
        raw_chunks = legal_chunks or [
            RetrievedItem(
                score=0.0,
                domain="Uploaded",
                source_file=source_file,
                chunk_id=f"{source_file}::temp_chunk_{index + 1}",
                chunk_index=index + 1,
                text=chunk,
                chunk_type="pdf_temp",
                document_title=Path(original_filename).stem,
            )
            for index, chunk in enumerate(
                split_text_into_chunks(
                    text=clean_legal_text(text),
                    chunk_size=PDF_TEMP_CHUNK_SIZE,
                    overlap=PDF_TEMP_CHUNK_OVERLAP,
                )
            )
        ]
        chunks: list[RetrievedItem] = []
        for index, chunk in enumerate(raw_chunks):
            clean = chunk.text.strip()
            if len(clean) < 40:
                continue
            chunks.append(
                RetrievedItem(
                    score=0.0,
                    domain="Uploaded",
                    source_file=source_file,
                    chunk_id=chunk.chunk_id,
                    chunk_index=index + 1,
                    text=clean,
                    chunk_type=chunk.chunk_type or "pdf_temp",
                    article_id=chunk.article_id,
                    article_title=chunk.article_title,
                    clause_id=chunk.clause_id,
                    document_title=Path(original_filename).stem,
                )
            )
        return chunks

    def _search_temporary_pdf_chunks(
        self,
        *,
        chunks: list[RetrievedItem],
        query: str,
        top_k: int,
    ) -> list[RetrievedItem]:
        if not chunks:
            return []
        texts = [item.text for item in chunks]
        vectors = self.embedder.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        query_vec = self.embedder.encode(
            [self._expand_pdf_query(query)],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        query_vec = np.asarray(query_vec, dtype=np.float32)[0]
        scores = np.dot(vectors, query_vec)
        order = np.argsort(-scores)[: max(1, top_k)]
        results: list[RetrievedItem] = []
        for idx in order:
            item = chunks[int(idx)]
            results.append(
                RetrievedItem(
                    score=float(scores[int(idx)]),
                    domain=item.domain,
                    source_file=item.source_file,
                    chunk_id=item.chunk_id,
                    chunk_index=item.chunk_index,
                    text=item.text,
                    chunk_type=item.chunk_type,
                    article_id=item.article_id,
                    article_title=item.article_title,
                    clause_id=item.clause_id,
                    document_title=item.document_title,
                    document_number=item.document_number,
                    status=item.status,
                    expiry_date=item.expiry_date,
                    is_expired=item.is_expired,
                )
            )
        return results

    def _expand_pdf_query(self, query: str) -> str:
        normalized = query.lower()
        additions: list[str] = []
        expansions = {
            "phần mềm": [
                "chương trình máy tính",
                "quyền tác giả đối với chương trình máy tính",
                "tác phẩm",
            ],
            "nhãn hiệu": [
                "quyền sở hữu công nghiệp đối với nhãn hiệu",
                "văn bằng bảo hộ nhãn hiệu",
                "chuyển quyền sử dụng nhãn hiệu",
            ],
            "góp vốn": [
                "quyền sở hữu trí tuệ",
                "quyền tác giả",
                "quyền sở hữu công nghiệp",
                "chuyển giao quyền sở hữu",
            ],
        }
        for trigger, values in expansions.items():
            if trigger in normalized:
                additions.extend(values)
        return " ".join([query, *additions])

    def _safe_file_stem(self, filename: str) -> str:
        stem = Path(filename).stem or "uploaded_document"
        stem = re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE).strip("._-")
        return stem[:90] or "uploaded_document"

    def _merge_api_supplement(
        self,
        local_answer: str,
        supplement_answer: str,
        missing_documents: list[str],
    ) -> str:
        supplement = supplement_answer.strip()
        supplement = re.sub(
            r"(?is)^không tìm thấy tài liệu làm căn cứ cho câu hỏi trong kho nội bộ\s*",
            "",
            supplement,
        ).strip()
        if not re.match(r"(?i)^b[oổ]\s+sung\s+t[aà]i\s+li[eệ]u", supplement):
            heading = (
                "Bổ sung tài liệu cần tra cứu qua API: "
                f"{', '.join(missing_documents)}."
            )
            supplement = f"{heading}\n{supplement}" if supplement else heading
        return f"{local_answer.strip()}\n\n{supplement.strip()}".strip()

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
        question: str,
        snippet_chars: int,
    ) -> str:
        answer_text = self._build_concise_answer(results=results, question=question)
        if not answer_text:
            snippets: list[str] = []
            seen_snippets: set[str] = set()
            per_snippet_chars = max(500, min(snippet_chars, 900))
            for item in results[:2]:
                snippet = self._clean_snippet(item.text, snippet_chars=per_snippet_chars)
                key = re.sub(r"\W+", "", snippet[:140].lower())
                if not snippet or key in seen_snippets:
                    continue
                seen_snippets.add(key)
                snippets.append(snippet)
            answer_text = self._format_answer_block("\n\n".join(snippets))
        return answer_text.strip()

    def _format_answer_block(self, text: str) -> str:
        text = text.strip()
        if not text or text.lstrip().startswith("- "):
            return text
        units: list[str] = []
        for block in re.split(r"\n{2,}", text):
            block = re.sub(r"\s+", " ", block).strip()
            if block:
                units.extend(self._split_answer_unit(block))
        return self._format_answer_bullets(units)

    def _build_direct_legal_answer(
        self,
        *,
        results: list[RetrievedItem],
        question: str,
    ) -> str:
        question_text = question.lower()
        context_text = " ".join(item.text for item in results[:8]).lower()
        lines: list[str] = []

        asks_identity = any(
            phrase in question_text
            for phrase in (
                "căn cước",
                "hộ tịch",
                "mã số thuế",
                "thông tin cá nhân",
                "cải chính hộ tịch",
            )
        )
        asks_ip_capital = any(
            phrase in question_text
            for phrase in (
                "phần mềm",
                "chương trình máy tính",
                "nhãn hiệu",
                "sở hữu trí tuệ",
                "tài sản góp vốn",
            )
        )
        asks_fully_paid = any(
            phrase in question_text
            for phrase in (
                "góp đủ",
                "được coi là đã góp đủ",
                "khi nào phần vốn góp",
                "chuyển quyền sở hữu",
            )
        )

        if asks_identity and any(
            phrase in context_text
            for phrase in (
                "cơ sở dữ liệu quốc gia về dân cư",
                "cơ sở dữ liệu căn cước",
                "cải chính hộ tịch",
                "mã số thuế",
                "thông tin đăng ký thuế",
                "cấp đổi thẻ căn cước",
            )
        ):
            lines.append(
                "- Bạn nên cập nhật trước các thông tin đã thay đổi do cải chính hộ tịch, "
                "đặc biệt là thông tin căn cước/cơ sở dữ liệu dân cư và thông tin đăng ký thuế, "
                "để hồ sơ thành lập công ty, mã số thuế và giấy tờ giao dịch thống nhất."
            )

        if asks_ip_capital and any(
            phrase in context_text
            for phrase in (
                "quyền sở hữu trí tuệ",
                "quyền tác giả",
                "chương trình máy tính",
                "nhãn hiệu",
                "tài sản góp vốn",
                "quyền sở hữu công nghiệp",
            )
        ):
            lines.append(
                "- Quyền đối với phần mềm và nhãn hiệu có thể dùng để góp vốn nếu đó là "
                "quyền hợp pháp của bạn, được định giá bằng Đồng Việt Nam và thuộc nhóm "
                "quyền sở hữu trí tuệ/tài sản có thể chuyển giao cho công ty."
            )

        if asks_fully_paid and any(
            phrase in context_text
            for phrase in (
                "chuyển quyền sở hữu",
                "chuyển giao quyền sở hữu",
                "thanh toán xong",
                "góp đủ",
                "sang tên",
                "quyền sở hữu trí tuệ",
            )
        ):
            lines.append(
                "- Phần vốn góp bằng quyền sở hữu trí tuệ chỉ nên coi là đã góp đủ khi "
                "quyền đó đã được chuyển giao hợp pháp cho công ty theo đúng thủ tục, "
                "được công ty ghi nhận và hoàn tất việc định giá/góp vốn; chỉ cam kết góp "
                "hoặc mới nộp hồ sơ nội bộ thì chưa chắc đã đủ."
            )

        if not lines:
            return ""

        if asks_identity and asks_ip_capital and asks_fully_paid:
            lines.append(
                "- Thứ tự xử lý thực tế nên là: cập nhật thông tin cá nhân và thuế cho khớp, "
                "chuẩn bị chứng cứ quyền đối với phần mềm/nhãn hiệu, định giá tài sản góp vốn, "
                "rồi thực hiện thủ tục chuyển giao quyền cho công ty."
            )

        return "\n".join(lines)

    def _build_concise_answer(self, results: list[RetrievedItem], question: str) -> str:
        direct_answer = self._build_direct_legal_answer(results=results, question=question)
        if direct_answer:
            return direct_answer

        terms = self._query_terms(question)
        candidates: list[tuple[float, int, str]] = []
        seen: set[str] = set()
        for item_index, item in enumerate(results[:6]):
            for unit_index, unit in enumerate(self._answer_units(item.text)):
                normalized = re.sub(r"\W+", "", unit.lower())
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                score = self._answer_unit_score(unit=unit, terms=terms, rank=item_index)
                if score <= 0:
                    continue
                candidates.append((score, unit_index, unit))

        candidates.sort(key=lambda row: (-row[0], row[1]))
        selected: list[str] = []
        selected_keys: set[str] = set()
        for _, _, unit in candidates:
            key = re.sub(r"\W+", "", unit[:180].lower())
            if key in selected_keys:
                continue
            selected_keys.add(key)
            selected.append(unit)
            if len(selected) >= 4:
                break

        if not selected:
            return ""

        selected.sort(key=lambda text: self._unit_order_key(text))
        bullets = self._format_answer_bullets(selected)
        return self._trim_answer_length(bullets, max_chars=1400)

    def _answer_units(self, text: str) -> list[str]:
        clean = self._clean_snippet(text, snippet_chars=1800)
        raw_lines = [line.strip() for line in clean.splitlines() if line.strip()]
        units: list[str] = []
        current = ""
        for line in raw_lines:
            if LEGAL_HEADING_LINE_RE.match(line):
                continue
            line = re.sub(r"^[a-zà-ỹđ]\)\s+", "", line, flags=re.IGNORECASE).strip()
            if not line:
                continue
            starts_new = bool(LIST_MARKER_RE.match(line))
            if starts_new and current:
                units.append(current.strip())
                current = line
            elif current:
                current = f"{current} {line}"
            else:
                current = line
            if current.endswith((".", ";", ":")) and len(current) >= 90:
                units.append(current.strip())
                current = ""
        if current:
            units.append(current.strip())
        return [self._normalize_answer_unit(unit) for unit in units if len(unit) >= 45]

    def _normalize_answer_unit(self, unit: str) -> str:
        unit = re.sub(r"\s+", " ", unit).strip()
        unit = re.sub(r"^\d+[a-zA-Z]?\.\s+", "", unit)
        return unit

    def _format_answer_bullets(self, units: list[str]) -> str:
        lines: list[str] = []
        for unit in units:
            for sentence in self._split_answer_unit(unit):
                sentence = sentence.strip(" -")
                if not sentence:
                    continue
                if not sentence.endswith((".", ";", ":", "!", "?")):
                    sentence = f"{sentence}."
                lines.append(f"- {sentence}")
        return "\n".join(lines)

    def _split_answer_unit(self, unit: str) -> list[str]:
        unit = re.sub(r"\s+", " ", unit).strip()
        if len(unit) <= 260:
            return [unit]

        parts = [
            part.strip()
            for part in re.split(r"(?<=[.;:])\s+(?=[A-ZĐ0-9À-Ỹ])", unit)
            if part.strip()
        ]
        if len(parts) > 1:
            return parts

        chunks: list[str] = []
        current = ""
        for part in re.split(r",\s+", unit):
            candidate = f"{current}, {part}".strip(", ") if current else part
            if len(candidate) <= 230:
                current = candidate
                continue
            if current:
                chunks.append(current)
            current = part
        if current:
            chunks.append(current)
        return chunks or [unit]

    def _query_terms(self, question: str) -> set[str]:
        normalized = question.lower()
        expansions = {
            "góp đủ": ["thanh toán xong", "chuyển quyền sở hữu", "tài sản góp vốn"],
            "tiền mặt": ["đồng việt nam", "ngoại tệ", "vàng", "quyền sử dụng đất", "quyền sở hữu trí tuệ"],
            "không có giấy tờ": ["điều 138", "không vi phạm pháp luật đất đai", "không thuộc trường hợp đất được giao không đúng thẩm quyền"],
            "chưa có sổ đỏ": ["cấp giấy chứng nhận", "không có giấy tờ về quyền sử dụng đất"],
            "cập nhật thông tin": ["cơ sở dữ liệu quốc gia về dân cư", "cơ sở dữ liệu căn cước", "đổi thẻ"],
            "thay đổi thông tin": ["đổi thẻ", "cập nhật", "điều chỉnh"],
            "cải chính hộ tịch": ["cập nhật thông tin", "cơ sở dữ liệu quốc gia về dân cư", "cơ sở dữ liệu căn cước"],
            "thông tin căn cước": ["số định danh cá nhân", "cập nhật", "chỉnh sửa thông tin", "cấp đổi thẻ căn cước"],
            "mã số thuế": ["thay đổi thông tin đăng ký thuế", "khớp đúng với cơ sở dữ liệu quốc gia về dân cư"],
            "giấy tờ nhà đất": ["giấy chứng nhận", "người sử dụng đất", "quyền sử dụng đất"],
            "xác minh danh tính": ["nhận biết khách hàng", "xác thực danh tính", "phòng chống rửa tiền"],
            "tài khoản ngân hàng": ["ngân hàng", "xác minh danh tính", "nhận biết khách hàng"],
        }
        term_text = [normalized]
        for trigger, values in expansions.items():
            if trigger in normalized:
                term_text.extend(values)
        tokens = {
            token
            for token in WORD_RE.findall(" ".join(term_text).lower())
            if len(token) >= 3 and token not in {"theo", "trong", "trường", "hợp", "của", "với", "cho", "khi", "nào"}
        }
        phrases = {
            phrase
            for phrase in term_text[1:]
            if len(phrase.split()) >= 2
        }
        return tokens | phrases

    def _answer_unit_score(self, unit: str, terms: set[str], rank: int) -> float:
        normalized = unit.lower()
        score = max(0, 5 - rank)
        for term in terms:
            if term in normalized:
                score += 4 if " " in term else 1
        if any(marker in normalized for marker in ("được cấp", "phải", "được đổi", "chỉ được", "được coi là", "thanh toán xong", "cập nhật")):
            score += 2
        return float(score)

    def _unit_order_key(self, text: str) -> tuple[int, str]:
        match = re.match(r"^(\d+)[a-zA-Z]?\.", text)
        return (int(match.group(1)) if match else 999, text)

    def _trim_answer_length(self, answer: str, max_chars: int) -> str:
        if len(answer) <= max_chars:
            return answer
        cut = answer.rfind(". ", 0, max_chars)
        if cut < max_chars * 0.5:
            cut = answer.rfind("; ", 0, max_chars)
        if cut < max_chars * 0.5:
            return answer[:max_chars].rstrip(" ,;:")
        return answer[: cut + 1].strip()

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
        if item.domain == "Uploaded" or source_path.parts[:1] == ("Uploaded",):
            clause, article = self._section_label(item)
            filename = source_path.name or f"{stem}.pdf"
            if not filename.lower().endswith(".pdf"):
                filename = f"{filename}.pdf"
            parts: list[str] = []
            if clause:
                parts.append(f"khoản {clause}")
            if article:
                parts.append(f"Điều {article}")
            prefix = " ".join(parts).strip()
            citation = f"{prefix} trong {filename}" if prefix else f"trong {filename}"
            return {
                "citation": citation,
                "document_number": "",
            }

        domain = DOMAIN_LABELS.get(item.domain, item.domain or source_path.parent.name)
        document_label = self._document_label(item, stem)
        clause, article = self._section_label(item)
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

    def _section_label(self, item: RetrievedItem) -> tuple[str, str]:
        clause = getattr(item, "clause_id", "") or ""
        article = getattr(item, "article_id", "") or ""
        if article:
            return clause, article

        text = item.text
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


class DailyReportService:
    def __init__(self, store: ChatStore, rag_engine: RagEngine) -> None:
        self.store = store
        self.engine = rag_engine

    def build_report(self) -> tuple[str, Path]:
        now = datetime.now(self._timezone())
        report_date = now.strftime("%Y-%m-%d")
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        markdown = self._build_markdown(now)
        pdf_path = REPORTS_DIR / f"legal-watchlist-report-{report_date}.pdf"
        self._render_pdf(markdown=markdown, output_path=pdf_path)
        markdown_path = REPORTS_DIR / f"legal-watchlist-report-{report_date}.md"
        markdown_path.write_text(markdown, encoding="utf-8")
        return markdown, pdf_path

    def send_daily_report(
        self,
        recipients: list[str] | None = None,
    ) -> dict[str, Any]:
        subscribers = self.store.subscribers(active_only=True)
        recipient_list = recipients or [item["email"] for item in subscribers]
        recipient_list = sorted({email.strip().lower() for email in recipient_list if email.strip()})
        if not recipient_list:
            return {"sent": 0, "recipients": [], "reason": "no_active_subscribers"}
        if not SMTP_HOST or not SMTP_FROM_EMAIL:
            raise RuntimeError("Missing SMTP_HOST or SMTP_FROM_EMAIL in environment.")

        markdown, pdf_path = self.build_report()
        subject = f"Báo cáo danh mục theo dõi pháp luật - {datetime.now(self._timezone()):%d/%m/%Y}"
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = SMTP_FROM_EMAIL
        message["To"] = ", ".join(recipient_list)
        message.set_content(
            "Chào bạn,\n\n"
            "Hệ thống gửi kèm báo cáo danh mục theo dõi pháp luật hằng ngày.\n\n"
            "Nội dung tóm tắt Markdown:\n\n"
            f"{markdown[:3500]}\n\n"
            "File PDF đầy đủ được đính kèm trong email này.\n",
            charset="utf-8",
        )
        message.add_attachment(
            pdf_path.read_bytes(),
            maintype="application",
            subtype="pdf",
            filename=pdf_path.name,
        )

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as smtp:
            if SMTP_USE_TLS:
                smtp.starttls()
            if SMTP_USERNAME:
                smtp.login(SMTP_USERNAME, SMTP_PASSWORD)
            smtp.send_message(message)
        return {
            "sent": len(recipient_list),
            "recipients": recipient_list,
            "pdf_path": str(pdf_path),
        }

    def _build_markdown(self, now: datetime) -> str:
        recent_messages = self.store.recent_messages(limit=100)
        total = len(recent_messages)
        gemini_count = sum(1 for item in recent_messages if item.get("gemini_used"))
        local_count = sum(1 for item in recent_messages if item.get("local_used"))
        lines = [
            "# Báo cáo danh mục theo dõi pháp luật",
            "",
            f"- Thời điểm tạo: {now:%d/%m/%Y %H:%M} ({REPORT_TIMEZONE})",
            f"- Số lượt hỏi gần nhất được phân tích: {total}",
            f"- Lượt dùng căn cứ nội bộ: {local_count}",
            f"- Lượt cần Gemini/API: {gemini_count}",
            "",
            "## Danh mục theo dõi",
            "",
        ]
        for index, topic in enumerate(REPORT_WATCHLIST_TOPICS, start=1):
            analysis = self._analyze_topic(topic)
            lines.extend(
                [
                    f"### {index}. {topic}",
                    "",
                    f"- Chế độ trả lời: {analysis.mode}",
                    f"- Lý do: {analysis.reason}",
                    "",
                    self._compact_markdown_answer(analysis.answer),
                    "",
                ]
            )
        lines.extend(["## Hoạt động gần đây", ""])
        if not recent_messages:
            lines.append("- Chưa có lịch sử hỏi đáp.")
        else:
            for item in recent_messages[:10]:
                question = str(item.get("question", "")).strip()
                mode = str(item.get("mode", "")).strip()
                created_at = str(item.get("created_at", "")).strip()
                lines.append(f"- {created_at}: `{mode}` - {question[:180]}")
        return "\n".join(lines).strip() + "\n"

    def _analyze_topic(self, topic: str) -> ChatResponse:
        try:
            return self.engine.query(
                ChatRequest(
                    message=topic,
                    top_k=5,
                    gemini_fallback=False,
                    snippet_chars=900,
                )
            )
        except Exception as exc:
            return ChatResponse(
                mode="report_error",
                answer=f"- Không thể phân tích chủ đề này: {exc}",
                reason=str(exc),
                local_used=False,
                gemini_used=False,
                sources=[],
                expired_sources=[],
                low_confidence_sources=[],
                metadata={},
            )

    def _compact_markdown_answer(self, answer: str) -> str:
        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        kept: list[str] = []
        for line in lines:
            kept.append(line if line.startswith(("-", "#")) else f"- {line}")
            if len(kept) >= 8:
                break
        return "\n".join(kept) if kept else "- Không có nội dung."

    def _render_pdf(self, markdown: str, output_path: Path) -> None:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required to render PDF reports.") from exc

        document = fitz.open()
        page = document.new_page(width=595, height=842)
        margin = 54
        y = margin
        line_height = 15
        max_width = 78
        font_file = self._pdf_font_file()
        font_name = "dejavu" if font_file else "helv"
        for raw_line in markdown.splitlines():
            if y > 790:
                page = document.new_page(width=595, height=842)
                y = margin
            line = raw_line.strip()
            if not line:
                y += line_height
                continue
            font_size = 16 if line.startswith("# ") else 13 if line.startswith("##") else 11
            text = line.lstrip("#").strip()
            wrapped = self._wrap_text(text, max_width=max_width)
            for wrapped_line in wrapped:
                if y > 790:
                    page = document.new_page(width=595, height=842)
                    y = margin
                page.insert_text(
                    (margin, y),
                    wrapped_line,
                    fontsize=font_size,
                    fontname=font_name,
                    fontfile=font_file,
                )
                y += line_height + (4 if font_size > 11 else 0)
        document.save(output_path)
        document.close()

    def _pdf_font_file(self) -> str | None:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]
        for candidate in candidates:
            if Path(candidate).exists():
                return candidate
        return None

    def _wrap_text(self, text: str, max_width: int) -> list[str]:
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) <= max_width:
                current = candidate
                continue
            if current:
                lines.append(current)
            current = word
        if current:
            lines.append(current)
        return lines or [text]

    def _timezone(self) -> ZoneInfo:
        try:
            return ZoneInfo(REPORT_TIMEZONE)
        except Exception:
            return ZoneInfo("Asia/Ho_Chi_Minh")


def _next_report_run(now: datetime) -> datetime:
    hour_text, _, minute_text = REPORT_DAILY_TIME.partition(":")
    hour = int(hour_text or "7")
    minute = int(minute_text or "0")
    target = datetime.combine(now.date(), time(hour=hour, minute=minute), tzinfo=now.tzinfo)
    if target <= now:
        target += timedelta(days=1)
    return target


async def _daily_report_scheduler() -> None:
    timezone_info = report_service._timezone()
    while True:
        now = datetime.now(timezone_info)
        next_run = _next_report_run(now)
        await asyncio.sleep(max(1.0, (next_run - now).total_seconds()))
        try:
            await asyncio.to_thread(report_service.send_daily_report)
        except Exception as exc:
            print(f"[REPORT_SCHEDULER] Failed to send report: {exc}")


chat_store = ChatStore(CHAT_DB_PATH)
engine = RagEngine(chat_store)
report_service = DailyReportService(chat_store, engine)

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
        "report_scheduler_enabled": REPORT_SCHEDULER_ENABLED,
        "report_daily_time": REPORT_DAILY_TIME,
        "report_timezone": REPORT_TIMEZONE,
        "report_subscribers": len(chat_store.subscribers(active_only=True)),
    }


@app.get("/history")
def history(limit: int = 30) -> dict[str, Any]:
    return {
        "items": chat_store.recent(limit=limit),
        "db_path": str(CHAT_DB_PATH),
    }


@app.get("/reports/subscribers")
def report_subscribers(active_only: bool = True) -> dict[str, Any]:
    return {
        "items": chat_store.subscribers(active_only=active_only),
        "db_path": str(CHAT_DB_PATH),
    }


@app.post("/reports/subscribers")
def upsert_report_subscriber(request: ReportSubscriberRequest) -> dict[str, Any]:
    try:
        subscriber = chat_store.upsert_subscriber(
            email=request.email,
            name=request.name,
            active=request.active,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "subscriber": subscriber}


@app.post("/reports/send")
def send_report(request: ReportSendRequest) -> dict[str, Any]:
    if not request.force and not REPORT_SCHEDULER_ENABLED:
        raise HTTPException(
            status_code=400,
            detail="REPORT_SCHEDULER_ENABLED=false. Set force=true to send manually.",
        )
    try:
        result = report_service.send_daily_report(recipients=request.recipients)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"ok": True, **result}


@app.get("/reports/preview")
def preview_report() -> dict[str, Any]:
    markdown, pdf_path = report_service.build_report()
    return {
        "markdown": markdown,
        "pdf_path": str(pdf_path),
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


@app.post("/chat-with-pdf", response_model=PdfChatResponse)
async def chat_with_pdf(
    message: str = Form(...),
    domain: str = Form(""),
    top_k: int = Form(TOP_K),
    gemini_fallback: bool = Form(True),
    file: UploadFile = File(...),
) -> PdfChatResponse:
    filename = file.filename or "uploaded.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file PDF.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File PDF rỗng.")
    if len(content) > 30 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File PDF vượt quá 30MB.")

    target_domain = domain if domain in DOMAIN_LABELS else "Uploaded"
    safe_stem = engine._safe_file_stem(filename)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    upload_path = UPLOADS_DIR / target_domain / f"{timestamp}_{safe_stem}.pdf"
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    upload_path.write_bytes(content)

    try:
        response = await asyncio.to_thread(
            engine.query_pdf,
            pdf_path=upload_path,
            original_filename=filename,
            request=ChatRequest(
                message=message,
                domain=domain or None,
                top_k=top_k,
                gemini_fallback=gemini_fallback,
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return response


@app.on_event("startup")
async def start_report_scheduler() -> None:
    if REPORT_SCHEDULER_ENABLED:
        asyncio.create_task(_daily_report_scheduler())


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
