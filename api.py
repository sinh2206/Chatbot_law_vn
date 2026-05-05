from __future__ import annotations

import json
import logging
import smtplib
import sqlite3
import ssl
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from functools import lru_cache
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from chatbot_core import FallbackAlertContext, LegalOrchestrator
from config import (
    ADMIN_ALERT_LOG_FILE,
    ADMIN_EMAIL,
    ALERT_MAX_AVG_RESPONSE_MS,
    ALERT_MAX_FALLBACK_RATE,
    ALERT_MIN_ACCURACY,
    CORS_ALLOW_ORIGINS,
    DASHBOARD_DIR,
    DOMAIN_LABELS,
    FRONTEND_DIR,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_SENDER,
    SMTP_USERNAME,
    SMTP_USE_SSL,
    SMTP_USE_TLS,
    TELEMETRY_DB_FILE,
    ensure_directories,
)
from expiry import LegalDocumentMetadata

logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RetrievedChunkLog:
    agent_domain: str
    chunk_rank: int
    distance: float
    text_excerpt: str
    metadata: dict[str, Any]


@dataclass
class AgentTraceLog:
    domain: str
    mode: str
    fallback_reason: str
    sub_question: str
    retrieved_count: int
    citation_count: int
    source_count: int
    retrieved_chunks: list[RetrievedChunkLog] = field(default_factory=list)


@dataclass
class InteractionLog:
    interaction_id: str
    created_at: str
    start_time: str
    end_time: str
    response_time_ms: int
    question: str
    domain_requested: str
    domains_detected: list[str]
    answer: str
    fallback_used: bool
    multi_domain: bool
    agent_count: int
    rag_query_count: int
    rag_hit_count: int
    fallback_count: int
    error: str
    run_type: str
    agent_traces: list[AgentTraceLog] = field(default_factory=list)


@dataclass
class FeedbackLog:
    interaction_id: str
    helpful: bool | None
    rating: int | None
    comment: str


class TelemetryStore:
    def __init__(self, db_file: Path = TELEMETRY_DB_FILE) -> None:
        ensure_directories()
        self.db_file = Path(db_file)
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    @staticmethod
    def new_interaction_id() -> str:
        return uuid.uuid4().hex

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_file), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        return connection

    def _init_schema(self) -> None:
        with self._lock:
            connection = self._connect()
            try:
                cursor = connection.cursor()
                cursor.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS interactions (
                        interaction_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        start_time TEXT NOT NULL,
                        end_time TEXT NOT NULL,
                        response_time_ms INTEGER NOT NULL,
                        question TEXT NOT NULL,
                        domain_requested TEXT,
                        domains_detected_json TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        fallback_used INTEGER NOT NULL,
                        multi_domain INTEGER NOT NULL,
                        agent_count INTEGER NOT NULL,
                        rag_query_count INTEGER NOT NULL,
                        rag_hit_count INTEGER NOT NULL,
                        fallback_count INTEGER NOT NULL,
                        error TEXT NOT NULL,
                        run_type TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS agent_traces (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        interaction_id TEXT NOT NULL,
                        domain TEXT NOT NULL,
                        mode TEXT NOT NULL,
                        fallback_reason TEXT NOT NULL,
                        sub_question TEXT NOT NULL,
                        retrieved_count INTEGER NOT NULL,
                        citation_count INTEGER NOT NULL,
                        source_count INTEGER NOT NULL,
                        FOREIGN KEY(interaction_id) REFERENCES interactions(interaction_id)
                    );

                    CREATE TABLE IF NOT EXISTS retrieved_chunks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        interaction_id TEXT NOT NULL,
                        agent_domain TEXT NOT NULL,
                        chunk_rank INTEGER NOT NULL,
                        distance REAL NOT NULL,
                        text_excerpt TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        FOREIGN KEY(interaction_id) REFERENCES interactions(interaction_id)
                    );

                    CREATE TABLE IF NOT EXISTS feedback (
                        interaction_id TEXT PRIMARY KEY,
                        helpful INTEGER,
                        rating INTEGER,
                        comment TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY(interaction_id) REFERENCES interactions(interaction_id)
                    );

                    CREATE TABLE IF NOT EXISTS test_runs (
                        run_id TEXT PRIMARY KEY,
                        started_at TEXT NOT NULL,
                        finished_at TEXT NOT NULL,
                        mode TEXT NOT NULL,
                        total_cases INTEGER NOT NULL,
                        passed_cases INTEGER NOT NULL,
                        avg_score REAL NOT NULL,
                        avg_response_ms REAL NOT NULL,
                        fallback_rate REAL NOT NULL,
                        level_breakdown_json TEXT NOT NULL,
                        notes TEXT NOT NULL,
                        citation_valid_rate REAL NOT NULL DEFAULT 0,
                        citation_f1_avg REAL NOT NULL DEFAULT 0,
                        entity_f1_avg REAL NOT NULL DEFAULT 0,
                        accuracy REAL NOT NULL DEFAULT 0,
                        mean_score REAL NOT NULL DEFAULT 0,
                        report_markdown_path TEXT NOT NULL DEFAULT '',
                        report_html_path TEXT NOT NULL DEFAULT '',
                        report_csv_path TEXT NOT NULL DEFAULT ''
                    );

                    CREATE TABLE IF NOT EXISTS test_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        case_id TEXT NOT NULL,
                        repeat_index INTEGER NOT NULL DEFAULT 1,
                        level INTEGER NOT NULL,
                        domain TEXT NOT NULL,
                        question TEXT NOT NULL,
                        expected_answer TEXT NOT NULL,
                        expected_keywords_json TEXT NOT NULL,
                        expected_citations_json TEXT NOT NULL,
                        response_text TEXT NOT NULL,
                        citations_json TEXT NOT NULL,
                        fallback_used INTEGER NOT NULL,
                        response_time_ms INTEGER NOT NULL,
                        score REAL NOT NULL,
                        passed INTEGER NOT NULL,
                        reason TEXT NOT NULL,
                        heuristic_score REAL NOT NULL DEFAULT 0,
                        llm_score REAL,
                        citation_precision REAL NOT NULL DEFAULT 0,
                        citation_recall REAL NOT NULL DEFAULT 0,
                        citation_f1 REAL NOT NULL DEFAULT 0,
                        entity_precision REAL NOT NULL DEFAULT 0,
                        entity_recall REAL NOT NULL DEFAULT 0,
                        entity_f1 REAL NOT NULL DEFAULT 0,
                        citation_valid INTEGER NOT NULL DEFAULT 0,
                        manual_score REAL,
                        manual_passed INTEGER,
                        manual_note TEXT NOT NULL DEFAULT '',
                        manual_updated_at TEXT,
                        grader_mode TEXT NOT NULL DEFAULT 'heuristic',
                        FOREIGN KEY(run_id) REFERENCES test_runs(run_id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_interactions_created_at
                        ON interactions(created_at);
                    CREATE INDEX IF NOT EXISTS idx_interactions_run_type
                        ON interactions(run_type);
                    CREATE INDEX IF NOT EXISTS idx_agent_traces_interaction
                        ON agent_traces(interaction_id);
                    CREATE INDEX IF NOT EXISTS idx_feedback_updated_at
                        ON feedback(updated_at);
                    CREATE INDEX IF NOT EXISTS idx_test_results_run_id
                        ON test_results(run_id);
                    CREATE INDEX IF NOT EXISTS idx_test_results_case_id
                        ON test_results(case_id);
                    """
                )

                self._ensure_column(
                    connection,
                    "test_runs",
                    "citation_valid_rate REAL NOT NULL DEFAULT 0",
                )
                self._ensure_column(
                    connection,
                    "test_runs",
                    "citation_f1_avg REAL NOT NULL DEFAULT 0",
                )
                self._ensure_column(
                    connection,
                    "test_runs",
                    "entity_f1_avg REAL NOT NULL DEFAULT 0",
                )
                self._ensure_column(
                    connection,
                    "test_runs",
                    "accuracy REAL NOT NULL DEFAULT 0",
                )
                self._ensure_column(
                    connection,
                    "test_runs",
                    "mean_score REAL NOT NULL DEFAULT 0",
                )
                self._ensure_column(
                    connection,
                    "test_runs",
                    "report_markdown_path TEXT NOT NULL DEFAULT ''",
                )
                self._ensure_column(
                    connection,
                    "test_runs",
                    "report_html_path TEXT NOT NULL DEFAULT ''",
                )
                self._ensure_column(
                    connection,
                    "test_runs",
                    "report_csv_path TEXT NOT NULL DEFAULT ''",
                )

                self._ensure_column(
                    connection,
                    "test_results",
                    "heuristic_score REAL NOT NULL DEFAULT 0",
                )
                self._ensure_column(
                    connection,
                    "test_results",
                    "repeat_index INTEGER NOT NULL DEFAULT 1",
                )
                self._ensure_column(connection, "test_results", "llm_score REAL")
                self._ensure_column(
                    connection,
                    "test_results",
                    "citation_precision REAL NOT NULL DEFAULT 0",
                )
                self._ensure_column(
                    connection,
                    "test_results",
                    "citation_recall REAL NOT NULL DEFAULT 0",
                )
                self._ensure_column(
                    connection,
                    "test_results",
                    "citation_f1 REAL NOT NULL DEFAULT 0",
                )
                self._ensure_column(
                    connection,
                    "test_results",
                    "entity_precision REAL NOT NULL DEFAULT 0",
                )
                self._ensure_column(
                    connection,
                    "test_results",
                    "entity_recall REAL NOT NULL DEFAULT 0",
                )
                self._ensure_column(
                    connection,
                    "test_results",
                    "entity_f1 REAL NOT NULL DEFAULT 0",
                )
                self._ensure_column(
                    connection,
                    "test_results",
                    "citation_valid INTEGER NOT NULL DEFAULT 0",
                )
                self._ensure_column(connection, "test_results", "manual_score REAL")
                self._ensure_column(connection, "test_results", "manual_passed INTEGER")
                self._ensure_column(
                    connection,
                    "test_results",
                    "manual_note TEXT NOT NULL DEFAULT ''",
                )
                self._ensure_column(connection, "test_results", "manual_updated_at TEXT")
                self._ensure_column(
                    connection,
                    "test_results",
                    "grader_mode TEXT NOT NULL DEFAULT 'heuristic'",
                )

                connection.commit()
            finally:
                connection.close()

    @staticmethod
    def _ensure_column(connection: sqlite3.Connection, table: str, column_definition: str) -> None:
        column_name = column_definition.split()[0]
        columns = {
            row["name"]
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column_name not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column_definition}")

    def save_interaction(self, record: InteractionLog) -> None:
        with self._lock:
            connection = self._connect()
            try:
                cursor = connection.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO interactions (
                        interaction_id, created_at, start_time, end_time, response_time_ms,
                        question, domain_requested, domains_detected_json, answer,
                        fallback_used, multi_domain, agent_count, rag_query_count,
                        rag_hit_count, fallback_count, error, run_type
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.interaction_id,
                        record.created_at,
                        record.start_time,
                        record.end_time,
                        record.response_time_ms,
                        record.question,
                        record.domain_requested,
                        json.dumps(record.domains_detected, ensure_ascii=False),
                        record.answer,
                        int(record.fallback_used),
                        int(record.multi_domain),
                        record.agent_count,
                        record.rag_query_count,
                        record.rag_hit_count,
                        record.fallback_count,
                        record.error,
                        record.run_type,
                    ),
                )

                cursor.execute(
                    "DELETE FROM agent_traces WHERE interaction_id = ?",
                    (record.interaction_id,),
                )
                cursor.execute(
                    "DELETE FROM retrieved_chunks WHERE interaction_id = ?",
                    (record.interaction_id,),
                )

                for trace in record.agent_traces:
                    cursor.execute(
                        """
                        INSERT INTO agent_traces (
                            interaction_id, domain, mode, fallback_reason, sub_question,
                            retrieved_count, citation_count, source_count
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record.interaction_id,
                            trace.domain,
                            trace.mode,
                            trace.fallback_reason,
                            trace.sub_question,
                            trace.retrieved_count,
                            trace.citation_count,
                            trace.source_count,
                        ),
                    )

                    for chunk in trace.retrieved_chunks:
                        cursor.execute(
                            """
                            INSERT INTO retrieved_chunks (
                                interaction_id, agent_domain, chunk_rank, distance,
                                text_excerpt, metadata_json
                            )
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                record.interaction_id,
                                chunk.agent_domain,
                                chunk.chunk_rank,
                                chunk.distance,
                                chunk.text_excerpt,
                                json.dumps(chunk.metadata, ensure_ascii=False),
                            ),
                        )

                connection.commit()
            finally:
                connection.close()

    def save_feedback(self, feedback: FeedbackLog) -> None:
        with self._lock:
            connection = self._connect()
            try:
                connection.execute(
                    """
                    INSERT INTO feedback (interaction_id, helpful, rating, comment, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(interaction_id)
                    DO UPDATE SET
                        helpful = excluded.helpful,
                        rating = excluded.rating,
                        comment = excluded.comment,
                        updated_at = excluded.updated_at
                    """,
                    (
                        feedback.interaction_id,
                        None if feedback.helpful is None else int(feedback.helpful),
                        feedback.rating,
                        feedback.comment,
                        utc_now_iso(),
                    ),
                )
                connection.commit()
            finally:
                connection.close()

    def interaction_exists(self, interaction_id: str) -> bool:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT interaction_id FROM interactions WHERE interaction_id = ?",
                (interaction_id,),
            ).fetchone()
            return row is not None
        finally:
            connection.close()

    def save_test_run(
        self,
        run_id: str,
        started_at: str,
        finished_at: str,
        mode: str,
        total_cases: int,
        passed_cases: int,
        avg_score: float,
        avg_response_ms: float,
        fallback_rate: float,
        level_breakdown: dict[str, Any],
        notes: str,
        citation_valid_rate: float = 0.0,
        citation_f1_avg: float = 0.0,
        entity_f1_avg: float = 0.0,
        accuracy: float = 0.0,
        mean_score: float = 0.0,
        report_markdown_path: str = "",
        report_html_path: str = "",
        report_csv_path: str = "",
    ) -> None:
        with self._lock:
            connection = self._connect()
            try:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO test_runs (
                        run_id, started_at, finished_at, mode, total_cases,
                        passed_cases, avg_score, avg_response_ms, fallback_rate,
                        level_breakdown_json, notes, citation_valid_rate,
                        citation_f1_avg, entity_f1_avg, accuracy, mean_score,
                        report_markdown_path, report_html_path, report_csv_path
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        started_at,
                        finished_at,
                        mode,
                        total_cases,
                        passed_cases,
                        avg_score,
                        avg_response_ms,
                        fallback_rate,
                        json.dumps(level_breakdown, ensure_ascii=False),
                        notes,
                        citation_valid_rate,
                        citation_f1_avg,
                        entity_f1_avg,
                        accuracy,
                        mean_score,
                        report_markdown_path,
                        report_html_path,
                        report_csv_path,
                    ),
                )
                connection.commit()
            finally:
                connection.close()

    def save_test_result(
        self,
        run_id: str,
        case_id: str,
        repeat_index: int,
        level: int,
        domain: str,
        question: str,
        expected_answer: str,
        expected_keywords: list[str],
        expected_citations: list[str],
        response_text: str,
        citations: list[str],
        fallback_used: bool,
        response_time_ms: int,
        score: float,
        passed: bool,
        reason: str,
        heuristic_score: float = 0.0,
        llm_score: float | None = None,
        citation_precision: float = 0.0,
        citation_recall: float = 0.0,
        citation_f1: float = 0.0,
        entity_precision: float = 0.0,
        entity_recall: float = 0.0,
        entity_f1: float = 0.0,
        citation_valid: bool = False,
        grader_mode: str = "heuristic",
    ) -> None:
        with self._lock:
            connection = self._connect()
            try:
                connection.execute(
                    """
                    INSERT INTO test_results (
                        run_id, case_id, repeat_index, level, domain, question, expected_answer,
                        expected_keywords_json, expected_citations_json,
                        response_text, citations_json, fallback_used,
                        response_time_ms, score, passed, reason,
                        heuristic_score, llm_score,
                        citation_precision, citation_recall, citation_f1,
                        entity_precision, entity_recall, entity_f1,
                        citation_valid, grader_mode
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        case_id,
                        repeat_index,
                        level,
                        domain,
                        question,
                        expected_answer,
                        json.dumps(expected_keywords, ensure_ascii=False),
                        json.dumps(expected_citations, ensure_ascii=False),
                        response_text,
                        json.dumps(citations, ensure_ascii=False),
                        int(fallback_used),
                        response_time_ms,
                        score,
                        int(passed),
                        reason,
                        heuristic_score,
                        llm_score,
                        citation_precision,
                        citation_recall,
                        citation_f1,
                        entity_precision,
                        entity_recall,
                        entity_f1,
                        int(citation_valid),
                        grader_mode,
                    ),
                )
                connection.commit()
            finally:
                connection.close()

    def update_test_result_manual_review(
        self,
        run_id: str,
        case_id: str,
        repeat_index: int,
        manual_score: float | None,
        manual_passed: bool | None,
        manual_note: str,
    ) -> None:
        with self._lock:
            connection = self._connect()
            try:
                connection.execute(
                    """
                    UPDATE test_results
                    SET manual_score = ?,
                        manual_passed = ?,
                        manual_note = ?,
                        manual_updated_at = ?
                    WHERE run_id = ? AND case_id = ? AND repeat_index = ?
                    """,
                    (
                        manual_score,
                        None if manual_passed is None else int(manual_passed),
                        manual_note,
                        utc_now_iso(),
                        run_id,
                        case_id,
                        repeat_index,
                    ),
                )
                connection.commit()
            finally:
                connection.close()

    def get_latest_test_run_id(self) -> str | None:
        row = self.query_one(
            "SELECT run_id FROM test_runs ORDER BY finished_at DESC LIMIT 1"
        )
        return str(row["run_id"]) if row else None

    def get_test_run(self, run_id: str) -> dict[str, Any] | None:
        row = self.query_one("SELECT * FROM test_runs WHERE run_id = ?", (run_id,))
        if not row:
            return None

        return {key: row[key] for key in row.keys()}

    def get_test_results(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.query(
            """
            SELECT *
            FROM test_results
            WHERE run_id = ?
            ORDER BY id ASC
            """,
            (run_id,),
        )

        payload: list[dict[str, Any]] = []
        for row in rows:
            payload.append(
                {
                    "id": int(row["id"]),
                    "run_id": row["run_id"],
                    "case_id": row["case_id"],
                    "repeat_index": int(row["repeat_index"]),
                    "level": int(row["level"]),
                    "domain": row["domain"],
                    "question": row["question"],
                    "expected_answer": row["expected_answer"],
                    "expected_keywords": json.loads(row["expected_keywords_json"]),
                    "expected_citations": json.loads(row["expected_citations_json"]),
                    "response_text": row["response_text"],
                    "citations": json.loads(row["citations_json"]),
                    "fallback_used": bool(row["fallback_used"]),
                    "response_time_ms": int(row["response_time_ms"]),
                    "score": float(row["score"]),
                    "passed": bool(row["passed"]),
                    "reason": row["reason"],
                    "heuristic_score": float(row["heuristic_score"]),
                    "llm_score": None if row["llm_score"] is None else float(row["llm_score"]),
                    "citation_precision": float(row["citation_precision"]),
                    "citation_recall": float(row["citation_recall"]),
                    "citation_f1": float(row["citation_f1"]),
                    "entity_precision": float(row["entity_precision"]),
                    "entity_recall": float(row["entity_recall"]),
                    "entity_f1": float(row["entity_f1"]),
                    "citation_valid": bool(row["citation_valid"]),
                    "manual_score": None
                    if row["manual_score"] is None
                    else float(row["manual_score"]),
                    "manual_passed": None
                    if row["manual_passed"] is None
                    else bool(row["manual_passed"]),
                    "manual_note": row["manual_note"],
                    "manual_updated_at": row["manual_updated_at"],
                    "grader_mode": row["grader_mode"],
                }
            )

        return payload

    def recompute_test_run_summary(self, run_id: str) -> None:
        results = self.get_test_results(run_id)
        if not results:
            return

        total = len(results)
        passed = 0
        scores: list[float] = []
        latencies: list[float] = []
        fallback_count = 0
        citation_valid_count = 0
        citation_f1_values: list[float] = []
        entity_f1_values: list[float] = []

        grouped: dict[int, list[dict[str, Any]]] = {}

        for item in results:
            effective_passed = (
                bool(item["manual_passed"])
                if item["manual_passed"] is not None
                else bool(item["passed"])
            )
            effective_score = (
                float(item["manual_score"])
                if item["manual_score"] is not None
                else float(item["score"])
            )

            passed += int(effective_passed)
            scores.append(effective_score)
            latencies.append(float(item["response_time_ms"]))
            fallback_count += int(bool(item["fallback_used"]))
            citation_valid_count += int(bool(item["citation_valid"]))
            citation_f1_values.append(float(item["citation_f1"]))
            entity_f1_values.append(float(item["entity_f1"]))

            grouped.setdefault(int(item["level"]), []).append(
                {
                    "passed": effective_passed,
                    "score": effective_score,
                }
            )

        level_breakdown: dict[str, Any] = {}
        for level, entries in grouped.items():
            level_total = len(entries)
            level_passed = sum(1 for entry in entries if entry["passed"])
            level_breakdown[str(level)] = {
                "total": level_total,
                "passed": level_passed,
                "accuracy": (level_passed / level_total) if level_total else 0.0,
                "avg_score": mean([float(entry["score"]) for entry in entries]),
            }

        with self._lock:
            connection = self._connect()
            try:
                connection.execute(
                    """
                    UPDATE test_runs
                    SET passed_cases = ?,
                        avg_score = ?,
                        mean_score = ?,
                        avg_response_ms = ?,
                        fallback_rate = ?,
                        citation_valid_rate = ?,
                        citation_f1_avg = ?,
                        entity_f1_avg = ?,
                        accuracy = ?,
                        level_breakdown_json = ?
                    WHERE run_id = ?
                    """,
                    (
                        passed,
                        mean(scores),
                        mean(scores),
                        mean(latencies),
                        fallback_count / total,
                        citation_valid_count / total,
                        mean(citation_f1_values),
                        mean(entity_f1_values),
                        passed / total,
                        json.dumps(level_breakdown, ensure_ascii=False),
                        run_id,
                    ),
                )
                connection.commit()
            finally:
                connection.close()

    def query(self, sql: str, parameters: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        connection = self._connect()
        try:
            rows = connection.execute(sql, parameters).fetchall()
            return rows
        finally:
            connection.close()

    def query_one(self, sql: str, parameters: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        connection = self._connect()
        try:
            row = connection.execute(sql, parameters).fetchone()
            return row
        finally:
            connection.close()


class AdminNotifier:
    def __init__(self, log_file: Path = ADMIN_ALERT_LOG_FILE) -> None:
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._sent_cache: set[str] = set()

    def notify_expired_document(self, record: LegalDocumentMetadata, domain_label: str) -> None:
        expiry = record.expiry_date.isoformat() if record.expiry_date else "(không rõ)"
        message = (
            f"Văn bản [{record.file_name} | {record.document_number}] thuộc lĩnh vực "
            f"[{domain_label}] đã hết hiệu lực từ ngày [{expiry}]. "
            "Vui lòng cập nhật văn bản thay thế vào hệ thống."
        )
        dedupe_key = f"expired::{record.domain.lower()}::{record.file_name.lower()}::{expiry}"
        self._notify(message=message, subject="[Chatbot] Cảnh báo văn bản hết hiệu lực", dedupe_key=dedupe_key)

    def notify_fallback(self, context: FallbackAlertContext) -> None:
        docs = ", ".join(context.expired_documents) if context.expired_documents else "(không xác định)"
        message = (
            "Kích hoạt Web Search Fallback do dữ liệu trong kho không còn phù hợp.\n"
            f"- Lĩnh vực: {context.domain_label} ({context.domain})\n"
            f"- Lý do: {context.reason}\n"
            f"- Câu hỏi: {context.question}\n"
            f"- Văn bản liên quan: {docs}\n"
            "Đề nghị cập nhật văn bản mới vào hệ thống và chạy lại: python vector_store.py --build"
        )
        dedupe_key = f"fallback::{context.domain.lower()}::{context.question.strip().lower()}::{context.reason.lower()}"
        self._notify(message=message, subject="[Chatbot] Kích hoạt Web Fallback", dedupe_key=dedupe_key)

    def _notify(self, message: str, subject: str, dedupe_key: str) -> None:
        if dedupe_key in self._sent_cache:
            return

        self._sent_cache.add(dedupe_key)
        self._append_log(subject=subject, message=message)

        if ADMIN_EMAIL and SMTP_HOST:
            try:
                self._send_email(subject=subject, message=message)
            except Exception as exc:
                logger.warning("Send admin email failed: %s", exc)

    def _append_log(self, subject: str, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {subject}\n{message}\n\n"
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(entry)

    def _send_email(self, subject: str, message: str) -> None:
        if not ADMIN_EMAIL:
            return

        email_message = EmailMessage()
        email_message["From"] = SMTP_SENDER or SMTP_USERNAME
        email_message["To"] = ADMIN_EMAIL
        email_message["Subject"] = subject
        email_message.set_content(message)

        if SMTP_USE_SSL:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
                if SMTP_USERNAME:
                    server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.send_message(email_message)
            return

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            if SMTP_USE_TLS:
                context = ssl.create_default_context()
                server.starttls(context=context)
            if SMTP_USERNAME:
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(email_message)


@dataclass
class AlertItem:
    level: str
    code: str
    message: str
    value: float
    threshold: float


class DashboardMetricsService:
    def __init__(self, telemetry: TelemetryStore) -> None:
        self.telemetry = telemetry

    def summary(self) -> dict[str, Any]:
        row = self.telemetry.query_one(
            """
            SELECT
                COUNT(*) AS total_questions,
                COALESCE(AVG(response_time_ms), 0) AS avg_response_ms,
                COALESCE(MAX(response_time_ms), 0) AS max_response_ms,
                COALESCE(SUM(fallback_used), 0) AS fallback_count,
                COALESCE(SUM(multi_domain), 0) AS multi_domain_count,
                COALESCE(SUM(rag_query_count), 0) AS rag_query_count,
                COALESCE(SUM(rag_hit_count), 0) AS rag_hit_count
            FROM interactions
            WHERE run_type = 'user'
            """
        )

        total = int(row["total_questions"]) if row else 0
        fallback_count = int(row["fallback_count"]) if row else 0
        multi_domain_count = int(row["multi_domain_count"]) if row else 0
        rag_query_count = int(row["rag_query_count"]) if row else 0
        rag_hit_count = int(row["rag_hit_count"]) if row else 0

        feedback_row = self.telemetry.query_one(
            """
            SELECT
                COUNT(*) AS total_feedback,
                COALESCE(SUM(CASE WHEN helpful = 1 THEN 1 ELSE 0 END), 0) AS helpful_count,
                COALESCE(SUM(CASE WHEN helpful = 0 THEN 1 ELSE 0 END), 0) AS unhelpful_count,
                COALESCE(AVG(rating), 0) AS avg_rating
            FROM feedback
            """
        )

        throughput_row = self.telemetry.query_one(
            """
            SELECT COUNT(*) AS total_24h
            FROM interactions
            WHERE run_type = 'user' AND created_at >= ?
            """,
            (self._cutoff_iso(hours=24),),
        )
        total_24h = int(throughput_row["total_24h"]) if throughput_row else 0

        response_times = [
            int(row_item["response_time_ms"])
            for row_item in self.telemetry.query(
                "SELECT response_time_ms FROM interactions WHERE run_type = 'user'"
            )
        ]
        p95 = self._percentile(response_times, 95)

        return {
            "total_questions": total,
            "avg_response_ms": float(row["avg_response_ms"] if row else 0),
            "max_response_ms": int(row["max_response_ms"] if row else 0),
            "p95_response_ms": int(p95),
            "fallback_rate": self._safe_ratio(fallback_count, total),
            "multi_domain_rate": self._safe_ratio(multi_domain_count, total),
            "rag_query_count": rag_query_count,
            "rag_hit_rate": self._safe_ratio(rag_hit_count, rag_query_count),
            "throughput_qph_24h": round(total_24h / 24.0, 3),
            "feedback": {
                "total": int(feedback_row["total_feedback"] if feedback_row else 0),
                "helpful_ratio": self._safe_ratio(
                    int(feedback_row["helpful_count"] if feedback_row else 0),
                    int(feedback_row["total_feedback"] if feedback_row else 0),
                ),
                "unhelpful_ratio": self._safe_ratio(
                    int(feedback_row["unhelpful_count"] if feedback_row else 0),
                    int(feedback_row["total_feedback"] if feedback_row else 0),
                ),
                "avg_rating": float(feedback_row["avg_rating"] if feedback_row else 0),
            },
        }

    def timeseries(self, days: int = 14) -> list[dict[str, Any]]:
        rows = self.telemetry.query(
            """
            SELECT
                DATE(created_at) AS day,
                COUNT(*) AS total_questions,
                COALESCE(AVG(response_time_ms), 0) AS avg_response_ms,
                COALESCE(SUM(fallback_used), 0) AS fallback_count,
                COALESCE(SUM(multi_domain), 0) AS multi_domain_count
            FROM interactions
            WHERE run_type = 'user' AND created_at >= ?
            GROUP BY DATE(created_at)
            ORDER BY day ASC
            """,
            (self._cutoff_iso(days=days),),
        )

        series: list[dict[str, Any]] = []
        for row in rows:
            total_questions = int(row["total_questions"])
            fallback_count = int(row["fallback_count"])
            multi_domain_count = int(row["multi_domain_count"])
            series.append(
                {
                    "day": row["day"],
                    "total_questions": total_questions,
                    "avg_response_ms": float(row["avg_response_ms"]),
                    "fallback_rate": self._safe_ratio(fallback_count, total_questions),
                    "multi_domain_rate": self._safe_ratio(multi_domain_count, total_questions),
                }
            )

        return series

    def domain_breakdown(self, days: int = 30) -> list[dict[str, Any]]:
        rows = self.telemetry.query(
            """
            SELECT
                t.domain AS domain,
                COUNT(*) AS total_subqueries,
                COALESCE(SUM(CASE WHEN t.mode = 'web_fallback' THEN 1 ELSE 0 END), 0) AS fallback_count,
                COALESCE(SUM(CASE WHEN t.retrieved_count > 0 THEN 1 ELSE 0 END), 0) AS retrieval_hit_count,
                COALESCE(AVG(t.retrieved_count), 0) AS avg_retrieved_chunks
            FROM agent_traces t
            INNER JOIN interactions i ON i.interaction_id = t.interaction_id
            WHERE i.run_type = 'user' AND i.created_at >= ?
            GROUP BY t.domain
            ORDER BY total_subqueries DESC
            """,
            (self._cutoff_iso(days=days),),
        )

        breakdown: list[dict[str, Any]] = []
        for row in rows:
            total_subqueries = int(row["total_subqueries"])
            fallback_count = int(row["fallback_count"])
            retrieval_hit_count = int(row["retrieval_hit_count"])

            breakdown.append(
                {
                    "domain": row["domain"],
                    "total_subqueries": total_subqueries,
                    "fallback_rate": self._safe_ratio(fallback_count, total_subqueries),
                    "retrieval_hit_rate": self._safe_ratio(retrieval_hit_count, total_subqueries),
                    "avg_retrieved_chunks": float(row["avg_retrieved_chunks"]),
                }
            )

        return breakdown

    def latency_histogram(self) -> list[dict[str, Any]]:
        rows = self.telemetry.query(
            "SELECT response_time_ms FROM interactions WHERE run_type = 'user'"
        )
        values = [int(row["response_time_ms"]) for row in rows]

        buckets = [
            ("0-1000", 0, 1000),
            ("1000-2000", 1000, 2000),
            ("2000-5000", 2000, 5000),
            ("5000-10000", 5000, 10000),
            ("10000+", 10000, 10**9),
        ]

        result: list[dict[str, Any]] = []
        for name, lower, upper in buckets:
            count = sum(1 for value in values if lower <= value < upper)
            result.append({"bucket": name, "count": count})

        return result

    def low_feedback_questions(self, limit: int = 15) -> list[dict[str, Any]]:
        rows = self.telemetry.query(
            """
            SELECT
                i.interaction_id,
                i.question,
                i.response_time_ms,
                i.fallback_used,
                f.helpful,
                f.rating,
                f.comment,
                f.updated_at
            FROM feedback f
            INNER JOIN interactions i ON i.interaction_id = f.interaction_id
            WHERE f.helpful = 0 OR (f.rating IS NOT NULL AND f.rating <= 2)
            ORDER BY f.updated_at DESC
            LIMIT ?
            """,
            (limit,),
        )

        return [
            {
                "interaction_id": row["interaction_id"],
                "question": row["question"],
                "response_time_ms": int(row["response_time_ms"]),
                "fallback_used": bool(row["fallback_used"]),
                "helpful": None if row["helpful"] is None else bool(row["helpful"]),
                "rating": row["rating"],
                "comment": row["comment"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def latest_test_run(self) -> dict[str, Any]:
        run = self.telemetry.query_one(
            """
            SELECT *
            FROM test_runs
            ORDER BY finished_at DESC
            LIMIT 1
            """
        )
        if not run:
            return {"available": False}

        rows = self.telemetry.query(
            """
            SELECT level, COUNT(*) AS total, SUM(passed) AS passed, AVG(score) AS avg_score
            FROM test_results
            WHERE run_id = ?
            GROUP BY level
            ORDER BY level
            """,
            (run["run_id"],),
        )

        levels = [
            {
                "level": int(row["level"]),
                "total": int(row["total"]),
                "passed": int(row["passed"] or 0),
                "accuracy": self._safe_ratio(int(row["passed"] or 0), int(row["total"])),
                "avg_score": float(row["avg_score"] or 0),
            }
            for row in rows
        ]

        return {
            "available": True,
            "run_id": run["run_id"],
            "mode": run["mode"],
            "started_at": run["started_at"],
            "finished_at": run["finished_at"],
            "total_cases": int(run["total_cases"]),
            "passed_cases": int(run["passed_cases"]),
            "accuracy": self._safe_ratio(int(run["passed_cases"]), int(run["total_cases"])),
            "avg_score": float(run["avg_score"]),
            "mean_score": float(run["mean_score"] if "mean_score" in run.keys() else run["avg_score"]),
            "avg_response_ms": float(run["avg_response_ms"]),
            "fallback_rate": float(run["fallback_rate"]),
            "citation_valid_rate": float(
                run["citation_valid_rate"] if "citation_valid_rate" in run.keys() else 0
            ),
            "citation_f1_avg": float(
                run["citation_f1_avg"] if "citation_f1_avg" in run.keys() else 0
            ),
            "entity_f1_avg": float(
                run["entity_f1_avg"] if "entity_f1_avg" in run.keys() else 0
            ),
            "report_markdown_path": (
                str(run["report_markdown_path"]) if "report_markdown_path" in run.keys() else ""
            ),
            "report_html_path": (
                str(run["report_html_path"]) if "report_html_path" in run.keys() else ""
            ),
            "report_csv_path": (
                str(run["report_csv_path"]) if "report_csv_path" in run.keys() else ""
            ),
            "level_breakdown": json.loads(run["level_breakdown_json"]),
            "levels": levels,
        }

    def run_history(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.telemetry.query(
            """
            SELECT run_id, mode, started_at, finished_at, total_cases,
                   passed_cases, avg_score, avg_response_ms, fallback_rate,
                   citation_valid_rate, citation_f1_avg, entity_f1_avg
            FROM test_runs
            ORDER BY finished_at DESC
            LIMIT ?
            """,
            (limit,),
        )

        return [
            {
                "run_id": row["run_id"],
                "mode": row["mode"],
                "started_at": row["started_at"],
                "finished_at": row["finished_at"],
                "total_cases": int(row["total_cases"]),
                "passed_cases": int(row["passed_cases"]),
                "accuracy": self._safe_ratio(int(row["passed_cases"]), int(row["total_cases"])),
                "avg_score": float(row["avg_score"]),
                "avg_response_ms": float(row["avg_response_ms"]),
                "fallback_rate": float(row["fallback_rate"]),
                "citation_valid_rate": float(row["citation_valid_rate"] or 0),
                "citation_f1_avg": float(row["citation_f1_avg"] or 0),
                "entity_f1_avg": float(row["entity_f1_avg"] or 0),
            }
            for row in rows
        ]

    def frequent_test_failures(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.telemetry.query(
            """
            SELECT
                case_id,
                question,
                COUNT(*) AS fail_count,
                AVG(COALESCE(manual_score, score)) AS avg_score
            FROM test_results
            WHERE COALESCE(manual_passed, passed) = 0
            GROUP BY case_id, question
            ORDER BY fail_count DESC, avg_score ASC
            LIMIT ?
            """,
            (limit,),
        )

        return [
            {
                "case_id": row["case_id"],
                "question": row["question"],
                "fail_count": int(row["fail_count"]),
                "avg_score": float(row["avg_score"] or 0),
            }
            for row in rows
        ]

    def alerts(self) -> list[dict[str, Any]]:
        alerts: list[AlertItem] = []

        summary = self.summary()
        latest_test = self.latest_test_run()

        avg_response = float(summary["avg_response_ms"])
        fallback_rate = float(summary["fallback_rate"])

        if avg_response > ALERT_MAX_AVG_RESPONSE_MS:
            alerts.append(
                AlertItem(
                    level="warning",
                    code="HIGH_LATENCY",
                    message="Thời gian phản hồi trung bình vượt ngưỡng",
                    value=avg_response,
                    threshold=float(ALERT_MAX_AVG_RESPONSE_MS),
                )
            )

        if fallback_rate > ALERT_MAX_FALLBACK_RATE:
            alerts.append(
                AlertItem(
                    level="warning",
                    code="HIGH_FALLBACK",
                    message="Tỷ lệ fallback đang cao",
                    value=fallback_rate,
                    threshold=float(ALERT_MAX_FALLBACK_RATE),
                )
            )

        if latest_test.get("available"):
            accuracy = float(latest_test["accuracy"])
            accuracy_percent = accuracy * 100
            if accuracy_percent < ALERT_MIN_ACCURACY:
                alerts.append(
                    AlertItem(
                        level="critical",
                        code="LOW_ACCURACY",
                        message="Độ chính xác test suite thấp hơn ngưỡng",
                        value=accuracy_percent,
                        threshold=float(ALERT_MIN_ACCURACY),
                    )
                )

        return [
            {
                "level": alert.level,
                "code": alert.code,
                "message": alert.message,
                "value": alert.value,
                "threshold": alert.threshold,
            }
            for alert in alerts
        ]

    @staticmethod
    def _safe_ratio(numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    @staticmethod
    def _percentile(values: list[int], percentile: int) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * (percentile / 100)
        lower = int(k)
        upper = min(lower + 1, len(sorted_values) - 1)
        if lower == upper:
            return float(sorted_values[lower])
        fraction = k - lower
        return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction

    @staticmethod
    def _cutoff_iso(days: int = 0, hours: int = 0) -> str:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days, hours=hours)
        return cutoff.isoformat()


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        orchestrator = get_orchestrator()
        orchestrator.expiry_monitor.run_if_due(force=True)
    except Exception:
        pass
    yield


app = FastAPI(title="Legal Multi-Agent RAG API", version="4.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

if DASHBOARD_DIR.exists():
    app.mount("/dashboard-static", StaticFiles(directory=str(DASHBOARD_DIR)), name="dashboard")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3)
    domain: str | None = None
    run_type: str = Field(default="user")


class ChatResponse(BaseModel):
    interaction_id: str
    answer: str
    domains: list[str]
    citations: list[str]
    sources: list[dict[str, str]]
    routed_sub_questions: list[dict[str, str]]
    fallback_used: bool
    agent_modes: list[dict[str, str]]
    response_time_ms: int


class FeedbackRequest(BaseModel):
    interaction_id: str = Field(..., min_length=6)
    helpful: bool | None = None
    rating: int | None = Field(default=None, ge=1, le=5)
    comment: str = Field(default="", max_length=1000)


class FeedbackResponse(BaseModel):
    status: str


class TestReviewRequest(BaseModel):
    run_id: str = Field(..., min_length=6)
    case_id: str = Field(..., min_length=3)
    repeat_index: int = Field(..., ge=1)
    manual_score: float | None = Field(default=None, ge=0, le=10)
    manual_passed: bool | None = None
    manual_note: str = Field(default="", max_length=2000)


@lru_cache(maxsize=1)
def get_admin_notifier() -> AdminNotifier:
    return AdminNotifier()


@lru_cache(maxsize=1)
def get_orchestrator() -> LegalOrchestrator:
    return LegalOrchestrator.from_defaults(notifier=get_admin_notifier())


@lru_cache(maxsize=1)
def get_telemetry() -> TelemetryStore:
    return TelemetryStore()


@lru_cache(maxsize=1)
def get_metrics_service() -> DashboardMetricsService:
    return DashboardMetricsService(telemetry=get_telemetry())


def _resolve_orchestrator(domain: str | None) -> LegalOrchestrator:
    base = get_orchestrator()
    if not domain:
        return base

    if domain not in DOMAIN_LABELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid domain. Supported: {', '.join(DOMAIN_LABELS.keys())}",
        )

    return _get_domain_orchestrator(domain)


@lru_cache(maxsize=8)
def _get_domain_orchestrator(domain: str) -> LegalOrchestrator:
    base = get_orchestrator()
    return LegalOrchestrator(
        vector_store=base.vector_store,
        domains=[domain],
        metadata_registry=base.metadata_registry,
        notifier=base.notifier,
    )


@app.get("/")
def index() -> FileResponse:
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_file)


@app.get("/dashboard")
def dashboard_page() -> FileResponse:
    dashboard_file = DASHBOARD_DIR / "index.html"
    if not dashboard_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return FileResponse(dashboard_file)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
@app.post("/ask", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    telemetry = get_telemetry()

    interaction_id = telemetry.new_interaction_id()
    started_at = utc_now_iso()
    started_perf = perf_counter()

    answer = ""
    domains: list[str] = []
    citations: list[str] = []
    sources: list[dict[str, str]] = []
    routed_sub_questions: list[dict[str, str]] = []
    modes: list[dict[str, str]] = []
    agent_traces: list[AgentTraceLog] = []
    fallback_used = False
    error_message = ""

    run_type = payload.run_type.strip().lower()
    if run_type not in {"user", "test"}:
        raise HTTPException(status_code=400, detail="run_type must be 'user' or 'test'")

    try:
        orchestrator = _resolve_orchestrator(payload.domain)
        if orchestrator.vector_store.count() == 0:
            raise RuntimeError("Vector store is empty. Run `python vector_store.py --build` first.")

        result = orchestrator.answer(payload.question)

        routed_sub_questions = [
            {"domain": domain, "question": question}
            for domain, question in result.routed_sub_questions
        ]

        modes = [
            {
                "domain": agent_answer.domain,
                "mode": agent_answer.mode,
                "fallback_reason": agent_answer.fallback_reason,
            }
            for agent_answer in result.agent_answers
        ]

        fallback_used = any(item["mode"] == "web_fallback" for item in modes)
        domains = result.intent.domains or ([payload.domain] if payload.domain else [])

        answer = result.merged.answer
        citations = result.merged.citations
        sources = result.merged.sources

        for agent_answer in result.agent_answers:
            chunk_logs = [
                RetrievedChunkLog(
                    agent_domain=agent_answer.domain,
                    chunk_rank=index,
                    distance=chunk.distance,
                    text_excerpt=chunk.text[:1200],
                    metadata=chunk.metadata,
                )
                for index, chunk in enumerate(agent_answer.retrieved_chunks, start=1)
            ]

            agent_traces.append(
                AgentTraceLog(
                    domain=agent_answer.domain,
                    mode=agent_answer.mode,
                    fallback_reason=agent_answer.fallback_reason,
                    sub_question=agent_answer.question,
                    retrieved_count=len(agent_answer.retrieved_chunks),
                    citation_count=len(agent_answer.citations),
                    source_count=len(agent_answer.sources),
                    retrieved_chunks=chunk_logs,
                )
            )

    except HTTPException as exc:
        error_message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        raise
    except Exception as exc:
        error_message = str(exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        ended_at = utc_now_iso()
        response_time_ms = int((perf_counter() - started_perf) * 1000)

        rag_query_count = len(agent_traces)
        rag_hit_count = sum(1 for trace in agent_traces if trace.retrieved_count > 0)
        fallback_count = sum(1 for trace in agent_traces if trace.mode == "web_fallback")

        interaction = InteractionLog(
            interaction_id=interaction_id,
            created_at=started_at,
            start_time=started_at,
            end_time=ended_at,
            response_time_ms=response_time_ms,
            question=payload.question,
            domain_requested=payload.domain or "",
            domains_detected=domains,
            answer=answer,
            fallback_used=fallback_used,
            multi_domain=(len(set(domains)) > 1),
            agent_count=len(agent_traces),
            rag_query_count=rag_query_count,
            rag_hit_count=rag_hit_count,
            fallback_count=fallback_count,
            error=error_message,
            run_type=run_type,
            agent_traces=agent_traces,
        )
        try:
            telemetry.save_interaction(interaction)
        except Exception as exc:
            logger.warning("Telemetry save_interaction failed: %s", exc)

    response_time_ms = int((perf_counter() - started_perf) * 1000)
    return ChatResponse(
        interaction_id=interaction_id,
        answer=answer,
        domains=domains,
        citations=citations,
        sources=sources,
        routed_sub_questions=routed_sub_questions,
        fallback_used=fallback_used,
        agent_modes=modes,
        response_time_ms=response_time_ms,
    )


@app.post("/feedback", response_model=FeedbackResponse)
def save_feedback(payload: FeedbackRequest) -> FeedbackResponse:
    telemetry = get_telemetry()
    if not telemetry.interaction_exists(payload.interaction_id):
        raise HTTPException(status_code=404, detail="interaction_id not found")

    telemetry.save_feedback(
        FeedbackLog(
            interaction_id=payload.interaction_id,
            helpful=payload.helpful,
            rating=payload.rating,
            comment=payload.comment.strip(),
        )
    )
    return FeedbackResponse(status="ok")


@app.get("/metrics/summary")
def metrics_summary() -> dict[str, object]:
    return get_metrics_service().summary()


@app.get("/metrics/timeseries")
def metrics_timeseries(days: int = 14) -> dict[str, object]:
    return {"items": get_metrics_service().timeseries(days=max(1, min(days, 180)))}


@app.get("/metrics/domain-breakdown")
def metrics_domain_breakdown(days: int = 30) -> dict[str, object]:
    return {"items": get_metrics_service().domain_breakdown(days=max(1, min(days, 365)))}


@app.get("/metrics/latency-histogram")
def metrics_latency_histogram() -> dict[str, object]:
    return {"items": get_metrics_service().latency_histogram()}


@app.get("/metrics/low-feedback")
def metrics_low_feedback(limit: int = 15) -> dict[str, object]:
    return {"items": get_metrics_service().low_feedback_questions(limit=max(1, min(limit, 100)))}


@app.get("/metrics/tests/latest")
def metrics_tests_latest() -> dict[str, object]:
    return get_metrics_service().latest_test_run()


@app.get("/metrics/tests/history")
def metrics_tests_history(limit: int = 20) -> dict[str, object]:
    return {"items": get_metrics_service().run_history(limit=max(1, min(limit, 100)))}


@app.get("/metrics/tests/frequent-failures")
def metrics_tests_frequent_failures(limit: int = 20) -> dict[str, object]:
    return {
        "items": get_metrics_service().frequent_test_failures(limit=max(1, min(limit, 100)))
    }


@app.get("/metrics/alerts")
def metrics_alerts() -> dict[str, object]:
    return {"items": get_metrics_service().alerts()}


@app.get("/metrics/tests/run/{run_id}")
def metrics_test_run_details(run_id: str) -> dict[str, object]:
    telemetry = get_telemetry()
    run = telemetry.get_test_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run_id not found")

    return {
        "run": run,
        "results": telemetry.get_test_results(run_id),
    }


@app.post("/metrics/tests/review")
def metrics_test_review(payload: TestReviewRequest) -> dict[str, str]:
    telemetry = get_telemetry()
    run = telemetry.get_test_run(payload.run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run_id not found")

    telemetry.update_test_result_manual_review(
        run_id=payload.run_id,
        case_id=payload.case_id,
        repeat_index=payload.repeat_index,
        manual_score=payload.manual_score,
        manual_passed=payload.manual_passed,
        manual_note=payload.manual_note.strip(),
    )
    telemetry.recompute_test_run_summary(payload.run_id)
    return {"status": "ok"}
