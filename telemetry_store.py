from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from config import TELEMETRY_DB_FILE, ensure_directories


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

        return {
            key: row[key]
            for key in row.keys()
        }

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
