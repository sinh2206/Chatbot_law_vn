from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Any

from config import ALERT_MAX_AVG_RESPONSE_MS, ALERT_MAX_FALLBACK_RATE, ALERT_MIN_ACCURACY
from telemetry_store import TelemetryStore


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


def aggregate_level_breakdown(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for item in results:
        level = int(item["level"])
        grouped.setdefault(level, []).append(item)

    summary: dict[str, Any] = {}
    for level, entries in grouped.items():
        total = len(entries)
        passed = sum(1 for entry in entries if entry["passed"])
        avg_score = mean([float(entry["score"]) for entry in entries]) if entries else 0.0
        summary[str(level)] = {
            "total": total,
            "passed": passed,
            "accuracy": passed / total if total else 0.0,
            "avg_score": avg_score,
        }

    return summary
