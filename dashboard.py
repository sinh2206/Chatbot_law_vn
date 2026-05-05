from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
import streamlit as st

from api import TelemetryStore
from config import TEST_API_BASE_URL, TEST_DATASET_FILE
from test_runner import difficulty_label, run_suite

st.set_page_config(page_title="Legal Chatbot Test Dashboard", layout="wide")


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
                COALESCE(SUM(multi_domain), 0) AS multi_domain_count
            FROM interactions
            WHERE run_type = 'user'
            """
        )

        total = int(row["total_questions"]) if row else 0
        fallback_count = int(row["fallback_count"]) if row else 0
        multi_domain_count = int(row["multi_domain_count"]) if row else 0

        return {
            "total_questions": total,
            "avg_response_ms": float(row["avg_response_ms"] if row else 0),
            "max_response_ms": int(row["max_response_ms"] if row else 0),
            "fallback_rate": self._safe_ratio(fallback_count, total),
            "multi_domain_rate": self._safe_ratio(multi_domain_count, total),
        }

    def timeseries(self, days: int = 14) -> list[dict[str, Any]]:
        rows = self.telemetry.query(
            """
            SELECT
                DATE(created_at) AS day,
                COUNT(*) AS total_questions,
                COALESCE(AVG(response_time_ms), 0) AS avg_response_ms,
                COALESCE(SUM(fallback_used), 0) AS fallback_count
            FROM interactions
            WHERE run_type = 'user' AND created_at >= ?
            GROUP BY DATE(created_at)
            ORDER BY day ASC
            """,
            (self._cutoff_iso(days=days),),
        )

        items: list[dict[str, Any]] = []
        for row in rows:
            total = int(row["total_questions"])
            items.append(
                {
                    "day": row["day"],
                    "total_questions": total,
                    "avg_response_ms": float(row["avg_response_ms"]),
                    "fallback_rate": self._safe_ratio(int(row["fallback_count"]), total),
                }
            )
        return items

    @staticmethod
    def _safe_ratio(numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    @staticmethod
    def _cutoff_iso(days: int = 0, hours: int = 0) -> str:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days, hours=hours)
        return cutoff.isoformat()


def resolved_score(item: dict[str, Any]) -> float:
    manual_score = item.get("manual_score")
    return float(manual_score if manual_score is not None else item.get("score", 0.0))


def resolved_passed(item: dict[str, Any]) -> bool:
    manual_passed = item.get("manual_passed")
    if manual_passed is not None:
        return bool(manual_passed)
    return bool(item.get("passed", False))


def summarize(results: list[dict[str, Any]]) -> dict[str, float]:
    total = len(results)
    if total == 0:
        return {
            "accuracy": 0.0,
            "mean_score": 0.0,
            "avg_latency_ms": 0.0,
            "fallback_rate": 0.0,
            "citation_valid_rate": 0.0,
            "entity_f1_avg": 0.0,
        }

    accuracy = sum(1 for item in results if resolved_passed(item)) / total
    mean_score_value = mean([resolved_score(item) for item in results])
    avg_latency = mean([float(item.get("response_time_ms", 0.0)) for item in results])
    fallback_rate = sum(1 for item in results if item.get("fallback_used")) / total
    citation_valid_rate = sum(1 for item in results if item.get("citation_valid")) / total
    entity_f1_avg = mean([float(item.get("entity_f1", 0.0)) for item in results])

    return {
        "accuracy": accuracy,
        "mean_score": mean_score_value,
        "avg_latency_ms": avg_latency,
        "fallback_rate": fallback_rate,
        "citation_valid_rate": citation_valid_rate,
        "entity_f1_avg": entity_f1_avg,
    }


def build_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in results:
        rows.append(
            {
                "case_id": item["case_id"],
                "repeat": item["repeat_index"],
                "level": item["level"],
                "difficulty": difficulty_label(int(item["level"])),
                "domain": item["domain"],
                "question": item["question"],
                "resolved_passed": resolved_passed(item),
                "resolved_score": resolved_score(item),
                "raw_score": float(item["score"]),
                "manual_score": item["manual_score"],
                "manual_passed": item["manual_passed"],
                "latency_ms": float(item["response_time_ms"]),
                "fallback": bool(item["fallback_used"]),
                "citation_valid": bool(item["citation_valid"]),
                "citation_f1": float(item["citation_f1"]),
                "entity_f1": float(item["entity_f1"]),
                "reason": item["reason"],
                "manual_note": item["manual_note"],
            }
        )

    return pd.DataFrame(rows)


def render_kpis(summary: dict[str, float], total: int) -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Tổng test", f"{total}")
    c2.metric("Độ chính xác", f"{summary['accuracy'] * 100:.2f}%")
    c3.metric("Mean score", f"{summary['mean_score']:.2f}/10")
    c4.metric("Avg latency", f"{summary['avg_latency_ms']:.0f} ms")
    c5.metric("Fallback rate", f"{summary['fallback_rate'] * 100:.2f}%")
    c6.metric("Citation valid", f"{summary['citation_valid_rate'] * 100:.2f}%")


def render_charts(df: pd.DataFrame) -> None:
    if df.empty:
        return

    domain_group = (
        df.groupby("domain", as_index=False)
        .agg(
            accuracy=("resolved_passed", "mean"),
            mean_score=("resolved_score", "mean"),
        )
        .sort_values("accuracy", ascending=False)
    )
    difficulty_group = (
        df.groupby("difficulty", as_index=False)
        .agg(
            accuracy=("resolved_passed", "mean"),
            mean_score=("resolved_score", "mean"),
        )
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Độ chính xác theo lĩnh vực")
        st.bar_chart(domain_group.set_index("domain")[["accuracy"]])

        st.subheader("Độ khó theo cấp")
        st.bar_chart(difficulty_group.set_index("difficulty")[["accuracy"]])

    with right:
        st.subheader("Thời gian phản hồi")
        st.line_chart(df[["latency_ms"]])

        st.subheader("F1 thực thể pháp lý theo câu")
        st.line_chart(df[["entity_f1", "citation_f1"]])


def render_runtime_metrics(metrics_service: DashboardMetricsService) -> None:
    st.subheader("Runtime Metrics (User Chat)")
    summary = metrics_service.summary()
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("User Questions", f"{summary['total_questions']}")
    col2.metric("Avg Latency", f"{summary['avg_response_ms']:.0f} ms")
    col3.metric("Max Latency", f"{summary['max_response_ms']:.0f} ms")
    col4.metric("Fallback Rate", f"{summary['fallback_rate'] * 100:.2f}%")
    col5.metric("Multi-domain Rate", f"{summary['multi_domain_rate'] * 100:.2f}%")

    ts = metrics_service.timeseries(days=14)
    if ts:
        ts_df = pd.DataFrame(ts)
        left, right = st.columns(2)
        with left:
            st.caption("Số câu hỏi theo ngày")
            st.line_chart(ts_df.set_index("day")[["total_questions"]])
        with right:
            st.caption("Avg latency theo ngày")
            st.line_chart(ts_df.set_index("day")[["avg_response_ms"]])


def render_manual_review(
    telemetry: TelemetryStore,
    run_id: str,
    results: list[dict[str, Any]],
) -> None:
    st.subheader("Đánh giá thủ công / bán tự động")
    if not results:
        st.info("Chưa có dữ liệu test để review.")
        return

    case_options = [
        f"{item['case_id']} | r{item['repeat_index']} | {item['question'][:80]}"
        for item in results
    ]
    selected_label = st.selectbox("Chọn case cần chỉnh", options=case_options)
    selected_index = case_options.index(selected_label)
    selected = results[selected_index]

    st.markdown("**Câu hỏi**")
    st.write(selected["question"])

    st.markdown("**Đáp án chuẩn (ground truth)**")
    st.write(selected["expected_answer"])

    st.markdown("**Trả lời chatbot**")
    st.write(selected["response_text"])

    st.markdown("**Trích dẫn chatbot**")
    st.write("; ".join(selected.get("citations", [])) or "(không có)")

    c1, c2, c3 = st.columns(3)
    with c1:
        manual_score = st.number_input(
            "Manual score (0-10)",
            min_value=0.0,
            max_value=10.0,
            value=float(selected["manual_score"] if selected["manual_score"] is not None else selected["score"]),
            step=0.1,
        )
    with c2:
        default_pass = selected["manual_passed"]
        if default_pass is None:
            default_pass = bool(selected["passed"])
        manual_passed = st.checkbox("Manual passed", value=bool(default_pass))
    with c3:
        keep_null = st.checkbox("Xóa override", value=False)

    manual_note = st.text_area(
        "Ghi chú review",
        value=selected.get("manual_note", ""),
        height=120,
    )

    if st.button("Lưu đánh giá thủ công", type="primary"):
        telemetry.update_test_result_manual_review(
            run_id=run_id,
            case_id=selected["case_id"],
            repeat_index=int(selected["repeat_index"]),
            manual_score=None if keep_null else float(manual_score),
            manual_passed=None if keep_null else bool(manual_passed),
            manual_note=manual_note.strip(),
        )
        telemetry.recompute_test_run_summary(run_id)
        st.success("Đã lưu review thủ công.")
        st.rerun()


def render_exports(run: dict[str, Any]) -> None:
    st.subheader("Báo cáo xuất ra")
    md = str(run.get("report_markdown_path", ""))
    html = str(run.get("report_html_path", ""))
    csv_path = str(run.get("report_csv_path", ""))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption("Markdown")
        if md and Path(md).exists():
            st.code(md)
            st.download_button(
                "Tải report.md",
                data=Path(md).read_text(encoding="utf-8"),
                file_name=Path(md).name,
                mime="text/markdown",
            )
        else:
            st.write("(chưa có)")

    with col2:
        st.caption("HTML")
        if html and Path(html).exists():
            st.code(html)
            st.download_button(
                "Tải report.html",
                data=Path(html).read_text(encoding="utf-8"),
                file_name=Path(html).name,
                mime="text/html",
            )
        else:
            st.write("(chưa có)")

    with col3:
        st.caption("CSV")
        if csv_path and Path(csv_path).exists():
            st.code(csv_path)
            st.download_button(
                "Tải details.csv",
                data=Path(csv_path).read_bytes(),
                file_name=Path(csv_path).name,
                mime="text/csv",
            )
        else:
            st.write("(chưa có)")


def run_dashboard() -> None:
    st.title("Dashboard đánh giá chatbot pháp luật")

    telemetry = TelemetryStore()
    metrics_service = DashboardMetricsService(telemetry)

    with st.sidebar:
        st.header("Cấu hình chạy test")
        mode = st.selectbox("Mode", ["smoke", "monitoring", "full"], index=0)
        monitoring_size = st.slider("Monitoring size", min_value=5, max_value=100, value=12)
        repeats = st.slider("Số lần lặp", min_value=1, max_value=5, value=1)
        use_llm_grader = st.checkbox("Bật LLM grader", value=False)
        runner_mode = st.selectbox("Runner mode", ["api", "direct"], index=0)
        api_base_url = st.text_input("API base URL", value=TEST_API_BASE_URL)
        dataset_file = st.text_input("Dataset file", value=str(TEST_DATASET_FILE))
        allow_direct_fallback = st.checkbox("API lỗi -> fallback direct", value=True)
        send_email = st.checkbox("Gửi email report (nếu SMTP có cấu hình)", value=False)

        run_clicked = st.button("Chạy test suite", type="primary")

    if run_clicked:
        with st.spinner("Đang chạy test suite..."):
            result = run_suite(
                mode=mode,
                monitoring_size=monitoring_size,
                seed=42,
                dataset_file=dataset_file,
                use_llm_grader=use_llm_grader,
                runner_mode=runner_mode,
                api_base_url=api_base_url,
                api_timeout_seconds=180,
                repeats=repeats,
                allow_direct_fallback=allow_direct_fallback,
                export_markdown=True,
                export_html=True,
                export_csv=True,
                send_report_email=send_email,
            )
        st.session_state["last_run_id"] = result["run_id"]
        st.success(f"Đã chạy xong test suite. Run ID: {result['run_id']}")

    render_runtime_metrics(metrics_service)
    st.markdown("---")

    run_rows = telemetry.query(
        """
        SELECT run_id, finished_at, mode
        FROM test_runs
        ORDER BY finished_at DESC
        LIMIT 100
        """
    )

    if not run_rows:
        st.info("Chưa có test run nào. Hãy bấm 'Chạy test suite'.")
        return

    run_options = [f"{row['run_id']} | {row['mode']} | {row['finished_at']}" for row in run_rows]
    preferred_run_id = st.session_state.get("last_run_id") or str(run_rows[0]["run_id"])
    default_index = 0
    for index, row in enumerate(run_rows):
        if str(row["run_id"]) == preferred_run_id:
            default_index = index
            break

    selected_run_option = st.selectbox(
        "Chọn test run",
        options=run_options,
        index=default_index,
    )
    selected_run_id = selected_run_option.split(" | ")[0]

    run = telemetry.get_test_run(selected_run_id)
    if run is None:
        st.warning("Không tải được dữ liệu run hiện tại.")
        return

    results = telemetry.get_test_results(selected_run_id)
    summary = summarize(results)
    df = build_dataframe(results)

    st.caption(f"Run ID hiện tại: {selected_run_id}")
    render_kpis(summary, total=len(results))

    st.markdown("---")
    render_charts(df)

    st.markdown("---")
    st.subheader("Bảng chi tiết theo câu hỏi")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    render_manual_review(telemetry=telemetry, run_id=selected_run_id, results=results)

    st.markdown("---")
    render_exports(run)


if __name__ == "__main__":
    run_dashboard()
