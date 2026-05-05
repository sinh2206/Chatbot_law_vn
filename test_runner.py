from __future__ import annotations

import argparse
import csv
import json
import random
import re
import smtplib
import ssl
import time
import unicodedata
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from api import (
    AgentTraceLog,
    InteractionLog,
    RetrievedChunkLog,
    TelemetryStore,
    utc_now_iso,
)
from chatbot_core import GeminiClient, LegalOrchestrator, parse_json_from_text
from config import (
    ADMIN_EMAIL,
    DOMAIN_LABELS,
    REPORTS_DIR,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_SENDER,
    SMTP_USERNAME,
    SMTP_USE_SSL,
    SMTP_USE_TLS,
    TEST_ALLOW_DIRECT_FALLBACK,
    TEST_API_BASE_URL,
    TEST_API_TIMEOUT_SECONDS,
    TEST_DATASET_FILE,
    TEST_PASS_SCORE,
    TEST_REPORT_SEND_EMAIL,
    TEST_RUNNER_MODE,
    TEST_USE_LLM_GRADER,
)
from testsuite.default_cases import default_test_cases

DOC_PATTERN = re.compile(r"\b\d{1,4}/\d{4}/[A-Za-z0-9-]+\b", re.IGNORECASE)
ARTICLE_PATTERN = re.compile(r"\b(?:điều|dieu)\s+\d+[a-z]?\b", re.IGNORECASE)


@dataclass
class EvaluatedCase:
    case_id: str
    repeat_index: int
    level: int
    domain: str
    question: str
    expected_answer: str
    expected_keywords: list[str]
    expected_citations: list[str]
    answer: str
    citations: list[str]
    fallback_used: bool
    response_time_ms: int
    heuristic_score: float
    llm_score: float | None
    score: float
    passed: bool
    reason: str
    citation_precision: float
    citation_recall: float
    citation_f1: float
    entity_precision: float
    entity_recall: float
    entity_f1: float
    citation_valid: bool
    interaction_id: str
    grader_mode: str


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


def difficulty_label(level: int) -> str:
    if level == 1:
        return "Easy"
    if level == 2:
        return "Medium"
    return "Hard"


def load_test_cases(dataset_file: str) -> list[dict[str, Any]]:
    path = TEST_DATASET_FILE if not dataset_file else Path(dataset_file)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError("Test dataset must be a JSON list")
        return payload

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = default_test_cases()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return payload


def select_cases(mode: str, cases: list[dict[str, Any]], monitoring_size: int, seed: int) -> list[dict[str, Any]]:
    if mode == "full":
        return cases
    if mode == "smoke":
        level_1 = [case for case in cases if int(case.get("level", 0)) == 1]
        return level_1[: min(12, len(level_1))]
    if mode == "monitoring":
        random.seed(seed)
        population = cases[:]
        size = min(len(population), max(1, monitoring_size))
        return random.sample(population, size)
    raise ValueError(f"Unsupported mode: {mode}")


def difficulty_bucket(level: int) -> str:
    if level == 1:
        return "easy"
    if level == 2:
        return "medium"
    return "hard"


def primary_domain(domain_field: str) -> str:
    return domain_field.split(",")[0].strip()


def build_balanced_subset(cases: list[dict[str, object]], target_size: int, seed: int) -> list[dict[str, object]]:
    if target_size <= 0 or target_size >= len(cases):
        return cases

    random.seed(seed)

    buckets: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for case in cases:
        level = int(case.get("level", 1))
        domain = primary_domain(str(case.get("domain", "")))
        buckets[(domain, difficulty_bucket(level))].append(case)

    bucket_keys = sorted(buckets.keys())
    per_bucket = max(1, target_size // max(1, len(bucket_keys)))

    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()

    for key in bucket_keys:
        items = buckets[key][:]
        random.shuffle(items)
        take = min(len(items), per_bucket)
        for item in items[:take]:
            case_id = str(item.get("case_id", ""))
            if case_id and case_id not in selected_ids:
                selected.append(item)
                selected_ids.add(case_id)

    if len(selected) >= target_size:
        return selected[:target_size]

    remaining = [item for item in cases if str(item.get("case_id", "")) not in selected_ids]
    random.shuffle(remaining)
    selected.extend(remaining[: max(0, target_size - len(selected))])
    return selected


def bootstrap_testset(target_size: int, seed: int, export_all: bool, output_file: str = "") -> Path:
    path = TEST_DATASET_FILE if not output_file else Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    full_cases = default_test_cases()
    payload = full_cases if export_all else build_balanced_subset(full_cases, target_size, seed)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    lowered = strip_accents(text.lower())
    lowered = re.sub(r"[^a-z0-9/\-\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def extract_doc_ids(text: str) -> set[str]:
    return {match.group(0).upper() for match in DOC_PATTERN.finditer(text)}


def extract_articles(text: str) -> set[str]:
    normalized = strip_accents(text.lower())
    return {
        re.sub(r"\s+", " ", match.group(0)).strip()
        for match in ARTICLE_PATTERN.finditer(normalized)
    }


def compute_set_metrics(expected: set[str], actual: set[str]) -> tuple[float, float, float]:
    if not expected and not actual:
        return 1.0, 1.0, 1.0
    if not actual:
        return 0.0, 0.0, 0.0
    if not expected:
        return 1.0, 1.0, 1.0

    matched = len(expected & actual)
    precision = matched / len(actual) if actual else 0.0
    recall = matched / len(expected) if expected else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate_citations(expected_citations: list[str], answer_text: str, citations: list[str]) -> tuple[float, float, float, bool]:
    expected_docs: set[str] = set()
    for item in expected_citations:
        expected_docs |= extract_doc_ids(item)

    actual_docs: set[str] = set()
    for item in citations:
        actual_docs |= extract_doc_ids(item)
    actual_docs |= extract_doc_ids(answer_text)

    precision, recall, f1 = compute_set_metrics(expected_docs, actual_docs)
    citation_valid = (recall >= 1.0 and precision > 0) if expected_docs else len(actual_docs) > 0
    return precision, recall, f1, citation_valid


def evaluate_legal_entities(
    expected_answer: str,
    expected_citations: list[str],
    answer_text: str,
    citations: list[str],
) -> tuple[float, float, float]:
    expected_entities: set[str] = set()
    actual_entities: set[str] = set()

    for item in expected_citations:
        for doc in extract_doc_ids(item):
            expected_entities.add(f"doc:{doc}")
        for article in extract_articles(item):
            expected_entities.add(f"article:{article}")

    for doc in extract_doc_ids(expected_answer):
        expected_entities.add(f"doc:{doc}")
    for article in extract_articles(expected_answer):
        expected_entities.add(f"article:{article}")

    source_text = "\n".join(citations + [answer_text])
    for doc in extract_doc_ids(source_text):
        actual_entities.add(f"doc:{doc}")
    for article in extract_articles(source_text):
        actual_entities.add(f"article:{article}")

    return compute_set_metrics(expected_entities, actual_entities)


def evaluate_heuristic(
    level: int,
    expected_keywords: list[str],
    answer_text: str,
    fallback_used: bool,
    citation_f1: float,
    entity_f1: float,
    citation_valid: bool,
) -> tuple[float, str]:
    normalized_answer = normalize_text(answer_text)

    keyword_hits = 0
    for keyword in expected_keywords:
        marker = normalize_text(keyword)
        if marker and marker in normalized_answer:
            keyword_hits += 1

    keyword_score = keyword_hits / len(expected_keywords) if expected_keywords else 0.0
    words = len(answer_text.split())
    coherence = 1.0 if words >= 110 else 0.85 if words >= 60 else 0.65 if words >= 30 else 0.35
    citation_bonus = 1.0 if citation_valid else 0.5 * citation_f1
    fallback_component = 0.85 if fallback_used else 1.0

    if level <= 1:
        combined = (
            0.55 * keyword_score
            + 0.25 * citation_f1
            + 0.10 * coherence
            + 0.10 * citation_bonus
        )
    elif level == 2:
        combined = (
            0.45 * keyword_score
            + 0.25 * citation_f1
            + 0.20 * entity_f1
            + 0.10 * coherence
        )
    else:
        combined = (
            0.35 * keyword_score
            + 0.20 * citation_f1
            + 0.25 * entity_f1
            + 0.10 * coherence
            + 0.10 * fallback_component
        )

    score = max(0.0, min(10.0, combined * 10.0))
    reason = (
        f"keyword={keyword_score:.2f}, citation_f1={citation_f1:.2f}, entity_f1={entity_f1:.2f}, "
        f"coherence={coherence:.2f}, fallback_component={fallback_component:.2f}, "
        f"citation_valid={int(citation_valid)}"
    )
    return score, reason


def evaluate_with_llm(
    llm: GeminiClient,
    question: str,
    expected_answer: str,
    expected_citations: list[str],
    actual_answer: str,
) -> tuple[float, str]:
    prompt = f"""
Bạn là giám khảo đánh giá chatbot pháp lý Việt Nam.

Câu hỏi: {question}
Đáp án chuẩn: {expected_answer}
Trích dẫn chuẩn: {'; '.join(expected_citations) if expected_citations else '(không yêu cầu)'}
Câu trả lời chatbot: {actual_answer}

Chấm điểm từ 0 đến 10 theo tiêu chí:
1) Đúng bản chất pháp lý
2) Đầy đủ luận điểm chính
3) Trích dẫn căn cứ phù hợp
4) Mạch lạc và có cảnh báo giới hạn khi thiếu dữ liệu

Trả JSON duy nhất:
{{
  "score": 7.5,
  "reason": "..."
}}
""".strip()

    raw = llm.generate(prompt=prompt, temperature=0.0)
    payload = parse_json_from_text(raw)
    score = float(payload.get("score", 0.0))
    reason = str(payload.get("reason", ""))
    return max(0.0, min(10.0, score)), reason


def parse_domain_for_request(domain: str) -> str | None:
    cleaned = domain.strip()
    if not cleaned:
        return None
    if "," in cleaned:
        return None
    return cleaned if cleaned in DOMAIN_LABELS else None


def ask_via_api(base_url: str, question: str, domain: str, timeout_seconds: int) -> dict[str, Any]:
    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "requests package is required for API mode. Install dependencies first."
        ) from exc

    payload: dict[str, Any] = {"question": question, "run_type": "test"}
    domain_request = parse_domain_for_request(domain)
    if domain_request:
        payload["domain"] = domain_request

    started = perf_counter()
    response = requests.post(
        f"{base_url}/chat",
        json=payload,
        timeout=timeout_seconds,
    )
    elapsed_ms = int((perf_counter() - started) * 1000)

    if response.status_code >= 400:
        raise RuntimeError(f"API call failed {response.status_code}: {response.text}")

    data = response.json()
    if "response_time_ms" not in data:
        data["response_time_ms"] = elapsed_ms
    return data


def build_agent_traces(result: Any) -> list[AgentTraceLog]:
    traces: list[AgentTraceLog] = []
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
        traces.append(
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
    return traces


def ask_via_direct(base_orchestrator: LegalOrchestrator, question: str, domain: str) -> tuple[dict[str, Any], list[AgentTraceLog]]:
    domain_request = parse_domain_for_request(domain)
    if domain_request:
        orchestrator = LegalOrchestrator(
            vector_store=base_orchestrator.vector_store,
            domains=[domain_request],
            metadata_registry=base_orchestrator.metadata_registry,
            notifier=base_orchestrator.notifier,
        )
    else:
        orchestrator = base_orchestrator

    started = perf_counter()
    result = orchestrator.answer(question)
    elapsed_ms = int((perf_counter() - started) * 1000)

    traces = build_agent_traces(result)
    fallback_used = any(agent.mode == "web_fallback" for agent in result.agent_answers)

    payload: dict[str, Any] = {
        "interaction_id": "",
        "answer": result.merged.answer,
        "domains": result.intent.domains,
        "citations": result.merged.citations,
        "sources": result.merged.sources,
        "routed_sub_questions": [
            {"domain": domain_name, "question": sub_q}
            for domain_name, sub_q in result.routed_sub_questions
        ],
        "fallback_used": fallback_used,
        "agent_modes": [
            {
                "domain": agent.domain,
                "mode": agent.mode,
                "fallback_reason": agent.fallback_reason,
            }
            for agent in result.agent_answers
        ],
        "response_time_ms": elapsed_ms,
    }
    return payload, traces


def domain_accuracy(cases: list[EvaluatedCase]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[EvaluatedCase]] = {}
    for case in cases:
        primary = case.domain.split(",")[0].strip()
        grouped.setdefault(primary, []).append(case)

    output: dict[str, dict[str, float]] = {}
    for domain, entries in grouped.items():
        total = len(entries)
        passed = sum(1 for entry in entries if entry.passed)
        output[domain] = {
            "total": total,
            "passed": passed,
            "accuracy": (passed / total) if total else 0.0,
            "avg_score": mean([entry.score for entry in entries]) if entries else 0.0,
        }
    return output


def difficulty_accuracy(cases: list[EvaluatedCase]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[EvaluatedCase]] = {}
    for case in cases:
        grouped.setdefault(difficulty_label(case.level), []).append(case)

    output: dict[str, dict[str, float]] = {}
    for label, entries in grouped.items():
        total = len(entries)
        passed = sum(1 for entry in entries if entry.passed)
        output[label] = {
            "total": total,
            "passed": passed,
            "accuracy": (passed / total) if total else 0.0,
            "avg_score": mean([entry.score for entry in entries]) if entries else 0.0,
        }
    return output


def build_markdown_report(
    run_id: str,
    mode: str,
    started_at: str,
    finished_at: str,
    cases: list[EvaluatedCase],
) -> str:
    total = len(cases)
    passed = sum(1 for item in cases if item.passed)
    accuracy = (passed / total) * 100 if total else 0.0
    avg_score = mean([item.score for item in cases]) if cases else 0.0
    avg_latency_s = (mean([item.response_time_ms for item in cases]) / 1000.0) if cases else 0.0
    fallback_rate = sum(1 for item in cases if item.fallback_used) / total if total else 0.0
    citation_valid_rate = sum(1 for item in cases if item.citation_valid) / total if total else 0.0
    citation_f1_avg = mean([item.citation_f1 for item in cases]) if cases else 0.0
    entity_f1_avg = mean([item.entity_f1 for item in cases]) if cases else 0.0

    domain_stats = domain_accuracy(cases)
    difficulty_stats = difficulty_accuracy(cases)

    lines: list[str] = []
    lines.append("# BAO CAO HIEU NANG CHATBOT")
    lines.append("")
    lines.append(f"- Run ID: `{run_id}`")
    lines.append(f"- Mode: `{mode}`")
    lines.append(f"- Bat dau: `{started_at}`")
    lines.append(f"- Ket thuc: `{finished_at}`")
    lines.append(f"- Tong so test: `{total}`")
    lines.append("")
    lines.append("## KET QUA")
    lines.append(f"- Do chinh xac trung binh: **{accuracy:.2f}%**")
    lines.append(f"- Mean score: **{avg_score:.2f}/10**")
    lines.append(f"- Thoi gian phan hoi TB: **{avg_latency_s:.2f} giay**")
    lines.append(f"- Ty le fallback search: **{fallback_rate * 100:.2f}%**")
    lines.append(f"- Ty le trich dan hop le: **{citation_valid_rate * 100:.2f}%**")
    lines.append(f"- Citation F1 trung binh: **{citation_f1_avg * 100:.2f}%**")
    lines.append(f"- Legal entity F1 trung binh: **{entity_f1_avg * 100:.2f}%**")
    lines.append("")

    lines.append("## Do Chinh Xac Theo Linh Vuc")
    for domain_code, stats in sorted(domain_stats.items()):
        label = DOMAIN_LABELS.get(domain_code, domain_code)
        lines.append(
            f"- {label}: {stats['accuracy'] * 100:.2f}% (avg score {stats['avg_score']:.2f})"
        )
    lines.append("")

    lines.append("## Do Chinh Xac Theo Do Kho")
    for label in ["Easy", "Medium", "Hard"]:
        if label not in difficulty_stats:
            continue
        stats = difficulty_stats[label]
        lines.append(
            f"- {label}: {stats['accuracy'] * 100:.2f}% (avg score {stats['avg_score']:.2f})"
        )
    lines.append("")

    failed = [item for item in cases if not item.passed]
    lines.append("## Chi Tiet Cau Hoi Sai")
    if not failed:
        lines.append("- Khong co cau sai trong run nay.")
    else:
        for index, item in enumerate(failed[:30], start=1):
            lines.append(
                f"{index}. [{item.case_id}] {item.question} -> score {item.score:.2f}; ly do: {item.reason}"
            )
    return "\n".join(lines)


def build_report_paths(run_id: str) -> dict[str, Path]:
    root = REPORTS_DIR / run_id
    return {
        "root": root,
        "markdown": root / "report.md",
        "html": root / "report.html",
        "csv": root / "details.csv",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_markdown(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def markdown_to_basic_html(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    html_lines = [
        "<!doctype html>",
        "<html lang='vi'><head><meta charset='utf-8'><title>Chatbot Report</title>",
        "<style>body{font-family:Segoe UI,Arial,sans-serif;max-width:980px;margin:20px auto;line-height:1.5;color:#1f2d2b;}h1,h2{color:#0e5e4e;}li{margin:4px 0;}code{background:#edf3f1;padding:1px 4px;border-radius:4px;}</style>",
        "</head><body>",
    ]

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# "):
            html_lines.append(f"<h1>{stripped[2:]}</h1>")
        elif stripped.startswith("## "):
            html_lines.append(f"<h2>{stripped[3:]}</h2>")
        elif stripped.startswith("- "):
            html_lines.append(f"<li>{stripped[2:]}</li>")
        elif stripped:
            html_lines.append(f"<p>{stripped}</p>")
        else:
            html_lines.append("<br />")

    html_lines.append("</body></html>")
    return "\n".join(html_lines)


def write_html(path: Path, html: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def send_email_report_if_configured(
    subject: str,
    text_body: str,
    html_body: str | None = None,
    attachments: list[Path] | None = None,
) -> bool:
    if not (ADMIN_EMAIL and SMTP_HOST):
        return False

    email_message = EmailMessage()
    email_message["From"] = SMTP_SENDER or SMTP_USERNAME
    email_message["To"] = ADMIN_EMAIL
    email_message["Subject"] = subject
    email_message.set_content(text_body)

    if html_body:
        email_message.add_alternative(html_body, subtype="html")

    for attachment in attachments or []:
        if not attachment.exists() or not attachment.is_file():
            continue
        data = attachment.read_bytes()
        maintype = "application"
        subtype = "octet-stream"
        if attachment.suffix.lower() == ".csv":
            maintype, subtype = "text", "csv"
        elif attachment.suffix.lower() == ".md":
            maintype, subtype = "text", "markdown"
        elif attachment.suffix.lower() == ".html":
            maintype, subtype = "text", "html"

        email_message.add_attachment(
            data,
            maintype=maintype,
            subtype=subtype,
            filename=attachment.name,
        )

    if SMTP_USE_SSL:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
            if SMTP_USERNAME:
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(email_message)
        return True

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        if SMTP_USE_TLS:
            context = ssl.create_default_context()
            server.starttls(context=context)
        if SMTP_USERNAME:
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(email_message)
    return True


def run_suite(
    mode: str,
    monitoring_size: int,
    seed: int,
    dataset_file: str,
    use_llm_grader: bool,
    runner_mode: str,
    api_base_url: str,
    api_timeout_seconds: int,
    repeats: int,
    allow_direct_fallback: bool,
    export_markdown: bool,
    export_html: bool,
    export_csv: bool,
    send_report_email: bool,
) -> dict[str, Any]:
    telemetry = TelemetryStore()
    cases = load_test_cases(dataset_file=dataset_file)
    selected_cases = select_cases(
        mode=mode,
        cases=cases,
        monitoring_size=monitoring_size,
        seed=seed,
    )

    run_id = uuid.uuid4().hex
    started_at = utc_now_iso()
    llm_grader = GeminiClient() if use_llm_grader else None

    base_orchestrator: LegalOrchestrator | None = None
    if runner_mode == "direct" or allow_direct_fallback:
        base_orchestrator = LegalOrchestrator.from_defaults()
        if base_orchestrator.vector_store.count() == 0:
            raise RuntimeError("Vector store is empty. Run `python vector_store.py --build` first.")

    evaluated_cases: list[EvaluatedCase] = []

    for repeat_index in range(1, max(1, repeats) + 1):
        for case in selected_cases:
            case_id = str(case.get("case_id", ""))
            level = int(case.get("level", 1))
            domain = str(case.get("domain", ""))
            question = str(case.get("question", "")).strip()
            expected_answer = str(case.get("expected_answer", ""))
            expected_keywords = [str(item) for item in case.get("expected_keywords", [])]
            expected_citations = [str(item) for item in case.get("expected_citations", [])]

            if not question:
                continue

            run_answer: dict[str, Any]
            agent_traces: list[AgentTraceLog] = []
            start_time = utc_now_iso()
            used_mode = runner_mode

            try:
                if runner_mode == "api":
                    run_answer = ask_via_api(
                        base_url=api_base_url,
                        question=question,
                        domain=domain,
                        timeout_seconds=api_timeout_seconds,
                    )
                else:
                    if base_orchestrator is None:
                        base_orchestrator = LegalOrchestrator.from_defaults()
                    run_answer, agent_traces = ask_via_direct(
                        base_orchestrator=base_orchestrator,
                        question=question,
                        domain=domain,
                    )
            except Exception as exc:
                if runner_mode == "api" and allow_direct_fallback:
                    used_mode = "direct_fallback"
                    if base_orchestrator is None:
                        base_orchestrator = LegalOrchestrator.from_defaults()
                    run_answer, agent_traces = ask_via_direct(
                        base_orchestrator=base_orchestrator,
                        question=question,
                        domain=domain,
                    )
                else:
                    raise RuntimeError(f"Failed to run case {case_id}: {exc}") from exc

            answer = str(run_answer.get("answer", ""))
            citations = [str(item) for item in run_answer.get("citations", [])]
            fallback_used = bool(run_answer.get("fallback_used", False))
            response_time_ms = int(run_answer.get("response_time_ms", 0))

            citation_precision, citation_recall, citation_f1, citation_valid = evaluate_citations(
                expected_citations=expected_citations,
                answer_text=answer,
                citations=citations,
            )
            entity_precision, entity_recall, entity_f1 = evaluate_legal_entities(
                expected_answer=expected_answer,
                expected_citations=expected_citations,
                answer_text=answer,
                citations=citations,
            )
            heuristic_score, heuristic_reason = evaluate_heuristic(
                level=level,
                expected_keywords=expected_keywords,
                answer_text=answer,
                fallback_used=fallback_used,
                citation_f1=citation_f1,
                entity_f1=entity_f1,
                citation_valid=citation_valid,
            )

            llm_score: float | None = None
            final_score = heuristic_score
            reason = heuristic_reason
            grader_mode = "heuristic"

            if llm_grader is not None:
                grader_mode = "hybrid"
                try:
                    llm_score, llm_reason = evaluate_with_llm(
                        llm=llm_grader,
                        question=question,
                        expected_answer=expected_answer,
                        expected_citations=expected_citations,
                        actual_answer=answer,
                    )
                    final_score = 0.55 * heuristic_score + 0.45 * llm_score
                    reason = f"heuristic=({heuristic_reason}) | llm=({llm_reason})"
                except Exception as exc:
                    reason = f"{heuristic_reason} | llm_error={exc}"

            passed = final_score >= TEST_PASS_SCORE
            interaction_id = str(run_answer.get("interaction_id", "")).strip()
            if not interaction_id:
                interaction_id = telemetry.new_interaction_id()

            if runner_mode != "api" or used_mode == "direct_fallback":
                telemetry.save_interaction(
                    InteractionLog(
                        interaction_id=interaction_id,
                        created_at=start_time,
                        start_time=start_time,
                        end_time=utc_now_iso(),
                        response_time_ms=response_time_ms,
                        question=question,
                        domain_requested=domain,
                        domains_detected=[str(item) for item in run_answer.get("domains", [])],
                        answer=answer,
                        fallback_used=fallback_used,
                        multi_domain=("," in domain),
                        agent_count=len(agent_traces),
                        rag_query_count=len(agent_traces),
                        rag_hit_count=sum(1 for item in agent_traces if item.retrieved_count > 0),
                        fallback_count=sum(1 for item in agent_traces if item.mode == "web_fallback"),
                        error="",
                        run_type="test",
                        agent_traces=agent_traces,
                    )
                )

            evaluated = EvaluatedCase(
                case_id=case_id,
                repeat_index=repeat_index,
                level=level,
                domain=domain,
                question=question,
                expected_answer=expected_answer,
                expected_keywords=expected_keywords,
                expected_citations=expected_citations,
                answer=answer,
                citations=citations,
                fallback_used=fallback_used,
                response_time_ms=response_time_ms,
                heuristic_score=heuristic_score,
                llm_score=llm_score,
                score=final_score,
                passed=passed,
                reason=reason,
                citation_precision=citation_precision,
                citation_recall=citation_recall,
                citation_f1=citation_f1,
                entity_precision=entity_precision,
                entity_recall=entity_recall,
                entity_f1=entity_f1,
                citation_valid=citation_valid,
                interaction_id=interaction_id,
                grader_mode=grader_mode if used_mode != "direct_fallback" else "direct_fallback",
            )
            evaluated_cases.append(evaluated)

            telemetry.save_test_result(
                run_id=run_id,
                case_id=case_id,
                repeat_index=repeat_index,
                level=level,
                domain=domain,
                question=question,
                expected_answer=expected_answer,
                expected_keywords=expected_keywords,
                expected_citations=expected_citations,
                response_text=answer,
                citations=citations,
                fallback_used=fallback_used,
                response_time_ms=response_time_ms,
                score=final_score,
                passed=passed,
                reason=reason,
                heuristic_score=heuristic_score,
                llm_score=llm_score,
                citation_precision=citation_precision,
                citation_recall=citation_recall,
                citation_f1=citation_f1,
                entity_precision=entity_precision,
                entity_recall=entity_recall,
                entity_f1=entity_f1,
                citation_valid=citation_valid,
                grader_mode=evaluated.grader_mode,
            )

    finished_at = utc_now_iso()
    total_cases = len(evaluated_cases)
    passed_cases = sum(1 for item in evaluated_cases if item.passed)
    accuracy = (passed_cases / total_cases) if total_cases else 0.0
    avg_score = mean([item.score for item in evaluated_cases]) if evaluated_cases else 0.0
    avg_response_ms = mean([item.response_time_ms for item in evaluated_cases]) if evaluated_cases else 0.0
    fallback_rate = (
        sum(1 for item in evaluated_cases if item.fallback_used) / total_cases if total_cases else 0.0
    )
    citation_valid_rate = (
        sum(1 for item in evaluated_cases if item.citation_valid) / total_cases if total_cases else 0.0
    )
    citation_f1_avg = mean([item.citation_f1 for item in evaluated_cases]) if evaluated_cases else 0.0
    entity_f1_avg = mean([item.entity_f1 for item in evaluated_cases]) if evaluated_cases else 0.0

    level_breakdown = aggregate_level_breakdown(
        [{"level": item.level, "passed": item.passed, "score": item.score} for item in evaluated_cases]
    )

    notes = (
        f"mode={mode}; runner_mode={runner_mode}; repeats={repeats}; "
        f"use_llm_grader={use_llm_grader}; dataset={dataset_file or str(TEST_DATASET_FILE)}"
    )

    report_paths = build_report_paths(run_id)
    markdown_report = build_markdown_report(
        run_id=run_id,
        mode=mode,
        started_at=started_at,
        finished_at=finished_at,
        cases=evaluated_cases,
    )

    markdown_path: Path | None = report_paths["markdown"] if export_markdown else None
    html_path: Path | None = report_paths["html"] if export_html else None
    csv_path: Path | None = report_paths["csv"] if export_csv else None

    if markdown_path:
        write_markdown(markdown_path, markdown_report)

    html_report = markdown_to_basic_html(markdown_report)
    if html_path:
        write_html(html_path, html_report)

    csv_rows = [
        {
            "run_id": run_id,
            "case_id": item.case_id,
            "repeat_index": item.repeat_index,
            "level": item.level,
            "difficulty": difficulty_label(item.level),
            "domain": item.domain,
            "question": item.question,
            "expected_answer": item.expected_answer,
            "expected_citations": " | ".join(item.expected_citations),
            "response_text": item.answer,
            "citations": " | ".join(item.citations),
            "fallback_used": int(item.fallback_used),
            "response_time_ms": item.response_time_ms,
            "heuristic_score": round(item.heuristic_score, 4),
            "llm_score": "" if item.llm_score is None else round(item.llm_score, 4),
            "final_score": round(item.score, 4),
            "passed": int(item.passed),
            "citation_valid": int(item.citation_valid),
            "citation_precision": round(item.citation_precision, 4),
            "citation_recall": round(item.citation_recall, 4),
            "citation_f1": round(item.citation_f1, 4),
            "entity_precision": round(item.entity_precision, 4),
            "entity_recall": round(item.entity_recall, 4),
            "entity_f1": round(item.entity_f1, 4),
            "reason": item.reason,
            "interaction_id": item.interaction_id,
            "grader_mode": item.grader_mode,
        }
        for item in evaluated_cases
    ]

    if csv_path:
        write_csv(csv_path, csv_rows)

    if send_report_email:
        attachments = [path for path in [markdown_path, html_path, csv_path] if path and path.exists()]
        send_email_report_if_configured(
            subject=f"[Chatbot Report] run_id={run_id}",
            text_body=markdown_report,
            html_body=html_report,
            attachments=attachments,
        )

    telemetry.save_test_run(
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        mode=mode,
        total_cases=total_cases,
        passed_cases=passed_cases,
        avg_score=avg_score,
        avg_response_ms=avg_response_ms,
        fallback_rate=fallback_rate,
        level_breakdown=level_breakdown,
        notes=notes,
        citation_valid_rate=citation_valid_rate,
        citation_f1_avg=citation_f1_avg,
        entity_f1_avg=entity_f1_avg,
        accuracy=accuracy,
        mean_score=avg_score,
        report_markdown_path=str(markdown_path) if markdown_path else "",
        report_html_path=str(html_path) if html_path else "",
        report_csv_path=str(csv_path) if csv_path else "",
    )

    return {
        "run_id": run_id,
        "mode": mode,
        "started_at": started_at,
        "finished_at": finished_at,
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "accuracy": accuracy,
        "avg_score": avg_score,
        "mean_score": avg_score,
        "avg_response_ms": avg_response_ms,
        "fallback_rate": fallback_rate,
        "citation_valid_rate": citation_valid_rate,
        "citation_f1_avg": citation_f1_avg,
        "entity_f1_avg": entity_f1_avg,
        "report_markdown_path": str(markdown_path) if markdown_path else "",
        "report_html_path": str(html_path) if html_path else "",
        "report_csv_path": str(csv_path) if csv_path else "",
    }


def run_schedule(
    schedule_plan: str,
    args: argparse.Namespace,
) -> None:
    if schedule_plan == "hourly":
        interval_seconds = 3600
        mode = "monitoring"
    elif schedule_plan == "daily":
        interval_seconds = 86400
        mode = "full"
    else:
        interval_seconds = 86400 * 7
        mode = "full"

    while True:
        print(f"[{datetime.now().isoformat()}] scheduled run: plan={schedule_plan}, mode={mode}")
        result = run_suite(
            mode=mode,
            monitoring_size=args.monitoring_size,
            seed=args.seed,
            dataset_file=args.dataset_file,
            use_llm_grader=(args.llm_grader or TEST_USE_LLM_GRADER),
            runner_mode=args.runner_mode,
            api_base_url=args.api_base_url,
            api_timeout_seconds=args.api_timeout_seconds,
            repeats=max(1, args.repeats),
            allow_direct_fallback=(args.allow_direct_fallback or TEST_ALLOW_DIRECT_FALLBACK),
            export_markdown=not args.no_export_markdown,
            export_html=not args.no_export_html,
            export_csv=not args.no_export_csv,
            send_report_email=(args.send_report_email or TEST_REPORT_SEND_EMAIL),
        )
        print(f"run_id={result['run_id']} accuracy={result['accuracy'] * 100:.2f}%")

        if args.once:
            return
        print(f"Sleep {interval_seconds} seconds...")
        time.sleep(interval_seconds)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Unified test runner (suite + bootstrap + scheduler + reporting).")
    parser.add_argument("--mode", choices=["smoke", "full", "monitoring"], default="smoke")
    parser.add_argument("--monitoring-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-file", type=str, default="")
    parser.add_argument("--llm-grader", action="store_true")
    parser.add_argument("--runner-mode", choices=["api", "direct"], default=TEST_RUNNER_MODE)
    parser.add_argument("--api-base-url", type=str, default=TEST_API_BASE_URL)
    parser.add_argument("--api-timeout-seconds", type=int, default=TEST_API_TIMEOUT_SECONDS)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--allow-direct-fallback", action="store_true")
    parser.add_argument("--no-export-markdown", action="store_true")
    parser.add_argument("--no-export-html", action="store_true")
    parser.add_argument("--no-export-csv", action="store_true")
    parser.add_argument("--send-report-email", action="store_true")

    parser.add_argument("--bootstrap", action="store_true", help="Bootstrap test_cases.json and exit.")
    parser.add_argument("--bootstrap-size", type=int, default=45, help="Target size for bootstrap subset.")
    parser.add_argument("--bootstrap-all", action="store_true", help="Export full default test set.")
    parser.add_argument("--bootstrap-output", type=str, default="")

    parser.add_argument("--schedule", choices=["hourly", "daily", "weekly"], default="")
    parser.add_argument("--once", action="store_true", help="Run one cycle in scheduler mode then exit.")

    args = parser.parse_args()

    if args.bootstrap:
        output = bootstrap_testset(
            target_size=max(1, args.bootstrap_size),
            seed=args.seed,
            export_all=args.bootstrap_all,
            output_file=args.bootstrap_output,
        )
        print(f"Bootstrapped test set -> {output}")
        return

    if args.schedule:
        run_schedule(args.schedule, args)
        return

    result = run_suite(
        mode=args.mode,
        monitoring_size=args.monitoring_size,
        seed=args.seed,
        dataset_file=args.dataset_file,
        use_llm_grader=(args.llm_grader or TEST_USE_LLM_GRADER),
        runner_mode=args.runner_mode,
        api_base_url=args.api_base_url,
        api_timeout_seconds=args.api_timeout_seconds,
        repeats=max(1, args.repeats),
        allow_direct_fallback=(args.allow_direct_fallback or TEST_ALLOW_DIRECT_FALLBACK),
        export_markdown=not args.no_export_markdown,
        export_html=not args.no_export_html,
        export_csv=not args.no_export_csv,
        send_report_email=(args.send_report_email or TEST_REPORT_SEND_EMAIL),
    )

    print(f"run_id={result['run_id']}")
    print(f"mode={result['mode']}")
    print(f"total_cases={result['total_cases']}")
    print(f"passed_cases={result['passed_cases']}")
    print(f"accuracy={result['accuracy'] * 100:.2f}%")
    print(f"mean_score={result['mean_score']:.2f}")
    print(f"avg_response_ms={result['avg_response_ms']:.2f}")
    print(f"fallback_rate={result['fallback_rate'] * 100:.2f}%")
    print(f"citation_valid_rate={result['citation_valid_rate'] * 100:.2f}%")
    print(f"entity_f1_avg={result['entity_f1_avg'] * 100:.2f}%")
    if result["report_markdown_path"]:
        print(f"report_markdown={result['report_markdown_path']}")
    if result["report_html_path"]:
        print(f"report_html={result['report_html_path']}")
    if result["report_csv_path"]:
        print(f"report_csv={result['report_csv_path']}")


if __name__ == "__main__":
    _cli()
