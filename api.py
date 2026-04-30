from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import CORS_ALLOW_ORIGINS, DASHBOARD_DIR, DOMAIN_LABELS, FRONTEND_DIR
from metrics_dashboard import DashboardMetricsService
from orchestrator import LegalOrchestrator
from telemetry_store import (
    AgentTraceLog,
    FeedbackLog,
    InteractionLog,
    RetrievedChunkLog,
    TelemetryStore,
    utc_now_iso,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        orchestrator = get_orchestrator()
        orchestrator.expiry_monitor.run_if_due(force=True)
    except Exception:
        # Keep API process alive even if store is not ready.
        pass
    yield


app = FastAPI(title="Legal Multi-Agent RAG API", version="3.0.0", lifespan=lifespan)

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
def get_orchestrator() -> LegalOrchestrator:
    return LegalOrchestrator.from_defaults()


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
            raise RuntimeError("Vector store is empty. Run `python build_store.py` first.")

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
        except Exception:
            # Do not break request flow because of telemetry persistence errors.
            pass

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
