from __future__ import annotations

from expiry import parse_date
from test_runner import aggregate_level_breakdown
from vector_store import split_text_into_chunks


def test_split_text_into_chunks_returns_overlap_chunks() -> None:
    text = " ".join([f"token{i}" for i in range(300)])
    chunks = split_text_into_chunks(text=text, chunk_size=120, overlap=20)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_parse_date_supports_multiple_formats() -> None:
    assert parse_date("2026-05-01") is not None
    assert parse_date("01/05/2026") is not None
    assert parse_date("01-05-2026") is not None
    assert parse_date("") is None


def test_aggregate_level_breakdown() -> None:
    payload = [
        {"level": 1, "passed": True, "score": 8.0},
        {"level": 1, "passed": False, "score": 4.0},
        {"level": 2, "passed": True, "score": 7.0},
    ]
    summary = aggregate_level_breakdown(payload)

    assert summary["1"]["total"] == 2
    assert summary["1"]["passed"] == 1
    assert summary["2"]["total"] == 1
