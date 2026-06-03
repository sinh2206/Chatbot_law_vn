from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
import sys
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DEFAULT_EMBEDDING_MODEL_FALLBACK, FINETUNE_DIR  # noqa: E402


def setup_stdout_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_no} is invalid JSON") from exc
    return rows


def p95(values: list[int]) -> int:
    if not values:
        return 0
    idx = max(0, int(0.95 * len(values)) - 1)
    return sorted(values)[idx]


def pct(part: int, total: int) -> float:
    return (part / total * 100.0) if total else 0.0


def has_alpha_questionmark_alpha(text: str) -> bool:
    for idx in range(1, len(text) - 1):
        if text[idx] != "?":
            continue
        if text[idx - 1].isalpha() and text[idx + 1].isalpha():
            return True
    return False


def calc_basic_stats(rows: list[dict[str, object]]) -> dict[str, object]:
    q_lens = [len(str(r.get("query", ""))) for r in rows]
    p_lens = [len(str(r.get("positive", ""))) for r in rows]
    domains = Counter(str(r.get("domain", "UNKNOWN")) for r in rows)

    duplicate_count = 0
    seen: set[tuple[str, str]] = set()
    for r in rows:
        key = (str(r.get("query", "")).strip().lower(), str(r.get("positive", "")).strip())
        if key in seen:
            duplicate_count += 1
        else:
            seen.add(key)

    suspicious_questionmark = 0
    for r in rows:
        text = f"{r.get('query', '')} {r.get('positive', '')}"
        if has_alpha_questionmark_alpha(text):
            suspicious_questionmark += 1

    return {
        "rows": len(rows),
        "query_char_mean": round(statistics.mean(q_lens), 2) if q_lens else 0,
        "query_char_median": statistics.median(q_lens) if q_lens else 0,
        "query_char_p95": p95(q_lens),
        "positive_char_mean": round(statistics.mean(p_lens), 2) if p_lens else 0,
        "positive_char_median": statistics.median(p_lens) if p_lens else 0,
        "positive_char_p95": p95(p_lens),
        "duplicate_pairs": duplicate_count,
        "duplicate_pairs_pct": round(pct(duplicate_count, len(rows)), 2),
        "domains": dict(domains),
        "suspicious_questionmark_rows": suspicious_questionmark,
    }


def estimate_token_stats(
    rows: list[dict[str, object]],
    model_name: str,
    max_seq_length: int,
    sample_limit: int,
) -> dict[str, object]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing transformers dependency. Install requirements first:\n"
            f'  "{sys.executable}" -m pip install -r requirements.txt'
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    subset = rows[:sample_limit] if sample_limit > 0 else rows
    if not subset:
        return {
            "sampled_rows": 0,
            "query_token_mean": 0,
            "query_token_median": 0,
            "query_token_p95": 0,
            "positive_token_mean": 0,
            "positive_token_median": 0,
            "positive_token_p95": 0,
            "positive_truncated_count": 0,
            "positive_truncated_pct": 0,
            "model_name": model_name,
            "max_seq_length": max_seq_length,
        }

    q_tokens: list[int] = []
    p_tokens: list[int] = []
    truncated = 0
    for row in subset:
        q = str(row.get("query", ""))
        p = str(row.get("positive", ""))
        q_len = len(tokenizer.encode(q, add_special_tokens=True))
        p_len = len(tokenizer.encode(p, add_special_tokens=True))
        q_tokens.append(q_len)
        p_tokens.append(p_len)
        if p_len > max_seq_length:
            truncated += 1

    return {
        "sampled_rows": len(subset),
        "query_token_mean": round(statistics.mean(q_tokens), 2),
        "query_token_median": statistics.median(q_tokens),
        "query_token_p95": p95(q_tokens),
        "positive_token_mean": round(statistics.mean(p_tokens), 2),
        "positive_token_median": statistics.median(p_tokens),
        "positive_token_p95": p95(p_tokens),
        "positive_truncated_count": truncated,
        "positive_truncated_pct": round(pct(truncated, len(subset)), 2),
        "model_name": model_name,
        "max_seq_length": max_seq_length,
    }


def source_overlap(
    left_rows: Iterable[dict[str, object]],
    right_rows: Iterable[dict[str, object]],
    left_name: str,
    right_name: str,
) -> dict[str, object]:
    left_sources = {str(r.get("source_file", "")) for r in left_rows if str(r.get("source_file", "")).strip()}
    right_sources = {str(r.get("source_file", "")) for r in right_rows if str(r.get("source_file", "")).strip()}
    overlap = left_sources.intersection(right_sources)
    return {
        f"{left_name}_source_count": len(left_sources),
        f"{right_name}_source_count": len(right_sources),
        "overlap_source_count": len(overlap),
        "overlap_sources_preview": sorted(list(overlap))[:10],
    }


def main() -> None:
    setup_stdout_utf8()
    parser = argparse.ArgumentParser(description="Validate embedding fine-tune dataset quality.")
    parser.add_argument("--train-file", default=str(FINETUNE_DIR / "train_pairs.jsonl"))
    parser.add_argument("--valid-file", default=str(FINETUNE_DIR / "valid_pairs.jsonl"))
    parser.add_argument("--test-file", default=str(FINETUNE_DIR / "test_pairs.jsonl"))
    parser.add_argument("--model-name", default=DEFAULT_EMBEDDING_MODEL_FALLBACK)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--sample-limit", type=int, default=1500)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    if args.max_seq_length <= 0:
        raise ValueError("--max-seq-length must be > 0")

    train_rows = load_jsonl(Path(args.train_file).resolve())
    valid_rows = load_jsonl(Path(args.valid_file).resolve())
    test_path = Path(args.test_file).resolve()
    test_rows = load_jsonl(test_path) if test_path.exists() else []

    train_basic = calc_basic_stats(train_rows)
    valid_basic = calc_basic_stats(valid_rows)
    test_basic = calc_basic_stats(test_rows)
    overlap_train_valid = source_overlap(train_rows, valid_rows, "train", "valid")
    overlap_train_test = source_overlap(train_rows, test_rows, "train", "test")
    overlap_valid_test = source_overlap(valid_rows, test_rows, "valid", "test")

    train_token = estimate_token_stats(
        rows=train_rows,
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        sample_limit=args.sample_limit,
    )

    has_leakage = (
        overlap_train_valid["overlap_source_count"] > 0
        or overlap_train_test["overlap_source_count"] > 0
        or overlap_valid_test["overlap_source_count"] > 0
    )

    report = {
        "train_basic": train_basic,
        "valid_basic": valid_basic,
        "test_basic": test_basic,
        "train_token_estimate": train_token,
        "train_valid_source_overlap": overlap_train_valid,
        "train_test_source_overlap": overlap_train_test,
        "valid_test_source_overlap": overlap_valid_test,
        "recommendations": {
            "max_seq_length": args.max_seq_length,
            "truncate_warning": (
                "High truncation ratio detected."
                if train_token["positive_truncated_pct"] > 20
                else "Truncation ratio is acceptable."
            ),
            "leakage_warning": (
                "Source leakage detected across splits."
                if has_leakage
                else "No source-file leakage detected."
            ),
        },
    }

    if args.output_json.strip():
        out = Path(args.output_json).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved report: {out}")

    print("=== DATA VALIDATION REPORT ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
