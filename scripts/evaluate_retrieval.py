from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

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


def make_doc_id(source_file: str, article_id: str, positive: str) -> str:
    h = hashlib.sha1(positive.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{source_file}::{article_id}::{h}"


def build_retrieval_data(
    test_rows: list[dict[str, object]],
    extra_rows: list[dict[str, object]],
    max_extra_docs: int,
) -> tuple[list[str], list[str], list[str], list[str], dict[str, set[str]]]:
    corpus_texts: list[str] = []
    corpus_ids: list[str] = []
    corpus_seen: set[str] = set()

    queries: list[str] = []
    query_ids: list[str] = []
    relevant_docs: dict[str, set[str]] = {}

    for idx, row in enumerate(test_rows):
        query = str(row.get("query", "")).strip()
        positive = str(row.get("positive", "")).strip()
        source_file = str(row.get("source_file", ""))
        article_id = str(row.get("article_id", ""))
        if not query or not positive:
            continue

        doc_id = make_doc_id(source_file=source_file, article_id=article_id, positive=positive)
        if doc_id not in corpus_seen:
            corpus_seen.add(doc_id)
            corpus_ids.append(doc_id)
            corpus_texts.append(positive)

        qid = f"q{idx}"
        query_ids.append(qid)
        queries.append(query)
        relevant_docs[qid] = {doc_id}

    added_extra = 0
    for row in extra_rows:
        if max_extra_docs > 0 and added_extra >= max_extra_docs:
            break
        positive = str(row.get("positive", "")).strip()
        source_file = str(row.get("source_file", ""))
        article_id = str(row.get("article_id", ""))
        if not positive:
            continue
        doc_id = make_doc_id(source_file=source_file, article_id=article_id, positive=positive)
        if doc_id in corpus_seen:
            continue
        corpus_seen.add(doc_id)
        corpus_ids.append(doc_id)
        corpus_texts.append(positive)
        added_extra += 1

    return query_ids, queries, corpus_ids, corpus_texts, relevant_docs


def find_rank_of_relevant(sorted_doc_ids: list[str], relevant: set[str]) -> int | None:
    for rank, doc_id in enumerate(sorted_doc_ids, start=1):
        if doc_id in relevant:
            return rank
    return None


def evaluate_ranking(
    query_ids: list[str],
    query_embeddings: np.ndarray,
    corpus_ids: list[str],
    corpus_embeddings: np.ndarray,
    relevant_docs: dict[str, set[str]],
    ks: list[int],
) -> dict[str, object]:
    if query_embeddings.ndim != 2 or corpus_embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D arrays.")
    if query_embeddings.shape[1] != corpus_embeddings.shape[1]:
        raise ValueError("Query and corpus embeddings dimensions must match.")

    scores = np.dot(query_embeddings, corpus_embeddings.T)
    k_max = max(ks + [10])

    recalls = {k: 0 for k in ks}
    mrr_sum = 0.0
    ndcg_sum = 0.0
    query_count = len(query_ids)

    for i, qid in enumerate(query_ids):
        row_scores = scores[i]
        top_indices = np.argsort(-row_scores)[:k_max]
        ranked_doc_ids = [corpus_ids[idx] for idx in top_indices]
        relevant = relevant_docs[qid]
        rank = find_rank_of_relevant(ranked_doc_ids, relevant)

        for k in ks:
            if rank is not None and rank <= k:
                recalls[k] += 1

        if rank is not None and rank <= 10:
            mrr_sum += 1.0 / rank
            ndcg_sum += 1.0 / np.log2(rank + 1)

    metrics: dict[str, object] = {
        "query_count": query_count,
        "corpus_count": len(corpus_ids),
        "embedding_dim": int(query_embeddings.shape[1]),
        "mrr_at_10": (mrr_sum / query_count) if query_count else 0.0,
        "ndcg_at_10": (ndcg_sum / query_count) if query_count else 0.0,
    }
    for k in ks:
        metrics[f"recall_at_{k}"] = (recalls[k] / query_count) if query_count else 0.0
    return metrics


def main() -> None:
    setup_stdout_utf8()
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality on test split using dense embeddings."
    )
    parser.add_argument("--model-name", default=DEFAULT_EMBEDDING_MODEL_FALLBACK)
    parser.add_argument("--test-file", default=str(FINETUNE_DIR / "test_pairs.jsonl"))
    parser.add_argument("--train-file", default=str(FINETUNE_DIR / "train_pairs.jsonl"))
    parser.add_argument("--valid-file", default=str(FINETUNE_DIR / "valid_pairs.jsonl"))
    parser.add_argument("--max-extra-docs", type=int, default=4000)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--corpus-batch-size", type=int, default=64)
    parser.add_argument("--k-values", default="1,3,5,10")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    if args.query_batch_size <= 0 or args.corpus_batch_size <= 0:
        raise ValueError("Batch sizes must be > 0")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Missing sentence-transformers dependency. Install requirements first:\n"
            f'  "{sys.executable}" -m pip install -r requirements.txt'
        ) from exc

    test_rows = load_jsonl(Path(args.test_file).resolve())
    train_rows = load_jsonl(Path(args.train_file).resolve()) if Path(args.train_file).exists() else []
    valid_rows = load_jsonl(Path(args.valid_file).resolve()) if Path(args.valid_file).exists() else []

    if not test_rows:
        raise RuntimeError("Test split is empty. Re-run bootstrap with non-zero --test-ratio.")

    query_ids, queries, corpus_ids, corpus_texts, relevant_docs = build_retrieval_data(
        test_rows=test_rows,
        extra_rows=train_rows + valid_rows,
        max_extra_docs=max(0, args.max_extra_docs),
    )

    if not queries or not corpus_texts:
        raise RuntimeError("Failed to build retrieval benchmark data.")

    model = SentenceTransformer(args.model_name)
    tokenizer_limit = getattr(getattr(model, "tokenizer", None), "model_max_length", None)
    config_limit = getattr(getattr(model[0], "auto_model", None), "config", None)
    if config_limit is not None:
        config_limit = getattr(config_limit, "max_position_embeddings", None)
    limits: list[int] = []
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 100000:
        limits.append(int(tokenizer_limit))
    if isinstance(config_limit, int) and 0 < config_limit < 100000:
        limits.append(max(8, int(config_limit) - 2))
    if limits:
        safe_max_seq_length = min(limits)
        if model.max_seq_length != safe_max_seq_length:
            print(
                f"[WARN] Adjusting model.max_seq_length from {model.max_seq_length} "
                f"to safe limit {safe_max_seq_length} for model: {args.model_name}"
            )
            model.max_seq_length = safe_max_seq_length

    query_vec = model.encode(
        queries,
        batch_size=args.query_batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    corpus_vec = model.encode(
        corpus_texts,
        batch_size=args.corpus_batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    ks = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    metrics = evaluate_ranking(
        query_ids=query_ids,
        query_embeddings=np.asarray(query_vec, dtype=np.float32),
        corpus_ids=corpus_ids,
        corpus_embeddings=np.asarray(corpus_vec, dtype=np.float32),
        relevant_docs=relevant_docs,
        ks=ks,
    )
    metrics["model_name"] = args.model_name
    metrics["test_file"] = str(Path(args.test_file).resolve())

    if args.output_json.strip():
        out_path = Path(args.output_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved metrics: {out_path}")

    print("=== RETRIEVAL EVAL ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
