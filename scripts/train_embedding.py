from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL_FALLBACK,
    FINETUNE_DIR,
    MODELS_DIR,
    ensure_directories,
)


def setup_stdout_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def sample_rows(rows: list[dict[str, object]], max_samples: int, seed: int) -> list[dict[str, object]]:
    if max_samples <= 0 or len(rows) <= max_samples:
        return rows
    rng = random.Random(seed)
    cloned = rows[:]
    rng.shuffle(cloned)
    return cloned[:max_samples]


def build_eval_binary(rows: list[dict[str, object]], seed: int) -> tuple[list[str], list[str], list[int]]:
    if len(rows) < 2:
        return [], [], []
    rng = random.Random(seed)
    positives = rows[:]
    rng.shuffle(positives)

    sent1: list[str] = []
    sent2: list[str] = []
    labels: list[int] = []

    for row in positives:
        q = str(row.get("query", "")).strip()
        p = str(row.get("positive", "")).strip()
        if not q or not p:
            continue
        sent1.append(q)
        sent2.append(p)
        labels.append(1)

    shuffled = positives[:]
    rng.shuffle(shuffled)
    for row, neg_row in zip(positives, shuffled):
        q = str(row.get("query", "")).strip()
        p_neg = str(neg_row.get("positive", "")).strip()
        if not q or not p_neg:
            continue
        sent1.append(q)
        sent2.append(p_neg)
        labels.append(0)
    return sent1, sent2, labels


def train(
    model_name: str,
    train_rows: list[dict[str, object]],
    valid_rows: list[dict[str, object]],
    output_dir: Path,
    batch_size: int,
    epochs: int,
    lr: float,
    warmup_ratio: float,
    max_seq_length: int,
    use_amp: bool,
    seed: int,
) -> dict[str, object]:
    try:
        import torch
        from sentence_transformers import InputExample, SentenceTransformer
        from sentence_transformers.sentence_transformer.evaluation import (
            BinaryClassificationEvaluator,
        )
        from sentence_transformers.sentence_transformer.losses import (
            MultipleNegativesRankingLoss,
        )
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies for training. Install requirements first:\n"
            f'  "{sys.executable}" -m pip install -r requirements.txt'
        ) from exc

    if not train_rows:
        raise RuntimeError("No training rows found.")

    model = SentenceTransformer(model_name)

    tokenizer_limit = getattr(getattr(model, "tokenizer", None), "model_max_length", None)
    config_limit = getattr(getattr(model[0], "auto_model", None), "config", None)
    if config_limit is not None:
        config_limit = getattr(config_limit, "max_position_embeddings", None)

    limits: list[int] = []
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 100000:
        limits.append(int(tokenizer_limit))
    if isinstance(config_limit, int) and 0 < config_limit < 100000:
        # Reserve slots for special tokens.
        limits.append(max(8, int(config_limit) - 2))

    hard_limit = min(limits) if limits else None
    final_max_seq_length = min(max_seq_length, hard_limit) if hard_limit else max_seq_length
    if final_max_seq_length != max_seq_length:
        print(
            f"[WARN] max_seq_length={max_seq_length} is larger than model limit={hard_limit}. "
            f"Using {final_max_seq_length}."
        )
    model.max_seq_length = final_max_seq_length

    train_examples: list[InputExample] = []
    for row in train_rows:
        query = str(row.get("query", "")).strip()
        positive = str(row.get("positive", "")).strip()
        if not query or not positive:
            continue
        train_examples.append(InputExample(texts=[query, positive]))
    if not train_examples:
        raise RuntimeError("Training rows are invalid after filtering empty query/positive.")

    train_loader = DataLoader(train_examples, shuffle=True, batch_size=batch_size, drop_last=False)
    train_loss = MultipleNegativesRankingLoss(model=model)
    warmup_steps = int(len(train_loader) * max(1, epochs) * warmup_ratio)

    evaluator = None
    eval_steps = 0
    s1, s2, labels = build_eval_binary(valid_rows, seed=seed)
    if s1:
        evaluator = BinaryClassificationEvaluator(
            sentences1=s1,
            sentences2=s2,
            labels=labels,
            name="valid_binary",
        )
        eval_steps = max(50, len(train_loader) // 2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        output_path=str(output_dir),
        use_amp=use_amp and device == "cuda",
        evaluation_steps=eval_steps,
        show_progress_bar=True,
    )
    model.save(str(output_dir))

    return {
        "base_model": model_name,
        "output_model": str(output_dir),
        "device": device,
        "train_examples": len(train_examples),
        "valid_examples": len(valid_rows),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "warmup_steps": warmup_steps,
        "max_seq_length": final_max_seq_length,
    }


def main() -> None:
    setup_stdout_utf8()
    parser = argparse.ArgumentParser(
        description="Fine-tune Vietnamese embedding model for legal retrieval."
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_EMBEDDING_MODEL_FALLBACK,
        help="Base embedding model (HF repo id or local path).",
    )
    parser.add_argument("--train-file", default=str(FINETUNE_DIR / "train_pairs.jsonl"))
    parser.add_argument("--valid-file", default=str(FINETUNE_DIR / "valid_pairs.jsonl"))
    parser.add_argument(
        "--output-dir",
        default=str(MODELS_DIR / "vietnamese-embedding-legal"),
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-valid-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-amp", action="store_true")
    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.batch_size <= 1:
        raise ValueError("--batch-size must be > 1")
    if args.lr <= 0:
        raise ValueError("--lr must be > 0")
    if args.warmup_ratio < 0 or args.warmup_ratio >= 1:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if args.max_seq_length < 64:
        raise ValueError("--max-seq-length must be >= 64")

    ensure_directories()
    train_file = Path(args.train_file).resolve()
    valid_file = Path(args.valid_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(train_file)
    valid_rows = load_jsonl(valid_file)
    train_rows = sample_rows(train_rows, args.max_train_samples, seed=args.seed)
    valid_rows = sample_rows(valid_rows, args.max_valid_samples, seed=args.seed + 1)

    summary = train(
        model_name=args.model_name,
        train_rows=train_rows,
        valid_rows=valid_rows,
        output_dir=output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        max_seq_length=args.max_seq_length,
        use_amp=args.use_amp,
        seed=args.seed,
    )

    summary_path = output_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== FINETUNE SUMMARY ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"summary_file: {summary_path}")


if __name__ == "__main__":
    main()
