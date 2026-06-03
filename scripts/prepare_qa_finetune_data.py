from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DATA_DIR, FINETUNE_DIR, ensure_directories  # noqa: E402


PREFIX_RE = re.compile(
    r"^\s*(?:(?:cau|câu|q|question)\s*)?\d{1,4}\s*[\.\):\-]+\s*",
    re.IGNORECASE,
)
ANSWER_HEADING_RE = re.compile(
    r"^\s*(?:ket luan|kết luận|tra loi|trả lời|dap an|đáp án)\s*:\s*",
    re.IGNORECASE,
)


def setup_stdout_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def clean_item(text: str) -> str:
    text = " ".join(text.replace("\ufeff", "").split())
    for _ in range(4):
        before = text
        text = PREFIX_RE.sub("", text).strip()
        text = ANSWER_HEADING_RE.sub("", text).strip()
        if text == before:
            break
    return text


def read_items(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    items: list[str] = []
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        item = clean_item(line)
        if item:
            items.append(item)
    return items


def infer_test_questions_path(test_dir: Path, explicit_path: str) -> Path:
    if explicit_path:
        return Path(explicit_path).resolve()
    corrected = test_dir / "questions.txt"
    legacy_typo = test_dir / "questons.txt"
    return corrected if corrected.exists() else legacy_typo


def make_pairs(
    questions: list[str],
    answers: list[str],
    split_name: str,
) -> list[dict[str, object]]:
    if len(questions) != len(answers):
        raise RuntimeError(
            f"{split_name}: questions/answers count mismatch: "
            f"{len(questions)} != {len(answers)}"
        )
    rows: list[dict[str, object]] = []
    for idx, (query, positive) in enumerate(zip(questions, answers), start=1):
        if not query or not positive:
            continue
        rows.append(
            {
                "query": query,
                "positive": positive,
                "domain": "mixed",
                "source_file": f"manual_qa/{split_name}_{idx:04d}.txt",
                "article_id": f"{split_name}_{idx:04d}",
            }
        )
    return rows


def make_triplets(rows: list[dict[str, object]], seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    positives = [str(row["positive"]) for row in rows]
    triplets: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        if len(positives) <= 1:
            negative = str(row["positive"])
        else:
            neg_idx = rng.randrange(len(positives) - 1)
            if neg_idx >= idx:
                neg_idx += 1
            negative = positives[neg_idx]
        triplet = dict(row)
        triplet["negative"] = negative
        triplets.append(triplet)
    return triplets


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_train_valid(
    rows: list[dict[str, object]],
    valid_ratio: float,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not rows:
        return [], []
    if valid_ratio <= 0:
        return rows, []
    if valid_ratio >= 0.5:
        raise ValueError("--valid-ratio must be < 0.5")
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    valid_count = max(1, int(round(len(shuffled) * valid_ratio)))
    valid_rows = shuffled[:valid_count]
    train_rows = shuffled[valid_count:]
    return train_rows, valid_rows


def prepare(
    train_questions_path: Path,
    train_answers_path: Path,
    test_questions_path: Path,
    test_answers_path: Path,
    output_dir: Path,
    valid_ratio: float,
    seed: int,
) -> dict[str, object]:
    train_questions = read_items(train_questions_path)
    train_answers = read_items(train_answers_path)
    test_questions = read_items(test_questions_path)
    test_answers = read_items(test_answers_path)

    manual_train_rows = make_pairs(train_questions, train_answers, "train")
    test_rows = make_pairs(test_questions, test_answers, "test")
    train_rows, valid_rows = split_train_valid(manual_train_rows, valid_ratio, seed)

    write_jsonl(output_dir / "train_pairs.jsonl", train_rows)
    write_jsonl(output_dir / "valid_pairs.jsonl", valid_rows)
    write_jsonl(output_dir / "test_pairs.jsonl", test_rows)
    write_jsonl(output_dir / "train_triplets.jsonl", make_triplets(train_rows, seed))
    write_jsonl(output_dir / "valid_triplets.jsonl", make_triplets(valid_rows, seed + 1))
    write_jsonl(output_dir / "test_triplets.jsonl", make_triplets(test_rows, seed + 2))

    summary = {
        "source": "manual_qa_txt",
        "train_questions_path": str(train_questions_path),
        "train_answers_path": str(train_answers_path),
        "test_questions_path": str(test_questions_path),
        "test_answers_path": str(test_answers_path),
        "manual_train_pairs_total": len(manual_train_rows),
        "train_pairs": len(train_rows),
        "valid_pairs": len(valid_rows),
        "test_pairs": len(test_rows),
        "valid_ratio": valid_ratio,
        "seed": seed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    setup_stdout_utf8()
    parser = argparse.ArgumentParser(
        description="Prepare fine-tune JSONL files from manual question/answer txt files."
    )
    parser.add_argument(
        "--train-questions",
        default=str(DATA_DIR / "train" / "questions.txt"),
    )
    parser.add_argument(
        "--train-answers",
        default=str(DATA_DIR / "train" / "reference_answers.txt"),
    )
    parser.add_argument("--test-questions", default="")
    parser.add_argument(
        "--test-answers",
        default=str(DATA_DIR / "test" / "reference_answers.txt"),
    )
    parser.add_argument("--output-dir", default=str(FINETUNE_DIR))
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_directories()
    test_questions_path = infer_test_questions_path(
        test_dir=DATA_DIR / "test",
        explicit_path=args.test_questions,
    )
    summary = prepare(
        train_questions_path=Path(args.train_questions).resolve(),
        train_answers_path=Path(args.train_answers).resolve(),
        test_questions_path=test_questions_path.resolve(),
        test_answers_path=Path(args.test_answers).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )

    print("=== PREPARE QA FINETUNE DATA SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
