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

from config import FINETUNE_DIR, PROCESSED_DIR, ensure_directories  # noqa: E402


def setup_stdout_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    clean = "\n".join(lines)
    while "\n\n\n" in clean:
        clean = clean.replace("\n\n\n", "\n\n")
    return clean.strip()


def short_domain_label(domain: str) -> str:
    mapping = {
        "DoanhNghiep": "doanh nghiệp",
        "HoTich": "hộ tịch",
        "CCCD": "căn cước công dân",
        "DatDai": "đất đai",
        "Thue": "thuế",
    }
    return mapping.get(domain, domain.lower())


def split_text_by_articles(text: str) -> list[dict[str, str]]:
    pattern = re.compile(
        r"(?im)^\s*(Điều\s+\d+[A-Za-z0-9/-]*)(?:\s*[.:-]\s*|\s+)(.*)\s*$"
    )
    matches = list(pattern.finditer(text))
    results: list[dict[str, str]] = []

    if not matches:
        return results

    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = normalize_text(text[start:end])
        title = normalize_text(match.group(2))
        article = normalize_text(match.group(1))
        if not body:
            continue
        results.append(
            {
                "article": article,
                "title": title,
                "body": body,
            }
        )
    return results


def split_fallback_chunks(text: str, chunk_size: int = 1400, overlap: int = 180) -> list[str]:
    clean = normalize_text(text)
    if not clean:
        return []
    if len(clean) <= chunk_size:
        return [clean]

    chunks: list[str] = []
    start = 0
    total = len(clean)
    min_boundary = int(chunk_size * 0.6)

    while start < total:
        end = min(start + chunk_size, total)
        if end < total:
            boundary = clean.rfind("\n", start + min_boundary, end)
            if boundary == -1:
                boundary = clean.rfind(" ", start + min_boundary, end)
            if boundary > start:
                end = boundary
        piece = clean[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= total:
            break
        start = max(start + 1, end - overlap)
    return chunks


def build_query_variants(article: str, title: str, domain: str, max_queries: int) -> list[str]:
    domain_label = short_domain_label(domain)
    title_clean = title.strip(" .:-")
    queries = [
        f"{article} {title_clean}".strip(),
        f"Quy định về {title_clean} là gì?",
        f"{title_clean} được quy định như thế nào?",
        f"Theo pháp luật {domain_label}, {title_clean} gồm những nội dung gì?",
        f"Căn cứ {article}, nội dung chính của {title_clean} là gì?",
    ]

    deduped: list[str] = []
    seen = set()
    for query in queries:
        key = query.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(query)
        if len(deduped) >= max_queries:
            break
    return deduped


def write_jsonl(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def choose_negative(
    candidates: list[dict[str, object]],
    idx: int,
    by_domain: dict[str, list[int]],
    rng: random.Random,
) -> str:
    anchor = candidates[idx]
    domain = str(anchor["domain"])
    source_file = str(anchor["source_file"])

    cross_domain_indices = []
    for domain_name, indices in by_domain.items():
        if domain_name != domain:
            cross_domain_indices.extend(indices)

    if cross_domain_indices:
        chosen_idx = rng.choice(cross_domain_indices)
        return str(candidates[chosen_idx]["positive"])

    fallback_indices = [
        i for i, row in enumerate(candidates) if i != idx and str(row["source_file"]) != source_file
    ]
    if fallback_indices:
        chosen_idx = rng.choice(fallback_indices)
        return str(candidates[chosen_idx]["positive"])

    return str(anchor["positive"])


def build_dataset(
    processed_dir: Path,
    output_dir: Path,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
    max_queries_per_article: int,
    min_positive_chars: int,
    max_positive_chars: int,
    max_records: int,
    domains: set[str] | None,
) -> dict[str, int | str]:
    rng = random.Random(seed)

    source_files: list[Path] = []
    for path in processed_dir.rglob("*.txt"):
        if not path.is_file():
            continue
        rel = path.relative_to(processed_dir)
        if domains and (not rel.parts or rel.parts[0] not in domains):
            continue
        source_files.append(path)
    source_files.sort()

    if not source_files:
        raise RuntimeError(f"No .txt files found in {processed_dir}")

    candidates: list[dict[str, object]] = []

    for file_path in source_files:
        rel = file_path.relative_to(processed_dir)
        source_file = str(rel).replace("\\", "/")
        domain = rel.parts[0] if rel.parts else "Unknown"
        raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
        text = normalize_text(raw_text)
        if not text:
            continue

        articles = split_text_by_articles(text)
        if articles:
            for article_idx, article in enumerate(articles):
                body = article["body"][:max_positive_chars].strip()
                if len(body) < min_positive_chars:
                    continue
                title = article["title"] or f"Nội dung {article['article']}"
                queries = build_query_variants(
                    article=article["article"],
                    title=title,
                    domain=domain,
                    max_queries=max_queries_per_article,
                )
                for query in queries:
                    candidates.append(
                        {
                            "query": query,
                            "positive": body,
                            "domain": domain,
                            "source_file": source_file,
                            "article_id": f"{article['article']}#{article_idx}",
                        }
                    )
                    if len(candidates) >= max_records:
                        break
                if len(candidates) >= max_records:
                    break
        else:
            chunks = split_fallback_chunks(text=text, chunk_size=max_positive_chars)
            for chunk_idx, chunk in enumerate(chunks):
                chunk = chunk.strip()
                if len(chunk) < min_positive_chars:
                    continue
                query = f"Nội dung chính của {source_file} (đoạn {chunk_idx + 1}) là gì?"
                candidates.append(
                    {
                        "query": query,
                        "positive": chunk[:max_positive_chars],
                        "domain": domain,
                        "source_file": source_file,
                        "article_id": f"chunk#{chunk_idx}",
                    }
                )
                if len(candidates) >= max_records:
                    break
        if len(candidates) >= max_records:
            break

    if len(candidates) < 20:
        raise RuntimeError(
            "Generated too few training candidates (<20). Check OCR/text normalization quality first."
        )

    by_domain: dict[str, list[int]] = {}
    for idx, row in enumerate(candidates):
        by_domain.setdefault(str(row["domain"]), []).append(idx)

    triplets: list[dict[str, object]] = []
    for idx, row in enumerate(candidates):
        negative = choose_negative(candidates, idx, by_domain, rng)
        triplets.append(
            {
                "query": row["query"],
                "positive": row["positive"],
                "negative": negative,
                "domain": row["domain"],
                "source_file": row["source_file"],
                "article_id": row["article_id"],
            }
        )

    file_keys = sorted({str(item["source_file"]) for item in triplets})
    domain_to_files: dict[str, list[str]] = {}
    for row in triplets:
        domain_to_files.setdefault(str(row["domain"]), [])
        source_file = str(row["source_file"])
        if source_file not in domain_to_files[str(row["domain"])]:
            domain_to_files[str(row["domain"])].append(source_file)

    valid_file_set: set[str] = set()
    test_file_set: set[str] = set()
    for domain, files in sorted(domain_to_files.items()):
        local_files = files[:]
        rng.shuffle(local_files)
        if len(local_files) <= 1:
            # Avoid emptying a domain from train split when only 1 source file exists.
            continue

        n_files = len(local_files)
        valid_count = max(1, int(n_files * valid_ratio))
        test_count = int(n_files * test_ratio)
        if test_ratio > 0 and test_count == 0 and n_files >= 3:
            test_count = 1

        max_non_train = n_files - 1
        if valid_count + test_count > max_non_train:
            overflow = valid_count + test_count - max_non_train
            # Reduce test first, then valid.
            reduce_test = min(test_count, overflow)
            test_count -= reduce_test
            overflow -= reduce_test
            if overflow > 0:
                valid_count = max(1, valid_count - overflow)

        valid_file_set.update(local_files[:valid_count])
        if test_count > 0:
            start = valid_count
            end = valid_count + test_count
            test_file_set.update(local_files[start:end])

    train_pairs: list[dict[str, object]] = []
    valid_pairs: list[dict[str, object]] = []
    test_pairs: list[dict[str, object]] = []
    train_triplets: list[dict[str, object]] = []
    valid_triplets: list[dict[str, object]] = []
    test_triplets: list[dict[str, object]] = []

    for row in triplets:
        pair = {
            "query": row["query"],
            "positive": row["positive"],
            "domain": row["domain"],
            "source_file": row["source_file"],
            "article_id": row["article_id"],
        }
        source_file = str(row["source_file"])
        if source_file in test_file_set:
            test_pairs.append(pair)
            test_triplets.append(row)
        elif source_file in valid_file_set:
            valid_pairs.append(pair)
            valid_triplets.append(row)
        else:
            train_pairs.append(pair)
            train_triplets.append(row)

    write_jsonl(train_pairs, output_dir / "train_pairs.jsonl")
    write_jsonl(valid_pairs, output_dir / "valid_pairs.jsonl")
    write_jsonl(test_pairs, output_dir / "test_pairs.jsonl")
    write_jsonl(train_triplets, output_dir / "train_triplets.jsonl")
    write_jsonl(valid_triplets, output_dir / "valid_triplets.jsonl")
    write_jsonl(test_triplets, output_dir / "test_triplets.jsonl")

    summary = {
        "processed_dir": str(processed_dir),
        "output_dir": str(output_dir),
        "domains": sorted({str(row["domain"]) for row in triplets}),
        "source_files": len(file_keys),
        "records_total": len(triplets),
        "train_pairs": len(train_pairs),
        "valid_pairs": len(valid_pairs),
        "test_pairs": len(test_pairs),
        "train_triplets": len(train_triplets),
        "valid_triplets": len(valid_triplets),
        "test_triplets": len(test_triplets),
        "train_source_files": len(
            {str(item["source_file"]) for item in train_triplets}
        ),
        "valid_source_files": len(valid_file_set),
        "test_source_files": len(test_file_set),
        "seed": seed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    setup_stdout_utf8()
    parser = argparse.ArgumentParser(
        description="Bootstrap embedding fine-tune dataset from local legal text corpus."
    )
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--output-dir", default=str(FINETUNE_DIR))
    parser.add_argument("--domain", action="append", default=[])
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-queries-per-article", type=int, default=3)
    parser.add_argument("--min-positive-chars", type=int, default=120)
    parser.add_argument("--max-positive-chars", type=int, default=1400)
    parser.add_argument("--max-records", type=int, default=30000)
    args = parser.parse_args()

    if args.valid_ratio <= 0 or args.valid_ratio >= 0.9:
        raise ValueError("--valid-ratio should be in (0, 0.9)")
    if args.test_ratio < 0 or args.test_ratio >= 0.8:
        raise ValueError("--test-ratio should be in [0, 0.8)")
    if args.valid_ratio + args.test_ratio >= 0.9:
        raise ValueError("--valid-ratio + --test-ratio should be < 0.9")
    if args.max_queries_per_article <= 0:
        raise ValueError("--max-queries-per-article must be > 0")
    if args.min_positive_chars < 20:
        raise ValueError("--min-positive-chars should be >= 20")
    if args.max_positive_chars <= args.min_positive_chars:
        raise ValueError("--max-positive-chars must be > min-positive-chars")
    if args.max_records <= 0:
        raise ValueError("--max-records must be > 0")

    ensure_directories()
    processed_dir = Path(args.processed_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    domains = set(args.domain) if args.domain else None

    summary = build_dataset(
        processed_dir=processed_dir,
        output_dir=output_dir,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_queries_per_article=args.max_queries_per_article,
        min_positive_chars=args.min_positive_chars,
        max_positive_chars=args.max_positive_chars,
        max_records=args.max_records,
        domains=domains,
    )

    print("=== BOOTSTRAP FINETUNE DATA SUMMARY ===")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
