from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

from config import TEST_DATASET_FILE
from testsuite.default_cases import default_test_cases


def difficulty_bucket(level: int) -> str:
    if level == 1:
        return "easy"
    if level == 2:
        return "medium"
    return "hard"


def primary_domain(domain_field: str) -> str:
    return domain_field.split(",")[0].strip()


def build_balanced_subset(
    cases: list[dict[str, object]],
    target_size: int,
    seed: int,
) -> list[dict[str, object]]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate test suite JSON file.")
    parser.add_argument(
        "--target-size",
        type=int,
        default=45,
        help="Target number of test cases (recommended 30-50). Use --all for full set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subset generation.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export full test set instead of balanced subset.",
    )
    args = parser.parse_args()

    TEST_DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)

    full_cases = default_test_cases()
    payload = (
        full_cases
        if args.all
        else build_balanced_subset(full_cases, target_size=args.target_size, seed=args.seed)
    )

    with TEST_DATASET_FILE.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"Generated {len(payload)} test cases -> {TEST_DATASET_FILE.name}")


if __name__ == "__main__":
    main()
