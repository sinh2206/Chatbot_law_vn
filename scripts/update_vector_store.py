from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    PROCESSED_DIR,
    RAW_DOCS_DIR,
)
from scripts.build_vector_store import build_vector_store  # noqa: E402
from scripts.convert_docs_to_txt import convert_all  # noqa: E402


def setup_stdout_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main() -> None:
    setup_stdout_utf8()
    parser = argparse.ArgumentParser(
        description="Update vector store after adding/removing legal documents."
    )
    parser.add_argument("--source-dir", default=str(RAW_DOCS_DIR))
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--domain", action="append", default=[])
    parser.add_argument(
        "--scope",
        choices=["full", "domain"],
        default="full",
        help="full: rebuild index from all processed files; domain: index only selected domain(s)",
    )
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=EMBEDDING_BATCH_SIZE)
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    processed_dir = Path(args.processed_dir).resolve()
    domains = set(args.domain) if args.domain else None

    if not args.skip_convert:
        converted, skipped = convert_all(
            source_dir=source_dir,
            output_dir=processed_dir,
            domains=domains,
            overwrite=args.overwrite,
            clean_output=args.clean_output,
        )
        print(f"[CONVERT] converted={converted}, skipped={skipped}")

    index_domains = domains if args.scope == "domain" else None
    result = build_vector_store(
        processed_dir=processed_dir,
        domains=index_domains,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
    )

    print("\n=== UPDATE SUMMARY ===")
    print(f"scope: {args.scope}")
    print(f"domains: {sorted(domains) if domains else 'all'}")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
