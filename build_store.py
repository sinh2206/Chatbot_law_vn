from __future__ import annotations

import argparse
import logging

from embedding_store import create_vector_store
from legal_metadata import LegalMetadataRegistry
from load_and_chunk import load_and_chunk_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("build_store")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build legal vector store from local documents.")
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Do not clear existing collection before indexing.",
    )
    parser.add_argument(
        "--bootstrap-metadata",
        action="store_true",
        help="Generate metadata/legal_documents_metadata.csv template and exit.",
    )
    args = parser.parse_args()

    metadata_registry = LegalMetadataRegistry(autocreate_template=True)

    if args.bootstrap_metadata:
        output = metadata_registry.bootstrap_template_from_data(overwrite=True)
        logger.info("Generated metadata template: %s", output)
        return

    chunks = load_and_chunk_documents(metadata_registry=metadata_registry)
    if not chunks:
        logger.warning("No chunks were created. Check DATA_DIR and document reader dependencies.")
        return

    vector_store = create_vector_store()
    if not args.no_reset:
        logger.info("Reset existing collection before indexing...")
        vector_store.reset_collection()

    inserted = vector_store.upsert_chunks(chunks)
    logger.info("Indexed %d chunks", inserted)
    logger.info("Collection now has %d vectors", vector_store.count())

    report = metadata_registry.scan_expired_documents()
    logger.info(
        "Legal metadata scan: %d/%d documents are marked expired",
        len(report.expired_documents),
        report.total_documents,
    )


if __name__ == "__main__":
    main()
