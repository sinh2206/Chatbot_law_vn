from __future__ import annotations

import argparse
import logging

from config import DATA_DIR, DOMAIN_LABELS
from embedding_store import create_vector_store
from legal_metadata import LegalMetadataRegistry
from load_and_chunk import load_and_chunk_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("update_store")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update vector store after adding/replacing legal documents."
    )
    parser.add_argument(
        "--domain",
        type=str,
        help=f"Update only one domain: {', '.join(DOMAIN_LABELS.keys())}",
    )
    args = parser.parse_args()

    registry = LegalMetadataRegistry(autocreate_template=True)
    vector_store = create_vector_store()

    if args.domain:
        domain = args.domain.strip()
        if domain not in DOMAIN_LABELS:
            raise ValueError(f"Invalid domain: {domain}")

        domain_path = DATA_DIR / domain
        chunks = load_and_chunk_documents(root_dir=domain_path, metadata_registry=registry)
        removed = vector_store.delete_by_domain(domain)
        inserted = vector_store.upsert_chunks(chunks)

        logger.info("Domain update completed for %s", domain)
        logger.info("Removed %d old chunks", removed)
        logger.info("Inserted %d new chunks", inserted)
        logger.info("Collection count: %d", vector_store.count())
        return

    chunks = load_and_chunk_documents(metadata_registry=registry)
    vector_store.reset_collection()
    inserted = vector_store.upsert_chunks(chunks)

    logger.info("Full rebuild completed")
    logger.info("Inserted %d chunks", inserted)
    logger.info("Collection count: %d", vector_store.count())


if __name__ == "__main__":
    main()
