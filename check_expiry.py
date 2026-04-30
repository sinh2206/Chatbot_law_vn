from __future__ import annotations

import logging

from admin_notify import AdminNotifier
from expiry_monitor import ExpiryMonitor
from legal_metadata import LegalMetadataRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("check_expiry")


def main() -> None:
    registry = LegalMetadataRegistry(autocreate_template=True)
    notifier = AdminNotifier()
    monitor = ExpiryMonitor(registry=registry, notifier=notifier)

    report = monitor.run_if_due(force=True)
    if report is None:
        logger.info("Expiry scan skipped")
        return

    logger.info(
        "Expiry scan completed: %d/%d documents expired",
        len(report.expired_documents),
        report.total_documents,
    )


if __name__ == "__main__":
    main()
