from __future__ import annotations

from datetime import datetime, timedelta

from admin_notify import AdminNotifier
from config import DOMAIN_LABELS, EXPIRY_SCAN_INTERVAL_HOURS
from legal_metadata import ExpiryScanResult, LegalMetadataRegistry


class ExpiryMonitor:
    def __init__(
        self,
        registry: LegalMetadataRegistry,
        notifier: AdminNotifier,
        interval_hours: int = EXPIRY_SCAN_INTERVAL_HOURS,
    ) -> None:
        self.registry = registry
        self.notifier = notifier
        self.interval = timedelta(hours=max(1, interval_hours))
        self.last_scan_at: datetime | None = None

    def run_if_due(self, force: bool = False) -> ExpiryScanResult | None:
        now = datetime.now()
        if not force and self.last_scan_at and now - self.last_scan_at < self.interval:
            return None

        self.registry.refresh()
        report = self.registry.scan_expired_documents(today=now.date())
        self.last_scan_at = now

        for record in report.expired_documents:
            domain_label = DOMAIN_LABELS.get(record.domain, record.domain)
            self.notifier.notify_expired_document(record=record, domain_label=domain_label)

        return report
