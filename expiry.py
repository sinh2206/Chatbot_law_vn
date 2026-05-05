from __future__ import annotations

import argparse
import csv
import logging
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Protocol

from config import (
    ALLOWED_EXTENSIONS,
    DATA_DIR,
    DOMAIN_LABELS,
    EXPIRY_SCAN_INTERVAL_HOURS,
    LEGAL_METADATA_FILE,
)

logger = logging.getLogger(__name__)
DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y")


class ExpiryNotifier(Protocol):
    def notify_expired_document(self, record: "LegalDocumentMetadata", domain_label: str) -> None:
        ...


def parse_date(raw: str) -> date | None:
    value = raw.strip()
    if not value:
        return None

    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def to_iso_or_empty(value: date | None) -> str:
    return value.isoformat() if value else ""


@dataclass(frozen=True)
class LegalDocumentMetadata:
    domain: str
    file_name: str
    document_number: str
    issue_date: date | None
    effective_date: date | None
    expiry_date: date | None
    status_override: str
    replacement_document: str
    source_url: str
    notes: str

    def is_expired(self, today: date | None = None) -> bool:
        check_day = today or date.today()
        override = self.status_override.strip().lower()

        if override == "expired":
            return True
        if override == "active":
            return False
        if self.expiry_date and check_day > self.expiry_date:
            return True

        return False


@dataclass(frozen=True)
class ExpiryScanResult:
    checked_at: date
    total_documents: int
    expired_documents: list[LegalDocumentMetadata]


class LegalMetadataRegistry:
    def __init__(
        self,
        metadata_file: Path = LEGAL_METADATA_FILE,
        data_dir: Path = DATA_DIR,
        autocreate_template: bool = True,
    ) -> None:
        self.metadata_file = Path(metadata_file)
        self.data_dir = Path(data_dir)

        if autocreate_template and not self.metadata_file.exists():
            self.bootstrap_template_from_data(overwrite=False)

        self._records: list[LegalDocumentMetadata] = []
        self._by_file: dict[tuple[str, str], LegalDocumentMetadata] = {}
        self._by_number: dict[tuple[str, str], LegalDocumentMetadata] = {}
        self.refresh()

    def refresh(self) -> None:
        self._records = []
        self._by_file = {}
        self._by_number = {}

        if not self.metadata_file.exists():
            return

        with self.metadata_file.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                record = LegalDocumentMetadata(
                    domain=(row.get("domain") or "").strip(),
                    file_name=(row.get("file_name") or "").strip(),
                    document_number=(row.get("document_number") or "").strip(),
                    issue_date=parse_date(row.get("issue_date") or ""),
                    effective_date=parse_date(row.get("effective_date") or ""),
                    expiry_date=parse_date(row.get("expiry_date") or ""),
                    status_override=(row.get("status_override") or "").strip(),
                    replacement_document=(row.get("replacement_document") or "").strip(),
                    source_url=(row.get("source_url") or "").strip(),
                    notes=(row.get("notes") or "").strip(),
                )

                self._records.append(record)

                if record.domain and record.file_name:
                    file_key = (record.domain.lower(), record.file_name.lower())
                    self._by_file[file_key] = record

                if record.domain and record.document_number:
                    number_key = (record.domain.lower(), record.document_number.upper())
                    self._by_number[number_key] = record

    def find(self, domain: str, file_name: str, document_number: str = "") -> LegalDocumentMetadata | None:
        key_by_file = (domain.lower(), file_name.lower())
        if key_by_file in self._by_file:
            return self._by_file[key_by_file]

        if document_number:
            key_by_number = (domain.lower(), document_number.upper())
            return self._by_number.get(key_by_number)

        return None

    def enrich_chunk_metadata(self, metadata: dict[str, str]) -> dict[str, str]:
        domain = metadata.get("domain", "")
        file_name = metadata.get("file_name", "")
        document_number = metadata.get("document_number", "")

        record = self.find(domain=domain, file_name=file_name, document_number=document_number)
        if not record:
            return {
                **metadata,
                "issue_date": "",
                "effective_date": "",
                "expiry_date": "",
                "legal_status": "unknown",
                "replacement_document": "",
                "source_url": "",
            }

        status = "expired" if record.is_expired() else "active"
        return {
            **metadata,
            "issue_date": to_iso_or_empty(record.issue_date),
            "effective_date": to_iso_or_empty(record.effective_date),
            "expiry_date": to_iso_or_empty(record.expiry_date),
            "legal_status": status,
            "replacement_document": record.replacement_document,
            "source_url": record.source_url,
        }

    def get_status_from_chunk_metadata(self, metadata: dict[str, str]) -> str:
        legal_status = (metadata.get("legal_status") or "").strip().lower()
        if legal_status in {"active", "expired"}:
            expiry_date = parse_date(metadata.get("expiry_date", ""))
            if expiry_date and date.today() > expiry_date:
                return "expired"
            return legal_status

        domain = metadata.get("domain", "")
        file_name = metadata.get("file_name", "")
        document_number = metadata.get("document_number", "")
        record = self.find(domain=domain, file_name=file_name, document_number=document_number)
        if not record:
            return "unknown"

        return "expired" if record.is_expired() else "active"

    def scan_expired_documents(self, today: date | None = None) -> ExpiryScanResult:
        check_day = today or date.today()
        expired = [record for record in self._records if record.is_expired(check_day)]
        return ExpiryScanResult(
            checked_at=check_day,
            total_documents=len(self._records),
            expired_documents=expired,
        )

    def bootstrap_template_from_data(self, overwrite: bool = False) -> Path:
        if self.metadata_file.exists() and not overwrite:
            return self.metadata_file

        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, str]] = []

        files = sorted(
            path
            for path in self.data_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
        )
        for path in files:
            domain = path.parent.name
            document_number = infer_document_number(path.stem)
            rows.append(
                {
                    "domain": domain,
                    "file_name": path.name,
                    "document_number": document_number,
                    "issue_date": "",
                    "effective_date": "",
                    "expiry_date": "",
                    "status_override": "",
                    "replacement_document": "",
                    "source_url": "",
                    "notes": "",
                }
            )

        with self.metadata_file.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "domain",
                    "file_name",
                    "document_number",
                    "issue_date",
                    "effective_date",
                    "expiry_date",
                    "status_override",
                    "replacement_document",
                    "source_url",
                    "notes",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        return self.metadata_file


def infer_document_number(stem: str) -> str:
    match = re.search(
        r"(?P<number>\d{1,4})_(?P<year>\d{4})_(?P<code>[A-Za-z0-9-]+)",
        stem,
    )
    if match:
        return f"{match.group('number')}/{match.group('year')}/{match.group('code')}"

    parts = stem.split("_")
    if len(parts) <= 1:
        return stem

    numbered_parts = [part for part in parts[1:] if not part.isdigit() or len(part) < 6]
    if not numbered_parts:
        return stem

    return "/".join(numbered_parts[:3])


class ExpiryMonitor:
    def __init__(
        self,
        registry: LegalMetadataRegistry,
        notifier: ExpiryNotifier | None = None,
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

        if self.notifier is not None:
            for record in report.expired_documents:
                domain_label = DOMAIN_LABELS.get(record.domain, record.domain)
                self.notifier.notify_expired_document(record=record, domain_label=domain_label)

        return report


def _build_notifier_if_available() -> ExpiryNotifier | None:
    try:
        from api import AdminNotifier
    except Exception:
        return None

    try:
        return AdminNotifier()
    except Exception:
        return None


def run_monitor(
    force: bool,
    loop_forever: bool,
    interval_hours: int,
) -> None:
    registry = LegalMetadataRegistry(autocreate_template=True)
    notifier = _build_notifier_if_available()
    monitor = ExpiryMonitor(
        registry=registry,
        notifier=notifier,
        interval_hours=interval_hours,
    )

    while True:
        report = monitor.run_if_due(force=force)
        if report is None:
            logger.info("Expiry scan skipped (not due yet)")
        else:
            logger.info(
                "Expiry scan completed: %d/%d documents expired",
                len(report.expired_documents),
                report.total_documents,
            )

        if not loop_forever:
            return

        force = False
        sleep_seconds = max(300, interval_hours * 3600)
        logger.info("Next scan in %d seconds", sleep_seconds)
        time.sleep(sleep_seconds)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Expiry metadata monitor and scanner.")
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Generate metadata/legal_documents_metadata.csv template from data folder.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Run expiry monitor once.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Run expiry monitor in a loop.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force scan regardless of interval.",
    )
    parser.add_argument(
        "--interval-hours",
        type=int,
        default=EXPIRY_SCAN_INTERVAL_HOURS,
        help="Monitor interval in hours.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.bootstrap:
        output = LegalMetadataRegistry(autocreate_template=False).bootstrap_template_from_data(overwrite=False)
        logger.info("Metadata template ready: %s", output)
        return

    if args.monitor or args.watch:
        run_monitor(
            force=args.force or args.monitor,
            loop_forever=args.watch,
            interval_hours=max(1, args.interval_hours),
        )
        return

    parser.print_help()


if __name__ == "__main__":
    _cli()
