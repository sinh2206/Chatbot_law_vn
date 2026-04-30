from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import re

from config import ALLOWED_EXTENSIONS, DATA_DIR, LEGAL_METADATA_FILE

DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y")


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

                file_key = (record.domain.lower(), record.file_name.lower())
                if record.domain and record.file_name:
                    self._by_file[file_key] = record

                number_key = (record.domain.lower(), record.document_number.upper())
                if record.domain and record.document_number:
                    self._by_number[number_key] = record

    def find(
        self,
        domain: str,
        file_name: str,
        document_number: str = "",
    ) -> LegalDocumentMetadata | None:
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
            # Recheck by date in case vector store was built earlier.
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
