from __future__ import annotations

import logging
import smtplib
import ssl
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

from config import (
    ADMIN_ALERT_LOG_FILE,
    ADMIN_EMAIL,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_SENDER,
    SMTP_USERNAME,
    SMTP_USE_SSL,
    SMTP_USE_TLS,
)
from legal_metadata import LegalDocumentMetadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FallbackAlertContext:
    question: str
    domain: str
    domain_label: str
    reason: str
    expired_documents: list[str]


class AdminNotifier:
    def __init__(self, log_file: Path = ADMIN_ALERT_LOG_FILE) -> None:
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._sent_cache: set[str] = set()

    def notify_expired_document(self, record: LegalDocumentMetadata, domain_label: str) -> None:
        expiry = record.expiry_date.isoformat() if record.expiry_date else "(không rõ)"
        message = (
            f"Văn bản [{record.file_name} | {record.document_number}] thuộc lĩnh vực "
            f"[{domain_label}] đã hết hiệu lực từ ngày [{expiry}]. "
            "Vui lòng cập nhật văn bản thay thế vào hệ thống."
        )
        dedupe_key = f"expired::{record.domain.lower()}::{record.file_name.lower()}::{expiry}"
        self._notify(message=message, subject="[Chatbot] Cảnh báo văn bản hết hiệu lực", dedupe_key=dedupe_key)

    def notify_fallback(self, context: FallbackAlertContext) -> None:
        docs = ", ".join(context.expired_documents) if context.expired_documents else "(không xác định)"
        message = (
            "Kích hoạt Web Search Fallback do dữ liệu trong kho không còn phù hợp.\n"
            f"- Lĩnh vực: {context.domain_label} ({context.domain})\n"
            f"- Lý do: {context.reason}\n"
            f"- Câu hỏi: {context.question}\n"
            f"- Văn bản liên quan: {docs}\n"
            "Đề nghị cập nhật văn bản mới vào hệ thống và chạy lại build_store.py."
        )
        dedupe_key = f"fallback::{context.domain.lower()}::{context.question.strip().lower()}::{context.reason.lower()}"
        self._notify(message=message, subject="[Chatbot] Kích hoạt Web Fallback", dedupe_key=dedupe_key)

    def _notify(self, message: str, subject: str, dedupe_key: str) -> None:
        if dedupe_key in self._sent_cache:
            return

        self._sent_cache.add(dedupe_key)
        self._append_log(subject=subject, message=message)

        if ADMIN_EMAIL and SMTP_HOST:
            try:
                self._send_email(subject=subject, message=message)
            except Exception as exc:
                logger.warning("Send admin email failed: %s", exc)

    def _append_log(self, subject: str, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {subject}\n{message}\n\n"
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(entry)

    def _send_email(self, subject: str, message: str) -> None:
        if not ADMIN_EMAIL:
            return

        email_message = EmailMessage()
        email_message["From"] = SMTP_SENDER or SMTP_USERNAME
        email_message["To"] = ADMIN_EMAIL
        email_message["Subject"] = subject
        email_message.set_content(message)

        if SMTP_USE_SSL:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
                if SMTP_USERNAME:
                    server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.send_message(email_message)
            return

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            if SMTP_USE_TLS:
                context = ssl.create_default_context()
                server.starttls(context=context)
            if SMTP_USERNAME:
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(email_message)
