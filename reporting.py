from __future__ import annotations

import csv
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from config import (
    ADMIN_EMAIL,
    REPORTS_DIR,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_SENDER,
    SMTP_USERNAME,
    SMTP_USE_SSL,
    SMTP_USE_TLS,
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    return path


def write_markdown(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def markdown_to_basic_html(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    html_lines = [
        "<!doctype html>",
        "<html lang='vi'><head><meta charset='utf-8'><title>Chatbot Report</title>",
        "<style>body{font-family:Segoe UI,Arial,sans-serif;max-width:980px;margin:20px auto;line-height:1.5;color:#1f2d2b;}h1,h2{color:#0e5e4e;}pre{background:#f4f7f6;padding:12px;border-radius:8px;overflow:auto;}li{margin:4px 0;}table{border-collapse:collapse;}td,th{border:1px solid #cfd8d5;padding:6px 8px;}code{background:#edf3f1;padding:1px 4px;border-radius:4px;}</style>",
        "</head><body>",
    ]

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# "):
            html_lines.append(f"<h1>{stripped[2:]}</h1>")
        elif stripped.startswith("## "):
            html_lines.append(f"<h2>{stripped[3:]}</h2>")
        elif stripped.startswith("- "):
            html_lines.append(f"<li>{stripped[2:]}</li>")
        elif stripped:
            html_lines.append(f"<p>{stripped}</p>")
        else:
            html_lines.append("<br />")

    html_lines.append("</body></html>")
    return "\n".join(html_lines)


def write_html(path: Path, html: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def send_email_report_if_configured(
    subject: str,
    text_body: str,
    html_body: str | None = None,
    attachments: list[Path] | None = None,
) -> bool:
    if not (ADMIN_EMAIL and SMTP_HOST):
        return False

    email_message = EmailMessage()
    email_message["From"] = SMTP_SENDER or SMTP_USERNAME
    email_message["To"] = ADMIN_EMAIL
    email_message["Subject"] = subject
    email_message.set_content(text_body)

    if html_body:
        email_message.add_alternative(html_body, subtype="html")

    for attachment in attachments or []:
        if not attachment.exists() or not attachment.is_file():
            continue
        data = attachment.read_bytes()
        maintype = "application"
        subtype = "octet-stream"
        if attachment.suffix.lower() == ".csv":
            maintype, subtype = "text", "csv"
        elif attachment.suffix.lower() == ".md":
            maintype, subtype = "text", "markdown"
        elif attachment.suffix.lower() == ".html":
            maintype, subtype = "text", "html"

        email_message.add_attachment(
            data,
            maintype=maintype,
            subtype=subtype,
            filename=attachment.name,
        )

    if SMTP_USE_SSL:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
            if SMTP_USERNAME:
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(email_message)
        return True

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        if SMTP_USE_TLS:
            context = ssl.create_default_context()
            server.starttls(context=context)
        if SMTP_USERNAME:
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(email_message)

    return True


def build_report_paths(run_id: str) -> dict[str, Path]:
    root = REPORTS_DIR / run_id
    return {
        "root": root,
        "markdown": root / "report.md",
        "html": root / "report.html",
        "csv": root / "details.csv",
    }
