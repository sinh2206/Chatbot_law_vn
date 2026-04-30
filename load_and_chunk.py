from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from config import (
    ALLOWED_EXTENSIONS,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    DOMAIN_LABELS,
)
from legal_metadata import LegalMetadataRegistry, infer_document_number

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    text: str
    metadata: dict[str, str]


def list_document_files(root_dir: Path | None = None) -> list[Path]:
    root = Path(root_dir or DATA_DIR)
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    ]
    files.sort()
    return files


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]

    normalized_lines: list[str] = []
    previous_blank = False
    for line in lines:
        if line:
            normalized_lines.append(line)
            previous_blank = False
        elif not previous_blank:
            normalized_lines.append("")
            previous_blank = True

    return "\n".join(normalized_lines).strip()


def split_text_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    normalized = _normalize_text(text)
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []
    start = 0
    total_length = len(normalized)
    min_boundary = int(chunk_size * 0.6)

    while start < total_length:
        end = min(start + chunk_size, total_length)

        if end < total_length:
            boundary = normalized.rfind("\n", start + min_boundary, end)
            if boundary == -1:
                boundary = normalized.rfind(" ", start + min_boundary, end)
            if boundary > start:
                end = boundary

        piece = normalized[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= total_length:
            break

        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks


def _read_text_file(path: Path) -> str:
    encodings = ["utf-8-sig", "utf-8", "cp1258", "cp1252", "latin-1"]
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_docx_file(path: Path) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError(
            "python-docx is required to read .docx files. Install dependencies first."
        ) from exc

    doc = Document(str(path))
    paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)


def _extract_doc_with_textract(path: Path) -> str | None:
    try:
        import textract  # type: ignore
    except ImportError:
        return None

    try:
        raw = textract.process(str(path))
    except Exception:
        return None

    for encoding in ("utf-8", "cp1258", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _extract_doc_with_command(path: Path, command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:
        return None

    raw = completed.stdout
    for encoding in ("utf-8", "cp1258", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _extract_doc_with_word_com(path: Path) -> str | None:
    if os.name != "nt":
        return None

    try:
        import win32com.client  # type: ignore
    except ImportError:
        return None

    temp_txt = Path(tempfile.gettempdir()) / f"{path.stem}_{os.getpid()}_tmp.txt"
    word = None
    doc = None

    try:
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        doc = word.Documents.Open(str(path.resolve()))
        doc.SaveAs(str(temp_txt), FileFormat=2)
    except Exception:
        return None
    finally:
        if doc is not None:
            try:
                doc.Close(False)
            except Exception:
                pass
        if word is not None:
            try:
                word.Quit()
            except Exception:
                pass

    if not temp_txt.exists():
        return None

    try:
        return _read_text_file(temp_txt)
    finally:
        try:
            temp_txt.unlink(missing_ok=True)
        except OSError:
            pass


def _read_doc_file(path: Path) -> str:
    strategies = [
        _extract_doc_with_textract,
        lambda p: _extract_doc_with_command(p, ["antiword", str(p)]),
        lambda p: _extract_doc_with_command(p, ["catdoc", str(p)]),
        _extract_doc_with_word_com,
    ]

    for strategy in strategies:
        text = strategy(path)
        if text and text.strip():
            return text

    raise RuntimeError(
        "Cannot read .doc file. Install one of these options: "
        "(1) textract + antiword, (2) antiword/catdoc command, or (3) pywin32 + Microsoft Word."
    )


def extract_text(path: Path) -> str:
    extension = path.suffix.lower()

    if extension == ".txt":
        return _read_text_file(path)
    if extension == ".docx":
        return _read_docx_file(path)
    if extension == ".doc":
        return _read_doc_file(path)

    raise ValueError(f"Unsupported file extension: {extension}")


def _extract_article_hint(chunk_text: str) -> str:
    matches = re.findall(r"\bĐiều\s+\d+[A-Za-z]?", chunk_text, flags=re.IGNORECASE)
    if not matches:
        return ""

    deduped: list[str] = []
    seen: set[str] = set()
    for item in matches:
        key = item.lower()
        if key not in seen:
            deduped.append(item)
            seen.add(key)
        if len(deduped) >= 3:
            break

    return "; ".join(deduped)


def _build_chunk_id(path: Path, chunk_index: int) -> str:
    raw = f"{path.as_posix()}::{chunk_index}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def build_metadata(
    path: Path,
    chunk_index: int,
    chunk_text: str,
    metadata_registry: LegalMetadataRegistry | None = None,
) -> dict[str, str]:
    domain_code = path.parent.name
    base_metadata = {
        "domain": domain_code,
        "domain_label": DOMAIN_LABELS.get(domain_code, domain_code),
        "file_name": path.name,
        "file_stem": path.stem,
        "source_path": str(path),
        "document_type": path.stem.split("_")[0] if "_" in path.stem else "Unknown",
        "document_number": infer_document_number(path.stem),
        "article_hint": _extract_article_hint(chunk_text),
        "chunk_index": str(chunk_index),
    }

    if metadata_registry is None:
        return base_metadata

    return metadata_registry.enrich_chunk_metadata(base_metadata)


def load_and_chunk_documents(
    root_dir: Path | None = None,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    metadata_registry: LegalMetadataRegistry | None = None,
) -> list[ChunkRecord]:
    files = list_document_files(root_dir=root_dir)
    chunk_records: list[ChunkRecord] = []
    registry = metadata_registry or LegalMetadataRegistry(autocreate_template=True)

    logger.info("Found %d documents in %s", len(files), root_dir or DATA_DIR)

    for file_path in files:
        try:
            text = extract_text(file_path)
        except Exception as exc:
            logger.warning("Skip %s due to read error: %s", file_path, exc)
            continue

        chunks = split_text_into_chunks(text=text, chunk_size=chunk_size, overlap=overlap)
        for chunk_index, chunk_text in enumerate(chunks):
            chunk_records.append(
                ChunkRecord(
                    chunk_id=_build_chunk_id(file_path, chunk_index),
                    text=chunk_text,
                    metadata=build_metadata(
                        file_path,
                        chunk_index,
                        chunk_text,
                        metadata_registry=registry,
                    ),
                )
            )

    logger.info("Generated %d chunks", len(chunk_records))
    return chunk_records


def group_chunks_by_domain(chunks: Iterable[ChunkRecord]) -> dict[str, list[ChunkRecord]]:
    grouped: dict[str, list[ChunkRecord]] = {}
    for chunk in chunks:
        domain = chunk.metadata.get("domain", "Unknown")
        grouped.setdefault(domain, []).append(chunk)
    return grouped
