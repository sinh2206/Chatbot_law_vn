from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    ALLOWED_SOURCE_EXTENSIONS,
    PROCESSED_DIR,
    RAW_DOCS_DIR,
    TEXT_OUTPUT_EXTENSION,
    ensure_directories,
)


def setup_stdout_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def normalize_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    clean = "\n".join(lines)
    while "\n\n\n" in clean:
        clean = clean.replace("\n\n\n", "\n\n")
    return clean.strip()


def read_txt_file(path: Path) -> str:
    encodings = (
        "utf-8-sig",
        "utf-8",
        "utf-16",
        "utf-16le",
        "utf-16be",
        "cp1258",
        "cp1252",
        "latin-1",
    )
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def read_docx_file(path: Path) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError(
            "Missing python-docx. Install requirements first."
        ) from exc

    document = Document(str(path))
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def run_extract_command(path: Path, command: list[str]) -> str | None:
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


def text_quality_score(text: str) -> float:
    if not text:
        return -1.0

    normalized = normalize_text(text)
    if len(normalized) < 50:
        return -1.0

    letters = re.findall(r"[A-Za-zÀ-ỹĐđ]", normalized)
    if not letters:
        return -1.0

    total = len(normalized)
    question_ratio = normalized.count("?") / max(1, total)
    replacement_ratio = normalized.count("�") / max(1, total)
    mojibake_hint = sum(normalized.count(token) for token in ("Ã", "Ä", "Ì", "Æ", "�"))
    mojibake_ratio = mojibake_hint / max(1, total)

    vietnamese_letters = re.findall(r"[À-ỹĐđ]", normalized)
    vn_ratio = len(vietnamese_letters) / max(1, len(letters))

    # Favor long readable text with proper Vietnamese letters, penalize mojibake.
    score = 0.0
    score += min(3.0, len(normalized) / 20000.0)
    score += vn_ratio * 6.0
    score -= question_ratio * 10.0
    score -= replacement_ratio * 20.0
    score -= mojibake_ratio * 25.0
    return score


def read_doc_file(path: Path) -> str:
    candidates: list[tuple[float, str, str]] = []

    extractors = [
        ("antiword", ["antiword", str(path)]),
        ("catdoc", ["catdoc", str(path)]),
    ]
    for name, command in extractors:
        content = run_extract_command(path, command)
        if content and content.strip():
            candidates.append((text_quality_score(content), name, content))

    # Optional fallback for Windows machines with MS Word installed.
    com_text = read_doc_with_word_com(path)
    if com_text and com_text.strip():
        candidates.append((text_quality_score(com_text), "word_com", com_text))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_name, best_text = candidates[0]
        if best_score > 0:
            print(f"[DOC-PARSER] {path.name}: selected={best_name}, score={best_score:.2f}")
            return best_text

    raise RuntimeError(
        f"Cannot parse .doc file: {path.name}. "
        "Install antiword/catdoc and retry."
    )


def read_doc_with_word_com(path: Path) -> str | None:
    if sys.platform != "win32":
        return None

    try:
        import win32com.client  # type: ignore
    except Exception:
        return None

    temp_path = Path(tempfile.gettempdir()) / f"{path.stem}_{path.stat().st_size}_unicode.txt"
    app = None
    document = None
    try:
        app = win32com.client.Dispatch("Word.Application")
        app.Visible = False
        app.DisplayAlerts = 0
        document = app.Documents.Open(str(path.resolve()))
        # wdFormatUnicodeText = 7 to preserve Vietnamese characters.
        document.SaveAs(str(temp_path), FileFormat=7)
    except Exception:
        return None
    finally:
        if document is not None:
            try:
                document.Close(False)
            except Exception:
                pass
        if app is not None:
            try:
                app.Quit()
            except Exception:
                pass

    if not temp_path.exists():
        return None
    try:
        return read_txt_file(temp_path)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def read_document(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".txt":
        return read_txt_file(path)
    if ext == ".docx":
        return read_docx_file(path)
    if ext == ".doc":
        return read_doc_file(path)
    raise ValueError(f"Unsupported extension: {ext}")


def iter_source_documents(source_dir: Path, domains: set[str] | None) -> list[Path]:
    files: list[Path] = []
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED_SOURCE_EXTENSIONS:
            continue

        if domains:
            try:
                relative = path.relative_to(source_dir)
            except ValueError:
                continue
            if not relative.parts:
                continue
            if relative.parts[0] not in domains:
                continue

        files.append(path)

    files.sort()
    return files


def source_to_target_path(source_path: Path, source_root: Path, output_root: Path) -> Path:
    relative = source_path.relative_to(source_root)
    target_dir = output_root / relative.parent
    return target_dir / f"{source_path.stem}{TEXT_OUTPUT_EXTENSION}"


def convert_all(
    source_dir: Path,
    output_dir: Path,
    domains: set[str] | None,
    overwrite: bool,
    clean_output: bool,
) -> tuple[int, int]:
    ensure_directories()
    output_dir.mkdir(parents=True, exist_ok=True)

    if clean_output:
        if domains:
            for domain in domains:
                domain_dir = output_dir / domain
                if domain_dir.exists():
                    shutil.rmtree(domain_dir)
        else:
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

    sources = iter_source_documents(source_dir=source_dir, domains=domains)
    converted = 0
    skipped = 0

    for source in sources:
        target = source_to_target_path(source, source_dir, output_dir)
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.exists() and not overwrite:
            skipped += 1
            continue

        try:
            text = read_document(source)
            clean = normalize_text(text)
            if not clean:
                skipped += 1
                continue
            target.write_text(clean, encoding="utf-8")
            converted += 1
            print(f"[OK] {source} -> {target}")
        except Exception as exc:
            skipped += 1
            print(f"[SKIP] {source}: {exc}")

    return converted, skipped


def main() -> None:
    setup_stdout_utf8()
    parser = argparse.ArgumentParser(description="Convert .doc/.docx/.txt into normalized .txt files.")
    parser.add_argument("--source-dir", default=str(RAW_DOCS_DIR))
    parser.add_argument("--output-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--domain", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    domains = set(args.domain) if args.domain else None

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    converted, skipped = convert_all(
        source_dir=source_dir,
        output_dir=output_dir,
        domains=domains,
        overwrite=args.overwrite,
        clean_output=args.clean_output,
    )

    print("\n=== CONVERT SUMMARY ===")
    print(f"Source dir: {source_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Converted: {converted}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
