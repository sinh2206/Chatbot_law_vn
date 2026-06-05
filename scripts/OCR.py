from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import PROCESSED_DIR, RAW_DOCS_DIR, TEXT_OUTPUT_EXTENSION, ensure_directories  # noqa: E402

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_EXTENSIONS = {".pdf"}
OCR_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS


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


def iter_ocr_source_files(source_dir: Path, domains: set[str] | None) -> list[Path]:
    files: list[Path] = []
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in OCR_EXTENSIONS:
            continue

        if domains:
            try:
                relative = path.relative_to(source_dir)
            except ValueError:
                continue
            if not relative.parts or relative.parts[0] not in domains:
                continue

        files.append(path)
    files.sort()
    return files


def source_to_target_path(source_path: Path, source_root: Path, output_root: Path) -> Path:
    relative = source_path.relative_to(source_root)
    target_dir = output_root / relative.parent
    return target_dir / f"{source_path.stem}{TEXT_OUTPUT_EXTENSION}"


def load_image_bgr(path: Path):
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Missing OCR dependencies (opencv-python / numpy). "
            "Install requirements first."
        ) from exc

    # Use imdecode to support non-ASCII file paths on Windows.
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Cannot decode image bytes.")
    return image


def load_pdf_page_bgr(path: Path, page_index: int, dpi: int):
    try:
        import cv2
        import fitz
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Missing PDF OCR dependencies (PyMuPDF / opencv-python / numpy). "
            "Install requirements first."
        ) from exc

    scale = max(dpi, 72) / 72
    with fitz.open(path) as document:
        page = document.load_page(page_index)
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(scale, scale),
            alpha=False,
        )
        channels = pixmap.n
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height,
            pixmap.width,
            channels,
        )
        if channels == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR).copy()
        if channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
        return cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2BGR).copy()


def preprocess_image(image_bgr, mode: str):
    import cv2

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    if mode == "none":
        return gray
    if mode == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    if mode == "adaptive":
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )

    raise ValueError(f"Unsupported preprocess mode: {mode}")


def ocr_image_bgr(
    image_bgr,
    lang: str,
    psm: int,
    oem: int,
    preprocess_mode: str,
    tesseract_cmd: str,
    timeout: float,
) -> str:
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError(
            "Missing OCR dependency (pytesseract). Install requirements first."
        ) from exc

    if tesseract_cmd.strip():
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd.strip()

    processed = preprocess_image(image_bgr, mode=preprocess_mode)

    config = f"--oem {oem} --psm {psm}"
    timeout_arg = timeout if timeout > 0 else None

    try:
        text = pytesseract.image_to_string(
            processed,
            lang=lang,
            config=config,
            timeout=timeout_arg,
        )
    except RuntimeError as exc:
        raise RuntimeError(f"Tesseract timeout/error: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"OCR failed: {exc}") from exc

    return text


def ocr_image(
    path: Path,
    lang: str,
    psm: int,
    oem: int,
    preprocess_mode: str,
    tesseract_cmd: str,
    timeout: float,
) -> str:
    return ocr_image_bgr(
        image_bgr=load_image_bgr(path),
        lang=lang,
        psm=psm,
        oem=oem,
        preprocess_mode=preprocess_mode,
        tesseract_cmd=tesseract_cmd,
        timeout=timeout,
    )


def ocr_pdf(
    path: Path,
    lang: str,
    psm: int,
    oem: int,
    preprocess_mode: str,
    tesseract_cmd: str,
    timeout: float,
    dpi: int,
) -> str:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("Missing PDF OCR dependency (PyMuPDF). Install requirements first.") from exc

    page_texts: list[str] = []
    with fitz.open(path) as document:
        page_count = document.page_count

    for page_index in range(page_count):
        page_text = ocr_image_bgr(
            image_bgr=load_pdf_page_bgr(path=path, page_index=page_index, dpi=dpi),
            lang=lang,
            psm=psm,
            oem=oem,
            preprocess_mode=preprocess_mode,
            tesseract_cmd=tesseract_cmd,
            timeout=timeout,
        )
        page_text = normalize_text(page_text)
        if page_text:
            page_texts.append(f"--- Trang {page_index + 1} ---\n{page_text}")
    return "\n\n".join(page_texts)


def convert_all_images(
    source_dir: Path,
    output_dir: Path,
    domains: set[str] | None,
    overwrite: bool,
    clean_output: bool,
    lang: str,
    psm: int,
    oem: int,
    preprocess_mode: str,
    tesseract_cmd: str,
    timeout: float,
    min_chars: int,
    pdf_dpi: int,
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

    sources = iter_ocr_source_files(source_dir=source_dir, domains=domains)
    converted = 0
    skipped = 0

    for source in sources:
        target = source_to_target_path(source, source_dir, output_dir)
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.exists() and not overwrite:
            skipped += 1
            continue

        try:
            if source.suffix.lower() in PDF_EXTENSIONS:
                raw_text = ocr_pdf(
                    path=source,
                    lang=lang,
                    psm=psm,
                    oem=oem,
                    preprocess_mode=preprocess_mode,
                    tesseract_cmd=tesseract_cmd,
                    timeout=timeout,
                    dpi=pdf_dpi,
                )
            else:
                raw_text = ocr_image(
                    path=source,
                    lang=lang,
                    psm=psm,
                    oem=oem,
                    preprocess_mode=preprocess_mode,
                    tesseract_cmd=tesseract_cmd,
                    timeout=timeout,
                )
            clean = normalize_text(raw_text)
            if len(clean) < min_chars:
                skipped += 1
                print(f"[SKIP] {source}: OCR text too short ({len(clean)} chars)")
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
    parser = argparse.ArgumentParser(
        description="OCR legal document images/PDF scans into normalized .txt files."
    )
    parser.add_argument("--source-dir", default=str(RAW_DOCS_DIR))
    parser.add_argument("--output-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--domain", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--lang", default="vie+eng", help="Tesseract language pack(s), e.g. vie+eng")
    parser.add_argument("--psm", type=int, default=6, help="Tesseract PSM mode (default: 6)")
    parser.add_argument("--oem", type=int, default=3, help="Tesseract OEM mode (default: 3)")
    parser.add_argument(
        "--preprocess",
        choices=["none", "otsu", "adaptive"],
        default="adaptive",
        help="Image preprocess mode before OCR",
    )
    parser.add_argument(
        "--tesseract-cmd",
        default="",
        help="Full path to tesseract executable if not in PATH",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0,
        help="OCR timeout in seconds per image (0 = no timeout)",
    )
    parser.add_argument("--min-chars", type=int, default=30, help="Skip OCR output shorter than this")
    parser.add_argument("--pdf-dpi", type=int, default=220, help="Render DPI for scanned PDFs")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    domains = set(args.domain) if args.domain else None

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if args.psm < 0 or args.psm > 13:
        raise ValueError("--psm should be in [0, 13]")
    if args.oem < 0 or args.oem > 3:
        raise ValueError("--oem should be in [0, 3]")
    if args.min_chars < 1:
        raise ValueError("--min-chars must be >= 1")
    if args.pdf_dpi < 72:
        raise ValueError("--pdf-dpi must be >= 72")

    converted, skipped = convert_all_images(
        source_dir=source_dir,
        output_dir=output_dir,
        domains=domains,
        overwrite=args.overwrite,
        clean_output=args.clean_output,
        lang=args.lang,
        psm=args.psm,
        oem=args.oem,
        preprocess_mode=args.preprocess,
        tesseract_cmd=args.tesseract_cmd,
        timeout=args.timeout,
        min_chars=args.min_chars,
        pdf_dpi=args.pdf_dpi,
    )

    print("\n=== OCR SUMMARY ===")
    print(f"Source dir: {source_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Converted: {converted}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
