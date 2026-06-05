from __future__ import annotations

from dataclasses import dataclass, field
import re
import sys
from typing import Any


@dataclass(frozen=True)
class GeminiFallbackRequest:
    question: str
    reason: str
    domain: str | None
    fallback_notice: str = ""
    expired_sources: list[str] = field(default_factory=list)
    low_confidence_sources: list[str] = field(default_factory=list)
    available_local_documents: list[str] = field(default_factory=list)
    missing_documents: list[str] = field(default_factory=list)
    local_answer: str = ""


@dataclass(frozen=True)
class GeminiFallbackResult:
    answer: str
    sources: list[dict[str, str]]


def build_prompt(request: GeminiFallbackRequest) -> str:
    domain_text = request.domain or "khong gioi han"
    expired_text = "\n".join(f"- {item}" for item in request.expired_sources) or "- Khong co"
    low_conf_text = (
        "\n".join(f"- {item}" for item in request.low_confidence_sources) or "- Khong co"
    )
    available_text = (
        "\n".join(f"- {item}" for item in request.available_local_documents)
        or "- Khong co"
    )
    missing_text = (
        "\n".join(f"- {item}" for item in request.missing_documents)
        or "- Khong xac dinh cu the"
    )
    local_answer_text = request.local_answer.strip() or "Khong co"

    if request.missing_documents:
        return f"""
Ban la tro ly phap luat Viet Nam. He thong RAG noi bo da truy xuat duoc mot phan can cu.
Nhiem vu cua ban la BO SUNG ngan gon bang Gemini/Google Search grounding CHI cho cac van ban con thieu.

Cac van ban da co trong kho noi bo, KHONG duoc tim lai bang API:
{available_text}

Cac van ban con thieu, chi duoc tim nhung van ban nay:
{missing_text}

Tom tat cau tra loi noi bo da co:
{local_answer_text}

Linh vuc nguoi dung yeu cau: {domain_text}

Yeu cau bat buoc:
1. Tra loi bang tieng Viet, ngan gon, uu tien tiet kiem token.
2. Dong dau tien phai la: Bổ sung tài liệu cần tra cứu qua API: <ten cac van ban con thieu>.
3. Chi bo sung noi dung lien quan den van ban con thieu; khong lap lai cac can cu noi bo da co.
4. Neu cau hoi co nhieu phan, chi tra loi phan ma cac van ban con thieu dieu chinh.
5. Neu van ban con thieu da bi thay the/het hieu luc, noi ro van ban thay the neu tim duoc.
6. Dua ra muc "Tai lieu/can cu tham khao" o cuoi cau tra loi, gom ten van ban/trang va duong dan neu co.
7. Khong bia dat so dieu, khoan, so hieu van ban neu khong chac chan.

Cau hoi cua nguoi dung:
{request.question}
""".strip()

    return f"""
Ban la tro ly phap luat Viet Nam. He thong RAG noi bo da duoc thu truoc khi goi ban.
Ket luan cua RAG noi bo: KHONG CO CAN CU HOP LE DE TRA LOI TU KHO NOI BO.

Ly do fallback:
{request.reason}

Linh vuc nguoi dung yeu cau: {domain_text}

Tai lieu noi bo het hieu luc hoac khong duoc dung:
{expired_text}

Tai lieu noi bo co score thap/khong du tin cay:
{low_conf_text}

Yeu cau bat buoc:
1. Tra loi bang tieng Viet.
2. Dong dau tien phai viet dung nguyen van:
{request.fallback_notice}
3. Khong duoc noi rang cau tra loi duoc trich tu data/processed.
4. Sau dong thong bao, hay tim cau tra loi bang Gemini/Google Search grounding.
5. Dua ra muc "Tai lieu/can cu tham khao" o cuoi cau tra loi, gom ten van ban/trang va duong dan neu co.
   Neu khong co duong dan thi van phai ghi ten van ban lam can cu.
6. Khong bia dat so dieu, khoan, so hieu van ban neu khong chac chan.
7. Neu van de can van ban moi/cap nhat, hay noi ro can admin cap nhat kho tri thuc hoac can doi chieu nguon chinh thong hien hanh.
8. Tra loi ngan gon, thuc dung, neu khong du can cu thi neu huong kiem tra tiep theo thay vi ket luan chac chan.

Cau hoi cua nguoi dung:
{request.question}
""".strip()


def extract_grounding_sources(response: Any) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    for candidate in getattr(response, "candidates", []) or []:
        grounding_metadata = getattr(candidate, "grounding_metadata", None)
        if grounding_metadata is None:
            grounding_metadata = getattr(candidate, "groundingMetadata", None)
        chunks = getattr(grounding_metadata, "grounding_chunks", None)
        if chunks is None:
            chunks = getattr(grounding_metadata, "groundingChunks", None)
        for chunk in chunks or []:
            web = getattr(chunk, "web", None)
            if web is None:
                continue
            uri = str(getattr(web, "uri", "") or "").strip()
            title = str(getattr(web, "title", "") or "").strip()
            domain = str(getattr(web, "domain", "") or "").strip()
            key = uri or title
            if not key or key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "source": title or uri,
                    "title": title,
                    "url": uri,
                    "domain": domain,
                    "type": "gemini_google_search",
                }
            )
    return sources


REFERENCE_HEADING_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(?:t[àa]i\s+li[eệ]u|can\s+cu|căn\s+cứ).*(?:tham\s+kh[aả]o|can\s+cu|căn\s+cứ).*$",
    re.IGNORECASE,
)
REFERENCE_LINE_RE = re.compile(r"^\s*(?:[-*]|\d+[\.)])\s*(.+?)\s*$")
URL_RE = re.compile(r"https?://[^\s)\]]+")


def extract_text_reference_sources(text: str) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    in_reference_section = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if in_reference_section and sources:
                break
            continue
        if REFERENCE_HEADING_RE.match(line):
            in_reference_section = True
            continue
        if not in_reference_section:
            continue
        match = REFERENCE_LINE_RE.match(line)
        if not match:
            if sources:
                break
            continue
        value = match.group(1).strip()
        value = value.strip("* ")
        if not value:
            continue
        url_match = URL_RE.search(value)
        url = url_match.group(0) if url_match else ""
        title = URL_RE.sub("", value).strip(" -|.")
        sources.append(
            {
                "source": title or url,
                "title": title or url,
                "url": url,
                "domain": "",
                "type": "gemini_text_reference",
            }
        )
    return sources


def ensure_notice_prefix(answer: str, fallback_notice: str) -> str:
    if not fallback_notice.strip():
        return answer.strip()
    normalized_answer = answer.strip().lower()
    normalized_notice = fallback_notice.strip().lower()
    if normalized_answer.startswith(normalized_notice):
        return answer.strip()
    return f"{fallback_notice.strip()}\n\n{answer.strip()}"


def generate_gemini_fallback_answer(
    request: GeminiFallbackRequest,
    api_key: str,
    model_name: str,
) -> GeminiFallbackResult:
    if not api_key.strip():
        raise RuntimeError(
            "Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY in environment."
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "Missing google-genai package. Install requirements first:\n"
            f'  "{sys.executable}" -m pip install google-genai'
        ) from exc

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=build_prompt(request),
        config=types.GenerateContentConfig(
            temperature=0.2,
            tools=[types.Tool(googleSearch=types.GoogleSearch())],
        ),
    )
    text = getattr(response, "text", "") or ""
    if not text.strip():
        raise RuntimeError("Gemini returned an empty response.")
    answer = ensure_notice_prefix(text, request.fallback_notice)
    sources = extract_grounding_sources(response)
    if not sources:
        sources = extract_text_reference_sources(answer)
    return GeminiFallbackResult(answer=answer, sources=sources)
