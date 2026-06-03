from __future__ import annotations

from dataclasses import dataclass
import sys


@dataclass(frozen=True)
class GeminiFallbackRequest:
    question: str
    reason: str
    domain: str | None
    expired_sources: list[str]
    low_confidence_sources: list[str]


def build_prompt(request: GeminiFallbackRequest) -> str:
    domain_text = request.domain or "khong gioi han"
    expired_text = "\n".join(f"- {item}" for item in request.expired_sources) or "- Khong co"
    low_conf_text = (
        "\n".join(f"- {item}" for item in request.low_confidence_sources) or "- Khong co"
    )

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
2. Mo dau bang thong bao ngan: "Khong tim thay can cu hop le trong kho noi bo; cau tra loi duoi day la fallback tu Gemini."
3. Khong duoc noi rang cau tra loi duoc trich tu data/processed.
4. Khong bia dat so dieu, khoan, so hieu van ban neu khong chac chan.
5. Neu van de can van ban moi/cap nhat, hay noi ro can admin cap nhat kho tri thuc hoac can doi chieu nguon chinh thong hien hanh.
6. Tra loi ngan gon, thuc dung, neu khong du can cu thi neu huong kiem tra tiep theo thay vi ket luan chac chan.

Cau hoi cua nguoi dung:
{request.question}
""".strip()


def generate_gemini_fallback_answer(
    request: GeminiFallbackRequest,
    api_key: str,
    model_name: str,
) -> str:
    if not api_key.strip():
        raise RuntimeError(
            "Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY in environment."
        )

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "Missing google-genai package. Install requirements first:\n"
            f'  "{sys.executable}" -m pip install google-genai'
        ) from exc

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=build_prompt(request),
    )
    text = getattr(response, "text", "") or ""
    if not text.strip():
        raise RuntimeError("Gemini returned an empty response.")
    return text.strip()
