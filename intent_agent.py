from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import DOMAIN_KEYWORDS, DOMAIN_LABELS, INTENT_TEMPERATURE
from llm_client import GeminiClient, parse_json_from_text


@dataclass
class SubQuestion:
    question: str
    domain: str | None


@dataclass
class IntentResult:
    domains: list[str]
    sub_questions: list[SubQuestion]
    raw_model_output: str


class IntentAgent:
    def __init__(
        self,
        llm: GeminiClient | None = None,
        supported_domains: list[str] | None = None,
    ) -> None:
        self.llm = llm or GeminiClient()
        self.supported_domains = supported_domains or list(DOMAIN_LABELS.keys())

    def analyze(self, user_question: str) -> IntentResult:
        prompt = self._build_prompt(user_question)
        raw_output = self.llm.generate(prompt=prompt, temperature=INTENT_TEMPERATURE)

        try:
            payload = parse_json_from_text(raw_output)
            return self._normalize_payload(payload, raw_output, user_question)
        except Exception:
            fallback_domains = self._keyword_domains(user_question)
            fallback_sub_questions = [
                SubQuestion(question=user_question.strip(), domain=domain)
                for domain in (fallback_domains or [None])
            ]
            return IntentResult(
                domains=fallback_domains,
                sub_questions=fallback_sub_questions,
                raw_model_output=raw_output,
            )

    def _build_prompt(self, user_question: str) -> str:
        domain_guide = "\n".join(
            f"- {code}: {DOMAIN_LABELS.get(code, code)}"
            for code in self.supported_domains
        )

        return f"""
Bạn là bộ phân loại intent cho chatbot pháp luật Việt Nam.

Danh sách lĩnh vực hợp lệ:
{domain_guide}

Yêu cầu:
1) Xác định câu hỏi thuộc 1 hoặc nhiều lĩnh vực.
2) Nếu là câu hỏi phức hợp liên quan nhiều lĩnh vực, tách thành nhiều câu hỏi con.
3) Trả về DUY NHẤT JSON hợp lệ, không thêm markdown.

JSON schema:
{{
  "domains": ["DoanhNghiep", "Thue"],
  "sub_questions": [
    {{"domain": "DoanhNghiep", "question": "..."}},
    {{"domain": "Thue", "question": "..."}}
  ]
}}

Quy tắc:
- Chỉ dùng domain trong danh sách hợp lệ.
- Nếu không chắc, có thể để domain là null ở từng sub_question.
- Nếu câu hỏi đơn giản, trả 1 sub_question.

Câu hỏi người dùng: {user_question}
""".strip()

    def _normalize_payload(
        self,
        payload: dict[str, Any],
        raw_output: str,
        user_question: str,
    ) -> IntentResult:
        domains_raw = payload.get("domains") or []
        normalized_domains = [
            domain
            for domain in domains_raw
            if isinstance(domain, str) and domain in self.supported_domains
        ]

        sub_questions: list[SubQuestion] = []
        for item in payload.get("sub_questions", []):
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            domain = item.get("domain")
            if isinstance(domain, str) and domain not in self.supported_domains:
                domain = None
            if question:
                sub_questions.append(SubQuestion(question=question, domain=domain))

        if not sub_questions:
            fallback_domain = normalized_domains[0] if normalized_domains else None
            sub_questions = [
                SubQuestion(question=user_question.strip(), domain=fallback_domain)
            ]

        if not normalized_domains:
            inferred_from_sub = [sq.domain for sq in sub_questions if sq.domain]
            normalized_domains = sorted({d for d in inferred_from_sub if d})

        return IntentResult(
            domains=normalized_domains,
            sub_questions=sub_questions,
            raw_model_output=raw_output,
        )

    def _keyword_domains(self, question: str) -> list[str]:
        normalized = question.lower()
        detected: list[str] = []

        for domain, keywords in DOMAIN_KEYWORDS.items():
            if domain not in self.supported_domains:
                continue
            if any(keyword in normalized for keyword in keywords):
                detected.append(domain)

        return detected
