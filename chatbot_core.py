from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

from config import (
    DOMAIN_KEYWORDS,
    DOMAIN_LABELS,
    GEMINI_API_KEY,
    INTENT_TEMPERATURE,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    MERGE_TEMPERATURE,
    TOP_K,
    TRUSTED_WEB_DOMAINS,
    WEB_FALLBACK_ENABLED,
    WEB_SEARCH_BACKEND,
    WEB_SEARCH_MAX_RESULTS,
    require_gemini_key,
)
from expiry import ExpiryMonitor, LegalMetadataRegistry
from vector_store import ChromaVectorStore, SearchResult, create_vector_store


class GeminiClient:
    def __init__(self, model_name: str = LLM_MODEL, api_key: str | None = GEMINI_API_KEY) -> None:
        key = api_key or require_gemini_key()

        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError(
                "google-generativeai is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        genai.configure(api_key=key)
        self._genai = genai
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

    def generate(
        self,
        prompt: str,
        temperature: float,
        max_output_tokens: int = LLM_MAX_OUTPUT_TOKENS,
    ) -> str:
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }

        response = self.model.generate_content(
            prompt,
            generation_config=generation_config,
        )

        text = getattr(response, "text", None)
        if text:
            return text.strip()

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return ""

        parts: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    parts.append(part_text)

        return "\n".join(parts).strip()


def parse_json_from_text(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced_match:
        candidate = fenced_match.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if brace_match:
        candidate = brace_match.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError("Cannot parse JSON from model output")


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


@dataclass(frozen=True)
class WebSource:
    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class WebFallbackResult:
    answer: str
    sources: list[WebSource]


class WebSearchFallback:
    def __init__(
        self,
        llm: GeminiClient | None = None,
        backend: str = WEB_SEARCH_BACKEND,
        max_results: int = WEB_SEARCH_MAX_RESULTS,
        trusted_domains: list[str] | None = None,
    ) -> None:
        self.llm = llm or GeminiClient()
        self.backend = backend
        self.max_results = max(3, max_results)
        self.trusted_domains = trusted_domains or TRUSTED_WEB_DOMAINS

    def answer(self, question: str, domain_label: str) -> WebFallbackResult:
        sources = self.search(question=question, domain_label=domain_label)
        if not sources:
            return WebFallbackResult(
                answer=(
                    "Không tìm thấy nguồn web phù hợp để cập nhật thông tin pháp lý mới nhất. "
                    "Vui lòng thử lại hoặc cung cấp thêm ngữ cảnh."
                ),
                sources=[],
            )

        prompt = self._build_prompt(question=question, domain_label=domain_label, sources=sources)
        text = self.llm.generate(prompt=prompt, temperature=0.1)

        cited_lines = "\n".join(f"- {source.title}: {source.url}" for source in sources[:5])
        answer = f"{text.strip()}\n\nNguồn web tham khảo:\n{cited_lines}"

        return WebFallbackResult(answer=answer, sources=sources)

    def search(self, question: str, domain_label: str) -> list[WebSource]:
        query = (
            f"{question} pháp luật Việt Nam {domain_label} "
            "văn bản mới nhất hiệu lực"
        )

        rows: list[dict[str, Any]] = []
        if self.backend.lower() in {"ddgs", "duckduckgo", "duckduckgo_search"}:
            rows = self._search_with_ddgs(query=query)

        parsed = [self._to_source(row) for row in rows]
        sources = [item for item in parsed if item and item.url]
        return self._rank_sources(sources)[: self.max_results]

    def _search_with_ddgs(self, query: str) -> list[dict[str, Any]]:
        ddgs = None
        try:
            from duckduckgo_search import DDGS as LegacyDDGS  # type: ignore

            ddgs = LegacyDDGS()
        except Exception:
            try:
                from ddgs import DDGS as NewDDGS  # type: ignore

                ddgs = NewDDGS()
            except Exception:
                ddgs = None

        if ddgs is None:
            return []

        try:
            results = list(ddgs.text(query, max_results=self.max_results * 2))
        except Exception:
            return []

        normalized: list[dict[str, Any]] = []
        for row in results:
            if isinstance(row, dict):
                normalized.append(row)
        return normalized

    @staticmethod
    def _to_source(row: dict[str, Any]) -> WebSource | None:
        title = str(row.get("title") or "").strip()
        url = str(row.get("href") or row.get("url") or "").strip()
        snippet = str(row.get("body") or row.get("snippet") or "").strip()

        if not url:
            return None
        return WebSource(title=title or url, url=url, snippet=snippet)

    def _rank_sources(self, sources: list[WebSource]) -> list[WebSource]:
        trusted: list[WebSource] = []
        others: list[WebSource] = []

        for source in sources:
            host = (urlparse(source.url).hostname or "").lower()
            if any(domain in host for domain in self.trusted_domains):
                trusted.append(source)
            else:
                others.append(source)

        deduped: list[WebSource] = []
        seen: set[str] = set()
        for source in trusted + others:
            if source.url in seen:
                continue
            deduped.append(source)
            seen.add(source.url)

        return deduped

    @staticmethod
    def _build_prompt(question: str, domain_label: str, sources: list[WebSource]) -> str:
        source_blocks = []
        for index, source in enumerate(sources[:8], start=1):
            source_blocks.append(
                "\n".join(
                    [
                        f"[Nguồn {index}]",
                        f"Tiêu đề: {source.title}",
                        f"URL: {source.url}",
                        f"Tóm tắt: {source.snippet}",
                    ]
                )
            )

        joined_sources = "\n\n".join(source_blocks)

        return f"""
Bạn là trợ lý pháp lý Việt Nam.
Dựa trên các kết quả web bên dưới, hãy trả lời câu hỏi theo hướng cập nhật mới nhất.

Yêu cầu:
- Viết tiếng Việt rõ ràng.
- Chỉ dùng thông tin có trong các nguồn đã cho.
- Nếu nguồn mâu thuẫn, nêu rõ cần đối chiếu văn bản gốc.
- Ưu tiên thông tin về hiệu lực văn bản và văn bản thay thế.

Lĩnh vực: {domain_label}
Câu hỏi: {question}

Nguồn tìm kiếm:
{joined_sources}
""".strip()


@dataclass(frozen=True)
class FallbackAlertContext:
    question: str
    domain: str
    domain_label: str
    reason: str
    expired_documents: list[str]


class RuntimeNotifier(Protocol):
    def notify_expired_document(self, record: Any, domain_label: str) -> None:
        ...

    def notify_fallback(self, context: FallbackAlertContext) -> None:
        ...


class NullNotifier:
    def notify_expired_document(self, record: Any, domain_label: str) -> None:
        return None

    def notify_fallback(self, context: FallbackAlertContext) -> None:
        return None


@dataclass
class AgentAnswer:
    domain: str
    domain_label: str
    question: str
    answer: str
    citations: list[str]
    retrieved_chunks: list[SearchResult]
    mode: str
    sources: list[dict[str, str]]
    fallback_reason: str


class DomainRAGAgent:
    def __init__(
        self,
        domain: str,
        vector_store: ChromaVectorStore,
        llm: GeminiClient | None = None,
        top_k: int = TOP_K,
        metadata_registry: LegalMetadataRegistry | None = None,
        web_fallback: WebSearchFallback | None = None,
        notifier: RuntimeNotifier | None = None,
        enable_web_fallback: bool = WEB_FALLBACK_ENABLED,
    ) -> None:
        self.domain = domain
        self.domain_label = DOMAIN_LABELS.get(domain, domain)
        self.vector_store = vector_store
        self.llm = llm or GeminiClient()
        self.top_k = top_k
        self.metadata_registry = metadata_registry or LegalMetadataRegistry()
        self.web_fallback = web_fallback or WebSearchFallback(llm=self.llm)
        self.notifier = notifier or NullNotifier()
        self.enable_web_fallback = enable_web_fallback

    def answer(self, question: str) -> AgentAnswer:
        retrieved = self.vector_store.similarity_search(
            query=question,
            top_k=self.top_k,
            domain_filter=self.domain,
        )

        active_chunks, expired_chunks = self._split_active_and_expired_chunks(retrieved)

        if not retrieved:
            if self.enable_web_fallback:
                return self._fallback_answer(
                    question=question,
                    fallback_reason="no_local_context",
                    expired_chunks=[],
                )
            return self._empty_answer(question=question)

        if not active_chunks and expired_chunks:
            if self.enable_web_fallback:
                return self._fallback_answer(
                    question=question,
                    fallback_reason="all_retrieved_chunks_expired",
                    expired_chunks=expired_chunks,
                )

            return AgentAnswer(
                domain=self.domain,
                domain_label=self.domain_label,
                question=question,
                answer=(
                    "Các trích đoạn tìm được đều thuộc văn bản đã hết hiệu lực. "
                    "Vui lòng cập nhật kho văn bản để trả lời chính xác hơn."
                ),
                citations=[],
                retrieved_chunks=retrieved,
                mode="blocked_expired",
                sources=[],
                fallback_reason="all_retrieved_chunks_expired",
            )

        context_chunks = active_chunks if active_chunks else retrieved
        context_text = self._format_context(context_chunks)

        if not context_text:
            return self._empty_answer(question=question)

        prompt = self._build_prompt(question=question, context=context_text)
        answer_text = self.llm.generate(prompt=prompt, temperature=LLM_TEMPERATURE)

        if expired_chunks and active_chunks:
            answer_text = (
                f"{answer_text.strip()}\n\n"
                "Lưu ý: Một số văn bản cũ trong kết quả truy xuất đã hết hiệu lực và đã được loại bỏ khỏi phần lập luận."
            )

        return AgentAnswer(
            domain=self.domain,
            domain_label=self.domain_label,
            question=question,
            answer=answer_text.strip(),
            citations=self._extract_citations(context_chunks),
            retrieved_chunks=retrieved,
            mode="rag",
            sources=[],
            fallback_reason="",
        )

    def _fallback_answer(
        self,
        question: str,
        fallback_reason: str,
        expired_chunks: list[SearchResult],
    ) -> AgentAnswer:
        expired_documents = self._collect_expired_documents(expired_chunks)
        context = FallbackAlertContext(
            question=question,
            domain=self.domain,
            domain_label=self.domain_label,
            reason=fallback_reason,
            expired_documents=expired_documents,
        )
        self.notifier.notify_fallback(context)

        fallback_result = self.web_fallback.answer(question=question, domain_label=self.domain_label)
        sources = [
            {"title": source.title, "url": source.url, "snippet": source.snippet}
            for source in fallback_result.sources
        ]
        citations = [source.url for source in fallback_result.sources]

        return AgentAnswer(
            domain=self.domain,
            domain_label=self.domain_label,
            question=question,
            answer=fallback_result.answer,
            citations=citations,
            retrieved_chunks=expired_chunks,
            mode="web_fallback",
            sources=sources,
            fallback_reason=fallback_reason,
        )

    def _empty_answer(self, question: str) -> AgentAnswer:
        answer_text = f"Không tìm thấy dữ liệu phù hợp trong kho văn bản lĩnh vực {self.domain_label}."
        return AgentAnswer(
            domain=self.domain,
            domain_label=self.domain_label,
            question=question,
            answer=answer_text,
            citations=[],
            retrieved_chunks=[],
            mode="no_context",
            sources=[],
            fallback_reason="",
        )

    def _split_active_and_expired_chunks(
        self,
        results: list[SearchResult],
    ) -> tuple[list[SearchResult], list[SearchResult]]:
        active: list[SearchResult] = []
        expired: list[SearchResult] = []

        for result in results:
            status = self.metadata_registry.get_status_from_chunk_metadata(result.metadata)
            if status == "expired":
                expired.append(result)
            else:
                active.append(result)

        return active, expired

    def _build_prompt(self, question: str, context: str) -> str:
        return f"""
Bạn là chuyên gia pháp lý lĩnh vực {self.domain_label}.
Hãy trả lời câu hỏi dựa CHỈ dựa vào ngữ cảnh pháp lý được cung cấp.

Yêu cầu trả lời:
- Viết tiếng Việt, rõ ràng, mạch lạc.
- Nếu có thể, nêu căn cứ pháp lý theo mẫu: [Số hiệu văn bản - Điều/Khoản].
- Nếu ngữ cảnh không đủ, nói rõ phần còn thiếu, không bịa.

Câu hỏi:
{question}

Ngữ cảnh pháp lý:
{context}
""".strip()

    @staticmethod
    def _format_context(results: list[SearchResult]) -> str:
        blocks: list[str] = []
        for index, result in enumerate(results, start=1):
            metadata = result.metadata
            doc_number = metadata.get("document_number", "Không rõ số hiệu")
            article_hint = metadata.get("article_hint", "")
            source = metadata.get("file_name", "")
            legal_status = metadata.get("legal_status", "unknown")
            effective_date = metadata.get("effective_date", "")
            expiry_date = metadata.get("expiry_date", "")

            blocks.append(
                "\n".join(
                    [
                        f"[Trích đoạn {index}]",
                        f"Số hiệu: {doc_number}",
                        f"Nguồn: {source}",
                        f"Trạng thái hiệu lực: {legal_status}",
                        f"Ngày hiệu lực: {effective_date or 'N/A'}",
                        f"Ngày hết hiệu lực: {expiry_date or 'N/A'}",
                        f"Điều gợi ý: {article_hint or 'N/A'}",
                        f"Nội dung: {result.text}",
                    ]
                )
            )

        return "\n\n".join(blocks)

    @staticmethod
    def _extract_citations(results: list[SearchResult]) -> list[str]:
        citations: list[str] = []
        seen: set[str] = set()

        for result in results:
            metadata = result.metadata
            doc_number = metadata.get("document_number", "Không rõ số hiệu")
            article_hint = metadata.get("article_hint", "")

            citation = f"{doc_number}"
            if article_hint:
                citation = f"{citation} - {article_hint}"

            if citation not in seen:
                citations.append(citation)
                seen.add(citation)

        return citations

    @staticmethod
    def _collect_expired_documents(results: list[SearchResult]) -> list[str]:
        seen: set[str] = set()
        docs: list[str] = []
        for result in results:
            metadata = result.metadata
            doc_number = metadata.get("document_number", "Không rõ số hiệu")
            file_name = metadata.get("file_name", "")
            label = f"{doc_number} ({file_name})" if file_name else doc_number
            if label not in seen:
                docs.append(label)
                seen.add(label)
        return docs


@dataclass
class MergedAnswer:
    answer: str
    citations: list[str]
    sources: list[dict[str, str]]


class AnswerMerger:
    def __init__(self, llm: GeminiClient | None = None) -> None:
        self.llm = llm or GeminiClient()

    def merge(self, user_question: str, agent_answers: list[AgentAnswer]) -> MergedAnswer:
        if not agent_answers:
            return MergedAnswer(
                answer="Không có dữ liệu để trả lời câu hỏi.",
                citations=[],
                sources=[],
            )

        if len(agent_answers) == 1:
            answer = agent_answers[0]
            return MergedAnswer(
                answer=answer.answer,
                citations=answer.citations,
                sources=answer.sources,
            )

        prompt = self._build_prompt(user_question=user_question, agent_answers=agent_answers)
        merged_text = self.llm.generate(prompt=prompt, temperature=MERGE_TEMPERATURE)

        citations = self._collect_citations(agent_answers)
        sources = self._collect_sources(agent_answers)

        if citations:
            merged_text = (
                f"{merged_text.strip()}\n\nCăn cứ pháp lý / nguồn tham chiếu:\n"
                + "\n".join(f"- {item}" for item in citations)
            )

        return MergedAnswer(answer=merged_text.strip(), citations=citations, sources=sources)

    def _build_prompt(self, user_question: str, agent_answers: list[AgentAnswer]) -> str:
        blocks = []
        for answer in agent_answers:
            block = "\n".join(
                [
                    f"Lĩnh vực: {answer.domain_label} ({answer.domain})",
                    f"Câu hỏi con: {answer.question}",
                    f"Chế độ trả lời: {answer.mode}",
                    "Trả lời chuyên gia:",
                    answer.answer,
                    "Trích dẫn:",
                    "; ".join(answer.citations) if answer.citations else "(không có)",
                ]
            )
            blocks.append(block)

        joined_blocks = "\n\n---\n\n".join(blocks)

        return f"""
Bạn là bộ tổng hợp kết quả từ nhiều chuyên gia pháp lý.

Câu hỏi gốc của người dùng:
{user_question}

Các câu trả lời chuyên gia:
{joined_blocks}

Yêu cầu:
- Hợp nhất thành một câu trả lời duy nhất, mạch lạc.
- Không tự tạo căn cứ pháp lý mới ngoài nội dung đã có.
- Nếu có điểm chưa chắc chắn, nêu rõ.
- Giữ nguyên thông tin trích dẫn đã có.
- Viết tiếng Việt dễ hiểu.
""".strip()

    @staticmethod
    def _collect_citations(agent_answers: list[AgentAnswer]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []

        for answer in agent_answers:
            for citation in answer.citations:
                if citation not in seen:
                    ordered.append(citation)
                    seen.add(citation)

        return ordered

    @staticmethod
    def _collect_sources(agent_answers: list[AgentAnswer]) -> list[dict[str, str]]:
        seen_urls: set[str] = set()
        merged: list[dict[str, str]] = []

        for answer in agent_answers:
            for source in answer.sources:
                url = (source.get("url") or "").strip()
                if not url or url in seen_urls:
                    continue
                merged.append(source)
                seen_urls.add(url)

        return merged


@dataclass
class OrchestratorResult:
    user_question: str
    intent: IntentResult
    routed_sub_questions: list[tuple[str, str]]
    agent_answers: list[AgentAnswer]
    merged: MergedAnswer


class LegalOrchestrator:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        domains: list[str] | None = None,
        metadata_registry: LegalMetadataRegistry | None = None,
        notifier: RuntimeNotifier | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.domains = domains or list(DOMAIN_LABELS.keys())

        self.metadata_registry = metadata_registry or LegalMetadataRegistry()
        self.notifier = notifier or NullNotifier()
        self.expiry_monitor = ExpiryMonitor(
            registry=self.metadata_registry,
            notifier=self.notifier,
        )

        self.intent_agent = IntentAgent(supported_domains=self.domains)
        self.answer_merger = AnswerMerger()
        self.web_fallback = WebSearchFallback()

        self.domain_agents = {
            domain: DomainRAGAgent(
                domain=domain,
                vector_store=self.vector_store,
                metadata_registry=self.metadata_registry,
                web_fallback=self.web_fallback,
                notifier=self.notifier,
            )
            for domain in self.domains
        }

    @classmethod
    def from_defaults(cls, notifier: RuntimeNotifier | None = None) -> "LegalOrchestrator":
        vector_store = create_vector_store()
        return cls(vector_store=vector_store, notifier=notifier)

    def answer(self, user_question: str) -> OrchestratorResult:
        return asyncio.run(self.answer_async(user_question))

    async def answer_async(self, user_question: str) -> OrchestratorResult:
        self.expiry_monitor.run_if_due()

        intent_result = self.intent_agent.analyze(user_question)
        routed_sub_questions = self._route_sub_questions(
            original_question=user_question,
            intent_result=intent_result,
        )

        agent_answers = await self._dispatch_agents(routed_sub_questions)
        merged = self.answer_merger.merge(
            user_question=user_question,
            agent_answers=agent_answers,
        )

        return OrchestratorResult(
            user_question=user_question,
            intent=intent_result,
            routed_sub_questions=routed_sub_questions,
            agent_answers=agent_answers,
            merged=merged,
        )

    def _route_sub_questions(
        self,
        original_question: str,
        intent_result: IntentResult,
    ) -> list[tuple[str, str]]:
        candidates = intent_result.sub_questions or [
            SubQuestion(question=original_question.strip(), domain=None)
        ]

        routed: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        for sub_question in candidates:
            text = sub_question.question.strip() or original_question.strip()

            target_domains: list[str]
            if sub_question.domain and sub_question.domain in self.domains:
                target_domains = [sub_question.domain]
            elif intent_result.domains:
                target_domains = [d for d in intent_result.domains if d in self.domains]
            else:
                target_domains = list(self.domains)

            for domain in target_domains:
                pair = (domain, text)
                if pair not in seen:
                    routed.append(pair)
                    seen.add(pair)

        return routed

    async def _dispatch_agents(
        self,
        routed_sub_questions: list[tuple[str, str]],
    ) -> list[AgentAnswer]:
        tasks = [
            asyncio.to_thread(self.domain_agents[domain].answer, sub_question)
            for domain, sub_question in routed_sub_questions
            if domain in self.domain_agents
        ]

        if not tasks:
            return []

        results = await asyncio.gather(*tasks)
        return list(results)
