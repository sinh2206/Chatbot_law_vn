from __future__ import annotations

from dataclasses import dataclass

from admin_notify import AdminNotifier, FallbackAlertContext
from config import DOMAIN_LABELS, LLM_TEMPERATURE, TOP_K, WEB_FALLBACK_ENABLED
from embedding_store import ChromaVectorStore, SearchResult
from legal_metadata import LegalMetadataRegistry
from llm_client import GeminiClient
from web_search_fallback import WebSearchFallback


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
        notifier: AdminNotifier | None = None,
        enable_web_fallback: bool = WEB_FALLBACK_ENABLED,
    ) -> None:
        self.domain = domain
        self.domain_label = DOMAIN_LABELS.get(domain, domain)
        self.vector_store = vector_store
        self.llm = llm or GeminiClient()
        self.top_k = top_k
        self.metadata_registry = metadata_registry or LegalMetadataRegistry()
        self.web_fallback = web_fallback or WebSearchFallback(llm=self.llm)
        self.notifier = notifier or AdminNotifier()
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
        self, results: list[SearchResult]
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
