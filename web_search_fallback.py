from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from config import (
    TRUSTED_WEB_DOMAINS,
    WEB_SEARCH_BACKEND,
    WEB_SEARCH_MAX_RESULTS,
)
from llm_client import GeminiClient


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

        cited_lines = "\n".join(
            f"- {source.title}: {source.url}" for source in sources[:5]
        )
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
