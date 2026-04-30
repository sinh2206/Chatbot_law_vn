from __future__ import annotations

from dataclasses import dataclass

from config import MERGE_TEMPERATURE
from llm_client import GeminiClient
from rag_agent import AgentAnswer


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
