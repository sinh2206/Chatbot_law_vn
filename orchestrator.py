from __future__ import annotations

import asyncio
from dataclasses import dataclass

from admin_notify import AdminNotifier
from config import DOMAIN_LABELS
from embedding_store import ChromaVectorStore, create_vector_store
from expiry_monitor import ExpiryMonitor
from intent_agent import IntentAgent, IntentResult, SubQuestion
from legal_metadata import LegalMetadataRegistry
from merge_answer import AnswerMerger, MergedAnswer
from rag_agent import AgentAnswer, DomainRAGAgent
from web_search_fallback import WebSearchFallback


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
        notifier: AdminNotifier | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.domains = domains or list(DOMAIN_LABELS.keys())

        self.metadata_registry = metadata_registry or LegalMetadataRegistry()
        self.notifier = notifier or AdminNotifier()
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
    def from_defaults(cls) -> "LegalOrchestrator":
        vector_store = create_vector_store()
        return cls(vector_store=vector_store)

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
