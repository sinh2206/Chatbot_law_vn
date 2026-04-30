from __future__ import annotations

import logging

from orchestrator import LegalOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def main() -> None:
    orchestrator = LegalOrchestrator.from_defaults()

    if orchestrator.vector_store.count() == 0:
        print("Vector store đang trống. Hãy chạy: python build_store.py")
        return

    print("Chatbot pháp luật đã sẵn sàng. Gõ 'exit' để thoát.")

    while True:
        user_question = input("\nBạn: ").strip()
        if not user_question:
            continue
        if user_question.lower() in {"exit", "quit", "q"}:
            print("Kết thúc phiên làm việc.")
            break

        try:
            result = orchestrator.answer(user_question)
        except Exception as exc:
            print(f"Lỗi khi xử lý câu hỏi: {exc}")
            continue

        domains = result.intent.domains or ["(không xác định)"]
        print(f"\n[Intent] Lĩnh vực nhận diện: {', '.join(domains)}")
        fallback_domains = [
            answer.domain_label
            for answer in result.agent_answers
            if answer.mode == "web_fallback"
        ]
        if fallback_domains:
            print(f"[Fallback] Đã dùng web search cho: {', '.join(fallback_domains)}")
        print("\nTrợ lý:")
        print(result.merged.answer)


if __name__ == "__main__":
    main()
