from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import RagConfig
from .memory_agent import MemoryAgent
from .memory_manager import MemoryManager
from .maintenance_coordinator import MaintenanceCoordinator
from .maintenance_schemas import AuditRunResult, AuditScope
from .message_processor import RAGMessageProcessor
from .qwen_judge import MemoryJudge
from .rag_message import RAGMessage
from .retriever import RagRetriever
from .schemas import ChapterReplacementResult, IngestResult, MemoryFact


class NovelRagSystem:
    def __init__(
        self,
        root_dir: str | Path = "rag_data",
        *,
        config: RagConfig | None = None,
        memory_agent: MemoryAgent | None = None,
        judge: MemoryJudge | None = None,
    ) -> None:
        self.config = config or RagConfig(root_dir=root_dir)
        self.memory_agent = memory_agent or MemoryAgent()
        self.manager = MemoryManager(
            self.config,
            memory_agent=self.memory_agent,
            judge=judge,
        )
        self.retriever = RagRetriever(self.config)
        self.maintenance = MaintenanceCoordinator(
            self.config,
            self.memory_agent,
        )
        self.message_processor = RAGMessageProcessor(self.config)

    def ingest(
        self,
        book_id: str,
        task_code: str,
        writer_result: dict[str, Any],
        *,
        chapter_id: int = 0,
    ) -> IngestResult:
        return self.manager.ingest_writer_result(
            book_id,
            task_code,
            writer_result,
            chapter_id=chapter_id,
        )

    def replace_chapter(
        self,
        book_id: str,
        task_code: str,
        writer_result: dict[str, Any],
        *,
        chapter_id: int,
        facts_override: list[MemoryFact] | None = None,
    ) -> ChapterReplacementResult:
        """Replace one chapter and rebuild every memory owned by that chapter."""
        return self.manager.replace_chapter(
            book_id,
            task_code,
            writer_result,
            chapter_id=chapter_id,
            facts_override=facts_override,
        )

    def retrieve_context(
        self,
        book_id: str,
        user_input: str,
        *,
        current_chapter: int,
        character_names: list[str] | None = None,
        item_names: list[str] | None = None,
        event_names: list[str] | None = None,
        token_budget: int | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        context = self.retriever.retrieve(
            book_id,
            user_input,
            current_chapter=current_chapter,
            character_names=character_names,
            item_names=item_names,
            event_names=event_names,
            token_budget=token_budget,
        )
        return context.model_dump()

    def build_writer_payload(
        self,
        book_id: str,
        user_input: str,
        *,
        current_chapter: int,
        token_budget: int | None = None,
    ) -> dict[str, Any]:
        return {
            "u": user_input.strip(),
            "c": self.retrieve_context(
                book_id,
                user_input,
                current_chapter=current_chapter,
                token_budget=token_budget,
            ),
        }

    def refresh_importance(self, book_id: str, current_chapter: int) -> None:
        self.manager.refresh_importance(book_id, current_chapter)

    def cleanup_deleted(self, book_id: str) -> int:
        return self.manager.cleanup_deleted(book_id)

    def audit_book_memories(
        self,
        book_id: str,
        *,
        apply: bool = False,
        scope: AuditScope | dict[str, Any] | None = None,
    ) -> AuditRunResult:
        return self.maintenance.run(book_id, apply=apply, scope=scope)

    def rollback_memory_audit(self, book_id: str, snapshot_id: str) -> None:
        self.maintenance.rollback(book_id, snapshot_id)

    def apply_saved_memory_audit(
        self,
        book_id: str,
        run_id: str,
    ) -> AuditRunResult:
        return self.maintenance.apply_saved_plan(book_id, run_id)

    def process_rag_message(self, message: RAGMessage) -> RAGMessage:
        return self.message_processor.handle(message)

    def rollback_rag_message(self, book_id: str, snapshot_id: str) -> None:
        self.message_processor.rollback(book_id, snapshot_id)
