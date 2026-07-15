from __future__ import annotations

from .book_lock import BookDatabaseLock
from .config import RagConfig
from .maintenance_schemas import AuditCoverage, PatchPlan
from .patch_executor import PatchApplicationError, PatchExecutor
from .rag_message import RAGMessage


class RAGMessageProcessor:
    """The only boundary that turns RAGMessage operations into database writes."""

    agent_name = "rag_processor"

    def __init__(self, config: RagConfig) -> None:
        self.config = config
        self.executor = PatchExecutor(config)

    def handle(self, message: RAGMessage) -> RAGMessage:
        if message.message_type != "request":
            return self._error(message, "RAG processor only accepts request messages")
        if message.receiver != self.agent_name:
            return self._error(message, f"RAGMessage receiver must be {self.agent_name}")
        if message.action not in {"rag.operations.preview", "rag.operations.apply"}:
            return self._error(message, f"Unsupported RAG action: {message.action}")
        if not message.book_id:
            return self._error(message, "RAGMessage is missing book_id")
        if not message.operations:
            return self._error(message, "RAGMessage contains no operations")

        applying = message.action == "rag.operations.apply" and not message.dry_run
        if applying and message.approval != "confirmed":
            return message.response(
                sender=self.agent_name,
                action="rag.operations.result",
                status="need_user_input",
                error="Database mutations require approval=confirmed",
            )

        try:
            plan = self._build_plan(message)
            with BookDatabaseLock(self.config, message.book_id):
                if applying:
                    snapshot_id = self.executor.apply(plan)
                else:
                    self.executor.dry_run(plan)
                    snapshot_id = None
        except PatchApplicationError as exc:
            return self._error(
                message,
                str(exc),
                metadata={"snapshot_id": exc.snapshot_id, "rolled_back": True},
            )
        except Exception as exc:
            return self._error(message, str(exc))

        return message.response(
            sender=self.agent_name,
            action="rag.operations.result",
            payload={
                "validated": True,
                "applied": applying,
                "operation_ids": [item.operation_id for item in message.operations],
                "snapshot_id": snapshot_id,
            },
            metadata={"snapshot_id": snapshot_id} if snapshot_id else {},
        )

    def rollback(self, book_id: str, snapshot_id: str) -> None:
        with BookDatabaseLock(self.config, book_id):
            self.executor.rollback(book_id, snapshot_id)

    @staticmethod
    def _build_plan(message: RAGMessage) -> PatchPlan:
        operations = [item.to_audit_operation() for item in message.operations]
        referenced = sorted(
            {
                memory_id
                for operation in operations
                for memory_id in operation.referenced_memory_ids()
            }
        )
        coverage = AuditCoverage(
            total_memory_ids=referenced,
            assigned_memory_ids=referenced,
            reviewed_memory_ids=referenced,
        )
        return PatchPlan(
            book_id=message.book_id or "",
            run_id=message.task_id,
            operations=operations,
            coverage=coverage,
        )

    def _error(
        self,
        message: RAGMessage,
        error: str,
        *,
        metadata: dict | None = None,
    ) -> RAGMessage:
        return message.response(
            sender=self.agent_name,
            action="rag.operations.result",
            status="error",
            error=error,
            metadata=metadata,
        )
