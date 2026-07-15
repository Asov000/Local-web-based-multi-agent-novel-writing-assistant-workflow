from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag import MemoryAgent, NovelRagSystem, RAGMessage, RAGOperation
from rag.config import RagConfig
from rag.repository import BookRepository


class RAGMessageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.system = NovelRagSystem(self.root)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _message(
        self,
        operations: list[RAGOperation],
        *,
        apply: bool,
        approved: bool = True,
    ) -> RAGMessage:
        return RAGMessage(
            sender="memory_agent",
            receiver="rag_processor",
            action="rag.operations.apply" if apply else "rag.operations.preview",
            book_id="book_rag_message",
            approval="confirmed" if approved else "pending",
            dry_run=not apply,
            operations=operations,
        )

    @staticmethod
    def _create(memory_id: str, content: str) -> RAGOperation:
        return RAGOperation(
            operation="create",
            created_memory_id=memory_id,
            store_type="chapter_memory",
            memory_type="event",
            content=content,
            source_chapter=1,
            raw_importance=0.8,
            reason="test create",
        )

    def _repository(self) -> BookRepository:
        return BookRepository(RagConfig(self.root), "book_rag_message")

    def test_preview_does_not_write_and_apply_requires_confirmation(self) -> None:
        operation = self._create("memory_preview", "preview memory")
        preview = self.system.process_rag_message(
            self._message([operation], apply=False)
        )
        self.assertEqual(preview.status, "ok")
        self.assertFalse(preview.payload["applied"])
        self.assertIsNone(self._repository().get("memory_preview"))

        rejected = self.system.process_rag_message(
            self._message([operation], apply=True, approved=False)
        )
        self.assertEqual(rejected.status, "need_user_input")
        self.assertIsNone(self._repository().get("memory_preview"))

    def test_create_update_soft_delete_and_rollback(self) -> None:
        created = self.system.process_rag_message(
            self._message(
                [self._create("memory_lifecycle", "old content")],
                apply=True,
            )
        )
        self.assertEqual(created.status, "ok")
        memory = self._repository().get("memory_lifecycle")
        self.assertIsNotNone(memory)
        self.assertEqual(memory.content, "old content")

        updated = self.system.process_rag_message(
            self._message(
                [
                    RAGOperation(
                        operation="update",
                        memory_id="memory_lifecycle",
                        expected_versions={"memory_lifecycle": 1},
                        new_content="new content",
                        metadata_patch={"source": "test"},
                        reason="test update",
                    )
                ],
                apply=True,
            )
        )
        self.assertEqual(updated.status, "ok")
        memory = self._repository().get("memory_lifecycle")
        self.assertEqual(memory.content, "new content")
        self.assertEqual(memory.version, 2)

        deleted = self.system.process_rag_message(
            self._message(
                [
                    RAGOperation(
                        operation="delete",
                        memory_id="memory_lifecycle",
                        expected_versions={"memory_lifecycle": 2},
                        reason="test delete",
                    )
                ],
                apply=True,
            )
        )
        self.assertEqual(deleted.status, "ok")
        snapshot_id = deleted.payload["snapshot_id"]
        self.assertEqual(self._repository().get("memory_lifecycle").status, "deleted")

        self.system.rollback_rag_message("book_rag_message", snapshot_id)
        restored = self._repository().get("memory_lifecycle")
        self.assertEqual(restored.status, "active")
        self.assertEqual(restored.content, "new content")
        self.assertEqual(restored.version, 2)

    def test_compress_archives_sources_and_can_rollback(self) -> None:
        setup = self.system.process_rag_message(
            self._message(
                [
                    self._create("memory_source_1", "first event"),
                    self._create("memory_source_2", "second event"),
                ],
                apply=True,
            )
        )
        self.assertEqual(setup.status, "ok")

        compressed = self.system.process_rag_message(
            self._message(
                [
                    RAGOperation(
                        operation="compress",
                        source_ids=["memory_source_1", "memory_source_2"],
                        created_memory_id="memory_summary",
                        expected_versions={
                            "memory_source_1": 1,
                            "memory_source_2": 1,
                        },
                        summary="two events happened in sequence",
                        reason="reduce retrieval context",
                    )
                ],
                apply=True,
            )
        )
        self.assertEqual(compressed.status, "ok")
        repository = self._repository()
        self.assertEqual(repository.get("memory_source_1").status, "archived")
        self.assertEqual(repository.get("memory_source_2").status, "archived")
        summary = repository.get("memory_summary")
        self.assertEqual(summary.memory_type, "compressed_summary")
        self.assertEqual(
            summary.metadata["compressed_from"],
            ["memory_source_1", "memory_source_2"],
        )

        self.system.rollback_rag_message(
            "book_rag_message",
            compressed.payload["snapshot_id"],
        )
        repository = self._repository()
        self.assertIsNone(repository.get("memory_summary"))
        self.assertEqual(repository.get("memory_source_1").status, "active")
        self.assertEqual(repository.get("memory_source_2").status, "active")

    def test_memory_agent_operation_plan_is_strict_and_preview_only(self) -> None:
        class PlannerClient:
            def invoke_json(self, system_prompt: str, payload: dict) -> dict:
                return {
                    "schema_version": "rag.message.v1",
                    "sender": "qwen_model",
                    "receiver": "memory_agent",
                    "message_type": "response",
                    "status": "ok",
                    "approval": "pending",
                    "dry_run": True,
                    "payload": {
                        "summary": "one proposed operation",
                        "operations": [
                            {
                                "operation": "create",
                                "created_memory_id": "memory_planned",
                                "store_type": "chapter_memory",
                                "memory_type": "event",
                                "content": "planned memory",
                                "raw_importance": 0.7,
                                "reason": "new event",
                            }
                        ],
                    },
                }

        agent = MemoryAgent(PlannerClient())
        planned = agent.handle_message(
            RAGMessage(
                sender="control_agent",
                receiver="memory_agent",
                action="rag.memory.operations.plan",
                book_id="book_rag_message",
                payload={"instruction": "add event", "memories": []},
            )
        )
        self.assertEqual(planned.status, "ok")
        self.assertEqual(planned.approval, "pending")
        self.assertTrue(planned.dry_run)
        self.assertEqual(planned.operations[0].operation, "create")
        self.assertEqual(
            planned.metadata["model_metadata"]["repaired_envelope_fields"],
            ["action", "operations_from_payload"],
        )

        preview = self.system.process_rag_message(
            RAGMessage(
                sender="memory_agent",
                receiver="rag_processor",
                action="rag.operations.preview",
                book_id="book_rag_message",
                operations=planned.operations,
            )
        )
        self.assertEqual(preview.status, "ok")
        self.assertIsNone(self._repository().get("memory_planned"))


if __name__ == "__main__":
    unittest.main()
