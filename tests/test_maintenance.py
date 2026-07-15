from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag import MemoryAgent, NovelRagSystem
from rag.audit_scanner import AuditScanner
from rag.book_lock import BookDatabaseLock, BookLockError
from rag.config import RagConfig
from rag.entity_partition import EntityGraphPartitioner
from rag.maintenance_schemas import (
    AuditCoverage,
    AuditFinding,
    AuditOperation,
    AuditScope,
    PatchPlan,
)
from rag.patch_executor import (
    PatchApplicationError,
    PatchExecutor,
    PatchValidationError,
)
from rag.repository import BookRepository
from rag.retriever import estimate_tokens
from rag.schemas import AtomicMemory
from rag.snapshot_manager import SnapshotError


class CompleteAuditClient:
    def __init__(self) -> None:
        self.audit_calls = 0
        self.cross_audit_calls = 0

    def invoke_json(self, system_prompt: str, payload: dict) -> dict:
        payload = payload.get("payload", payload)
        if "primary_memory_ids" in payload:
            self.audit_calls += 1
            return {
                "packet_id": payload["packet_id"],
                "reviewed_memory_ids": payload["primary_memory_ids"],
                "findings": [],
                "operations": [],
            }
        if "candidates" in payload:
            self.cross_audit_calls += 1
            return {
                "packet_id": payload["packet_id"],
                "reviewed_candidate_ids": [
                    candidate["candidate_id"]
                    for candidate in payload["candidates"]
                ],
                "findings": [],
                "operations": [],
            }
        return {
            "reviewed_finding_ids": [
                finding["finding_id"] for finding in payload.get("findings", [])
            ],
            "findings": [],
            "operations": payload.get("proposed_operations", []),
        }


class RetryOmissionClient(CompleteAuditClient):
    def invoke_json(self, system_prompt: str, payload: dict) -> dict:
        payload = payload.get("payload", payload)
        if "primary_memory_ids" not in payload:
            return super().invoke_json(system_prompt, payload)
        self.audit_calls += 1
        primary = payload["primary_memory_ids"]
        reviewed = primary if payload.get("retry_for_memory_ids") else primary[:-1]
        return {
            "packet_id": payload["packet_id"],
            "reviewed_memory_ids": reviewed,
            "findings": [],
            "operations": [],
        }


class PersistentOmissionClient(CompleteAuditClient):
    def invoke_json(self, system_prompt: str, payload: dict) -> dict:
        payload = payload.get("payload", payload)
        if "primary_memory_ids" not in payload:
            return super().invoke_json(system_prompt, payload)
        self.audit_calls += 1
        return {
            "packet_id": payload["packet_id"],
            "reviewed_memory_ids": [],
            "findings": [],
            "operations": [],
        }


class FindingAuditClient(CompleteAuditClient):
    def __init__(self, *, omit_reconcile: bool = False) -> None:
        super().__init__()
        self.reconcile_calls = 0
        self.omit_reconcile = omit_reconcile

    def invoke_json(self, system_prompt: str, payload: dict) -> dict:
        payload = payload.get("payload", payload)
        if "primary_memory_ids" in payload:
            self.audit_calls += 1
            primary = payload["primary_memory_ids"]
            findings = []
            if primary:
                findings.append(
                    {
                        "code": "cross_entity_check",
                        "severity": "warning",
                        "summary": "需要跨实体确认",
                        "memory_ids": [primary[0]],
                        "evidence": ["共享事件"],
                    }
                )
            return {
                "packet_id": payload["packet_id"],
                "reviewed_memory_ids": primary,
                "findings": findings,
                "operations": [],
            }
        if "candidates" in payload:
            return super().invoke_json(system_prompt, payload)
        self.reconcile_calls += 1
        reviewed = [] if self.omit_reconcile else [
            finding["finding_id"] for finding in payload.get("findings", [])
        ]
        return {
            "reviewed_finding_ids": reviewed,
            "findings": [],
            "operations": [],
        }


class PersistentCrossOmissionClient(CompleteAuditClient):
    def invoke_json(self, system_prompt: str, payload: dict) -> dict:
        payload = payload.get("payload", payload)
        if "candidates" not in payload:
            return super().invoke_json(system_prompt, payload)
        self.cross_audit_calls += 1
        return {
            "packet_id": payload["packet_id"],
            "reviewed_candidate_ids": [],
            "findings": [],
            "operations": [],
        }


class DriftedHealthyAuditClient(CompleteAuditClient):
    def invoke_json(self, system_prompt: str, payload: dict) -> dict:
        payload = payload.get("payload", payload)
        if "primary_memory_ids" not in payload:
            return super().invoke_json(system_prompt, payload)
        self.audit_calls += 1
        findings = [
            {
                "memory_id": memory_id,
                "status": "consistent",
                "notes": "Memory is internally consistent and needs no change.",
            }
            for memory_id in payload["primary_memory_ids"]
        ]
        operations = [
            {
                "operation_type": "no_op",
                "target_memory_id": memory_id,
                "expected_versions": ["1"],
            }
            for memory_id in payload["primary_memory_ids"]
        ]
        return {
            "packet_id": payload["packet_id"],
            "reviewed_memory_ids": payload["primary_memory_ids"],
            "findings": findings,
            "operations": operations,
        }


class InvalidThenValidAuditSchemaClient(CompleteAuditClient):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[dict] = []
        self.prompts: list[str] = []

    def invoke_json(self, system_prompt: str, payload: dict) -> dict:
        self.requests.append(payload)
        self.prompts.append(system_prompt)
        payload = payload.get("payload", payload)
        if "primary_memory_ids" not in payload:
            return super().invoke_json(system_prompt, payload)
        self.audit_calls += 1
        if self.audit_calls == 1:
            memory_id = payload["primary_memory_ids"][0]
            return {
                "packet_id": payload["packet_id"],
                "reviewed_memory_ids": payload["primary_memory_ids"],
                "findings": [],
                "operations": [
                    {
                        "operation_type": "archive",
                        "target_memory_id": memory_id,
                        "expected_versions": {memory_id: 1},
                    }
                ],
            }
        return {
            "packet_id": payload["packet_id"],
            "reviewed_memory_ids": payload["primary_memory_ids"],
            "findings": [],
            "operations": [],
        }


class CoverageFeedbackAuditClient(CompleteAuditClient):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[dict] = []
        self.prompts: list[str] = []

    def invoke_json(self, system_prompt: str, payload: dict) -> dict:
        self.requests.append(payload)
        self.prompts.append(system_prompt)
        business_payload = payload.get("payload", payload)
        if "primary_memory_ids" not in business_payload:
            return super().invoke_json(system_prompt, payload)
        self.audit_calls += 1
        primary = business_payload["primary_memory_ids"]
        feedback = (payload.get("metadata") or {}).get("validation_feedback")
        reviewed = primary if feedback else primary[:-1]
        return {
            "packet_id": business_payload["packet_id"],
            "reviewed_memory_ids": reviewed,
            "findings": [],
            "operations": [],
        }


class ScopedComparisonClient(CompleteAuditClient):
    def __init__(self, *, fail_candidate_suffix: str = "") -> None:
        super().__init__()
        self.comparison_calls = 0
        self.comparison_payloads: list[dict] = []
        self.fail_candidate_suffix = fail_candidate_suffix

    def invoke_json(self, system_prompt: str, payload: dict) -> dict:
        business_payload = payload.get("payload", payload)
        if "query_memory_id" not in business_payload:
            return super().invoke_json(system_prompt, payload)
        self.comparison_calls += 1
        self.comparison_payloads.append(business_payload)
        reviewed = list(business_payload["candidate_memory_ids"])
        if self.fail_candidate_suffix and any(
            memory_id.endswith(self.fail_candidate_suffix) for memory_id in reviewed
        ):
            reviewed = reviewed[:-1]
        return {
            "batch_id": business_payload["batch_id"],
            "query_memory_id": business_payload["query_memory_id"],
            "reviewed_candidate_ids": reviewed,
            "findings": [],
            "operations": [],
        }


def make_memory(
    memory_id: str,
    book_id: str,
    store_type: str,
    memory_type: str,
    content: str,
    *,
    chapter: int = 1,
    status: str = "active",
    entity_name: str | None = None,
    field: str | None = None,
    is_current: bool = True,
    hook_status: str | None = None,
    mention_count: int = 1,
) -> AtomicMemory:
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return AtomicMemory(
        memory_id=memory_id,
        book_id=book_id,
        store_type=store_type,
        memory_type=memory_type,
        content=content,
        source_chapter=chapter,
        last_mentioned_chapter=chapter,
        mention_count=mention_count,
        raw_importance=0.8,
        effective_importance=0.8,
        content_hash=content_hash,
        status=status,
        entity_name=entity_name,
        field=field,
        is_current=is_current,
        hook_status=hook_status,
    )


def complete_coverage(memory_ids: list[str]) -> AuditCoverage:
    return AuditCoverage(
        total_memory_ids=memory_ids,
        assigned_memory_ids=memory_ids,
        reviewed_memory_ids=memory_ids,
    )


class MaintenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.config = RagConfig(self.root)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _repository(self, book_id: str = "book_a") -> BookRepository:
        return BookRepository(self.config, book_id)

    def _save(self, repository: BookRepository, memory: AtomicMemory) -> None:
        repository.store(memory.store_type).save(memory)

    def test_scoped_chapter_audit_uses_one_query_and_at_most_nine_candidates(self) -> None:
        repository = self._repository()
        for chapter_id in range(1, 14):
            self._save(
                repository,
                make_memory(
                    f"chapter_scope_{chapter_id}",
                    "book_a",
                    "chapter_memory",
                    "character_state",
                    f"林舟在第{chapter_id}章的目标发生变化",
                    chapter=chapter_id,
                    entity_name="林舟",
                    field="goal",
                ),
            )
        client = ScopedComparisonClient()
        system = NovelRagSystem(
            config=self.config,
            memory_agent=MemoryAgent(client),
        )

        result = system.audit_book_memories(
            "book_a",
            scope={"mode": "chapters", "chapter_ids": [1]},
        )

        self.assertTrue(result.coverage.complete)
        self.assertEqual(result.coverage.total_memory_ids, ["chapter_scope_1"])
        self.assertEqual(result.semantic_candidate_count, 12)
        self.assertEqual(result.semantic_candidate_reviewed_count, 12)
        self.assertEqual(result.comparison_batch_count, 2)
        self.assertEqual(client.comparison_calls, 2)
        self.assertTrue(
            all(
                len(payload["candidates"]) <= 9
                and len(payload["candidates"]) + 1 <= 10
                for payload in client.comparison_payloads
            )
        )

    def test_scoped_global_audit_only_uses_canon_memories(self) -> None:
        repository = self._repository()
        for memory in (
            make_memory(
                "canon_scope_1",
                "book_a",
                "canon_memory",
                "world_rule",
                "月族血脉可以开启青铜门",
                entity_name="月族",
                field="door_rule",
            ),
            make_memory(
                "canon_scope_2",
                "book_a",
                "canon_memory",
                "world_rule",
                "只有月族直系血脉能够开启青铜门",
                chapter=2,
                entity_name="月族",
                field="door_rule",
            ),
            make_memory(
                "chapter_outside_scope",
                "book_a",
                "chapter_memory",
                "event",
                "林舟抵达青铜门",
                chapter=2,
                entity_name="林舟",
                field="location",
            ),
        ):
            self._save(repository, memory)
        client = ScopedComparisonClient()
        system = NovelRagSystem(
            config=self.config,
            memory_agent=MemoryAgent(client),
        )

        result = system.audit_book_memories(
            "book_a",
            scope={"mode": "global", "chapter_ids": []},
        )

        self.assertCountEqual(
            result.coverage.total_memory_ids,
            ["canon_scope_1", "canon_scope_2"],
        )
        self.assertEqual(result.semantic_candidate_count, 1)
        self.assertEqual(client.comparison_calls, 1)
        compared_ids = {
            client.comparison_payloads[0]["query_memory_id"],
            *client.comparison_payloads[0]["candidate_memory_ids"],
        }
        self.assertEqual(compared_ids, {"canon_scope_1", "canon_scope_2"})

    def test_scoped_batch_failure_preserves_partial_result_and_blocks_apply(self) -> None:
        repository = self._repository()
        for chapter_id in range(1, 12):
            self._save(
                repository,
                make_memory(
                    f"partial_scope_{chapter_id}",
                    "book_a",
                    "chapter_memory",
                    "character_state",
                    f"陈玥在第{chapter_id}章的身份线索",
                    chapter=chapter_id,
                    entity_name="陈玥",
                    field="identity",
                ),
            )
        client = ScopedComparisonClient(fail_candidate_suffix="11")
        system = NovelRagSystem(
            config=self.config,
            memory_agent=MemoryAgent(client),
        )

        result = system.audit_book_memories(
            "book_a",
            scope={"mode": "chapters", "chapter_ids": [1]},
        )

        self.assertFalse(result.coverage.complete)
        self.assertFalse(result.semantic_candidate_complete)
        self.assertGreater(result.semantic_candidate_reviewed_count, 0)
        self.assertLess(
            result.semantic_candidate_reviewed_count,
            result.semantic_candidate_count,
        )
        self.assertIn("comparison_incomplete", result.blocking_issue_ids)
        self.assertTrue(result.validation_errors)

    def test_chapter_scope_rejects_more_than_five_chapters(self) -> None:
        with self.assertRaises(ValueError):
            AuditScope.model_validate(
                {"mode": "chapters", "chapter_ids": [1, 2, 3, 4, 5, 6]}
            )

    def test_audit_finding_normalizes_structured_evidence_to_text(self) -> None:
        finding = AuditFinding.model_validate(
            {
                "code": "duplicate_memory",
                "summary": "内容重复",
                "memory_ids": ["memory_a", "memory_b"],
                "evidence": [
                    {"memory_id": "memory_a", "reason": "exact_content_match"}
                ],
            }
        )
        self.assertEqual(len(finding.evidence), 1)
        self.assertIn('"memory_id": "memory_a"', finding.evidence[0])

    def test_scanner_and_partitioner_cover_each_memory_exactly_once(self) -> None:
        repository = self._repository()
        first = make_memory(
            "chapter_1", "book_a", "chapter_memory", "event", "林舟进入遗迹"
        )
        second = make_memory(
            "chapter_2", "book_a", "chapter_memory", "event", "顾清得到钥匙"
        )
        self._save(repository, first)
        self._save(repository, second)
        char_id = repository.index.resolve_or_create("character", "林舟")
        item_id = repository.index.resolve_or_create("item", "青铜钥匙")
        repository.index.link(char_id, first.memory_id, first.store_type, "subject", 1)
        repository.index.link(item_id, first.memory_id, first.store_type, "related", 1)
        repository.index.link(item_id, second.memory_id, second.store_type, "subject", 1)

        scan = AuditScanner().scan(repository)
        partitioner = EntityGraphPartitioner(
            max_primary_memories=1,
            max_packet_tokens=1200,
            max_global_tokens=300,
        )
        owners = partitioner.assign_owners(scan)
        packets = partitioner.build_packets(scan, owners)
        assigned = [memory_id for packet in packets for memory_id in packet.primary_memory_ids]
        full_record_ids = [
            str(record["memory_id"])
            for packet in packets
            for record in packet.memories
        ]
        self.assertCountEqual(assigned, scan.all_memory_ids)
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertCountEqual(full_record_ids, scan.all_memory_ids)
        self.assertEqual(len(full_record_ids), len(set(full_record_ids)))
        self.assertEqual(owners[first.memory_id], char_id)
        self.assertTrue(
            all(
                record["role"] == "primary"
                for packet in packets
                for record in packet.memories
            )
        )
        self.assertTrue(any(packet.related_memory_refs for packet in packets))
        self.assertTrue(
            any(packet.related_heads for packet in packets),
            "共享记忆应该把其他实体作为关联上下文保留下来",
        )
        for packet in packets:
            serialized = json.dumps(packet.model_dump(mode="json"), ensure_ascii=False)
            self.assertLessEqual(estimate_tokens(serialized), 1400)

    def test_dense_index_graph_keeps_full_memory_payload_unique(self) -> None:
        repository = self._repository()
        head_ids = [
            repository.index.resolve_or_create("character", f"角色{index}")
            for index in range(6)
        ]
        for index in range(48):
            memory = make_memory(
                f"event_{index:03d}",
                "book_a",
                "chapter_memory",
                "event",
                f"角色{index % 6}在第{index + 1}幕完成事件{index}",
                chapter=index + 1,
            )
            self._save(repository, memory)
            owner_index = index % len(head_ids)
            for head_index, head_id in enumerate(head_ids):
                repository.index.link(
                    head_id,
                    memory.memory_id,
                    memory.store_type,
                    "subject" if head_index == owner_index else "related",
                    memory.source_chapter,
                )

        scan = AuditScanner().scan(repository)
        partitioner = EntityGraphPartitioner(
            max_primary_memories=4,
            max_context_memories=5,
            max_packet_tokens=2600,
            max_global_tokens=300,
        )
        owners = partitioner.assign_owners(scan)
        packets = partitioner.build_packets(scan, owners)
        full_record_ids = [
            str(record["memory_id"])
            for packet in packets
            for record in packet.memories
        ]

        self.assertEqual(len(full_record_ids), 48)
        self.assertEqual(len(set(full_record_ids)), 48)
        self.assertLessEqual(
            sum(len(packet.related_memory_refs) for packet in packets),
            len(packets) * 5,
        )
        self.assertTrue(
            all(
                len(packet.context_memory_ids) == len(packet.related_memory_refs)
                for packet in packets
            )
        )

    def test_best_fit_bundles_small_owner_partitions(self) -> None:
        repository = self._repository()
        expected_owners: dict[str, str] = {}
        for index in range(8):
            memory = make_memory(
                f"small_event_{index}",
                "book_a",
                "chapter_memory",
                "event",
                f"独立事件{index}",
                chapter=index + 1,
            )
            self._save(repository, memory)
            head_id = repository.index.resolve_or_create("event", f"事件索引{index}")
            repository.index.link(
                head_id,
                memory.memory_id,
                memory.store_type,
                "subject",
                memory.source_chapter,
            )
            expected_owners[memory.memory_id] = head_id

        scan = AuditScanner().scan(repository)
        partitioner = EntityGraphPartitioner(
            max_primary_memories=4,
            max_context_memories=0,
            max_packet_tokens=2400,
            max_global_tokens=300,
        )
        owners = partitioner.assign_owners(scan)
        packets = partitioner.build_packets(scan, owners)

        self.assertEqual(owners, expected_owners)
        self.assertEqual(len(packets), 2)
        self.assertTrue(
            all(packet.focus_head["head_type"] == "owner_bundle" for packet in packets)
        )
        for packet in packets:
            for record in packet.memories:
                memory_id = str(record["memory_id"])
                self.assertEqual(record["owner_head_id"], expected_owners[memory_id])

    def test_sparse_cross_index_candidates_keep_timeline_relationships(self) -> None:
        repository = self._repository()
        old_state = make_memory(
            "state_goal_old",
            "book_a",
            "state_timeline_memory",
            "character_state",
            "林舟的目标是寻找妹妹",
            chapter=1,
            entity_name="林舟",
            field="goal",
            is_current=False,
        )
        new_state = make_memory(
            "state_goal_new",
            "book_a",
            "state_timeline_memory",
            "character_state",
            "林舟的目标改为阻止零频实验",
            chapter=2,
            entity_name="林舟",
            field="goal",
        )
        self._save(repository, old_state)
        self._save(repository, new_state)
        old_head = repository.index.resolve_or_create("event", "寻找妹妹")
        new_head = repository.index.resolve_or_create("event", "零频实验")
        repository.index.link(
            old_head, old_state.memory_id, old_state.store_type, "subject", 1
        )
        repository.index.link(
            new_head, new_state.memory_id, new_state.store_type, "subject", 2
        )

        scan = AuditScanner().scan(repository)
        partitioner = EntityGraphPartitioner(max_cross_candidates_per_memory=1)
        owners = partitioner.assign_owners(scan)
        packets = partitioner.build_cross_check_packets(scan, owners)
        candidates = [candidate for packet in packets for candidate in packet.candidates]

        self.assertEqual(len(candidates), 1)
        self.assertCountEqual(
            candidates[0].memory_ids,
            [old_state.memory_id, new_state.memory_id],
        )
        self.assertIn("same_entity_field_timeline", candidates[0].reason_codes)

    def test_apply_correctly_repairs_deterministic_issues_and_can_rollback(self) -> None:
        repository = self._repository()
        duplicate_target = make_memory(
            "chapter_target",
            "book_a",
            "chapter_memory",
            "event",
            "林舟打开青铜门",
            mention_count=3,
        )
        duplicate_source = make_memory(
            "chapter_source",
            "book_a",
            "chapter_memory",
            "event",
            "林舟打开青铜门",
        )
        old_state = make_memory(
            "state_old",
            "book_a",
            "state_timeline_memory",
            "character_state",
            "寻找妹妹",
            chapter=1,
            entity_name="林舟",
            field="goal",
        )
        new_state = make_memory(
            "state_new",
            "book_a",
            "state_timeline_memory",
            "character_state",
            "进入青铜门",
            chapter=2,
            entity_name="林舟",
            field="goal",
        )
        open_hook = make_memory(
            "hook_deleted",
            "book_a",
            "relation_hook_memory",
            "foreshadowing_open",
            "门后传来妹妹的声音",
            status="deleted",
            hook_status="open",
        )
        canon = make_memory(
            "canon_archived",
            "book_a",
            "canon_memory",
            "world_rule",
            "只有王族血脉可以开启青铜门",
            status="archived",
        )
        for memory in (
            duplicate_target,
            duplicate_source,
            old_state,
            new_state,
            open_hook,
            canon,
        ):
            self._save(repository, memory)

        system = NovelRagSystem(
            config=self.config,
            memory_agent=MemoryAgent(CompleteAuditClient()),
        )
        result = system.audit_book_memories("book_a", apply=True)
        self.assertTrue(result.applied)
        self.assertTrue(result.coverage.complete)
        self.assertIsNotNone(result.snapshot_id)

        repaired = self._repository()
        self.assertEqual(repaired.get("chapter_source").status, "deleted")
        self.assertEqual(repaired.get("chapter_target").mention_count, 4)
        self.assertFalse(repaired.get("state_old").is_current)
        self.assertEqual(repaired.get("state_old").status, "archived")
        self.assertTrue(repaired.get("state_new").is_current)
        self.assertEqual(repaired.get("hook_deleted").status, "active")
        self.assertEqual(repaired.get("canon_archived").status, "archived")

        system.rollback_memory_audit("book_a", result.snapshot_id or "")
        restored = self._repository()
        self.assertEqual(restored.get("chapter_source").status, "active")
        self.assertEqual(restored.get("chapter_target").mention_count, 3)
        self.assertTrue(restored.get("state_old").is_current)
        self.assertEqual(restored.get("hook_deleted").status, "deleted")
        self.assertEqual(restored.get("canon_archived").status, "archived")

    def test_apply_merges_exact_duplicate_canon_memories(self) -> None:
        repository = self._repository()
        target = make_memory(
            "canon_target",
            "book_a",
            "canon_memory",
            "material_character",
            "陈玥（角色）\n身份：友善的深渊恶魔",
            entity_name="陈玥",
            field="material_profile",
            mention_count=3,
        )
        source = make_memory(
            "canon_source",
            "book_a",
            "canon_memory",
            "material_character",
            "陈玥（角色）\n身份：友善的深渊恶魔",
            entity_name="陈玥",
            field="material_profile",
        )
        self._save(repository, target)
        self._save(repository, source)

        system = NovelRagSystem(
            config=self.config,
            memory_agent=MemoryAgent(CompleteAuditClient()),
        )
        result = system.audit_book_memories("book_a", apply=True)

        self.assertTrue(result.applied)
        repaired = self._repository()
        self.assertEqual(repaired.get("canon_target").status, "active")
        self.assertEqual(repaired.get("canon_target").mention_count, 4)
        self.assertEqual(repaired.get("canon_source").status, "deleted")
        self.assertIn(
            "canon_source",
            repaired.get("canon_target").metadata["merged_memory_ids"],
        )

    def test_patch_executor_allows_audited_canon_archive(self) -> None:
        repository = self._repository()
        canon = make_memory(
            "canon_obsolete",
            "book_a",
            "canon_memory",
            "world_rule",
            "旧版规则",
        )
        self._save(repository, canon)
        plan = PatchPlan(
            book_id="book_a",
            run_id="audit_canon_archive",
            operations=[
                AuditOperation(
                    operation="archive",
                    memory_id=canon.memory_id,
                    expected_versions={canon.memory_id: canon.version},
                    reason="审计确认该核心设定已经失效",
                )
            ],
            coverage=AuditCoverage(
                total_memory_ids=[canon.memory_id],
                assigned_memory_ids=[canon.memory_id],
                reviewed_memory_ids=[canon.memory_id],
            ),
        )

        PatchExecutor(self.config).apply(plan)

        self.assertEqual(self._repository().get(canon.memory_id).status, "archived")

    def test_saved_dry_run_plan_applies_without_repeating_model_audit(self) -> None:
        repository = self._repository()
        hook = make_memory(
            "hook_saved_plan",
            "book_a",
            "relation_hook_memory",
            "foreshadowing_open",
            "门后传来妹妹的声音",
            status="deleted",
            hook_status="open",
        )
        self._save(repository, hook)
        client = CompleteAuditClient()
        system = NovelRagSystem(
            config=self.config,
            memory_agent=MemoryAgent(client),
        )

        dry = system.audit_book_memories("book_a", apply=False)
        calls_after_dry = client.audit_calls
        self.assertGreater(dry.operation_count, 0)
        applied = system.apply_saved_memory_audit("book_a", dry.run_id)

        self.assertTrue(applied.applied)
        self.assertIsNotNone(applied.snapshot_id)
        self.assertEqual(client.audit_calls, calls_after_dry)
        self.assertEqual(self._repository().get(hook.memory_id).status, "active")

    def test_model_omission_is_retried_and_coverage_becomes_complete(self) -> None:
        repository = self._repository()
        for index in range(3):
            self._save(
                repository,
                make_memory(
                    f"chapter_{index}",
                    "book_a",
                    "chapter_memory",
                    "event",
                    f"事件{index}",
                ),
            )
        client = RetryOmissionClient()
        system = NovelRagSystem(
            config=self.config,
            memory_agent=MemoryAgent(client),
        )
        result = system.audit_book_memories("book_a", apply=False)
        self.assertTrue(result.coverage.complete)
        self.assertGreaterEqual(client.audit_calls, 2)

    def test_coverage_error_is_sent_back_in_chinese_rag_message(self) -> None:
        repository = self._repository()
        for memory_id, content in (
            ("chapter_feedback_1", "林舟进入遗迹"),
            ("chapter_feedback_2", "顾清留守城门"),
        ):
            self._save(
                repository,
                make_memory(
                    memory_id,
                    "book_a",
                    "chapter_memory",
                    "event",
                    content,
                ),
            )
        client = CoverageFeedbackAuditClient()
        system = NovelRagSystem(
            config=self.config,
            memory_agent=MemoryAgent(client),
        )

        result = system.audit_book_memories("book_a", apply=False)

        self.assertTrue(result.coverage.complete)
        self.assertEqual(client.audit_calls, 2)
        self.assertIn("最高优先级：覆盖要求", client.prompts[0])
        self.assertIn("只能返回一个完整的 rag.message.v1 JSON 对象", client.prompts[0])
        self.assertNotIn("Audit only the supplied", client.prompts[0])
        self.assertIn(
            '"reviewed_memory_ids": ["chapter_feedback_1", "chapter_feedback_2"]',
            client.prompts[0],
        )
        feedback = client.requests[1]["metadata"]["validation_feedback"]
        self.assertEqual(feedback["error_type"], "audit_coverage_omission")
        self.assertEqual(feedback["missing_ids"], ["chapter_feedback_2"])
        self.assertIn("逐条审阅", feedback["repair_instruction"])

    def test_persistent_model_omission_blocks_apply(self) -> None:
        repository = self._repository()
        self._save(
            repository,
            make_memory(
                "chapter_1", "book_a", "chapter_memory", "event", "林舟进入遗迹"
            ),
        )
        system = NovelRagSystem(
            config=self.config,
            memory_agent=MemoryAgent(PersistentOmissionClient()),
        )
        dry_result = system.audit_book_memories("book_a", apply=False)
        self.assertFalse(dry_result.coverage.complete)
        self.assertEqual(dry_result.coverage.unreviewed_memory_ids, ["chapter_1"])
        with self.assertRaises(PatchValidationError):
            system.audit_book_memories("book_a", apply=True)
        self.assertEqual(self._repository().get("chapter_1").status, "active")

    def test_cross_candidate_coverage_is_recorded(self) -> None:
        repository = self._repository()
        for memory in (
            make_memory(
                "state_old",
                "book_a",
                "state_timeline_memory",
                "character_state",
                "林舟仍在寻找妹妹",
                chapter=1,
                entity_name="林舟",
                field="goal",
                is_current=False,
            ),
            make_memory(
                "state_new",
                "book_a",
                "state_timeline_memory",
                "character_state",
                "林舟开始追查零频实验",
                chapter=2,
                entity_name="林舟",
                field="goal",
            ),
        ):
            self._save(repository, memory)
        client = CompleteAuditClient()
        system = NovelRagSystem(
            config=self.config,
            memory_agent=MemoryAgent(client),
        )

        result = system.audit_book_memories("book_a", apply=False)

        self.assertGreaterEqual(result.semantic_candidate_count, 1)
        self.assertEqual(
            result.semantic_candidate_reviewed_count,
            result.semantic_candidate_count,
        )
        self.assertTrue(result.semantic_candidate_complete)
        self.assertGreaterEqual(client.cross_audit_calls, 1)

    def test_persistent_cross_candidate_omission_blocks_apply(self) -> None:
        repository = self._repository()
        for memory in (
            make_memory(
                "state_old",
                "book_a",
                "state_timeline_memory",
                "character_state",
                "林舟仍在寻找妹妹",
                chapter=1,
                entity_name="林舟",
                field="goal",
                is_current=False,
            ),
            make_memory(
                "state_new",
                "book_a",
                "state_timeline_memory",
                "character_state",
                "林舟开始追查零频实验",
                chapter=2,
                entity_name="林舟",
                field="goal",
            ),
        ):
            self._save(repository, memory)
        client = PersistentCrossOmissionClient()
        system = NovelRagSystem(
            config=self.config,
            memory_agent=MemoryAgent(client),
        )

        result = system.audit_book_memories("book_a", apply=False)

        self.assertFalse(result.semantic_candidate_complete)
        self.assertIn("cross_check_incomplete", result.blocking_issue_ids)
        self.assertTrue(result.validation_errors)
        self.assertGreaterEqual(client.cross_audit_calls, 2)

    def test_qwen_healthy_field_drift_is_filtered_without_blocking_audit(self) -> None:
        repository = self._repository()
        self._save(
            repository,
            make_memory(
                "canon_healthy",
                "book_a",
                "canon_memory",
                "world_rule",
                "亡者不可复活",
            ),
        )
        client = DriftedHealthyAuditClient()
        system = NovelRagSystem(
            config=self.config,
            memory_agent=MemoryAgent(client),
        )

        result = system.audit_book_memories("book_a", apply=False)

        self.assertTrue(result.coverage.complete)
        self.assertEqual(result.model_finding_count, 0)
        self.assertEqual(result.operation_count, 0)
        self.assertEqual(client.audit_calls, 1)

    def test_invalid_audit_operation_schema_is_retried_before_failure(self) -> None:
        repository = self._repository()
        self._save(
            repository,
            make_memory(
                "chapter_retry_schema",
                "book_a",
                "chapter_memory",
                "event",
                "林舟进入遗迹",
            ),
        )
        client = InvalidThenValidAuditSchemaClient()
        system = NovelRagSystem(
            config=self.config,
            memory_agent=MemoryAgent(client),
        )

        result = system.audit_book_memories("book_a", apply=False)

        self.assertTrue(result.coverage.complete)
        self.assertEqual(result.operation_count, 0)
        self.assertEqual(client.audit_calls, 2)
        feedback = client.requests[1]["metadata"]["validation_feedback"]
        self.assertEqual(feedback["source"], "python_response_validator")
        self.assertEqual(feedback["error_type"], "ValidationError")
        self.assertIn("operations.0", feedback["error_message"])
        self.assertIn("格式修复重试", client.prompts[1])

    def test_cross_partition_findings_are_reconciled(self) -> None:
        repository = self._repository()
        first = make_memory(
            "chapter_1", "book_a", "chapter_memory", "event", "林舟交出钥匙"
        )
        second = make_memory(
            "chapter_2", "book_a", "chapter_memory", "event", "顾清得到钥匙"
        )
        self._save(repository, first)
        self._save(repository, second)
        char_one = repository.index.resolve_or_create("character", "林舟")
        char_two = repository.index.resolve_or_create("character", "顾清")
        repository.index.link(char_one, first.memory_id, first.store_type, "subject", 1)
        repository.index.link(char_two, second.memory_id, second.store_type, "subject", 1)
        client = FindingAuditClient()
        system = NovelRagSystem(
            config=self.config,
            memory_agent=MemoryAgent(client),
        )
        result = system.audit_book_memories("book_a", apply=False)
        self.assertTrue(result.coverage.complete)
        self.assertGreaterEqual(client.reconcile_calls, 1)
        self.assertNotIn("reconciliation_incomplete", result.blocking_issue_ids)

    def test_incomplete_cross_partition_reconciliation_blocks_apply(self) -> None:
        repository = self._repository()
        memory = make_memory(
            "chapter_1", "book_a", "chapter_memory", "event", "林舟交出钥匙"
        )
        self._save(repository, memory)
        client = FindingAuditClient(omit_reconcile=True)
        system = NovelRagSystem(
            config=self.config,
            memory_agent=MemoryAgent(client),
        )
        result = system.audit_book_memories("book_a", apply=False)
        self.assertIn("reconciliation_incomplete", result.blocking_issue_ids)
        self.assertTrue(result.validation_errors)

    def test_invalid_version_is_rejected_but_canon_archive_is_allowed(self) -> None:
        repository = self._repository()
        canon = make_memory(
            "canon_1", "book_a", "canon_memory", "world_rule", "亡者不可复活"
        )
        self._save(repository, canon)
        executor = PatchExecutor(self.config)
        coverage = complete_coverage([canon.memory_id])

        stale_plan = PatchPlan(
            book_id="book_a",
            run_id="audit_stale",
            coverage=coverage,
            operations=[
                AuditOperation(
                    operation="restore",
                    memory_id=canon.memory_id,
                    expected_versions={canon.memory_id: 99},
                    reason="测试过期版本",
                )
            ],
        )
        with self.assertRaises(PatchValidationError):
            executor.dry_run(stale_plan)

        archive_plan = PatchPlan(
            book_id="book_a",
            run_id="audit_canon",
            coverage=coverage,
            operations=[
                AuditOperation(
                    operation="archive",
                    memory_id=canon.memory_id,
                    expected_versions={canon.memory_id: canon.version},
                    reason="审计确认旧核心设定应归档",
                )
            ],
        )
        executor.dry_run(archive_plan)

    def test_failure_mid_apply_automatically_rolls_back(self) -> None:
        repository = self._repository()
        first = make_memory(
            "chapter_1", "book_a", "chapter_memory", "event", "事件一"
        )
        second = make_memory(
            "chapter_2", "book_a", "chapter_memory", "event", "事件二"
        )
        first.is_current = False
        second.is_current = False
        self._save(repository, first)
        self._save(repository, second)
        plan = PatchPlan(
            book_id="book_a",
            run_id="audit_failure",
            coverage=complete_coverage([first.memory_id, second.memory_id]),
            operations=[
                AuditOperation(
                    operation="archive",
                    memory_id=first.memory_id,
                    expected_versions={first.memory_id: 1},
                    reason="第一步",
                ),
                AuditOperation(
                    operation="archive",
                    memory_id=second.memory_id,
                    expected_versions={second.memory_id: 1},
                    reason="第二步",
                ),
            ],
        )
        executor = PatchExecutor(self.config)
        original = executor._apply_operation
        calls = 0

        def fail_on_second(repo, operation):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("模拟执行中断")
            return original(repo, operation)

        with patch.object(executor, "_apply_operation", side_effect=fail_on_second):
            with self.assertRaises(PatchApplicationError) as context:
                executor.apply(plan)
        executor.snapshots.verify("book_a", context.exception.snapshot_id)
        restored = self._repository()
        self.assertEqual(restored.get(first.memory_id).status, "active")
        self.assertEqual(restored.get(first.memory_id).version, 1)
        self.assertEqual(restored.get(second.memory_id).status, "active")

    def test_unknown_snapshot_cannot_be_rolled_back(self) -> None:
        repository = self._repository()
        self._save(
            repository,
            make_memory("chapter_1", "book_a", "chapter_memory", "event", "事件"),
        )
        with self.assertRaises(SnapshotError):
            PatchExecutor(self.config).rollback("book_a", "snapshot_not_found")

    def test_maintenance_lock_blocks_concurrent_ingestion(self) -> None:
        system = NovelRagSystem(config=self.config)
        with BookDatabaseLock(self.config, "book_a"):
            with self.assertRaises(BookLockError):
                system.ingest(
                    "book_a",
                    "BD",
                    {
                        "world_name": "测试世界",
                        "background": "永夜笼罩大陆 0.8/T",
                        "rules": [],
                        "factions": [],
                        "locations": [],
                        "conflict": "寻找光明 0.7/F",
                    },
                )

    def test_tampered_snapshot_is_rejected_without_changing_current_data(self) -> None:
        repository = self._repository()
        memory = make_memory(
            "chapter_1", "book_a", "chapter_memory", "event", "事件"
        )
        memory.is_current = False
        self._save(repository, memory)
        executor = PatchExecutor(self.config)
        plan = PatchPlan(
            book_id="book_a",
            run_id="audit_tamper",
            coverage=complete_coverage([memory.memory_id]),
            operations=[
                AuditOperation(
                    operation="archive",
                    memory_id=memory.memory_id,
                    expected_versions={memory.memory_id: 1},
                    reason="生成可回滚变更",
                )
            ],
        )
        snapshot_id = executor.apply(plan)
        snapshot_book = (
            self.root
            / ".maintenance_snapshots"
            / "book_a"
            / snapshot_id
            / "book"
        )
        snapshot_file = next(path for path in snapshot_book.rglob("*") if path.is_file())
        with snapshot_file.open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaises(SnapshotError):
            executor.rollback("book_a", snapshot_id)
        self.assertEqual(self._repository().get(memory.memory_id).status, "archived")


if __name__ == "__main__":
    unittest.main()
