from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .audit_scanner import ScanResult
from .entity_partition import EntityGraphPartitioner
from .index_store import normalize_name
from .maintenance_schemas import (
    AuditIssue,
    AuditScope,
    MemoryComparisonCandidate,
    MemoryComparisonPacket,
)
from .schemas import AtomicMemory


@dataclass(slots=True)
class ScopedAuditPlan:
    scope: AuditScope
    query_memory_ids: list[str]
    batches: list[MemoryComparisonPacket]
    deterministic_issues: list[AuditIssue]

    @property
    def comparison_pair_count(self) -> int:
        return sum(len(batch.candidate_memory_ids) for batch in self.batches)


class ScopedComparisonPlanner:
    """Build small, non-repeating semantic comparison batches for one audit scope."""

    def __init__(self, *, max_candidates_per_batch: int = 9) -> None:
        self.max_candidates_per_batch = max(1, min(9, max_candidates_per_batch))

    def build(self, scan: ScanResult, scope: AuditScope) -> ScopedAuditPlan:
        memories = {
            memory_id: memory
            for memory_id, memory in scan.memories.items()
            if memory.status != "deleted"
        }
        query_memory_ids = self._query_memory_ids(memories, scope)
        if not query_memory_ids:
            raise ValueError("所选范围内没有可审计的有效记忆")

        query_set = set(query_memory_ids)
        memory_heads = self._memory_heads(scan)
        seen_pairs: set[tuple[str, str]] = set()
        batches: list[MemoryComparisonPacket] = []

        for query_memory_id in query_memory_ids:
            query = memories[query_memory_id]
            ranked: list[tuple[float, str, list[str]]] = []
            for candidate_memory_id, candidate in memories.items():
                if candidate_memory_id == query_memory_id:
                    continue
                if scope.mode == "global" and candidate_memory_id not in query_set:
                    continue
                pair = tuple(sorted((query_memory_id, candidate_memory_id)))
                if pair in seen_pairs:
                    continue
                relevance = self._relevance(
                    query,
                    candidate,
                    memory_heads.get(query_memory_id, set()),
                    memory_heads.get(candidate_memory_id, set()),
                )
                if relevance is None:
                    continue
                score, reason_codes = relevance
                ranked.append((score, candidate_memory_id, reason_codes))

            ranked.sort(key=lambda item: (-item[0], item[1]))
            candidates: list[MemoryComparisonCandidate] = []
            for score, candidate_memory_id, reason_codes in ranked:
                pair = tuple(sorted((query_memory_id, candidate_memory_id)))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                candidates.append(
                    MemoryComparisonCandidate(
                        memory_id=candidate_memory_id,
                        reason_codes=reason_codes,
                        score=score,
                        memory=self._memory_record(memories[candidate_memory_id]),
                    )
                )

            for index in range(0, len(candidates), self.max_candidates_per_batch):
                chunk = candidates[index : index + self.max_candidates_per_batch]
                batches.append(
                    MemoryComparisonPacket(
                        book_id=scan.book_id,
                        scope=scope,
                        query_memory_id=query_memory_id,
                        query_memory=self._memory_record(query),
                        candidate_memory_ids=[item.memory_id for item in chunk],
                        candidates=chunk,
                    )
                )

        return ScopedAuditPlan(
            scope=scope,
            query_memory_ids=query_memory_ids,
            batches=batches,
            deterministic_issues=self._issues_for_scope(
                scan.issues,
                query_set,
                include_index_only=scope.mode == "book",
            ),
        )

    @staticmethod
    def _query_memory_ids(
        memories: dict[str, AtomicMemory],
        scope: AuditScope,
    ) -> list[str]:
        chapter_ids = set(scope.chapter_ids)
        selected = [
            memory
            for memory in memories.values()
            if (
                scope.mode == "book"
                or (scope.mode == "global" and memory.store_type == "canon_memory")
                or (
                    scope.mode == "chapters"
                    and memory.source_chapter in chapter_ids
                )
            )
        ]
        selected.sort(
            key=lambda memory: (
                memory.source_chapter,
                memory.store_type,
                memory.memory_id,
            )
        )
        return [memory.memory_id for memory in selected]

    @classmethod
    def _relevance(
        cls,
        query: AtomicMemory,
        candidate: AtomicMemory,
        query_heads: set[str],
        candidate_heads: set[str],
    ) -> tuple[float, list[str]] | None:
        # Exact duplicates are handled deterministically and do not need a model call.
        if query.content_hash == candidate.content_hash and query.content == candidate.content:
            return None

        reasons: list[str] = []
        score = 0.0
        similarity = EntityGraphPartitioner._content_similarity(
            query.content,
            candidate.content,
        )
        shared_heads = query_heads & candidate_heads
        same_type = query.memory_type == candidate.memory_type
        same_entity_field = bool(
            query.entity_name
            and candidate.entity_name
            and query.field
            and candidate.field
            and normalize_name(query.entity_name) == normalize_name(candidate.entity_name)
            and query.field.casefold() == candidate.field.casefold()
        )
        protected_context = bool(
            candidate.store_type == "canon_memory"
            or (candidate.store_type == "state_timeline_memory" and candidate.is_current)
            or candidate.hook_status == "open"
        )

        if same_entity_field:
            reasons.append("same_entity_field_timeline")
            score = max(score, 1.0)
        if shared_heads and (
            similarity >= (0.18 if same_type else 0.42)
            or protected_context
            or query.store_type == "canon_memory"
        ):
            reasons.append("shared_index_context")
            score = max(score, min(0.95, 0.45 + similarity * 0.5))
        if similarity >= (0.56 if same_type else 0.72):
            reasons.append("semantic_overlap")
            score = max(score, min(0.98, 0.55 + similarity * 0.45))

        if not reasons:
            return None
        return min(1.0, score), reasons

    @staticmethod
    def _memory_heads(scan: ScanResult) -> dict[str, set[str]]:
        result: dict[str, set[str]] = defaultdict(set)
        for link in scan.links:
            memory_id = str(link.get("memory_id") or "")
            head_id = str(link.get("head_id") or "")
            if memory_id and head_id:
                result[memory_id].add(head_id)
        return result

    @staticmethod
    def _memory_record(memory: AtomicMemory) -> dict[str, object]:
        return memory.model_dump(mode="json")

    @staticmethod
    def _issues_for_scope(
        issues: list[AuditIssue],
        query_memory_ids: set[str],
        *,
        include_index_only: bool,
    ) -> list[AuditIssue]:
        return [
            issue
            for issue in issues
            if (
                bool(set(issue.memory_ids) & query_memory_ids)
                or (include_index_only and not issue.memory_ids)
            )
        ]
