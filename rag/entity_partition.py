from __future__ import annotations

import re
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

from .audit_scanner import ScanResult
from .index_store import normalize_name
from .maintenance_schemas import (
    AuditIssue,
    AuditPacket,
    CrossAuditCandidate,
    CrossAuditPacket,
    RelatedMemoryRef,
)
from .retriever import estimate_tokens, truncate_to_token_budget
from .schemas import AtomicMemory


class EntityGraphPartitioner:
    ROLE_WEIGHTS = {
        "subject": 100.0,
        "participant": 70.0,
        "owner": 70.0,
        "target": 55.0,
        "object": 55.0,
        "related": 30.0,
    }

    def __init__(
        self,
        *,
        max_primary_memories: int = 12,
        max_context_memories: int = 12,
        max_packet_tokens: int = 4200,
        max_global_tokens: int = 900,
        max_cross_candidates: int = 96,
        max_cross_candidates_per_memory: int = 2,
        max_cross_packet_candidates: int = 6,
    ) -> None:
        self.max_primary_memories = max_primary_memories
        self.max_context_memories = max_context_memories
        self.max_packet_tokens = max_packet_tokens
        self.max_global_tokens = max_global_tokens
        self.max_cross_candidates = max_cross_candidates
        self.max_cross_candidates_per_memory = max_cross_candidates_per_memory
        self.max_cross_packet_candidates = max_cross_packet_candidates

    def assign_owners(self, scan: ScanResult) -> dict[str, str]:
        """Choose one stable audit owner without changing retrieval links."""
        heads_by_id = {str(head["head_id"]): head for head in scan.heads}
        links_by_memory = self._links_by_memory(scan)
        owners: dict[str, str] = {}
        for memory_id in scan.all_memory_ids:
            links = links_by_memory.get(memory_id, [])
            candidate_ids = sorted(
                {
                    str(link["head_id"])
                    for link in links
                    if str(link["head_id"]) in heads_by_id
                }
            )
            if not candidate_ids:
                owners[memory_id] = "__unlinked__"
                continue
            memory = scan.memories.get(memory_id)
            ranked = sorted(
                candidate_ids,
                key=lambda head_id: (
                    -self._owner_score(
                        memory,
                        heads_by_id[head_id],
                        [
                            link
                            for link in links
                            if str(link["head_id"]) == head_id
                        ],
                    ),
                    head_id,
                ),
            )
            owners[memory_id] = ranked[0]
        return owners

    def build_packets(
        self,
        scan: ScanResult,
        owners: dict[str, str] | None = None,
    ) -> list[AuditPacket]:
        owners = owners or self.assign_owners(scan)
        heads_by_id = {str(head["head_id"]): head for head in scan.heads}
        memory_heads = self._memory_heads(scan)
        links_by_memory = self._links_by_memory(scan)
        head_memories: dict[str, list[str]] = defaultdict(list)
        for memory_id, head_ids in memory_heads.items():
            for head_id in head_ids:
                head_memories[head_id].append(memory_id)

        primary_groups: dict[str, list[str]] = defaultdict(list)
        for memory_id in scan.all_memory_ids:
            primary_groups[owners.get(memory_id, "__unlinked__")].append(memory_id)

        global_context = self._build_global_context(scan, owners)
        packets: list[AuditPacket] = []
        for owner_ids, primary_ids in self._build_owner_bundles(
            scan,
            primary_groups,
        ):
            related_candidates = [
                memory_id
                for owner_id in owner_ids
                for memory_id in head_memories.get(owner_id, [])
            ]
            related_ids = self._select_related_ids(
                scan,
                primary_ids,
                related_candidates,
            )
            related_refs = self._related_memory_refs(
                scan,
                related_ids,
                owners,
                links_by_memory,
            )
            related_heads = self._related_heads(
                primary_ids + related_ids,
                memory_heads,
                heads_by_id,
                set(owner_ids),
            )
            packets.append(
                AuditPacket(
                    book_id=scan.book_id,
                    focus_head=self._bundle_focus_head(owner_ids, heads_by_id),
                    primary_memory_ids=primary_ids,
                    context_memory_ids=[ref.memory_id for ref in related_refs],
                    memories=self._compact_primary_memories(
                        scan,
                        primary_ids,
                        global_context,
                        owners,
                    ),
                    related_memory_refs=related_refs,
                    related_heads=related_heads,
                    global_context=global_context,
                    deterministic_issues=self._issues_for_packet(
                        scan.issues,
                        primary_ids,
                    ),
                )
            )

        index_only_issues = [
            issue
            for issue in scan.issues
            if not set(issue.memory_ids) & set(scan.all_memory_ids)
        ]
        if index_only_issues:
            packets.append(
                AuditPacket(
                    book_id=scan.book_id,
                    focus_head={"head_id": "__index__", "head_type": "index"},
                    primary_memory_ids=[],
                    memories=[],
                    global_context=global_context,
                    deterministic_issues=index_only_issues,
                )
            )
        return packets

    def build_cross_check_packets(
        self,
        scan: ScanResult,
        owners: dict[str, str] | None = None,
    ) -> list[CrossAuditPacket]:
        owners = owners or self.assign_owners(scan)
        candidates = self._cross_candidates(scan)
        if not candidates:
            return []
        global_context = self._build_global_context(scan, owners)
        packets: list[CrossAuditPacket] = []
        for chunk in self._chunk_cross_candidates(candidates):
            memory_ids = list(
                dict.fromkeys(
                    memory_id
                    for candidate in chunk
                    for memory_id in candidate.memory_ids
                )
            )
            packets.append(
                CrossAuditPacket(
                    book_id=scan.book_id,
                    candidates=chunk,
                    memories=[
                        self._full_memory_record(
                            scan,
                            memory_id,
                            role="cross_candidate",
                            owner_head_id=owners.get(memory_id, "__unlinked__"),
                            content_budget=500,
                        )
                        for memory_id in memory_ids
                    ],
                    global_context=global_context,
                )
            )
        return packets

    def build_global_context(
        self,
        scan: ScanResult,
        owners: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return self._build_global_context(
            scan,
            owners or self.assign_owners(scan),
        )

    def _owner_score(
        self,
        memory: AtomicMemory | None,
        head: dict[str, object],
        links: list[dict[str, object]],
    ) -> float:
        score = max(
            (
                self.ROLE_WEIGHTS.get(str(link.get("role") or "").casefold(), 10.0)
                for link in links
            ),
            default=0.0,
        )
        if memory is None:
            return score
        canonical_name = normalize_name(str(head.get("canonical_name") or ""))
        if memory.entity_name and normalize_name(memory.entity_name) == canonical_name:
            score += 80.0
        if str(head.get("head_type") or "") == self._preferred_head_type(memory):
            score += 40.0
        return score

    @staticmethod
    def _preferred_head_type(memory: AtomicMemory) -> str | None:
        memory_type = memory.memory_type.casefold()
        if memory.store_type == "state_timeline_memory" or "character" in memory_type:
            return "character"
        if "item" in memory_type:
            return "item"
        if any(
            marker in memory_type
            for marker in ("event", "summary", "conflict", "hook", "foreshadow")
        ):
            return "event"
        return None

    def _chunk_primary_ids(
        self,
        scan: ScanResult,
        memory_ids: list[str],
    ) -> list[list[str]]:
        chunks: list[list[str]] = []
        current: list[str] = []
        used = 0
        primary_budget = max(600, self.max_packet_tokens - self.max_global_tokens - 500)
        for memory_id in memory_ids:
            cost = self._primary_memory_cost(scan, memory_id)
            if current and (
                len(current) >= self.max_primary_memories
                or used + cost > primary_budget
            ):
                chunks.append(current)
                current = []
                used = 0
            current.append(memory_id)
            used += cost
        if current:
            chunks.append(current)
        return chunks

    def _build_owner_bundles(
        self,
        scan: ScanResult,
        primary_groups: dict[str, list[str]],
    ) -> list[tuple[list[str], list[str]]]:
        """Best-fit small owner partitions into fewer model calls without reassigning them."""
        items: list[tuple[str, list[str], int]] = []
        for owner_id, memory_ids in sorted(primary_groups.items()):
            ordered = sorted(
                memory_ids,
                key=lambda memory_id: self._memory_sort_key(scan, memory_id),
            )
            for chunk in self._chunk_primary_ids(scan, ordered):
                items.append(
                    (
                        owner_id,
                        chunk,
                        sum(
                            self._primary_memory_cost(scan, memory_id)
                            for memory_id in chunk
                        ),
                    )
                )

        items.sort(key=lambda item: (-item[2], -len(item[1]), item[0], item[1]))
        primary_budget = max(600, self.max_packet_tokens - self.max_global_tokens - 500)
        bins: list[dict[str, Any]] = []
        for owner_id, memory_ids, cost in items:
            fits = [
                (primary_budget - int(bundle["cost"]) - cost, index)
                for index, bundle in enumerate(bins)
                if len(bundle["memory_ids"]) + len(memory_ids)
                <= self.max_primary_memories
                and int(bundle["cost"]) + cost <= primary_budget
            ]
            if not fits:
                bins.append(
                    {
                        "owner_ids": [owner_id],
                        "memory_ids": list(memory_ids),
                        "cost": cost,
                    }
                )
                continue
            _, index = min(fits)
            bundle = bins[index]
            if owner_id not in bundle["owner_ids"]:
                bundle["owner_ids"].append(owner_id)
            bundle["memory_ids"].extend(memory_ids)
            bundle["cost"] = int(bundle["cost"]) + cost

        return [
            (list(bundle["owner_ids"]), list(bundle["memory_ids"]))
            for bundle in bins
        ]

    @staticmethod
    def _primary_memory_cost(scan: ScanResult, memory_id: str) -> int:
        memory = scan.memories.get(memory_id)
        content = memory.content if memory else str(
            scan.raw_records[memory_id].get("content", "")
        )
        return min(600, estimate_tokens(content)) + 120

    @staticmethod
    def _bundle_focus_head(
        owner_ids: list[str],
        heads_by_id: dict[str, dict[str, object]],
    ) -> dict[str, Any]:
        if len(owner_ids) == 1:
            owner_id = owner_ids[0]
            if owner_id in heads_by_id:
                return {
                    **dict(heads_by_id[owner_id]),
                    "ownership_strategy": "role_entity_type_score_v1",
                }
            return {
                "head_id": "__unlinked__",
                "head_type": "unlinked",
                "ownership_strategy": "unlinked",
            }
        return {
            "head_id": "__owner_bundle__",
            "head_type": "owner_bundle",
            "ownership_strategy": "unique_owner_best_fit_v1",
            "owner_head_ids": owner_ids,
            "owners": [
                {
                    "head_id": owner_id,
                    "head_type": str(
                        heads_by_id.get(owner_id, {}).get("head_type") or "unlinked"
                    ),
                    "canonical_name": str(
                        heads_by_id.get(owner_id, {}).get("canonical_name") or ""
                    ),
                }
                for owner_id in owner_ids
            ],
        }

    def _build_global_context(
        self,
        scan: ScanResult,
        owners: dict[str, str],
    ) -> dict[str, Any]:
        memories = list(scan.memories.values())
        groups = {
            "canon_refs": sorted(
                (memory for memory in memories if memory.store_type == "canon_memory"),
                key=lambda memory: memory.raw_importance,
                reverse=True,
            )[:20],
            "current_state_refs": sorted(
                (
                    memory
                    for memory in memories
                    if memory.store_type == "state_timeline_memory"
                    and memory.is_current
                    and memory.status != "deleted"
                ),
                key=lambda memory: (memory.effective_importance, memory.source_chapter),
                reverse=True,
            )[:30],
            "open_hook_refs": sorted(
                (memory for memory in memories if memory.hook_status == "open"),
                key=lambda memory: memory.effective_importance,
                reverse=True,
            )[:20],
            "timeline_checkpoint_refs": sorted(
                (memory for memory in memories if memory.memory_type == "chapter_summary"),
                key=lambda memory: memory.source_chapter,
                reverse=True,
            )[:10],
        }
        result: dict[str, Any] = {
            "book_id": scan.book_id,
            "memory_count": len(scan.all_memory_ids),
            "entity_count": len(scan.heads),
            "content_policy": "references_only",
        }
        remaining = self.max_global_tokens
        for name, items in groups.items():
            refs: list[dict[str, Any]] = []
            for memory in items:
                ref = {
                    "memory_id": memory.memory_id,
                    "owner_head_id": owners.get(memory.memory_id, "__unlinked__"),
                    "memory_type": memory.memory_type,
                    "source_chapter": memory.source_chapter,
                    "version": memory.version,
                }
                cost = estimate_tokens(str(ref))
                if cost > remaining:
                    continue
                refs.append(ref)
                remaining -= cost
            result[name] = refs
        return result

    def _select_related_ids(
        self,
        scan: ScanResult,
        primary_ids: list[str],
        candidates: list[str],
    ) -> list[str]:
        primary = set(primary_ids)
        unique = [
            memory_id
            for memory_id in dict.fromkeys(candidates)
            if memory_id not in primary and memory_id in scan.raw_records
        ]
        unique.sort(
            key=lambda memory_id: self._context_priority(scan.memories.get(memory_id)),
        )
        return unique[: self.max_context_memories]

    def _related_memory_refs(
        self,
        scan: ScanResult,
        memory_ids: list[str],
        owners: dict[str, str],
        links_by_memory: dict[str, list[dict[str, object]]],
    ) -> list[RelatedMemoryRef]:
        refs: list[RelatedMemoryRef] = []
        for memory_id in memory_ids:
            memory = scan.memories.get(memory_id)
            if memory is None:
                continue
            refs.append(
                RelatedMemoryRef(
                    memory_id=memory_id,
                    owner_head_id=owners.get(memory_id, "__unlinked__"),
                    memory_type=memory.memory_type,
                    source_chapter=memory.source_chapter,
                    effective_importance=memory.effective_importance,
                    version=memory.version,
                    relation_roles=sorted(
                        {
                            str(link.get("role") or "")
                            for link in links_by_memory.get(memory_id, [])
                            if link.get("role")
                        }
                    ),
                )
            )
        return refs

    @staticmethod
    def _context_priority(memory: AtomicMemory | None) -> tuple[int, float, int]:
        if memory is None:
            return (9, 0.0, 0)
        if memory.store_type == "canon_memory":
            priority = 1
        elif memory.is_current and memory.store_type == "state_timeline_memory":
            priority = 2
        elif memory.hook_status == "open":
            priority = 3
        elif memory.memory_type == "chapter_summary":
            priority = 4
        else:
            priority = 5
        return (priority, -memory.effective_importance, -memory.source_chapter)

    def _cross_candidates(self, scan: ScanResult) -> list[CrossAuditCandidate]:
        memories = scan.memories
        proposals: dict[tuple[str, str], dict[str, Any]] = {}

        def propose(
            left_id: str,
            right_id: str,
            reason: str,
            score: float,
            shared_heads: set[str] | None = None,
        ) -> None:
            if left_id == right_id or left_id not in memories or right_id not in memories:
                return
            pair = tuple(sorted((left_id, right_id)))
            item = proposals.setdefault(
                pair,
                {"reason_codes": set(), "score": 0.0, "shared_head_ids": set()},
            )
            item["reason_codes"].add(reason)
            item["score"] = max(float(item["score"]), score)
            item["shared_head_ids"].update(shared_heads or set())

        state_groups: dict[tuple[str, str], list[AtomicMemory]] = defaultdict(list)
        for memory in memories.values():
            if memory.entity_name and memory.field:
                state_groups[
                    (normalize_name(memory.entity_name), memory.field.casefold())
                ].append(memory)
        for group in state_groups.values():
            ordered = sorted(
                group,
                key=lambda memory: (
                    memory.source_chapter,
                    memory.version,
                    memory.memory_id,
                ),
            )
            for left, right in zip(ordered, ordered[1:]):
                if left.content_hash != right.content_hash:
                    propose(
                        left.memory_id,
                        right.memory_id,
                        "same_entity_field_timeline",
                        1.0,
                    )

        memory_heads = self._memory_heads(scan)
        head_memories: dict[str, list[str]] = defaultdict(list)
        for memory_id, head_ids in memory_heads.items():
            for head_id in head_ids:
                head_memories[head_id].append(memory_id)
        for head_id, memory_ids in head_memories.items():
            ordered_ids = sorted(
                set(memory_ids),
                key=lambda memory_id: (
                    -memories[memory_id].effective_importance,
                    -memories[memory_id].source_chapter,
                    memory_id,
                ),
            )
            for index, left_id in enumerate(ordered_ids):
                for right_id in ordered_ids[index + 1 : index + 4]:
                    similarity = self._content_similarity(
                        memories[left_id].content,
                        memories[right_id].content,
                    )
                    same_type = (
                        memories[left_id].memory_type == memories[right_id].memory_type
                    )
                    threshold = 0.42 if same_type else 0.58
                    if similarity >= threshold:
                        propose(
                            left_id,
                            right_id,
                            "shared_index_semantic_overlap",
                            min(0.95, 0.45 + similarity * 0.5),
                            {head_id},
                        )

        token_sets = {
            memory_id: self._content_tokens(memory.content)
            for memory_id, memory in memories.items()
        }
        token_memories: dict[str, list[str]] = defaultdict(list)
        for memory_id, tokens in token_sets.items():
            for token in tokens:
                token_memories[token].append(memory_id)
        pair_shared_counts: Counter[tuple[str, str]] = Counter()
        for memory_ids in token_memories.values():
            unique_ids = sorted(set(memory_ids))
            if len(unique_ids) > 12:
                continue
            pair_shared_counts.update(combinations(unique_ids, 2))
        for (left_id, right_id), shared_count in pair_shared_counts.items():
            if shared_count < 3:
                continue
            left_tokens = token_sets[left_id]
            right_tokens = token_sets[right_id]
            union = len(left_tokens | right_tokens)
            similarity = shared_count / union if union else 0.0
            same_type = memories[left_id].memory_type == memories[right_id].memory_type
            threshold = 0.56 if same_type else 0.72
            if similarity >= threshold:
                propose(
                    left_id,
                    right_id,
                    "global_near_duplicate",
                    min(0.98, 0.55 + similarity * 0.45),
                )

        ranked = sorted(
            proposals.items(),
            key=lambda item: (-float(item[1]["score"]), item[0]),
        )
        selected: list[CrossAuditCandidate] = []
        counts: Counter[str] = Counter()
        dynamic_limit = min(
            self.max_cross_candidates,
            max(8, len(memories) * self.max_cross_candidates_per_memory),
        )
        for (left_id, right_id), details in ranked:
            if (
                counts[left_id] >= self.max_cross_candidates_per_memory
                or counts[right_id] >= self.max_cross_candidates_per_memory
            ):
                continue
            selected.append(
                CrossAuditCandidate(
                    memory_ids=[left_id, right_id],
                    reason_codes=sorted(details["reason_codes"]),
                    score=min(1.0, float(details["score"])),
                    shared_head_ids=sorted(details["shared_head_ids"]),
                )
            )
            counts[left_id] += 1
            counts[right_id] += 1
            if len(selected) >= dynamic_limit:
                break
        return selected

    def _chunk_cross_candidates(
        self,
        candidates: list[CrossAuditCandidate],
    ) -> list[list[CrossAuditCandidate]]:
        return [
            candidates[index : index + self.max_cross_packet_candidates]
            for index in range(0, len(candidates), self.max_cross_packet_candidates)
        ]

    @staticmethod
    def _content_tokens(content: str) -> set[str]:
        clean = re.sub(r"\s+", "", content).casefold()
        clean = re.sub(r"[^\w\u4e00-\u9fff]", "", clean)
        if len(clean) < 2:
            return {clean} if clean else set()
        return {clean[index : index + 2] for index in range(len(clean) - 1)}

    @classmethod
    def _content_similarity(cls, left: str, right: str) -> float:
        left_tokens = cls._content_tokens(left)
        right_tokens = cls._content_tokens(right)
        union = len(left_tokens | right_tokens)
        return len(left_tokens & right_tokens) / union if union else 0.0

    def _compact_primary_memories(
        self,
        scan: ScanResult,
        primary_ids: list[str],
        global_context: dict[str, Any],
        owners: dict[str, str],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        remaining = max(
            500,
            self.max_packet_tokens - estimate_tokens(str(global_context)),
        )
        for memory_id in primary_ids:
            content_budget = max(30, min(600, remaining - 100))
            record = self._full_memory_record(
                scan,
                memory_id,
                role="primary",
                content_budget=content_budget,
                owner_head_id=owners.get(memory_id, "__unlinked__"),
            )
            records.append(record)
            remaining = max(0, remaining - estimate_tokens(str(record)))
        return records

    @staticmethod
    def _full_memory_record(
        scan: ScanResult,
        memory_id: str,
        *,
        role: str,
        content_budget: int,
        owner_head_id: str | None = None,
    ) -> dict[str, Any]:
        memory = scan.memories.get(memory_id)
        if memory:
            record = {
                "memory_id": memory.memory_id,
                "role": role,
                "store_type": memory.store_type,
                "memory_type": memory.memory_type,
                "content": truncate_to_token_budget(memory.content, content_budget),
                "source_chapter": memory.source_chapter,
                "last_mentioned_chapter": memory.last_mentioned_chapter,
                "status": memory.status,
                "entity_name": memory.entity_name,
                "field": memory.field,
                "is_current": memory.is_current,
                "hook_status": memory.hook_status,
                "raw_importance": memory.raw_importance,
                "effective_importance": memory.effective_importance,
                "character_ids": memory.character_ids,
                "item_ids": memory.item_ids,
                "event_ids": memory.event_ids,
                "version": memory.version,
            }
        else:
            raw = scan.raw_records[memory_id]
            record = {
                "memory_id": memory_id,
                "role": role,
                "invalid_record": True,
                "store_type": scan.memory_store_types.get(memory_id),
                "content": truncate_to_token_budget(
                    str(raw.get("content", "")),
                    min(300, content_budget),
                ),
                "status": raw.get("status"),
            }
        if owner_head_id is not None:
            record["owner_head_id"] = owner_head_id
        return record

    @staticmethod
    def _memory_sort_key(scan: ScanResult, memory_id: str) -> tuple[float, int, str]:
        memory = scan.memories.get(memory_id)
        if memory is None:
            return (0.0, 0, memory_id)
        return (-memory.effective_importance, memory.source_chapter, memory_id)

    @staticmethod
    def _related_heads(
        memory_ids: list[str],
        memory_heads: dict[str, list[str]],
        heads_by_id: dict[str, dict[str, object]],
        excluded_owner_ids: set[str],
    ) -> list[dict[str, Any]]:
        counter: Counter[str] = Counter()
        for memory_id in memory_ids:
            counter.update(set(memory_heads.get(memory_id, [])))
        return [
            {**dict(heads_by_id[head_id]), "shared_memory_count": count}
            for head_id, count in counter.most_common(12)
            if head_id not in excluded_owner_ids and head_id in heads_by_id
        ]

    @staticmethod
    def _issues_for_packet(
        issues: list[AuditIssue],
        primary_ids: list[str],
    ) -> list[AuditIssue]:
        primary = set(primary_ids)
        return [issue for issue in issues if primary & set(issue.memory_ids)]

    @staticmethod
    def _links_by_memory(scan: ScanResult) -> dict[str, list[dict[str, object]]]:
        result: dict[str, list[dict[str, object]]] = defaultdict(list)
        for link in scan.links:
            memory_id = str(link["memory_id"])
            if memory_id in scan.raw_records:
                result[memory_id].append(link)
        return result

    @staticmethod
    def _memory_heads(scan: ScanResult) -> dict[str, list[str]]:
        head_ids = {str(head["head_id"]) for head in scan.heads}
        result: dict[str, list[str]] = defaultdict(list)
        for link in scan.links:
            memory_id = str(link["memory_id"])
            head_id = str(link["head_id"])
            if memory_id in scan.raw_records and head_id in head_ids:
                result[memory_id].append(head_id)
        return {
            memory_id: sorted(set(memory_head_ids))
            for memory_id, memory_head_ids in result.items()
        }
