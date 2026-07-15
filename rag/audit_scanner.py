from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from .maintenance_schemas import AuditIssue, AuditOperation
from .repository import BookRepository, MEMORY_STORES
from .schemas import AtomicMemory


@dataclass(slots=True)
class ScanResult:
    book_id: str
    memories: dict[str, AtomicMemory]
    raw_records: dict[str, dict[str, Any]]
    memory_store_types: dict[str, str]
    all_memory_ids: list[str]
    heads: list[dict[str, object]]
    links: list[dict[str, object]]
    issues: list[AuditIssue]


class AuditScanner:
    def scan(self, repository: BookRepository) -> ScanResult:
        memories: dict[str, AtomicMemory] = {}
        raw_records: dict[str, dict[str, Any]] = {}
        memory_store_types: dict[str, str] = {}
        all_ids: list[str] = []
        issues: list[AuditIssue] = []

        for store_type in MEMORY_STORES:
            store = repository.store(store_type)
            for raw in store.raw_records():
                memory_id = str(raw["memory_id"])
                all_ids.append(memory_id)
                raw_records[memory_id] = dict(raw)
                memory_store_types[memory_id] = store_type
                try:
                    memory = AtomicMemory.model_validate_json(str(raw["payload_json"]))
                except Exception as exc:
                    issues.append(
                        AuditIssue(
                            code="invalid_memory_record",
                            severity="critical",
                            message="记忆JSON无法通过结构校验，禁止自动应用补丁",
                            memory_ids=[memory_id],
                            details={"store_type": store_type, "error": str(exc)},
                        )
                    )
                    continue
                memories[memory_id] = memory
                if memory.book_id != repository.book_id:
                    issues.append(
                        AuditIssue(
                            code="cross_book_record",
                            severity="critical",
                            message="记忆记录的book_id与所在数据库不一致",
                            memory_ids=[memory_id],
                            details={"record_book_id": memory.book_id},
                        )
                    )
                if memory.store_type != store_type:
                    issues.append(
                        AuditIssue(
                            code="store_type_mismatch",
                            severity="error",
                            message="记忆声明的store_type与实际数据库不一致",
                            memory_ids=[memory_id],
                            details={
                                "declared": memory.store_type,
                                "actual": store_type,
                            },
                        )
                    )
                if memory.last_mentioned_chapter < memory.source_chapter:
                    issues.append(
                        AuditIssue(
                            code="invalid_chapter_order",
                            severity="error",
                            message="最后提及章节早于来源章节",
                            memory_ids=[memory_id],
                        )
                    )

        duplicate_groups: dict[
            tuple[str, str, str, str, str | None, str | None],
            list[AtomicMemory],
        ] = defaultdict(list)
        for memory in memories.values():
            if memory.status != "deleted" and not (
                memory.store_type == "state_timeline_memory" and memory.is_current
            ):
                duplicate_groups[
                    (
                        memory.store_type,
                        memory.content_hash,
                        memory.memory_type,
                        memory.content,
                        memory.entity_name,
                        memory.field,
                    )
                ].append(memory)
        for group in duplicate_groups.values():
            if len(group) < 2:
                continue
            ordered = sorted(
                group,
                key=lambda item: (
                    item.status == "active",
                    item.mention_count,
                    item.source_chapter,
                ),
                reverse=True,
            )
            target, sources = ordered[0], ordered[1:]
            is_canon = target.store_type == "canon_memory"
            issues.append(
                AuditIssue(
                    code="duplicate_canon_memory" if is_canon else "duplicate_memory",
                    severity="error" if is_canon else "warning",
                    message=(
                        "多条核心设定正文及结构完全相同，可安全合并为一条"
                        if is_canon
                        else "多条记忆具有相同内容哈希，可合并为一条"
                    ),
                    memory_ids=[item.memory_id for item in ordered],
                    deterministic_operation=AuditOperation(
                        operation="merge",
                        source_ids=[item.memory_id for item in sources],
                        target_id=target.memory_id,
                        expected_versions={item.memory_id: item.version for item in ordered},
                        reason=(
                            "确定性扫描发现完全相同的核心设定记录"
                            if is_canon
                            else "确定性扫描发现同库同内容哈希记录"
                        ),
                    ),
                )
            )

        current_states: dict[tuple[str, str], list[AtomicMemory]] = defaultdict(list)
        for memory in memories.values():
            if (
                memory.store_type == "state_timeline_memory"
                and memory.is_current
                and memory.status != "deleted"
                and memory.entity_name
                and memory.field
            ):
                current_states[(memory.entity_name, memory.field)].append(memory)
        for (entity_name, field_name), group in current_states.items():
            if len(group) < 2:
                continue
            ordered = sorted(
                group,
                key=lambda item: (item.source_chapter, item.version, item.memory_id),
                reverse=True,
            )
            newest = ordered[0]
            for old in ordered[1:]:
                issues.append(
                    AuditIssue(
                        code="multiple_current_states",
                        severity="error",
                        message=f"{entity_name}的{field_name}存在多个当前状态",
                        memory_ids=[old.memory_id, newest.memory_id],
                        deterministic_operation=AuditOperation(
                            operation="supersede",
                            old_memory_id=old.memory_id,
                            new_memory_id=newest.memory_id,
                            expected_versions={
                                old.memory_id: old.version,
                                newest.memory_id: newest.version,
                            },
                            reason="保留来源章节最新的状态为当前状态",
                        ),
                    )
                )

        for memory in memories.values():
            if memory.hook_status == "open" and memory.status == "deleted":
                issues.append(
                    AuditIssue(
                        code="open_hook_deleted",
                        severity="error",
                        message="开放伏笔被标记为deleted，应恢复为active",
                        memory_ids=[memory.memory_id],
                        deterministic_operation=AuditOperation(
                            operation="restore",
                            memory_id=memory.memory_id,
                            expected_versions={memory.memory_id: memory.version},
                            reason="开放伏笔不得直接删除",
                        ),
                    )
                )
        heads = repository.index.list_heads()
        links = repository.index.list_links()
        head_ids = {str(head["head_id"]) for head in heads}
        links_by_memory: dict[str, list[dict[str, object]]] = defaultdict(list)
        for link in links:
            memory_id = str(link["memory_id"])
            head_id = str(link["head_id"])
            links_by_memory[memory_id].append(link)
            if memory_id not in raw_records:
                issues.append(
                    AuditIssue(
                        code="orphan_index_link",
                        severity="error",
                        message="索引指向不存在的记忆",
                        memory_ids=[memory_id],
                        head_ids=[head_id],
                        deterministic_operation=AuditOperation(
                            operation="remove_orphan_link",
                            memory_id=memory_id,
                            orphan_head_id=head_id,
                            orphan_role=str(link["role"]),
                            reason="删除指向不存在记忆的索引关联",
                        ),
                    )
                )
            elif str(link["store_type"]) != memory_store_types[memory_id]:
                issues.append(
                    AuditIssue(
                        code="index_store_mismatch",
                        severity="error",
                        message="索引记录的store_type与记忆实际数据库不一致",
                        memory_ids=[memory_id],
                        head_ids=[head_id],
                    )
                )
            if head_id not in head_ids:
                issues.append(
                    AuditIssue(
                        code="missing_index_head",
                        severity="critical",
                        message="索引关联引用了不存在的索引头",
                        memory_ids=[memory_id],
                        head_ids=[head_id],
                    )
                )

        for memory in memories.values():
            expected_head_ids = set(
                memory.character_ids + memory.item_ids + memory.event_ids
            )
            linked_head_ids = {
                str(link["head_id"]) for link in links_by_memory[memory.memory_id]
            }
            missing_heads = sorted(expected_head_ids - linked_head_ids)
            if missing_heads:
                issues.append(
                    AuditIssue(
                        code="missing_memory_links",
                        severity="warning",
                        message="记忆中的实体ID没有对应索引关联",
                        memory_ids=[memory.memory_id],
                        head_ids=missing_heads,
                        deterministic_operation=AuditOperation(
                            operation="relink",
                            memory_id=memory.memory_id,
                            add_head_ids=missing_heads,
                            expected_versions={memory.memory_id: memory.version},
                            reason="恢复记忆声明但索引缺失的实体关联",
                        ),
                    )
                )

        duplicate_id_counts = Counter(all_ids)
        for memory_id, count in duplicate_id_counts.items():
            if count > 1:
                issues.append(
                    AuditIssue(
                        code="duplicate_memory_id_across_stores",
                        severity="critical",
                        message="同一memory_id存在于多个数据库，禁止自动整理",
                        memory_ids=[memory_id],
                        details={"count": count},
                    )
                )

        return ScanResult(
            book_id=repository.book_id,
            memories=memories,
            raw_records=raw_records,
            memory_store_types=memory_store_types,
            all_memory_ids=sorted(set(all_ids)),
            heads=heads,
            links=links,
            issues=issues,
        )
