from __future__ import annotations

import hashlib

from .config import RagConfig
from .importance import effective_importance, lifecycle_status
from .maintenance_schemas import AuditOperation, PatchPlan
from .repository import BookRepository
from .schemas import AtomicMemory, ConflictRecord
from .snapshot_manager import SnapshotManager


class PatchValidationError(ValueError):
    pass


class PatchApplicationError(RuntimeError):
    def __init__(self, message: str, snapshot_id: str) -> None:
        super().__init__(message)
        self.snapshot_id = snapshot_id


class PatchValidator:
    def __init__(self, config: RagConfig) -> None:
        self.config = config

    def validate(self, plan: PatchPlan, repository: BookRepository) -> None:
        if plan.book_id != repository.book_id:
            raise PatchValidationError("补丁计划book_id不匹配")
        if not plan.coverage.complete:
            raise PatchValidationError("审计覆盖率不完整，禁止应用补丁")
        if plan.blocking_issue_ids:
            raise PatchValidationError(
                f"存在无法安全自动处理的问题: {plan.blocking_issue_ids}"
            )
        operation_ids = [operation.operation_id for operation in plan.operations]
        if len(operation_ids) != len(set(operation_ids)):
            raise PatchValidationError("补丁包含重复operation_id")
        created_ids = [
            operation.created_memory_id
            for operation in plan.operations
            if operation.operation in {"create", "compress"}
        ]
        if len(created_ids) != len(set(created_ids)):
            raise PatchValidationError("patch contains duplicate created_memory_id")
        for operation in plan.operations:
            self._validate_operation(operation, repository)

    def _validate_operation(
        self,
        operation: AuditOperation,
        repository: BookRepository,
    ) -> None:
        if operation.confidence < 0.65 and operation.operation != "no_op":
            raise PatchValidationError(f"补丁置信度过低: {operation.operation_id}")
        referenced_ids = operation.referenced_memory_ids()
        memories: dict[str, AtomicMemory] = {}
        if operation.operation != "remove_orphan_link":
            for memory_id in referenced_ids:
                memory = repository.get(memory_id)
                if memory is None:
                    raise PatchValidationError(f"补丁引用不存在的记忆: {memory_id}")
                if memory.book_id != repository.book_id:
                    raise PatchValidationError(f"补丁跨书引用记忆: {memory_id}")
                memories[memory_id] = memory
            if operation.operation != "no_op":
                missing_versions = set(referenced_ids) - set(operation.expected_versions)
                if missing_versions:
                    raise PatchValidationError(
                        f"补丁缺少expected_versions: {sorted(missing_versions)}"
                    )
                for memory_id, expected in operation.expected_versions.items():
                    memory = memories.get(memory_id)
                    if memory is None or memory.version != expected:
                        raise PatchValidationError(f"记忆版本不一致: {memory_id}")

        if operation.operation == "create":
            if not operation.created_memory_id:
                raise PatchValidationError("create requires created_memory_id")
            if repository.get(operation.created_memory_id) is not None:
                raise PatchValidationError(
                    f"created_memory_id already exists: {operation.created_memory_id}"
                )
            if not operation.store_type or not operation.memory_type:
                raise PatchValidationError("create requires store_type and memory_type")
            if not (operation.content or "").strip():
                raise PatchValidationError("create requires non-empty content")
            if operation.raw_importance is None:
                raise PatchValidationError("create requires raw_importance")
            self._validate_head_ids(operation, repository)
        elif operation.operation == "update":
            if not operation.memory_id:
                raise PatchValidationError("update requires memory_id")
            if not any(
                (
                    operation.new_content is not None,
                    operation.raw_importance is not None,
                    bool(operation.metadata_patch),
                )
            ):
                raise PatchValidationError("update has no changed fields")
            if operation.new_content is not None and not operation.new_content.strip():
                raise PatchValidationError("updated content cannot be empty")
        elif operation.operation == "delete":
            if not operation.memory_id:
                raise PatchValidationError("delete requires memory_id")
            memory = memories[operation.memory_id]
            if memory.store_type == "canon_memory":
                raise PatchValidationError("canonical memory cannot be deleted automatically")
            if memory.store_type == "state_timeline_memory" and memory.is_current:
                raise PatchValidationError("current state cannot be deleted")
            if memory.hook_status == "open":
                raise PatchValidationError("open hook cannot be deleted")
        elif operation.operation == "compress":
            if len(operation.source_ids) < 2:
                raise PatchValidationError("compress requires at least two source memories")
            if not operation.created_memory_id:
                raise PatchValidationError("compress requires created_memory_id")
            if repository.get(operation.created_memory_id) is not None:
                raise PatchValidationError(
                    f"created_memory_id already exists: {operation.created_memory_id}"
                )
            if not (operation.summary or "").strip():
                raise PatchValidationError("compress requires a non-empty summary")
            group = [memories[item] for item in operation.source_ids]
            if any(memory.store_type == "canon_memory" for memory in group):
                raise PatchValidationError("canonical memory cannot be compressed")
            if len({memory.store_type for memory in group}) != 1:
                raise PatchValidationError("compression sources must share one store")
            if any(
                memory.store_type == "state_timeline_memory" and memory.is_current
                for memory in group
            ):
                raise PatchValidationError("current state cannot be compressed")
            if any(memory.hook_status == "open" for memory in group):
                raise PatchValidationError("open hooks cannot be compressed")
        elif operation.operation == "merge":
            if not operation.target_id or not operation.source_ids:
                raise PatchValidationError("merge必须提供target_id和source_ids")
            group = [memories[item] for item in operation.source_ids + [operation.target_id]]
            if len({memory.store_type for memory in group}) != 1:
                raise PatchValidationError("merge只能在同一记忆库内执行")
            if len({memory.content_hash for memory in group}) != 1:
                raise PatchValidationError("merge要求内容哈希一致")
            signatures = {
                (
                    memory.memory_type,
                    memory.content,
                    memory.entity_name,
                    memory.field,
                )
                for memory in group
            }
            if len(signatures) != 1:
                raise PatchValidationError("merge要求正文、类型、实体和字段完全一致")
        elif operation.operation == "supersede":
            if not operation.old_memory_id or not operation.new_memory_id:
                raise PatchValidationError("supersede缺少新旧记忆ID")
            old = memories[operation.old_memory_id]
            new = memories[operation.new_memory_id]
            if old.store_type != "state_timeline_memory" or new.store_type != old.store_type:
                raise PatchValidationError("supersede只允许状态时间线记忆")
            if (old.entity_name, old.field) != (new.entity_name, new.field):
                raise PatchValidationError("supersede的新旧状态实体或字段不一致")
            if new.source_chapter < old.source_chapter:
                raise PatchValidationError("不能用更早章节的状态覆盖较新状态")
        elif operation.operation == "archive":
            if not operation.memory_id:
                raise PatchValidationError("archive缺少memory_id")
            memory = memories[operation.memory_id]
            if memory.store_type == "state_timeline_memory" and memory.is_current:
                raise PatchValidationError("当前状态不能单独归档，必须使用supersede")
            if memory.hook_status == "open":
                raise PatchValidationError("开放伏笔禁止自动归档")
        elif operation.operation == "restore":
            if not operation.memory_id:
                raise PatchValidationError("restore缺少memory_id")
        elif operation.operation == "update_importance":
            if not operation.memory_id or operation.raw_importance is None:
                raise PatchValidationError("update_importance参数不完整")
            if memories[operation.memory_id].store_type == "canon_memory":
                raise PatchValidationError("权威记忆不计算动态重要度")
        elif operation.operation == "relink":
            if not operation.memory_id:
                raise PatchValidationError("relink缺少memory_id")
            for head_id in operation.add_head_ids + operation.remove_head_ids:
                if not repository.index.head_exists(head_id):
                    raise PatchValidationError(f"relink引用不存在的索引头: {head_id}")
        elif operation.operation == "flag_conflict":
            if len(operation.source_ids) < 2:
                raise PatchValidationError("flag_conflict至少需要两条记忆")
        elif operation.operation == "remove_orphan_link":
            if not operation.memory_id or not operation.orphan_head_id:
                raise PatchValidationError("remove_orphan_link参数不完整")
            if repository.get(operation.memory_id) is not None:
                raise PatchValidationError("目标记忆仍存在，不能按孤立索引删除")
            if not repository.index.head_exists(operation.orphan_head_id):
                raise PatchValidationError("孤立索引头本身不存在，需要人工修复")


    @staticmethod
    def _validate_head_ids(
        operation: AuditOperation,
        repository: BookRepository,
    ) -> None:
        for head_id in (
            operation.character_ids + operation.item_ids + operation.event_ids
        ):
            if not repository.index.head_exists(head_id):
                raise PatchValidationError(f"create references unknown index head: {head_id}")


class PatchExecutor:
    def __init__(self, config: RagConfig) -> None:
        self.config = config
        self.validator = PatchValidator(config)
        self.snapshots = SnapshotManager(config)

    def dry_run(self, plan: PatchPlan) -> None:
        repository = BookRepository(self.config, plan.book_id)
        self.validator.validate(plan, repository)

    def apply(self, plan: PatchPlan) -> str:
        repository = BookRepository(self.config, plan.book_id)
        self.validator.validate(plan, repository)
        snapshot_id = self.snapshots.create(plan.book_id)
        try:
            for operation in plan.operations:
                self._apply_operation(repository, operation)
        except Exception as exc:
            self.snapshots.rollback(plan.book_id, snapshot_id)
            raise PatchApplicationError(
                f"补丁执行失败，已自动回滚: {exc}",
                snapshot_id,
            ) from exc
        return snapshot_id

    def rollback(self, book_id: str, snapshot_id: str) -> None:
        self.snapshots.rollback(book_id, snapshot_id)

    def _apply_operation(
        self,
        repository: BookRepository,
        operation: AuditOperation,
    ) -> None:
        if operation.operation == "no_op":
            return
        if operation.operation == "create":
            self._create(repository, operation)
        elif operation.operation == "update":
            self._update(repository, operation)
        elif operation.operation == "delete":
            self._delete(repository, operation)
        elif operation.operation == "compress":
            self._compress(repository, operation)
        elif operation.operation == "merge":
            self._merge(repository, operation)
        elif operation.operation == "supersede":
            self._supersede(repository, operation)
        elif operation.operation == "relink":
            self._relink(repository, operation)
        elif operation.operation in {"archive", "restore"}:
            self._set_status(repository, operation)
        elif operation.operation == "update_importance":
            self._update_importance(repository, operation)
        elif operation.operation == "flag_conflict":
            self._flag_conflict(repository, operation)
        elif operation.operation == "remove_orphan_link":
            repository.index.remove_link(
                operation.orphan_head_id or "",
                operation.memory_id or "",
                operation.orphan_role,
            )
        else:
            raise PatchValidationError(f"未实现的补丁操作: {operation.operation}")

    def _create(
        self,
        repository: BookRepository,
        operation: AuditOperation,
    ) -> None:
        raw_importance = float(operation.raw_importance)
        source_chapter = operation.source_chapter
        dynamic_importance = (
            1.0
            if operation.store_type == "canon_memory"
            else effective_importance(
                raw_importance,
                1.0,
                source_chapter,
                source_chapter,
            )
        )
        memory = AtomicMemory(
            memory_id=operation.created_memory_id or "",
            book_id=repository.book_id,
            store_type=operation.store_type,  # type: ignore[arg-type]
            memory_type=operation.memory_type or "memory",
            content=(operation.content or "").strip(),
            character_ids=list(dict.fromkeys(operation.character_ids)),
            item_ids=list(dict.fromkeys(operation.item_ids)),
            event_ids=list(dict.fromkeys(operation.event_ids)),
            source_chapter=source_chapter,
            last_mentioned_chapter=source_chapter,
            raw_importance=raw_importance,
            effective_importance=dynamic_importance,
            content_hash=self._content_hash(operation.content or ""),
            entity_name=operation.entity_name,
            field=operation.field,
            metadata={**operation.metadata_patch, "created_by": "rag_message"},
        )
        repository.store(memory.store_type).save(memory)
        self._link_memory(repository, memory)

    def _update(
        self,
        repository: BookRepository,
        operation: AuditOperation,
    ) -> None:
        memory = self._require(repository, operation.memory_id)
        if operation.new_content is not None:
            memory.content = operation.new_content.strip()
            memory.content_hash = self._content_hash(memory.content)
        if operation.raw_importance is not None:
            memory.raw_importance = float(operation.raw_importance)
            memory.effective_importance = effective_importance(
                memory.raw_importance,
                memory.type_weight,
                memory.last_mentioned_chapter,
                max(memory.last_mentioned_chapter, operation.source_chapter),
            )
        memory.metadata.update(operation.metadata_patch)
        memory.version += 1
        repository.store(memory.store_type).save(memory)

    def _delete(
        self,
        repository: BookRepository,
        operation: AuditOperation,
    ) -> None:
        memory = self._require(repository, operation.memory_id)
        memory.status = "deleted"
        memory.is_current = False
        memory.metadata["deleted_reason"] = operation.reason
        memory.version += 1
        repository.store(memory.store_type).save(memory)

    def _compress(
        self,
        repository: BookRepository,
        operation: AuditOperation,
    ) -> None:
        sources = [self._require(repository, memory_id) for memory_id in operation.source_ids]
        source_chapter = max(source.source_chapter for source in sources)
        raw_importance = max(source.raw_importance for source in sources)
        memory = AtomicMemory(
            memory_id=operation.created_memory_id or "",
            book_id=repository.book_id,
            store_type=sources[0].store_type,
            memory_type=operation.memory_type or "compressed_summary",
            content=(operation.summary or "").strip(),
            character_ids=sorted({item for source in sources for item in source.character_ids}),
            item_ids=sorted({item for source in sources for item in source.item_ids}),
            event_ids=sorted({item for source in sources for item in source.event_ids}),
            source_chapter=source_chapter,
            last_mentioned_chapter=max(source.last_mentioned_chapter for source in sources),
            mention_count=sum(source.mention_count for source in sources),
            sample_count=sum(source.sample_count for source in sources),
            raw_importance=raw_importance,
            effective_importance=max(source.effective_importance for source in sources),
            content_hash=self._content_hash(operation.summary or ""),
            is_current=sources[0].store_type != "state_timeline_memory",
            metadata={
                **operation.metadata_patch,
                "compressed_from": [source.memory_id for source in sources],
                "created_by": "rag_message",
            },
        )
        repository.store(memory.store_type).save(memory)
        self._link_memory(repository, memory)
        for source in sources:
            source.status = "archived"
            source.is_current = False
            source.metadata["compressed_into"] = memory.memory_id
            source.version += 1
            repository.store(source.store_type).save(source)

    @staticmethod
    def _content_hash(content: str) -> str:
        return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()

    @staticmethod
    def _link_memory(repository: BookRepository, memory: AtomicMemory) -> None:
        for head_id in memory.character_ids + memory.item_ids + memory.event_ids:
            repository.index.link(
                head_id,
                memory.memory_id,
                memory.store_type,
                "related",
                memory.source_chapter,
            )

    @staticmethod
    def _require(repository: BookRepository, memory_id: str | None) -> AtomicMemory:
        memory = repository.get(memory_id or "")
        if memory is None:
            raise PatchValidationError(f"记忆不存在: {memory_id}")
        return memory

    def _merge(self, repository: BookRepository, operation: AuditOperation) -> None:
        target = self._require(repository, operation.target_id)
        sources = [self._require(repository, memory_id) for memory_id in operation.source_ids]
        target.mention_count += sum(source.mention_count for source in sources)
        target.sample_count += sum(source.sample_count for source in sources)
        target.last_mentioned_chapter = max(
            [target.last_mentioned_chapter]
            + [source.last_mentioned_chapter for source in sources]
        )
        target.raw_importance = max(
            [target.raw_importance] + [source.raw_importance for source in sources]
        )
        target.metadata["merged_memory_ids"] = sorted(
            set(target.metadata.get("merged_memory_ids", []))
            | {source.memory_id for source in sources}
        )
        target.version += 1
        for source in sources:
            repository.index.replace_memory_link_target(
                source.memory_id,
                target.memory_id,
                target.store_type,
            )
            source.status = "deleted"
            source.is_current = False
            source.version += 1
            repository.store(source.store_type).save(source)
        repository.store(target.store_type).save(target)

    def _supersede(self, repository: BookRepository, operation: AuditOperation) -> None:
        old = self._require(repository, operation.old_memory_id)
        new = self._require(repository, operation.new_memory_id)
        old.is_current = False
        old.status = "archived"
        old.version += 1
        new.is_current = True
        new.status = "active"
        new.version += 1
        repository.store(old.store_type).save(old)
        repository.store(new.store_type).save(new)

    def _relink(self, repository: BookRepository, operation: AuditOperation) -> None:
        memory = self._require(repository, operation.memory_id)
        heads = {
            str(head["head_id"]): str(head["head_type"])
            for head in repository.index.list_heads()
        }
        id_fields = {
            "character": memory.character_ids,
            "item": memory.item_ids,
            "event": memory.event_ids,
        }
        for head_id in operation.add_head_ids:
            repository.index.link(
                head_id,
                memory.memory_id,
                memory.store_type,
                "related",
                memory.source_chapter,
            )
            values = id_fields[heads[head_id]]
            if head_id not in values:
                values.append(head_id)
        for head_id in operation.remove_head_ids:
            repository.index.remove_link(head_id, memory.memory_id)
            values = id_fields[heads[head_id]]
            if head_id in values:
                values.remove(head_id)
        memory.version += 1
        repository.store(memory.store_type).save(memory)

    def _set_status(self, repository: BookRepository, operation: AuditOperation) -> None:
        memory = self._require(repository, operation.memory_id)
        memory.status = "archived" if operation.operation == "archive" else "active"
        memory.version += 1
        repository.store(memory.store_type).save(memory)

    def _update_importance(
        self,
        repository: BookRepository,
        operation: AuditOperation,
    ) -> None:
        memory = self._require(repository, operation.memory_id)
        current_chapter = max(
            (
                item.last_mentioned_chapter
                for item in repository.all_dynamic()
            ),
            default=memory.last_mentioned_chapter,
        )
        memory.raw_importance = float(operation.raw_importance)
        memory.effective_importance = effective_importance(
            memory.raw_importance,
            memory.type_weight,
            memory.last_mentioned_chapter,
            current_chapter,
        )
        memory.status = lifecycle_status(
            memory.effective_importance,
            self.config,
            open_hook=memory.hook_status == "open",
        )  # type: ignore[assignment]
        memory.version += 1
        repository.store(memory.store_type).save(memory)

    def _flag_conflict(
        self,
        repository: BookRepository,
        operation: AuditOperation,
    ) -> None:
        first = self._require(repository, operation.source_ids[0])
        second = self._require(repository, operation.source_ids[1])
        repository.conflicts.add(
            ConflictRecord(
                book_id=repository.book_id,
                memory_id=first.memory_id,
                fact_id=operation.operation_id,
                old_content=first.content,
                new_content=second.content,
                confidence=operation.confidence,
                source_chapter=max(first.source_chapter, second.source_chapter),
            )
        )
