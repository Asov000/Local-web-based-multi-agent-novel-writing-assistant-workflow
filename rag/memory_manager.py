from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .book_lock import BookDatabaseLock
from .candidate_sampler import sample_candidates
from .config import RagConfig
from .importance import effective_importance, lifecycle_status
from .index_store import AliasConflictError
from .memory_agent import MemoryAgent
from .normalizer import normalize_writer_result
from .qwen_judge import ConservativeMemoryJudge, MemoryJudge
from .rag_message import RAGMessage
from .repository import BookRepository, MEMORY_STORES
from .schemas import (
    AtomicMemory,
    ChapterReplacementResult,
    ConflictRecord,
    IngestResult,
    MemoryCompletionPayload,
    MemoryCompletionResult,
    MemoryFact,
    StoreType,
    new_id,
)
from .snapshot_manager import SnapshotManager
from .validator import apply_completed_fields, find_missing_fields


STATE_TYPES = {"character_state", "item_state", "state_change"}
RELATION_TYPES = {"relation", "foreshadowing_open", "foreshadowing_resolved"}


class MemoryManager:
    def __init__(
        self,
        config: RagConfig,
        *,
        memory_agent: MemoryAgent | None = None,
        judge: MemoryJudge | None = None,
    ) -> None:
        self.config = config
        self.memory_agent = memory_agent or MemoryAgent()
        self.judge = judge or ConservativeMemoryJudge()

    def ingest_writer_result(
        self,
        book_id: str,
        task_code: str,
        writer_result: dict[str, Any],
        *,
        chapter_id: int = 0,
    ) -> IngestResult:
        with BookDatabaseLock(self.config, book_id):
            return self._ingest_writer_result_locked(
                book_id,
                task_code,
                writer_result,
                chapter_id=chapter_id,
            )

    def _ingest_writer_result_locked(
        self,
        book_id: str,
        task_code: str,
        writer_result: dict[str, Any],
        *,
        chapter_id: int = 0,
        facts_override: list[MemoryFact] | None = None,
        document_revision: int = 1,
        allow_memory_agent_completion: bool = True,
    ) -> IngestResult:
        repository = BookRepository(self.config, book_id)
        if facts_override is not None:
            facts = facts_override
        else:
            stored_facts = writer_result.get("_memory_facts")
            if isinstance(stored_facts, list):
                facts = [MemoryFact.model_validate(item) for item in stored_facts]
            else:
                facts = normalize_writer_result(task_code, writer_result)
        if allow_memory_agent_completion:
            facts = self._complete_missing_fields(book_id, facts, writer_result)
        missing = find_missing_fields(facts)
        if missing:
            raise ValueError(f"章节记忆缺少必要字段: {', '.join(missing)}")

        candidates = self._candidate_memories(repository, facts, chapter_id)
        selected = sample_candidates(candidates, book_id, chapter_id, self.config)
        self._mark_sampled(repository, selected, chapter_id)
        matches = self.judge.judge(facts, selected) if selected else []
        updated_ids, conflict_ids = self._apply_matches(
            repository, facts, selected, matches, chapter_id
        )

        created_ids: list[str] = []
        for fact in facts:
            memory = self._build_memory(book_id, fact, chapter_id)
            existing = self._find_duplicate(repository, memory)
            if existing:
                self._remember_source(existing, chapter_id)
                existing.last_mentioned_chapter = chapter_id
                existing.mention_count += 1
                repository.store(existing.store_type).save(existing)
                updated_ids.append(existing.memory_id)
                continue
            self._supersede_current_state(repository, memory)
            repository.store(memory.store_type).save(memory)
            self._index_memory(repository, memory, fact, chapter_id)
            created_ids.append(memory.memory_id)

        summary = self._build_chapter_summary(book_id, task_code, writer_result, facts, chapter_id)
        if summary and not self._find_duplicate(repository, summary):
            repository.store("chapter_memory").save(summary)
            self._index_summary(repository, summary, facts, chapter_id)
            created_ids.append(summary.memory_id)

        self._archive_full_text(
            book_id,
            task_code,
            writer_result,
            chapter_id,
            revision=document_revision,
        )
        self.refresh_importance(book_id, chapter_id)
        return IngestResult(
            book_id=book_id,
            chapter_id=chapter_id,
            created_memory_ids=created_ids,
            updated_memory_ids=list(dict.fromkeys(updated_ids)),
            conflict_ids=conflict_ids,
            fact_count=len(facts),
        )

    def _complete_missing_fields(
        self,
        book_id: str,
        facts: list[MemoryFact],
        writer_result: dict[str, Any],
    ) -> list[MemoryFact]:
        missing = find_missing_fields(facts)
        if not missing:
            return facts
        payload = MemoryCompletionPayload(
            missing_fields=missing,
            known_fields={"facts": [fact.model_dump() for fact in facts]},
            text=str(writer_result.get("text") or ""),
        )
        request = RAGMessage(
            sender="rag_manager",
            receiver=self.memory_agent.agent_name,
            action="rag.memory.complete",
            book_id=book_id,
            payload=payload.model_dump(),
        )
        response = self.memory_agent.handle_message(request)
        if response.status != "ok":
            raise RuntimeError(response.error or "MemoryAgent补全失败")
        result = MemoryCompletionResult.model_validate(response.payload or {})
        return apply_completed_fields(facts, result.completed_fields)

    def _candidate_memories(
        self,
        repository: BookRepository,
        facts: list[MemoryFact],
        chapter_id: int,
    ) -> list[AtomicMemory]:
        head_ids: list[str] = []
        for fact in facts:
            for head_type, names in (
                ("character", fact.character_names),
                ("item", fact.item_names),
                ("event", fact.event_names),
            ):
                for name in names:
                    try:
                        head_id = repository.index.resolve(head_type, name)  # type: ignore[arg-type]
                    except AliasConflictError:
                        continue
                    if head_id:
                        head_ids.append(head_id)
        links = repository.index.find_memory_links(list(dict.fromkeys(head_ids)))
        return [
            memory
            for memory in repository.memories_from_links(links)
            if memory.status != "deleted" and memory.source_chapter != chapter_id
        ]

    def _mark_sampled(
        self,
        repository: BookRepository,
        memories: list[AtomicMemory],
        chapter_id: int,
    ) -> None:
        for memory in memories:
            memory.sample_count += 1
            memory.last_sampled_chapter = chapter_id
            repository.store(memory.store_type).save(memory)

    def _apply_matches(
        self,
        repository: BookRepository,
        facts: list[MemoryFact],
        candidates: list[AtomicMemory],
        matches: list[Any],
        chapter_id: int,
    ) -> tuple[list[str], list[str]]:
        candidate_map = {memory.memory_id: memory for memory in candidates}
        fact_map = {fact.fact_id: fact for fact in facts}
        updated: list[str] = []
        conflicts: list[str] = []
        for match in matches:
            memory = candidate_map.get(match.memory_id)
            if not memory or match.confidence < self.config.qwen_confidence_threshold:
                continue
            if match.status in {"confirmed", "updated", "referenced"}:
                self._remember_source(memory, chapter_id)
                memory.last_mentioned_chapter = chapter_id
                memory.mention_count += 1
                repository.store(memory.store_type).save(memory)
                updated.append(memory.memory_id)
            elif match.status == "conflict" and memory.store_type == "canon_memory":
                for fact_id in match.matched_fact_ids:
                    fact = fact_map.get(fact_id)
                    if not fact:
                        continue
                    conflict = ConflictRecord(
                        book_id=repository.book_id,
                        memory_id=memory.memory_id,
                        fact_id=fact.fact_id,
                        old_content=memory.content,
                        new_content=fact.content,
                        confidence=match.confidence,
                        source_chapter=chapter_id,
                    )
                    repository.conflicts.add(conflict)
                    conflicts.append(conflict.conflict_id)
        return updated, conflicts

    def _store_type_for(self, fact: MemoryFact) -> StoreType:
        if (
            fact.canon_candidate
            and fact.memory_scope == "permanent"
            and (fact.raw_importance or 0.0) >= 0.8
        ):
            return "canon_memory"
        if fact.fact_type in STATE_TYPES:
            return "state_timeline_memory"
        if fact.fact_type in RELATION_TYPES:
            return "relation_hook_memory"
        return "chapter_memory"

    def _build_memory(
        self,
        book_id: str,
        fact: MemoryFact,
        chapter_id: int,
    ) -> AtomicMemory:
        store_type = self._store_type_for(fact)
        memory_type = fact.fact_type
        if fact.hook_status:
            memory_type = f"foreshadowing_{fact.hook_status}"
        weight = self.config.type_weights.get(memory_type, 1.0)
        raw = float(fact.raw_importance)
        effective = 1.0 if store_type == "canon_memory" else effective_importance(
            raw, weight, chapter_id, chapter_id
        )
        content_hash = hashlib.sha256(
            json.dumps(
                {
                    "book_id": book_id,
                    "store_type": store_type,
                    "memory_type": memory_type,
                    "content": fact.content,
                    "entity": fact.entity_name,
                    "field": fact.field,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        prefix = {
            "canon_memory": "canon",
            "chapter_memory": "chapter",
            "state_timeline_memory": "state",
            "relation_hook_memory": "relation",
        }[store_type]
        return AtomicMemory(
            memory_id=new_id(prefix),
            book_id=book_id,
            store_type=store_type,
            memory_type=memory_type,
            content=fact.content,
            source_chapter=chapter_id,
            last_mentioned_chapter=chapter_id,
            raw_importance=raw,
            effective_importance=effective,
            type_weight=weight,
            entity_name=fact.entity_name,
            field=fact.field,
            hook_status=fact.hook_status,
            content_hash=content_hash,
            metadata={
                "fact_id": fact.fact_id,
                "source_field": fact.source_field,
                "old_value": fact.old_value,
                "new_value": fact.new_value,
                "source_chapters": [chapter_id],
            },
        )

    @staticmethod
    def _remember_source(memory: AtomicMemory, chapter_id: int) -> None:
        sources = [
            int(value)
            for value in memory.metadata.get("source_chapters", [])
            if str(value).isdigit()
        ]
        if not sources:
            sources = [memory.source_chapter]
        if chapter_id not in sources:
            sources.append(chapter_id)
        memory.metadata["source_chapters"] = sorted(set(sources))

    def _find_duplicate(
        self,
        repository: BookRepository,
        memory: AtomicMemory,
    ) -> AtomicMemory | None:
        for existing in repository.store(memory.store_type).list_memories(statuses=None):
            if existing.status != "deleted" and existing.content_hash == memory.content_hash:
                return existing
        return None

    def _supersede_current_state(
        self,
        repository: BookRepository,
        memory: AtomicMemory,
    ) -> None:
        if memory.store_type != "state_timeline_memory":
            return
        if not memory.entity_name or not memory.field:
            return
        store = repository.store("state_timeline_memory")
        previous = store.current_state(memory.entity_name, memory.field)
        if previous and previous.memory_id != memory.memory_id:
            previous.is_current = False
            previous.status = "archived"
            store.save(previous)

    def _index_memory(
        self,
        repository: BookRepository,
        memory: AtomicMemory,
        fact: MemoryFact,
        chapter_id: int,
    ) -> None:
        entity_ids: dict[str, list[str]] = {"character": [], "item": [], "event": []}
        names_by_type = {
            "character": fact.character_names,
            "item": fact.item_names,
            "event": fact.event_names or ([fact.content[:40]] if fact.fact_type == "event" else []),
        }
        for head_type, names in names_by_type.items():
            if head_type == "item" and memory.raw_importance < self.config.item_index_threshold:
                continue
            for name in names:
                head_id = repository.index.resolve_or_create(head_type, name)  # type: ignore[arg-type]
                entity_ids[head_type].append(head_id)
                repository.index.link(
                    head_id,
                    memory.memory_id,
                    memory.store_type,
                    "subject" if name == fact.entity_name else "related",
                    chapter_id,
                )
        memory.character_ids = entity_ids["character"]
        memory.item_ids = entity_ids["item"]
        memory.event_ids = entity_ids["event"]
        repository.store(memory.store_type).save(memory)

    def _index_summary(
        self,
        repository: BookRepository,
        summary: AtomicMemory,
        facts: list[MemoryFact],
        chapter_id: int,
    ) -> None:
        names = list(dict.fromkeys(name for fact in facts for name in fact.character_names))
        for name in names:
            head_id = repository.index.resolve_or_create("character", name)
            summary.character_ids.append(head_id)
            repository.index.link(
                head_id, summary.memory_id, summary.store_type, "participant", chapter_id
            )
        repository.store("chapter_memory").save(summary)

    def _build_chapter_summary(
        self,
        book_id: str,
        task_code: str,
        writer_result: dict[str, Any],
        facts: list[MemoryFact],
        chapter_id: int,
    ) -> AtomicMemory | None:
        if task_code.upper() not in {"CT", "NW", "RV"}:
            return None
        title = str(
            writer_result.get("chapter_title")
            or writer_result.get("title")
            or f"第{chapter_id}章"
        ).strip()
        key_facts = [
            fact.content
            for fact in facts
            if fact.fact_type in {"event", "state_change", "revision_change"}
        ][:6]
        if key_facts:
            content = f"{title}：" + "；".join(key_facts)
        else:
            text = str(writer_result.get("text") or "").strip()
            content = f"{title}：{text[:300]}" if text else title
        raw = max((fact.raw_importance or 0.0 for fact in facts), default=0.5)
        weight = self.config.type_weights["chapter_summary"]
        digest = hashlib.sha256(
            f"{book_id}:{chapter_id}:{content}".encode("utf-8")
        ).hexdigest()
        return AtomicMemory(
            memory_id=new_id("chapter"),
            book_id=book_id,
            store_type="chapter_memory",
            memory_type="chapter_summary",
            content=content,
            source_chapter=chapter_id,
            last_mentioned_chapter=chapter_id,
            raw_importance=raw,
            effective_importance=effective_importance(raw, weight, chapter_id, chapter_id),
            type_weight=weight,
            content_hash=digest,
            metadata={"title": title},
        )

    def _archive_full_text(
        self,
        book_id: str,
        task_code: str,
        writer_result: dict[str, Any],
        chapter_id: int,
        *,
        revision: int = 1,
    ) -> None:
        if not writer_result.get("text"):
            return
        folder = self.config.book_dir(book_id) / "documents"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"chapter_{chapter_id:06d}.json"
        path.write_text(
            json.dumps(
                {
                    "task_code": task_code,
                    "chapter_id": chapter_id,
                    "revision": revision,
                    "status": "active",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "result": writer_result,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
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
        if chapter_id < 1:
            raise ValueError("正文替换必须指定有效章节")
        book_dir = self.config.book_dir(book_id)
        existed_book = book_dir.is_dir()
        with BookDatabaseLock(self.config, book_id):
            snapshot_id = SnapshotManager(self.config).create(book_id) if existed_book else None
            try:
                revision, replaced_existing = self._archive_previous_document(
                    book_id, chapter_id
                )
                repository = BookRepository(self.config, book_id)
                retired = self._retire_chapter_sources(repository, chapter_id)
                title = str(
                    writer_result.get("chapter_title")
                    or writer_result.get("title")
                    or f"第{chapter_id}章"
                ).strip()
                text = str(writer_result.get("text") or "").strip()
                if not text:
                    raise ValueError("章节正文不能为空")
                result = self._ingest_writer_result_locked(
                    book_id,
                    task_code,
                    writer_result,
                    chapter_id=chapter_id,
                    facts_override=facts_override,
                    document_revision=revision,
                    allow_memory_agent_completion=False,
                )
                self._rebuild_current_states(repository)
                return ChapterReplacementResult(
                    **result.model_dump(),
                    replaced_existing=replaced_existing,
                    retired_memory_ids=retired,
                    revision=revision,
                    snapshot_id=snapshot_id,
                )
            except Exception:
                if snapshot_id:
                    SnapshotManager(self.config).rollback(book_id, snapshot_id)
                elif not existed_book and book_dir.exists():
                    shutil.rmtree(book_dir)
                raise

    def _archive_previous_document(
        self,
        book_id: str,
        chapter_id: int,
    ) -> tuple[int, bool]:
        document = (
            self.config.book_dir(book_id)
            / "documents"
            / f"chapter_{chapter_id:06d}.json"
        )
        if not document.is_file():
            return 1, False
        payload = json.loads(document.read_text(encoding="utf-8"))
        revision = int(payload.get("revision") or 1)
        history_dir = (
            self.config.book_dir(book_id)
            / "documents"
            / "history"
            / f"chapter_{chapter_id:06d}"
        )
        history_dir.mkdir(parents=True, exist_ok=True)
        old_payload = dict(payload)
        old_payload["status"] = "old"
        old_payload["superseded_at"] = datetime.now(timezone.utc).isoformat()
        (history_dir / f"revision_{revision:06d}.json").write_text(
            json.dumps(old_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return revision + 1, True

    def _retire_chapter_sources(
        self,
        repository: BookRepository,
        chapter_id: int,
    ) -> list[str]:
        retired: list[str] = []
        for store_type in MEMORY_STORES:
            store = repository.store(store_type)
            for memory in store.list_memories(statuses=None):
                if bool(memory.metadata.get("user_managed")):
                    continue
                sources = [
                    int(value)
                    for value in memory.metadata.get("source_chapters", [])
                    if str(value).isdigit()
                ] or [memory.source_chapter]
                if chapter_id not in sources:
                    continue
                remaining = sorted(set(sources) - {chapter_id})
                if remaining:
                    memory.metadata["source_chapters"] = remaining
                    memory.source_chapter = min(remaining)
                    memory.last_mentioned_chapter = max(remaining)
                    memory.mention_count = max(1, len(remaining))
                else:
                    memory.status = "deleted"
                    memory.is_current = False
                    repository.index.remove_memory_links(memory.memory_id)
                    retired.append(memory.memory_id)
                memory.version += 1
                store.save(memory)
        return retired

    @staticmethod
    def _rebuild_current_states(repository: BookRepository) -> None:
        store = repository.store("state_timeline_memory")
        groups: dict[tuple[str, str], list[AtomicMemory]] = {}
        for memory in store.list_memories(statuses=None):
            if memory.status == "deleted" or not memory.entity_name or not memory.field:
                continue
            groups.setdefault((memory.entity_name, memory.field), []).append(memory)
        for memories in groups.values():
            current = max(memories, key=lambda item: item.source_chapter)
            for memory in memories:
                should_be_current = memory.memory_id == current.memory_id
                if memory.is_current != should_be_current:
                    memory.is_current = should_be_current
                    memory.version += 1
                if should_be_current and memory.status == "archived":
                    memory.status = "active"
                store.save(memory)

    def refresh_importance(self, book_id: str, current_chapter: int) -> None:
        repository = BookRepository(self.config, book_id)
        for memory in repository.all_dynamic():
            if memory.status == "deleted":
                continue
            memory.effective_importance = effective_importance(
                memory.raw_importance,
                memory.type_weight,
                memory.last_mentioned_chapter,
                current_chapter,
            )
            if memory.store_type == "state_timeline_memory" and not memory.is_current:
                memory.status = (
                    "deleted"
                    if memory.effective_importance < self.config.delete_threshold
                    else "archived"
                )
            else:
                memory.status = lifecycle_status(
                    memory.effective_importance,
                    self.config,
                    open_hook=memory.hook_status == "open",
                )  # type: ignore[assignment]
            repository.store(memory.store_type).save(memory)
            if memory.status == "deleted":
                repository.index.remove_memory_links(memory.memory_id)

    def cleanup_deleted(self, book_id: str) -> int:
        repository = BookRepository(self.config, book_id)
        return sum(
            repository.store(store_type).delete_physical()
            for store_type in MEMORY_STORES
            if store_type != "canon_memory"
        )
