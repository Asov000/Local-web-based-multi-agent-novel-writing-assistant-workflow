from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import RagConfig
from .repository import BookRepository
from .retriever import (
    estimate_tokens,
    truncate_tail_to_token_budget,
    truncate_to_token_budget,
)
from .schemas import AtomicMemory


class ControlContextBuilder:
    def __init__(self, config: RagConfig) -> None:
        self.config = config

    def build(
        self,
        book_id: str,
        query: str,
        *,
        current_chapter: int,
        top_k: int = 20,
        token_budget: int | None = None,
        include_previous_chapter: bool = False,
    ) -> dict[str, Any]:
        repository = BookRepository(self.config, book_id)
        previous = (
            self._previous_chapter(book_id, current_chapter)
            if include_previous_chapter
            else {}
        )
        previous_text = str(previous.get("text") or "")
        continuity_budget = 0
        ending_excerpt = ""
        budget = token_budget or self.config.default_context_token_budget
        if previous_text:
            continuity_budget = min(1600, max(600, budget // 3))
            ending_excerpt = truncate_tail_to_token_budget(
                previous_text,
                continuity_budget,
            )
        search_text = f"{query}\n{ending_excerpt}" if ending_excerpt else query
        head_ids = list(dict.fromkeys(repository.index.heads_in_text(search_text)))
        links = repository.index.find_memory_links(head_ids)
        indexed = repository.memories_from_links(links)

        canon = repository.store("canon_memory").list_memories(statuses=("active",))
        current_states = [
            memory
            for memory in repository.store("state_timeline_memory").list_memories(
                statuses=("active", "archived")
            )
            if memory.is_current
        ]
        previous_summaries = [
            memory
            for memory in repository.store("chapter_memory").list_memories(
                statuses=("active", "archived"),
                memory_types=("chapter_summary",),
                descending_chapter=True,
            )
            if memory.source_chapter < current_chapter
        ][:1]
        open_hooks = [
            memory
            for memory in repository.store("relation_hook_memory").list_memories(
                statuses=("active", "archived"),
                descending_chapter=True,
            )
            if memory.hook_status == "open"
        ]

        mandatory_ids = {
            memory.memory_id
            for memory in canon[: self.config.max_canon_memories]
            + current_states[:10]
            + previous_summaries
            + open_hooks[:10]
        }
        candidates = {
            memory.memory_id: memory
            for memory in indexed + canon + current_states + previous_summaries + open_hooks
            if memory.status != "deleted"
        }
        ranked = sorted(
            candidates.values(),
            key=lambda memory: (
                memory.memory_id not in mandatory_ids,
                -self._score(memory, set(head_ids), current_chapter),
                -memory.source_chapter,
            ),
        )

        memory_budget = max(400, budget - estimate_tokens(ending_excerpt) - 80)
        selected: list[tuple[AtomicMemory, float, bool]] = []
        used = 0
        for memory in ranked:
            if len(selected) >= top_k and memory.memory_id not in mandatory_ids:
                continue
            score = self._score(memory, set(head_ids), current_chapter)
            cost = estimate_tokens(memory.content) + 24
            remaining = memory_budget - used
            if remaining <= 24:
                break
            if cost > remaining:
                compact = truncate_to_token_budget(memory.content, remaining - 24)
                if not compact:
                    continue
                memory = memory.model_copy(update={"content": compact})
                cost = estimate_tokens(memory.content) + 24
            selected.append((memory, score, memory.memory_id in mandatory_ids))
            used += cost

        memory_items = [
            {
                "memory_id": memory.memory_id,
                "content": memory.content,
                "store_type": memory.store_type,
                "memory_type": memory.memory_type,
                "source_chapter": memory.source_chapter,
                "effective_importance": memory.effective_importance,
                "score": round(score, 4),
                "mandatory": mandatory,
            }
            for memory, score, mandatory in selected
        ]
        context: dict[str, Any] = {
            "memories": memory_items,
            "retrieval_trace": {
                "head_ids": head_ids,
                "candidate_count": len(candidates),
                "selected_count": len(memory_items),
                "top_k": top_k,
                "token_budget": budget,
                "memory_token_budget": memory_budget,
                "estimated_memory_tokens": used,
                "estimated_continuity_tokens": estimate_tokens(ending_excerpt),
            },
        }
        if include_previous_chapter and previous:
            context["continuity"] = {
                "schema_version": "1.0",
                "book_id": book_id,
                "source_chapter_id": previous.get("chapter_id", 0),
                "source_chapter_title": previous.get("title", ""),
                "ending_excerpt": ending_excerpt,
                "excerpt_strategy": "chapter_tail",
            }
        return context

    def build_continuation_overview_material(self, book_id: str) -> dict[str, Any]:
        latest = self.latest_chapter(book_id)
        if not latest:
            raise FileNotFoundError(f"书籍 {book_id} 没有可续写的已归档章节")
        repository = BookRepository(self.config, book_id)
        summaries = repository.store("chapter_memory").list_memories(
            statuses=("active", "archived"),
            memory_types=("chapter_summary",),
            descending_chapter=True,
        )[:20]
        summaries.reverse()
        canon = repository.store("canon_memory").list_memories(statuses=("active",))[
            : self.config.max_canon_memories
        ]
        states = [
            memory
            for memory in repository.store("state_timeline_memory").list_memories(
                statuses=("active", "archived")
            )
            if memory.is_current
        ][:15]
        hooks = [
            memory
            for memory in repository.store("relation_hook_memory").list_memories(
                statuses=("active", "archived"),
                descending_chapter=True,
            )
            if memory.hook_status == "open"
        ][:10]
        latest_text = str(latest.get("text") or "")
        return {
            "book_id": book_id,
            "latest_chapter": {
                "chapter_id": latest["chapter_id"],
                "title": latest.get("title", ""),
            },
            "chapter_summaries": [
                {
                    "chapter_id": memory.source_chapter,
                    "content": memory.content[:600],
                }
                for memory in summaries
            ],
            "canon": [memory.content[:500] for memory in canon],
            "current_states": [memory.content[:500] for memory in states],
            "open_hooks": [memory.content[:500] for memory in hooks],
            "ending_preview": latest_text[-500:].strip(),
        }

    def latest_chapter(
        self,
        book_id: str,
        *,
        before_chapter: int | None = None,
    ) -> dict[str, Any]:
        documents = self.config.book_dir(book_id) / "documents"
        if not documents.is_dir():
            return {}
        latest: dict[str, Any] = {}
        for document in documents.glob("chapter_*.json"):
            try:
                file_chapter = int(document.stem.removeprefix("chapter_"))
            except ValueError:
                continue
            if before_chapter is not None and file_chapter >= before_chapter:
                continue
            data = self._read_chapter_document(document, file_chapter)
            if data and int(data["chapter_id"]) > int(latest.get("chapter_id", -1)):
                latest = data
        return latest

    def _previous_chapter(self, book_id: str, current_chapter: int) -> dict[str, Any]:
        if current_chapter <= 0:
            return {}
        document = (
            self.config.book_dir(book_id)
            / "documents"
            / f"chapter_{current_chapter - 1:06d}.json"
        )
        if not document.is_file():
            return self.latest_chapter(book_id, before_chapter=current_chapter)
        return self._read_chapter_document(document, current_chapter - 1)

    @staticmethod
    def _read_chapter_document(document: Path, fallback_chapter: int) -> dict[str, Any]:
        try:
            data = json.loads(document.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        result = data.get("result") if isinstance(data.get("result"), dict) else {}
        return {
            "chapter_id": data.get("chapter_id", fallback_chapter),
            "title": result.get("chapter_title") or result.get("title") or "",
            "text": result.get("text") or "",
        }

    @staticmethod
    def _score(
        memory: AtomicMemory,
        matched_head_ids: set[str],
        current_chapter: int,
    ) -> float:
        memory_heads = set(
            memory.character_ids + memory.item_ids + memory.event_ids
        )
        entity_match = 1.0 if memory_heads & matched_head_ids else 0.0
        type_priority = {
            "canon_memory": 1.0,
            "state_timeline_memory": 0.9,
            "relation_hook_memory": 0.8,
            "chapter_memory": 0.7,
        }.get(memory.store_type, 0.5)
        gap = max(0, current_chapter - memory.last_mentioned_chapter)
        recency = 1.0 / (1.0 + gap / 5.0)
        return (
            0.40 * memory.effective_importance
            + 0.25 * entity_match
            + 0.20 * type_priority
            + 0.15 * recency
        )
