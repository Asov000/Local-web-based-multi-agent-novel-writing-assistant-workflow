from __future__ import annotations

from typing import Any

from .config import RagConfig
from .index_store import AliasConflictError
from .repository import BookRepository
from .schemas import AtomicMemory, RagContext


def estimate_tokens(text: str) -> int:
    non_ascii = sum(1 for char in text if ord(char) > 127)
    ascii_count = max(0, len(text) - non_ascii)
    return max(1, non_ascii + (ascii_count + 3) // 4)


def truncate_to_token_budget(text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    if estimate_tokens(text) <= budget:
        return text
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(text[:middle]) <= budget:
            low = middle
        else:
            high = middle - 1
    return text[:low].rstrip()


def truncate_tail_to_token_budget(text: str, budget: int) -> str:
    """Keep the end of a document within a token budget."""
    if budget <= 0:
        return ""
    if estimate_tokens(text) <= budget:
        return text
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(text[-middle:]) <= budget:
            low = middle
        else:
            high = middle - 1
    return text[-low:].lstrip() if low else ""


class RagRetriever:
    def __init__(self, config: RagConfig) -> None:
        self.config = config

    def retrieve(
        self,
        book_id: str,
        query: str,
        *,
        current_chapter: int,
        character_names: list[str] | None = None,
        item_names: list[str] | None = None,
        event_names: list[str] | None = None,
        token_budget: int | None = None,
    ) -> RagContext:
        repository = BookRepository(self.config, book_id)
        head_ids = repository.index.heads_in_text(query)
        for head_type, names in (
            ("character", character_names or []),
            ("item", item_names or []),
            ("event", event_names or []),
        ):
            for name in names:
                try:
                    head_id = repository.index.resolve(head_type, name)  # type: ignore[arg-type]
                except AliasConflictError:
                    continue
                if head_id:
                    head_ids.append(head_id)
        links = repository.index.find_memory_links(list(dict.fromkeys(head_ids)))
        candidates = repository.memories_from_links(links)

        canon_fallback = sorted(
            repository.store("canon_memory").list_memories(statuses=("active",)),
            key=lambda memory: memory.raw_importance,
            reverse=True,
        )[: self.config.max_canon_memories]
        current_state_fallback = sorted(
            (
                memory
                for memory in repository.store("state_timeline_memory").list_memories(
                    statuses=("active", "archived"),
                )
                if memory.is_current
            ),
            key=lambda memory: memory.effective_importance,
            reverse=True,
        )[:10]

        previous = repository.store("chapter_memory").list_memories(
            statuses=("active", "archived"),
            memory_types=("chapter_summary",),
            descending_chapter=True,
            limit=3,
        )
        previous = [memory for memory in previous if memory.source_chapter < current_chapter][:1]
        open_hooks = [
            memory
            for memory in repository.store("relation_hook_memory").list_memories(
                statuses=("active", "archived"),
                descending_chapter=True,
            )
            if memory.hook_status == "open"
        ]
        combined = {
            memory.memory_id: memory
            for memory in candidates
            + canon_fallback
            + current_state_fallback
            + previous
            + open_hooks
            if memory.status != "deleted"
        }
        ranked = sorted(combined.values(), key=self._rank_key)
        budget = token_budget or self.config.default_context_token_budget
        selected: list[AtomicMemory] = []
        used = 0
        for memory in ranked:
            cost = estimate_tokens(memory.content) + 12
            remaining = budget - used
            if cost <= remaining:
                selected.append(memory)
                used += cost
                continue
            compact_content = truncate_to_token_budget(memory.content, remaining - 12)
            if compact_content:
                selected.append(memory.model_copy(update={"content": compact_content}))
                used = budget
            break

        context = RagContext()
        for memory in selected:
            item = {
                "memory_id": memory.memory_id,
                "content": memory.content,
                "source_chapter": memory.source_chapter,
            }
            if memory.store_type == "canon_memory":
                context.canon.append(item)
            elif memory.store_type == "state_timeline_memory" and memory.is_current:
                context.states.append(item)
            elif memory.memory_type == "relation":
                context.relations.append(item)
            elif memory.hook_status == "open":
                context.open_hooks.append(item)
            elif memory.store_type == "chapter_memory":
                context.recent_chapters.append(item)
        return context

    @staticmethod
    def _rank_key(memory: AtomicMemory) -> tuple[int, float, int]:
        if memory.store_type == "canon_memory":
            priority = 1
        elif memory.store_type == "state_timeline_memory" and memory.is_current:
            priority = 2
        elif memory.memory_type == "chapter_summary":
            priority = 3
        elif memory.memory_type == "event":
            priority = 4
        elif memory.hook_status == "open":
            priority = 5
        elif memory.memory_type == "relation":
            priority = 6
        else:
            priority = 7
        return (priority, -memory.effective_importance, -memory.source_chapter)
