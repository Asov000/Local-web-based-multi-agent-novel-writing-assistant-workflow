from __future__ import annotations

from collections.abc import Iterable

from .config import RagConfig
from .index_store import IndexStore
from .schemas import AtomicMemory, StoreType
from .storage import ConflictStore, SQLiteMemoryStore


MEMORY_STORES: tuple[StoreType, ...] = (
    "canon_memory",
    "chapter_memory",
    "state_timeline_memory",
    "relation_hook_memory",
)


class BookRepository:
    def __init__(self, config: RagConfig, book_id: str) -> None:
        self.config = config
        self.book_id = book_id
        self.stores = {
            store_type: SQLiteMemoryStore(config.database_path(book_id, store_type))
            for store_type in MEMORY_STORES
        }
        self.index = IndexStore(config.database_path(book_id, "index"), book_id)
        self.conflicts = ConflictStore(config.database_path(book_id, "conflicts"))

    def store(self, store_type: StoreType) -> SQLiteMemoryStore:
        return self.stores[store_type]

    def get(self, memory_id: str, store_type: str | None = None) -> AtomicMemory | None:
        if store_type in self.stores:
            return self.stores[store_type].get(memory_id)  # type: ignore[index]
        for store in self.stores.values():
            memory = store.get(memory_id)
            if memory:
                return memory
        return None

    def memories_from_links(
        self,
        links: Iterable[tuple[str, str]],
    ) -> list[AtomicMemory]:
        grouped: dict[str, list[str]] = {}
        for memory_id, store_type in links:
            grouped.setdefault(store_type, []).append(memory_id)
        memories: list[AtomicMemory] = []
        for store_type, memory_ids in grouped.items():
            if store_type in self.stores:
                memories.extend(self.stores[store_type].find_by_ids(memory_ids))  # type: ignore[index]
        return memories

    def all_dynamic(self) -> list[AtomicMemory]:
        memories: list[AtomicMemory] = []
        for store_type in MEMORY_STORES:
            if store_type == "canon_memory":
                continue
            memories.extend(self.stores[store_type].list_memories(statuses=None))
        return memories
