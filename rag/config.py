from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


STORE_NAMES = (
    "canon_memory",
    "chapter_memory",
    "state_timeline_memory",
    "relation_hook_memory",
    "index",
    "conflicts",
    "documents",
)


@dataclass(slots=True)
class RagConfig:
    root_dir: Path | str = Path("rag_data")
    max_judge_memories: int = 50
    max_canon_memories: int = 10
    top_importance_memories: int = 20
    random_memory_count: int = 20
    qwen_confidence_threshold: float = 0.65
    active_threshold: float = 0.35
    delete_threshold: float = 0.15
    item_index_threshold: float = 0.45
    default_context_token_budget: int = 4000
    type_weights: dict[str, float] = field(
        default_factory=lambda: {
            "chapter_summary": 0.8,
            "event": 0.9,
            "world_background": 0.9,
            "world_rule": 1.0,
            "character_identity": 1.0,
            "character_profile": 1.0,
            "character_state": 1.0,
            "item_state": 1.0,
            "state_change": 1.0,
            "relation": 1.0,
            "foreshadowing_open": 1.2,
            "foreshadowing_resolved": 0.6,
            "revision_change": 0.8,
        }
    )

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)

    def book_dir(self, book_id: str) -> Path:
        clean_id = book_id.strip()
        if not clean_id or clean_id in {".", ".."}:
            raise ValueError("book_id不能为空")
        if Path(clean_id).name != clean_id or "/" in clean_id or "\\" in clean_id:
            raise ValueError("book_id不能包含路径分隔符")
        return self.root_dir / clean_id

    def database_path(self, book_id: str, store_name: str) -> Path:
        if store_name not in STORE_NAMES:
            raise ValueError(f"未知数据库类型: {store_name}")
        folder = self.book_dir(book_id) / store_name
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{store_name}.sqlite3"
