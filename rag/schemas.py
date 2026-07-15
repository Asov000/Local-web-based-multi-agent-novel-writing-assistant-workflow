from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


TaskCode = Literal["BD", "CH", "CT", "NW", "RV"]
StoreType = Literal[
    "canon_memory",
    "chapter_memory",
    "state_timeline_memory",
    "relation_hook_memory",
]
MemoryStatus = Literal["active", "archived", "deleted"]
HeadType = Literal["character", "item", "event"]
MatchStatus = Literal["confirmed", "updated", "referenced", "conflict", "unrelated"]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class MemoryFact(BaseModel):
    fact_id: str = Field(default_factory=lambda: new_id("fact"))
    fact_type: str
    content: str
    character_names: list[str] = Field(default_factory=list)
    item_names: list[str] = Field(default_factory=list)
    event_names: list[str] = Field(default_factory=list)
    raw_importance: float | None = Field(default=None, ge=0.0, le=1.0)
    canon_candidate: bool = False
    memory_scope: Literal["temporary", "permanent"] = "temporary"
    entity_name: str | None = None
    field: str | None = None
    old_value: Any = None
    new_value: Any = None
    hook_status: Literal["open", "resolved", "abandoned"] | None = None
    source_field: str | None = None


class AtomicMemory(BaseModel):
    memory_id: str
    book_id: str
    store_type: StoreType
    memory_type: str
    content: str
    character_ids: list[str] = Field(default_factory=list)
    item_ids: list[str] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)
    source_chapter: int = Field(default=0, ge=0)
    last_mentioned_chapter: int = Field(default=0, ge=0)
    mention_count: int = Field(default=1, ge=0)
    sample_count: int = Field(default=0, ge=0)
    last_sampled_chapter: int = Field(default=0, ge=0)
    raw_importance: float = Field(ge=0.0, le=1.0)
    effective_importance: float = Field(ge=0.0, le=1.0)
    type_weight: float = Field(default=1.0, ge=0.0)
    status: MemoryStatus = "active"
    content_hash: str
    entity_name: str | None = None
    field: str | None = None
    is_current: bool = True
    hook_status: Literal["open", "resolved", "abandoned"] | None = None
    version: int = Field(default=1, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryMatch(BaseModel):
    memory_id: str
    matched_fact_ids: list[str] = Field(default_factory=list)
    status: MatchStatus
    confidence: float = Field(ge=0.0, le=1.0)


class MemoryCompletionPayload(BaseModel):
    missing_fields: list[str]
    known_fields: dict[str, Any]
    text: str


class MemoryCompletionResult(BaseModel):
    completed_fields: dict[str, Any] = Field(default_factory=dict)


class ChapterMemoryExtractionPayload(BaseModel):
    chapter_id: int = Field(ge=1)
    title: str
    text: str = Field(min_length=1)


class ChapterMemoryExtractionResult(BaseModel):
    facts: list[MemoryFact] = Field(default_factory=list)


class RagContext(BaseModel):
    canon: list[dict[str, Any]] = Field(default_factory=list)
    recent_chapters: list[dict[str, Any]] = Field(default_factory=list)
    states: list[dict[str, Any]] = Field(default_factory=list)
    relations: list[dict[str, Any]] = Field(default_factory=list)
    open_hooks: list[dict[str, Any]] = Field(default_factory=list)


class IngestResult(BaseModel):
    book_id: str
    chapter_id: int
    created_memory_ids: list[str] = Field(default_factory=list)
    updated_memory_ids: list[str] = Field(default_factory=list)
    conflict_ids: list[str] = Field(default_factory=list)
    fact_count: int = 0


class ChapterReplacementResult(IngestResult):
    replaced_existing: bool = False
    retired_memory_ids: list[str] = Field(default_factory=list)
    revision: int = Field(default=1, ge=1)
    snapshot_id: str | None = None


class ConflictRecord(BaseModel):
    conflict_id: str = Field(default_factory=lambda: new_id("conflict"))
    book_id: str
    memory_id: str
    fact_id: str
    old_content: str
    new_content: str
    confidence: float
    source_chapter: int
    status: Literal["pending", "resolved", "dismissed"] = "pending"
