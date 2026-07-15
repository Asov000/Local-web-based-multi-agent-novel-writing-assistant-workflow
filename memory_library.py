from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from rag.book_lock import BookDatabaseLock
from rag.config import RagConfig
from rag.repository import BookRepository, MEMORY_STORES
from rag.schemas import AtomicMemory, StoreType


STORE_LABELS: dict[str, str] = {
    "canon_memory": "核心设定",
    "chapter_memory": "章节记忆",
    "state_timeline_memory": "状态时间线",
    "relation_hook_memory": "关系与伏笔",
}


class UserMemoryInput(BaseModel):
    store_type: StoreType
    memory_type: str = Field(min_length=1, max_length=80)
    content: str = Field(min_length=1)
    raw_importance: float = Field(default=0.6, ge=0.0, le=1.0)
    source_chapter: int = Field(default=0, ge=0)
    status: Literal["active", "archived"] = "active"
    character_names: list[str] = Field(default_factory=list)
    item_names: list[str] = Field(default_factory=list)
    event_names: list[str] = Field(default_factory=list)
    entity_name: str | None = None
    field: str | None = None
    is_current: bool = True
    hook_status: Literal["open", "resolved", "abandoned"] | None = None
    note: str = ""

    @field_validator("memory_type", "content", mode="before")
    @classmethod
    def strip_required_text(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("entity_name", "field", mode="before")
    @classmethod
    def strip_optional_text(cls, value: Any) -> str | None:
        clean = str(value or "").strip()
        return clean or None

    @field_validator("character_names", "item_names", "event_names", mode="before")
    @classmethod
    def normalize_names(cls, value: Any) -> list[str]:
        values = value if isinstance(value, list) else []
        return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


class MemoryLibraryService:
    """Human-facing memory CRUD with format validation and local history."""

    def __init__(self, root_dir: str | Path = "rag_data") -> None:
        self.config = RagConfig(root_dir=Path(root_dir))

    def list(self, book_id: str, *, include_deleted: bool = False) -> dict[str, Any]:
        repository = BookRepository(self.config, book_id)
        heads = {
            str(item["head_id"]): str(item["canonical_name"])
            for item in repository.index.list_heads()
        }
        groups: list[dict[str, Any]] = []
        for store_type in MEMORY_STORES:
            statuses = None if include_deleted else ("active", "archived")
            memories = repository.store(store_type).list_memories(
                statuses=statuses,
                descending_chapter=True,
            )
            groups.append(
                {
                    "store_type": store_type,
                    "label": STORE_LABELS[store_type],
                    "memories": [self._payload(memory, heads) for memory in memories],
                }
            )
        return {"book_id": book_id, "groups": groups}

    def get(self, book_id: str, memory_id: str) -> dict[str, Any]:
        repository = BookRepository(self.config, book_id)
        memory = repository.get(memory_id)
        if memory is None:
            raise KeyError("记忆不存在")
        heads = {
            str(item["head_id"]): str(item["canonical_name"])
            for item in repository.index.list_heads()
        }
        return self._payload(memory, heads)

    def create(self, book_id: str, value: dict[str, Any]) -> dict[str, Any]:
        data = UserMemoryInput.model_validate(value)
        with BookDatabaseLock(self.config, book_id):
            repository = BookRepository(self.config, book_id)
            memory_id = f"memory_user_{uuid.uuid4().hex[:16]}"
            ids = self._resolve_ids(repository, data)
            now = datetime.now(timezone.utc).isoformat()
            memory = AtomicMemory(
                memory_id=memory_id,
                book_id=book_id,
                store_type=data.store_type,
                memory_type=data.memory_type,
                content=data.content,
                character_ids=ids["character"],
                item_ids=ids["item"],
                event_ids=ids["event"],
                source_chapter=data.source_chapter,
                last_mentioned_chapter=data.source_chapter,
                raw_importance=data.raw_importance,
                effective_importance=data.raw_importance,
                status=data.status,
                content_hash=self._hash(data.content),
                entity_name=data.entity_name,
                field=data.field,
                is_current=data.is_current,
                hook_status=data.hook_status,
                metadata={
                    "origin": "user",
                    "user_managed": True,
                    "user_note": data.note,
                    "created_at": now,
                    "updated_at": now,
                },
            )
            repository.store(data.store_type).save(memory)
            self._replace_links(repository, memory)
        return self.get(book_id, memory_id)

    def update(self, book_id: str, memory_id: str, value: dict[str, Any]) -> dict[str, Any]:
        data = UserMemoryInput.model_validate(value)
        with BookDatabaseLock(self.config, book_id):
            repository = BookRepository(self.config, book_id)
            current = repository.get(memory_id)
            if current is None:
                raise KeyError("记忆不存在")
            if current.store_type != data.store_type:
                raise ValueError("记忆所属板块不能直接修改，请新建后删除原记录")
            ids = self._resolve_ids(repository, data)
            self._write_history(book_id, current)
            metadata = dict(current.metadata)
            metadata.update(
                {
                    "origin": "user",
                    "user_managed": True,
                    "user_note": data.note,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            memory = current.model_copy(
                update={
                    "memory_type": data.memory_type,
                    "content": data.content,
                    "character_ids": ids["character"],
                    "item_ids": ids["item"],
                    "event_ids": ids["event"],
                    "source_chapter": data.source_chapter,
                    "last_mentioned_chapter": max(
                        current.last_mentioned_chapter, data.source_chapter
                    ),
                    "raw_importance": data.raw_importance,
                    "effective_importance": data.raw_importance,
                    "status": data.status,
                    "content_hash": self._hash(data.content),
                    "entity_name": data.entity_name,
                    "field": data.field,
                    "is_current": data.is_current,
                    "hook_status": data.hook_status,
                    "version": current.version + 1,
                    "metadata": metadata,
                }
            )
            repository.store(current.store_type).save(memory)
            self._replace_links(repository, memory)
        return self.get(book_id, memory_id)

    def delete(self, book_id: str, memory_id: str) -> dict[str, Any]:
        with BookDatabaseLock(self.config, book_id):
            repository = BookRepository(self.config, book_id)
            current = repository.get(memory_id)
            if current is None:
                raise KeyError("记忆不存在")
            self._write_history(book_id, current)
            metadata = dict(current.metadata)
            metadata.update(
                {
                    "origin": "user",
                    "user_managed": True,
                    "deleted_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            deleted = current.model_copy(
                update={
                    "status": "deleted",
                    "version": current.version + 1,
                    "metadata": metadata,
                }
            )
            repository.store(current.store_type).save(deleted)
            repository.index.remove_memory_links(memory_id)
        return {"memory_id": memory_id, "deleted": True}

    def history(self, book_id: str, memory_id: str) -> list[dict[str, Any]]:
        folder = self._history_dir(book_id, memory_id)
        if not folder.is_dir():
            return []
        versions: list[dict[str, Any]] = []
        for path in sorted(folder.glob("version_*.json"), reverse=True):
            try:
                versions.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return versions

    def create_from_material(self, book_id: str, material: dict[str, Any]) -> dict[str, Any]:
        return self.sync_from_material(book_id, material)["memory"]

    def sync_from_material(
        self,
        book_id: str,
        material: dict[str, Any],
        *,
        linked_memory_id: str = "",
        overwrite_user: bool = False,
    ) -> dict[str, Any]:
        data = UserMemoryInput.model_validate(self._material_memory_value(material))
        material_id = str(material.get("material_id") or "").strip()
        material_version = int(material.get("version") or 1)
        with BookDatabaseLock(self.config, book_id):
            repository = BookRepository(self.config, book_id)
            current = repository.get(linked_memory_id) if linked_memory_id else None
            if current is None and material_id:
                current = self._find_material_memory(repository, material_id)
            if current is not None and current.status == "deleted":
                current = None

            if current is not None:
                metadata = dict(current.metadata)
                legacy_note = f"由素材库同步：{material_id}"
                is_legacy_material_memory = (
                    current.version == 1
                    and str(metadata.get("user_note") or "") == legacy_note
                    and current.memory_type.startswith("material_")
                )
                if (
                    metadata.get("user_managed")
                    and not is_legacy_material_memory
                    and not overwrite_user
                ):
                    return {
                        "action": "conflict",
                        "memory": self._payload(current, self._head_names(repository)),
                    }
                content_hash = self._hash(data.content)
                if (
                    str(metadata.get("material_id") or "") == material_id
                    and int(metadata.get("material_version") or 0) == material_version
                    and current.content_hash == content_hash
                    and current.status == "active"
                ):
                    return {
                        "action": "unchanged",
                        "memory": self._payload(current, self._head_names(repository)),
                    }
                self._write_history(book_id, current)
                ids = self._resolve_ids(repository, data)
                metadata.update(
                    {
                        "origin": "material_sync",
                        "material_id": material_id,
                        "material_version": material_version,
                        "user_managed": False,
                        "user_note": data.note,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                memory = current.model_copy(
                    update={
                        "store_type": "canon_memory",
                        "memory_type": data.memory_type,
                        "content": data.content,
                        "character_ids": ids["character"],
                        "item_ids": ids["item"],
                        "event_ids": ids["event"],
                        "source_chapter": data.source_chapter,
                        "last_mentioned_chapter": max(
                            current.last_mentioned_chapter,
                            data.source_chapter,
                        ),
                        "raw_importance": data.raw_importance,
                        "effective_importance": data.raw_importance,
                        "status": "active",
                        "content_hash": self._hash(data.content),
                        "entity_name": data.entity_name,
                        "field": data.field,
                        "is_current": True,
                        "version": current.version + 1,
                        "metadata": metadata,
                    }
                )
                repository.store(current.store_type).save(memory)
                self._replace_links(repository, memory)
                action = "updated"
            else:
                ids = self._resolve_ids(repository, data)
                now = datetime.now(timezone.utc).isoformat()
                memory = AtomicMemory(
                    memory_id=f"memory_material_{uuid.uuid4().hex[:16]}",
                    book_id=book_id,
                    store_type="canon_memory",
                    memory_type=data.memory_type,
                    content=data.content,
                    character_ids=ids["character"],
                    item_ids=ids["item"],
                    event_ids=ids["event"],
                    source_chapter=data.source_chapter,
                    last_mentioned_chapter=data.source_chapter,
                    raw_importance=data.raw_importance,
                    effective_importance=data.raw_importance,
                    status="active",
                    content_hash=self._hash(data.content),
                    entity_name=data.entity_name,
                    field=data.field,
                    is_current=True,
                    metadata={
                        "origin": "material_sync",
                        "material_id": material_id,
                        "material_version": material_version,
                        "user_managed": False,
                        "user_note": data.note,
                        "created_at": now,
                        "updated_at": now,
                    },
                )
                repository.store("canon_memory").save(memory)
                self._replace_links(repository, memory)
                action = "created"
            payload = self._payload(memory, self._head_names(repository))
        return {"action": action, "memory": payload}

    @staticmethod
    def _find_material_memory(
        repository: BookRepository,
        material_id: str,
    ) -> AtomicMemory | None:
        for store_type in MEMORY_STORES:
            for memory in repository.store(store_type).list_memories(
                statuses=("active", "archived"),
                descending_chapter=True,
            ):
                if str(memory.metadata.get("material_id") or "") == material_id:
                    return memory
                if (
                    memory.version == 1
                    and memory.memory_type.startswith("material_")
                    and str(memory.metadata.get("user_note") or "")
                    == f"由素材库同步：{material_id}"
                ):
                    return memory
        return None

    @staticmethod
    def _head_names(repository: BookRepository) -> dict[str, str]:
        return {
            str(item["head_id"]): str(item["canonical_name"])
            for item in repository.index.list_heads()
        }

    def _material_memory_value(self, material: dict[str, Any]) -> dict[str, Any]:
        lines = [f"{material.get('name', '').strip()}（{material.get('category_label', '素材')}）"]
        lines.extend(
            f"{label}：{self._display_value(value)}"
            for label, value in material.get("display_fields", {}).items()
            if self._display_value(value)
        )
        category = str(material.get("category") or "background")
        character_names = [str(material.get("name"))] if category == "character" else []
        item_names = [str(material.get("name"))] if category == "item" else []
        return {
            "store_type": "canon_memory",
            "memory_type": f"material_{category}",
            "content": "\n".join(lines),
            "raw_importance": 0.85,
            "source_chapter": int(material.get("source_chapter") or 0),
            "character_names": character_names,
            "item_names": item_names,
            "entity_name": str(material.get("name") or "").strip() or None,
            "field": "material_profile",
            "note": f"由素材库同步：{material.get('material_id', '')}",
        }

    @staticmethod
    def _resolve_ids(repository: BookRepository, data: UserMemoryInput) -> dict[str, list[str]]:
        return {
            "character": [
                repository.index.resolve_or_create("character", name)
                for name in data.character_names
            ],
            "item": [
                repository.index.resolve_or_create("item", name)
                for name in data.item_names
            ],
            "event": [
                repository.index.resolve_or_create("event", name)
                for name in data.event_names
            ],
        }

    @staticmethod
    def _replace_links(repository: BookRepository, memory: AtomicMemory) -> None:
        repository.index.remove_memory_links(memory.memory_id)
        for role, ids in (
            ("character", memory.character_ids),
            ("item", memory.item_ids),
            ("event", memory.event_ids),
        ):
            for head_id in ids:
                repository.index.link(
                    head_id,
                    memory.memory_id,
                    memory.store_type,
                    role,
                    memory.source_chapter,
                )

    def _write_history(self, book_id: str, memory: AtomicMemory) -> None:
        folder = self._history_dir(book_id, memory.memory_id)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"version_{memory.version:06d}.json"
        if not path.exists():
            self._atomic_json(path, memory.model_dump(mode="json"))

    def _history_dir(self, book_id: str, memory_id: str) -> Path:
        return self.config.book_dir(book_id) / "memory_history" / memory_id

    @staticmethod
    def _payload(memory: AtomicMemory, heads: dict[str, str]) -> dict[str, Any]:
        payload = memory.model_dump(mode="json")
        payload.update(
            {
                "store_label": STORE_LABELS[memory.store_type],
                "character_names": [heads.get(item, item) for item in memory.character_ids],
                "item_names": [heads.get(item, item) for item in memory.item_ids],
                "event_names": [heads.get(item, item) for item in memory.event_ids],
                "note": str(memory.metadata.get("user_note") or ""),
                "user_managed": bool(memory.metadata.get("user_managed")),
            }
        )
        return payload

    @staticmethod
    def _hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _display_value(value: Any) -> str:
        if isinstance(value, list):
            return "、".join(str(item) for item in value if str(item).strip())
        return str(value or "").strip()

    @staticmethod
    def _atomic_json(path: Path, value: Any) -> None:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)
