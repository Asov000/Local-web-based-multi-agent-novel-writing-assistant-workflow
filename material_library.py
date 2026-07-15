from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from rag.config import RagConfig
from rag.normalizer import parse_scored_value


MaterialCategory = Literal["character", "background", "faction", "location", "item"]

MATERIAL_SCHEMAS: dict[str, dict[str, Any]] = {
    "character": {
        "label": "角色",
        "fields": [
            ("aliases", "别名", "list"), ("identity", "身份", "text"),
            ("appearance", "外貌", "text"), ("personality", "性格", "text"),
            ("background", "经历", "textarea"), ("goal", "目标", "text"),
            ("ability", "能力", "textarea"), ("relations", "关系", "list"),
            ("current_state", "当前状态", "text"),
        ],
    },
    "background": {
        "label": "背景",
        "fields": [
            ("era", "时代", "text"), ("overview", "世界概述", "textarea"),
            ("history", "历史", "textarea"), ("social_structure", "社会结构", "textarea"),
            ("rules", "世界规则", "list"), ("core_conflict", "核心冲突", "textarea"),
        ],
    },
    "faction": {
        "label": "势力",
        "fields": [
            ("nature", "性质", "text"), ("goal", "目标", "textarea"),
            ("leader", "首领", "text"), ("members", "成员", "list"),
            ("structure", "组织结构", "textarea"), ("territory", "活动范围", "text"),
            ("resources", "资源", "list"), ("allies", "盟友", "list"),
            ("enemies", "敌人", "list"), ("description", "描述", "textarea"),
        ],
    },
    "location": {
        "label": "地点",
        "fields": [
            ("location_type", "类型", "text"), ("region", "所属区域", "text"),
            ("environment", "环境", "textarea"), ("features", "特点", "list"),
            ("faction", "所属势力", "text"), ("entry_conditions", "进入条件", "textarea"),
            ("related_events", "相关事件", "list"), ("description", "描述", "textarea"),
        ],
    },
    "item": {
        "label": "物品",
        "fields": [
            ("item_type", "类型", "text"), ("description", "描述", "textarea"),
            ("function", "作用", "textarea"), ("holder", "持有者", "text"),
            ("origin", "来源", "textarea"), ("conditions", "使用条件", "textarea"),
            ("limitations", "限制", "textarea"), ("current_state", "当前状态", "text"),
        ],
    },
}


class MaterialInput(BaseModel):
    category: MaterialCategory
    name: str = Field(min_length=1, max_length=120)
    fields: dict[str, Any] = Field(default_factory=dict)
    note: str = ""

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, value: Any) -> str:
        return str(value or "").strip()


class MaterialLibrary:
    """Local, AI-independent material storage with fixed category schemas."""

    def __init__(self, root_dir: str | Path = "rag_data") -> None:
        self.config = RagConfig(root_dir=Path(root_dir))
        self._lock = threading.RLock()

    @staticmethod
    def schemas() -> dict[str, Any]:
        return {
            key: {
                "label": value["label"],
                "fields": [
                    {"key": field[0], "label": field[1], "type": field[2]}
                    for field in value["fields"]
                ],
            }
            for key, value in MATERIAL_SCHEMAS.items()
        }

    def list(self, book_id: str, *, include_deleted: bool = False) -> dict[str, Any]:
        records = self._read(book_id)
        if not include_deleted:
            records = [item for item in records if item.get("status") != "deleted"]
        records.sort(key=lambda item: (str(item.get("category")), str(item.get("name"))))
        return {"book_id": book_id, "materials": [self._decorate(item) for item in records]}

    def get(self, book_id: str, material_id: str) -> dict[str, Any]:
        for item in self._read(book_id):
            if item.get("material_id") == material_id:
                return self._decorate(item)
        raise KeyError("素材不存在")

    def save(
        self,
        book_id: str,
        value: dict[str, Any],
        *,
        material_id: str | None = None,
        origin: str = "user",
        source_chapter: int = 0,
    ) -> dict[str, Any]:
        data = MaterialInput.model_validate(value)
        fields = self._normalize_fields(data.category, data.fields)
        with self._lock:
            records = self._read(book_id)
            index = next(
                (i for i, item in enumerate(records) if item.get("material_id") == material_id),
                None,
            )
            now = datetime.now(timezone.utc).isoformat()
            if index is None:
                record = {
                    "material_id": material_id or f"material_{uuid.uuid4().hex[:16]}",
                    "book_id": book_id,
                    "category": data.category,
                    "name": data.name,
                    "fields": fields,
                    "note": data.note.strip(),
                    "origin": origin,
                    "source_chapter": max(0, int(source_chapter)),
                    "status": "active",
                    "version": 1,
                    "created_at": now,
                    "updated_at": now,
                }
                records.append(record)
            else:
                current = records[index]
                self._write_history(book_id, current)
                record = {
                    **current,
                    "category": data.category,
                    "name": data.name,
                    "fields": fields,
                    "note": data.note.strip(),
                    "origin": "user" if origin == "user" else current.get("origin", origin),
                    "status": "active",
                    "version": int(current.get("version") or 1) + 1,
                    "updated_at": now,
                }
                records[index] = record
            self._write(book_id, records)
        return self._decorate(record)

    def delete(self, book_id: str, material_id: str) -> dict[str, Any]:
        with self._lock:
            records = self._read(book_id)
            for index, current in enumerate(records):
                if current.get("material_id") != material_id:
                    continue
                self._write_history(book_id, current)
                records[index] = {
                    **current,
                    "status": "deleted",
                    "version": int(current.get("version") or 1) + 1,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                self._write(book_id, records)
                return {"material_id": material_id, "deleted": True}
        raise KeyError("素材不存在")

    def save_many(
        self,
        book_id: str,
        candidates: list[dict[str, Any]],
        *,
        origin: str,
        source_chapter: int = 0,
    ) -> list[dict[str, Any]]:
        return [
            self.save(
                book_id,
                candidate,
                origin=origin,
                source_chapter=source_chapter,
            )
            for candidate in candidates
        ]

    def apply_decisions(
        self,
        book_id: str,
        decisions: list[dict[str, Any]],
        *,
        origin: str,
        source_chapter: int = 0,
    ) -> list[dict[str, Any]]:
        saved: list[dict[str, Any]] = []
        for decision in decisions:
            action = str(decision.get("action") or "create").strip().lower()
            if action == "skip":
                continue
            if action not in {"create", "merge"}:
                raise ValueError("素材确认操作只能是create、merge或skip")
            value = decision.get("value")
            if not isinstance(value, dict):
                raise ValueError("待保存素材格式无效")
            incoming = MaterialInput.model_validate(value)
            save_value = incoming.model_dump(mode="json")
            material_id: str | None = None
            if action == "merge":
                material_id = str(decision.get("material_id") or "").strip()
                if not material_id:
                    raise ValueError("合并素材必须指定现有素材")
                current = self.get(book_id, material_id)
                if current["category"] != incoming.category:
                    raise ValueError("只能合并相同类型的素材")
                fields = dict(current.get("fields") or {})
                fields.update(
                    {
                        key: field_value
                        for key, field_value in incoming.fields.items()
                        if self._has_value(field_value)
                    }
                )
                save_value = {
                    "category": incoming.category,
                    "name": incoming.name or current["name"],
                    "fields": fields,
                    "note": incoming.note or str(current.get("note") or ""),
                }
            saved.append(
                self.save(
                    book_id,
                    save_value,
                    material_id=material_id,
                    origin=origin,
                    source_chapter=source_chapter,
                )
            )
        return saved

    def mark_synced(
        self,
        book_id: str,
        material_id: str,
        memory_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            records = self._read(book_id)
            for index, current in enumerate(records):
                if current.get("material_id") != material_id:
                    continue
                records[index] = {
                    **current,
                    "sync": {
                        "memory_id": memory_id,
                        "material_version": int(current.get("version") or 1),
                        "synced_at": datetime.now(timezone.utc).isoformat(),
                    },
                }
                self._write(book_id, records)
                return self._decorate(records[index])
        raise KeyError("素材不存在")

    def preview_writer_result(self, task_code: str, result: dict[str, Any]) -> list[dict[str, Any]]:
        code = task_code.strip().upper()
        if code == "BD":
            return self._world_candidates(result)
        if code == "CH":
            return self._character_candidates(result)
        raise ValueError("素材创作只支持世界观和人物任务")

    def save_review(
        self,
        book_id: str,
        *,
        chapter_id: int,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        review_id = f"review_{uuid.uuid4().hex[:16]}"
        payload = {
            "review_id": review_id,
            "book_id": book_id,
            "chapter_id": chapter_id,
            "status": "pending",
            "candidates": candidates,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self._reviews_dir(book_id) / f"{review_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_json(path, payload)
        return payload

    def list_reviews(self, book_id: str) -> list[dict[str, Any]]:
        folder = self._reviews_dir(book_id)
        if not folder.is_dir():
            return []
        result: list[dict[str, Any]] = []
        for path in sorted(folder.glob("review_*.json"), reverse=True):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if value.get("status") == "pending":
                result.append(value)
        return result

    def confirm_review(
        self,
        book_id: str,
        review_id: str,
        *,
        decisions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        path = self._reviews_dir(book_id) / f"{review_id}.json"
        if not path.is_file():
            raise KeyError("待确认素材不存在")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "pending":
            raise ValueError("该批素材已经处理")
        source_chapter = int(payload.get("chapter_id") or 0)
        if decisions is None:
            saved = self.save_many(
                book_id,
                list(payload.get("candidates") or []),
                origin="qwen_extract",
                source_chapter=source_chapter,
            )
        else:
            saved = self.apply_decisions(
                book_id,
                decisions,
                origin="qwen_extract",
                source_chapter=source_chapter,
            )
        payload["status"] = "confirmed"
        payload["confirmed_at"] = datetime.now(timezone.utc).isoformat()
        payload["decisions"] = decisions
        self._atomic_json(path, payload)
        return {"review_id": review_id, "saved": saved}

    @staticmethod
    def _has_value(value: Any) -> bool:
        if isinstance(value, list):
            return any(str(item).strip() for item in value)
        return bool(str(value or "").strip())

    @staticmethod
    def _world_candidates(result: dict[str, Any]) -> list[dict[str, Any]]:
        world_name = str(result.get("world_name") or "世界观").strip()
        background, _, _ = parse_scored_value(result.get("background"))
        conflict, _, _ = parse_scored_value(result.get("conflict"))
        rules = [parse_scored_value(item)[0] for item in result.get("rules", [])]
        candidates: list[dict[str, Any]] = [
            {
                "category": "background",
                "name": world_name,
                "fields": {
                    "overview": background,
                    "rules": [item for item in rules if item],
                    "core_conflict": conflict,
                },
            }
        ]
        for item in result.get("factions", []):
            text, _, _ = parse_scored_value(item)
            if text:
                candidates.append(
                    {"category": "faction", "name": MaterialLibrary._short_name(text), "fields": {"description": text}}
                )
        for item in result.get("locations", []):
            text, _, _ = parse_scored_value(item)
            if text:
                candidates.append(
                    {"category": "location", "name": MaterialLibrary._short_name(text), "fields": {"description": text}}
                )
        return candidates

    @staticmethod
    def _character_candidates(result: dict[str, Any]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for value in result.get("characters", []):
            if not isinstance(value, dict):
                continue
            name = str(value.get("name") or "").strip()
            if not name:
                continue
            fields: dict[str, Any] = {}
            mapping = {
                "role": "identity", "appearance": "appearance", "personality": "personality",
                "background": "background", "goal": "goal", "ability": "ability",
            }
            for source, target in mapping.items():
                text, _, _ = parse_scored_value(value.get(source))
                fields[target] = text
            fields["relations"] = [
                parse_scored_value(item)[0]
                for item in value.get("relations", [])
                if parse_scored_value(item)[0]
            ]
            candidates.append({"category": "character", "name": name, "fields": fields})
        return candidates

    @staticmethod
    def _short_name(text: str) -> str:
        first = re.split(r"[，。；：,:;]", text, maxsplit=1)[0].strip()
        return (first or text)[:40]

    @staticmethod
    def _normalize_fields(category: str, fields: dict[str, Any]) -> dict[str, Any]:
        schema = MATERIAL_SCHEMAS[category]
        normalized: dict[str, Any] = {}
        for key, _label, field_type in schema["fields"]:
            value = fields.get(key)
            if field_type == "list":
                if isinstance(value, str):
                    values = re.split(r"[,，、;；\n]+", value)
                else:
                    values = value if isinstance(value, list) else []
                normalized[key] = list(
                    dict.fromkeys(str(item).strip() for item in values if str(item).strip())
                )
            else:
                normalized[key] = str(value or "").strip()
        return normalized

    @staticmethod
    def _decorate(record: dict[str, Any]) -> dict[str, Any]:
        value = dict(record)
        schema = MATERIAL_SCHEMAS[str(record["category"])]
        value["category_label"] = schema["label"]
        labels = {key: label for key, label, _type in schema["fields"]}
        value["display_fields"] = {
            labels.get(key, key): field_value
            for key, field_value in dict(record.get("fields") or {}).items()
        }
        sync = value.get("sync") if isinstance(value.get("sync"), dict) else {}
        synced_version = int(sync.get("material_version") or 0)
        current_version = int(value.get("version") or 1)
        if sync.get("memory_id"):
            value["sync_status"] = (
                "synced" if synced_version == current_version else "pending"
            )
        else:
            value["sync_status"] = "unsynced"
        return value

    def _read(self, book_id: str) -> list[dict[str, Any]]:
        path = self._library_path(book_id)
        if not path.is_file():
            return []
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"素材库文件损坏: {path}") from exc
        return list(value.get("materials") or []) if isinstance(value, dict) else []

    def _write(self, book_id: str, records: list[dict[str, Any]]) -> None:
        path = self._library_path(book_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_json(path, {"schema_version": "1.0", "materials": records})

    def _write_history(self, book_id: str, record: dict[str, Any]) -> None:
        folder = self.config.book_dir(book_id) / "materials" / "history" / str(record["material_id"])
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"version_{int(record.get('version') or 1):06d}.json"
        if not path.exists():
            self._atomic_json(path, record)

    def _library_path(self, book_id: str) -> Path:
        return self.config.book_dir(book_id) / "materials" / "library.json"

    def _reviews_dir(self, book_id: str) -> Path:
        return self.config.book_dir(book_id) / "materials" / "reviews"

    @staticmethod
    def _atomic_json(path: Path, value: Any) -> None:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)
