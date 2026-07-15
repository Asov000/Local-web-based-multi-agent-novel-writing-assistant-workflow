from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator


RAG_MESSAGE_VERSION = "rag.message.v1"

RAGOperationType = Literal[
    "create",
    "update",
    "delete",
    "compress",
    "merge",
    "relink",
    "supersede",
    "archive",
    "restore",
    "update_importance",
    "flag_conflict",
    "remove_orphan_link",
    "no_op",
]


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class RAGOperation(BaseModel):
    """A model-proposed operation. Python validates it before any write."""

    operation_id: str = Field(default_factory=lambda: _id("op"))
    operation: RAGOperationType
    source_ids: list[str] = Field(default_factory=list)
    target_id: str | None = None
    memory_id: str | None = None
    old_memory_id: str | None = None
    new_memory_id: str | None = None
    created_memory_id: str | None = None
    add_head_ids: list[str] = Field(default_factory=list)
    remove_head_ids: list[str] = Field(default_factory=list)
    orphan_head_id: str | None = None
    orphan_role: str | None = None
    expected_versions: dict[str, int] = Field(default_factory=dict)

    store_type: str | None = None
    memory_type: str | None = None
    content: str | None = None
    new_content: str | None = None
    summary: str | None = None
    source_chapter: int = Field(default=0, ge=0)
    entity_name: str | None = None
    field: str | None = None
    character_ids: list[str] = Field(default_factory=list)
    item_ids: list[str] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)
    metadata_patch: dict[str, Any] = Field(default_factory=dict)

    raw_importance: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reason: str

    @model_validator(mode="after")
    def ensure_created_memory_id(self) -> "RAGOperation":
        if self.operation in {"create", "compress"} and not self.created_memory_id:
            self.created_memory_id = _id("memory")
        if self.operation == "create":
            if not self.store_type or not self.memory_type or not (self.content or "").strip():
                raise ValueError("create requires store_type, memory_type, and content")
            if self.raw_importance is None:
                raise ValueError("create requires raw_importance")
        elif self.operation == "update":
            self._require_existing(self.memory_id)
            if not any(
                (self.new_content is not None, self.raw_importance is not None, self.metadata_patch)
            ):
                raise ValueError("update requires at least one changed field")
        elif self.operation == "delete":
            self._require_existing(self.memory_id)
        elif self.operation == "compress":
            if len(self.source_ids) < 2 or not (self.summary or "").strip():
                raise ValueError("compress requires source_ids and summary")
            self._require_versions(self.source_ids)
        elif self.operation == "merge":
            if not self.target_id or not self.source_ids:
                raise ValueError("merge requires target_id and source_ids")
            self._require_versions([self.target_id, *self.source_ids])
        elif self.operation == "relink":
            self._require_existing(self.memory_id)
        elif self.operation == "supersede":
            if not self.old_memory_id or not self.new_memory_id:
                raise ValueError("supersede requires old_memory_id and new_memory_id")
            self._require_versions([self.old_memory_id, self.new_memory_id])
        elif self.operation in {"archive", "restore", "update_importance"}:
            self._require_existing(self.memory_id)
            if self.operation == "update_importance" and self.raw_importance is None:
                raise ValueError("update_importance requires raw_importance")
        elif self.operation == "flag_conflict":
            if len(self.source_ids) < 2:
                raise ValueError("flag_conflict requires at least two source_ids")
            self._require_versions(self.source_ids)
        elif self.operation == "remove_orphan_link":
            if not self.memory_id or not self.orphan_head_id:
                raise ValueError("remove_orphan_link requires memory_id and orphan_head_id")
        return self

    def _require_existing(self, memory_id: str | None) -> None:
        if not memory_id:
            raise ValueError(f"{self.operation} requires memory_id")
        self._require_versions([memory_id])

    def _require_versions(self, memory_ids: list[str]) -> None:
        missing = [item for item in memory_ids if item not in self.expected_versions]
        if missing:
            raise ValueError(f"{self.operation} is missing expected_versions for {missing}")

    def to_audit_operation(self):
        from .maintenance_schemas import AuditOperation

        return AuditOperation.model_validate(self.model_dump())


class RAGMessage(BaseModel):
    """Unified envelope for Qwen, MemoryAgent, and database operations."""

    schema_version: Literal["rag.message.v1"] = RAG_MESSAGE_VERSION
    message_id: str = Field(default_factory=lambda: _id("ragmsg"))
    task_id: str = Field(default_factory=lambda: _id("task"))
    parent_message_id: str | None = None
    sender: str
    receiver: str
    message_type: Literal["request", "response"] = "request"
    action: str
    status: Literal["pending", "ok", "need_user_input", "error"] = "pending"
    book_id: str | None = None
    approval: Literal["not_required", "pending", "confirmed"] = "not_required"
    dry_run: bool = True
    operations: list[RAGOperation] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    @classmethod
    def from_agent_message(
        cls,
        message: Any,
        *,
        book_id: str | None = None,
    ) -> "RAGMessage":
        if isinstance(message, cls):
            return message
        data = message.model_dump() if hasattr(message, "model_dump") else dict(message)
        return cls(
            message_id=data.get("message_id") or _id("ragmsg"),
            task_id=data.get("task_id") or _id("task"),
            parent_message_id=data.get("parent_message_id"),
            sender=str(data.get("sender") or "unknown"),
            receiver=str(data.get("receiver") or "unknown"),
            message_type=data.get("message_type", "request"),
            action=str(data.get("action") or "rag.legacy"),
            status=data.get("status", "pending"),
            book_id=book_id or data.get("book_id"),
            payload=data.get("payload") or {},
            metadata={**(data.get("metadata") or {}), "legacy_agent_message": True},
            error=data.get("error"),
        )

    def response(
        self,
        *,
        sender: str,
        action: str | None = None,
        status: Literal["ok", "need_user_input", "error"] = "ok",
        payload: dict[str, Any] | None = None,
        operations: list[RAGOperation] | None = None,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> "RAGMessage":
        return RAGMessage(
            task_id=self.task_id,
            parent_message_id=self.message_id,
            sender=sender,
            receiver=self.sender,
            message_type="response",
            action=action or self.action,
            status=status,
            book_id=self.book_id,
            approval=self.approval,
            dry_run=self.dry_run,
            operations=operations or [],
            payload=payload or {},
            metadata=metadata or {},
            error=error,
        )


def extract_rag_payload(
    raw: dict[str, Any],
    *,
    expected_action: str | None = None,
) -> tuple[dict[str, Any], RAGMessage | None]:
    """Validate a RAGMessage response while accepting legacy payloads during migration."""

    if raw.get("schema_version") != RAG_MESSAGE_VERSION:
        return raw, None
    candidate = dict(raw)
    repaired_fields: list[str] = []
    if expected_action and not candidate.get("action"):
        candidate["action"] = expected_action
        repaired_fields.append("action")
    if expected_action and ".operations." in expected_action:
        nested_operations = (candidate.get("payload") or {}).get("operations")
        if not candidate.get("operations") and isinstance(nested_operations, list):
            candidate["operations"] = nested_operations
            repaired_fields.append("operations_from_payload")
    if repaired_fields:
        candidate["metadata"] = {
            **(candidate.get("metadata") or {}),
            "repaired_envelope_fields": repaired_fields,
        }
    try:
        message = RAGMessage.model_validate(candidate)
    except ValidationError:
        if expected_action and ".operations." in expected_action:
            raise
        sanitized = dict(candidate)
        rejected_count = len(candidate.get("operations") or [])
        sanitized["operations"] = []
        sanitized["metadata"] = {
            **(candidate.get("metadata") or {}),
            "rejected_out_of_scope_operations": rejected_count,
        }
        message = RAGMessage.model_validate(sanitized)
    if message.message_type != "response":
        raise ValueError("Qwen RAGMessage must be a response")
    if message.status != "ok":
        raise ValueError(message.error or "Qwen RAGMessage returned an error")
    if expected_action and message.action != expected_action:
        raise ValueError(
            f"Unexpected RAGMessage action: {message.action}; expected {expected_action}"
        )
    return message.payload, message
