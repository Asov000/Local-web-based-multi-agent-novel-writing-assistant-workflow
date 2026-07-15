from __future__ import annotations

import json
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


AuditSeverity = Literal["info", "warning", "error", "critical"]
AuditScopeMode = Literal["book", "chapters", "global"]
AuditOperationType = Literal[
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


def maintenance_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


HEALTHY_FINDING_STATUSES = {
    "consistent",
    "healthy",
    "ok",
    "pass",
    "valid",
    "no_conflict",
    "no_issue",
    "unchanged",
}


def _filter_model_audit_collections(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    result = dict(value)
    findings = result.get("findings")
    if isinstance(findings, list):
        result["findings"] = [
            item
            for item in findings
            if not (
                isinstance(item, dict)
                and str(item.get("status") or "").strip().casefold()
                in HEALTHY_FINDING_STATUSES
            )
        ]
    operations = result.get("operations")
    if isinstance(operations, list):
        result["operations"] = [
            item
            for item in operations
            if not (
                isinstance(item, dict)
                and str(item.get("operation") or item.get("operation_type") or "")
                .strip()
                .casefold()
                == "no_op"
            )
        ]
    return result


class AuditOperation(BaseModel):
    operation_id: str = Field(default_factory=lambda: maintenance_id("op"))
    operation: AuditOperationType
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
    store_type: Literal[
        "canon_memory",
        "chapter_memory",
        "state_timeline_memory",
        "relation_hook_memory",
    ] | None = None
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

    @model_validator(mode="before")
    @classmethod
    def normalize_model_shape(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        result = dict(value)
        operation = str(
            result.get("operation") or result.get("operation_type") or ""
        ).strip()
        if operation and not result.get("operation"):
            result["operation"] = operation
        target_memory_id = result.get("target_memory_id")
        if target_memory_id:
            if operation == "merge" and not result.get("target_id"):
                result["target_id"] = target_memory_id
            elif operation not in {"create", "compress", "supersede"} and not result.get(
                "memory_id"
            ):
                result["memory_id"] = target_memory_id
        if not result.get("reason"):
            for alias in ("explanation", "description", "notes", "recommendation"):
                if str(result.get(alias) or "").strip():
                    result["reason"] = str(result[alias]).strip()
                    break
            if operation == "no_op" and not result.get("reason"):
                result["reason"] = "模型确认无需变更"
        versions = result.get("expected_versions")
        if isinstance(versions, list):
            mapped: dict[str, int] = {}
            for item in versions:
                if not isinstance(item, dict):
                    continue
                memory_id = str(
                    item.get("memory_id") or item.get("id") or ""
                ).strip()
                try:
                    version = int(item.get("version"))
                except (TypeError, ValueError):
                    continue
                if memory_id:
                    mapped[memory_id] = version
            if mapped or operation == "no_op":
                result["expected_versions"] = mapped
        return result

    def referenced_memory_ids(self) -> list[str]:
        values = [
            *self.source_ids,
            self.target_id,
            self.memory_id,
            self.old_memory_id,
            self.new_memory_id,
        ]
        return list(dict.fromkeys(value for value in values if value))


class AuditIssue(BaseModel):
    issue_id: str = Field(default_factory=lambda: maintenance_id("issue"))
    code: str
    severity: AuditSeverity
    message: str
    memory_ids: list[str] = Field(default_factory=list)
    head_ids: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    deterministic_operation: AuditOperation | None = None


class AuditFinding(BaseModel):
    finding_id: str = Field(default_factory=lambda: maintenance_id("finding"))
    code: str
    severity: AuditSeverity = "warning"
    summary: str
    memory_ids: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_model_shape(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        result = dict(value)
        if not result.get("memory_ids") and result.get("memory_id"):
            result["memory_ids"] = [str(result["memory_id"])]
        if not result.get("code"):
            for alias in ("issue_type", "category", "type", "status"):
                if str(result.get(alias) or "").strip():
                    result["code"] = str(result[alias]).strip().casefold()
                    break
        if not result.get("summary"):
            for alias in ("notes", "description", "message", "issue", "reason"):
                if str(result.get(alias) or "").strip():
                    result["summary"] = str(result[alias]).strip()
                    break
        evidence = result.get("evidence")
        if isinstance(evidence, str):
            result["evidence"] = [evidence]
        elif isinstance(evidence, list):
            result["evidence"] = [
                item
                if isinstance(item, str)
                else json.dumps(item, ensure_ascii=False, sort_keys=True)
                for item in evidence
                if item is not None
            ]
        severity = str(result.get("severity") or "").strip().casefold()
        result["severity"] = {
            "low": "info",
            "medium": "warning",
            "high": "error",
            "fatal": "critical",
        }.get(severity, severity or "warning")
        return result


class RelatedMemoryRef(BaseModel):
    memory_id: str
    owner_head_id: str
    memory_type: str
    source_chapter: int = Field(ge=0)
    effective_importance: float = Field(ge=0.0)
    version: int = Field(ge=1)
    relation_roles: list[str] = Field(default_factory=list)


class AuditScope(BaseModel):
    mode: AuditScopeMode = "book"
    chapter_ids: list[int] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def validate_scope(self) -> "AuditScope":
        self.chapter_ids = list(dict.fromkeys(self.chapter_ids))
        if any(chapter_id <= 0 for chapter_id in self.chapter_ids):
            raise ValueError("审计章节必须是大于0的整数")
        if self.mode == "chapters" and not self.chapter_ids:
            raise ValueError("章节审计至少需要选择一个章节")
        if self.mode != "chapters" and self.chapter_ids:
            raise ValueError("只有章节审计可以指定chapter_ids")
        return self


class MemoryComparisonCandidate(BaseModel):
    memory_id: str
    reason_codes: list[str] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    memory: dict[str, Any]


class MemoryComparisonPacket(BaseModel):
    batch_id: str = Field(default_factory=lambda: maintenance_id("compare_batch"))
    book_id: str
    scope: AuditScope
    query_memory_id: str
    query_memory: dict[str, Any]
    candidate_memory_ids: list[str] = Field(max_length=9)
    candidates: list[MemoryComparisonCandidate] = Field(max_length=9)

    @model_validator(mode="after")
    def validate_candidates(self) -> "MemoryComparisonPacket":
        candidate_ids = [candidate.memory_id for candidate in self.candidates]
        if not candidate_ids:
            raise ValueError("比较批次至少需要一条候选记忆")
        if candidate_ids != self.candidate_memory_ids:
            raise ValueError("candidate_memory_ids与candidates顺序不一致")
        if self.query_memory_id in candidate_ids:
            raise ValueError("查询记忆不能同时作为候选记忆")
        return self


class MemoryComparisonResult(BaseModel):
    batch_id: str
    query_memory_id: str
    reviewed_candidate_ids: list[str] = Field(default_factory=list)
    findings: list[AuditFinding] = Field(default_factory=list)
    operations: list[AuditOperation] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def remove_healthy_assessments(cls, value: Any) -> Any:
        return _filter_model_audit_collections(value)


class AuditPacket(BaseModel):
    packet_id: str = Field(default_factory=lambda: maintenance_id("packet"))
    book_id: str
    focus_head: dict[str, Any] | None = None
    primary_memory_ids: list[str]
    context_memory_ids: list[str] = Field(default_factory=list)
    memories: list[dict[str, Any]]
    related_memory_refs: list[RelatedMemoryRef] = Field(default_factory=list)
    related_heads: list[dict[str, Any]] = Field(default_factory=list)
    global_context: dict[str, Any] = Field(default_factory=dict)
    deterministic_issues: list[AuditIssue] = Field(default_factory=list)
    retry_for_memory_ids: list[str] = Field(default_factory=list)


class AuditAgentResult(BaseModel):
    packet_id: str
    reviewed_memory_ids: list[str] = Field(default_factory=list)
    findings: list[AuditFinding] = Field(default_factory=list)
    operations: list[AuditOperation] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def remove_healthy_assessments(cls, value: Any) -> Any:
        return _filter_model_audit_collections(value)


class CrossAuditCandidate(BaseModel):
    candidate_id: str = Field(default_factory=lambda: maintenance_id("candidate"))
    memory_ids: list[str]
    reason_codes: list[str]
    score: float = Field(ge=0.0, le=1.0)
    shared_head_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_pair(self) -> "CrossAuditCandidate":
        self.memory_ids = list(dict.fromkeys(self.memory_ids))
        if len(self.memory_ids) != 2:
            raise ValueError("cross audit candidate must reference exactly two memories")
        return self


class CrossAuditPacket(BaseModel):
    packet_id: str = Field(default_factory=lambda: maintenance_id("cross_packet"))
    book_id: str
    candidates: list[CrossAuditCandidate]
    memories: list[dict[str, Any]]
    global_context: dict[str, Any] = Field(default_factory=dict)


class CrossAuditResult(BaseModel):
    packet_id: str
    reviewed_candidate_ids: list[str] = Field(default_factory=list)
    findings: list[AuditFinding] = Field(default_factory=list)
    operations: list[AuditOperation] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def remove_healthy_assessments(cls, value: Any) -> Any:
        return _filter_model_audit_collections(value)


class ReconcilePayload(BaseModel):
    book_id: str
    global_context: dict[str, Any]
    findings: list[AuditFinding]
    proposed_operations: list[AuditOperation]


class ReconcileResult(BaseModel):
    reviewed_finding_ids: list[str] = Field(default_factory=list)
    findings: list[AuditFinding] = Field(default_factory=list)
    operations: list[AuditOperation] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def remove_healthy_assessments(cls, value: Any) -> Any:
        return _filter_model_audit_collections(value)


class AuditCoverage(BaseModel):
    total_memory_ids: list[str]
    assigned_memory_ids: list[str]
    reviewed_memory_ids: list[str]
    uncovered_memory_ids: list[str] = Field(default_factory=list)
    unreviewed_memory_ids: list[str] = Field(default_factory=list)
    duplicate_primary_memory_ids: list[str] = Field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not (
            self.uncovered_memory_ids
            or self.unreviewed_memory_ids
            or self.duplicate_primary_memory_ids
        )


class PatchPlan(BaseModel):
    plan_id: str = Field(default_factory=lambda: maintenance_id("plan"))
    book_id: str
    run_id: str
    scope: AuditScope = Field(default_factory=AuditScope)
    operations: list[AuditOperation] = Field(default_factory=list)
    coverage: AuditCoverage
    blocking_issue_ids: list[str] = Field(default_factory=list)


class AuditRunResult(BaseModel):
    run_id: str
    book_id: str
    scope: AuditScope = Field(default_factory=AuditScope)
    dry_run: bool
    applied: bool
    snapshot_id: str | None = None
    artifact_dir: str
    deterministic_issue_count: int
    model_finding_count: int
    operation_count: int
    coverage: AuditCoverage
    semantic_candidate_count: int = Field(default=0, ge=0)
    semantic_candidate_reviewed_count: int = Field(default=0, ge=0)
    semantic_candidate_complete: bool = True
    comparison_batch_count: int = Field(default=0, ge=0)
    blocking_issue_ids: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
