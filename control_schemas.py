from __future__ import annotations

import uuid
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from rag.schemas import TaskCode


ControlIntentName = Literal[
    "refine_setting",
    "start_writing",
    "revise_draft",
    "approve_draft",
    "continue_writing",
    "request_memory_audit",
    "confirm_audit",
    "confirm_audit_apply",
    "cancel",
    "general_question",
    "unknown",
]

ControlPhase = Literal[
    "task_selection",
    "setting_input",
    "refine_confirmation",
    "writing",
    "draft_review",
    "revision",
    "archive",
    "post_archive",
    "audit_confirmation",
    "audit_dry_run",
    "audit_apply_confirmation",
    "completed",
    "cancelled",
]


class ControlIntent(BaseModel):
    intent: ControlIntentName
    confidence: float = Field(ge=0.0, le=1.0)
    feedback: str = ""
    target_hint: str = ""
    is_logic_error: bool = False
    needs_rag: bool = False
    entities: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification: str = ""

    @field_validator("intent", mode="before")
    @classmethod
    def normalize_intent(cls, value: Any) -> Any:
        clean = str(value or "").strip()
        aliases = {
            "细化设定": "refine_setting",
            "扩写设定": "refine_setting",
            "开始写作": "start_writing",
            "创作": "start_writing",
            "修改草稿": "revise_draft",
            "修改": "revise_draft",
            "通过": "approve_draft",
            "批准草稿": "approve_draft",
            "续写": "continue_writing",
            "整理记忆": "request_memory_audit",
            "整理数据库": "request_memory_audit",
            "取消": "cancel",
            "提问": "general_question",
            "未知": "unknown",
        }
        return aliases.get(clean, clean or "unknown")

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: Any) -> float:
        if isinstance(value, str):
            clean = value.strip().rstrip("%")
            try:
                number = float(clean)
            except ValueError:
                return 0.0
            return number / 100.0 if number > 1.0 else number
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @field_validator("entities", mode="before")
    @classmethod
    def normalize_entities(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            values = re.split(r"[,，、;；\n]+", value)
        elif isinstance(value, list):
            values = value
        elif isinstance(value, dict):
            values = []
            entity_keys = {
                "entity",
                "entities",
                "character",
                "characters",
                "item",
                "items",
                "event",
                "events",
                "实体",
                "人物",
                "角色",
                "物品",
                "事件",
            }
            for key, item in value.items():
                if str(key).strip().casefold() not in entity_keys:
                    continue
                if isinstance(item, list):
                    values.extend(item)
                elif isinstance(item, str) and len(item.strip()) <= 40:
                    values.append(item)
        else:
            values = [value]
        return list(
            dict.fromkeys(
                text
                for item in values
                if (text := str(item).strip()) and len(text) <= 80
            )
        )

    @field_validator(
        "is_logic_error",
        "needs_rag",
        "needs_clarification",
        mode="before",
    )
    @classmethod
    def normalize_boolean(cls, value: Any) -> bool:
        if isinstance(value, str):
            clean = value.strip().casefold()
            if clean in {"true", "yes", "1", "是", "需要"}:
                return True
            if clean in {"false", "no", "0", "否", "不需要", ""}:
                return False
        return bool(value)


class ControlWriterPayload(BaseModel):
    schema_version: str = "1.0"
    task_code: TaskCode
    operation: Literal["generate", "revise", "extend"]
    user_input: str
    context: dict[str, Any] = Field(default_factory=dict)
    original_result: dict[str, Any] | None = None
    revision: dict[str, Any] = Field(default_factory=dict)
    output_contract: dict[str, Any] = Field(default_factory=dict)

    @field_validator("context", mode="after")
    @classmethod
    def validate_context_contract(cls, value: dict[str, Any]) -> dict[str, Any]:
        result = dict(value)
        if "continuity" in result:
            result["continuity"] = ControlContinuityContext.model_validate(
                result["continuity"]
            ).model_dump(mode="json")
        if "plot_overview" in result:
            result["plot_overview"] = ControlPlotOverviewContext.model_validate(
                result["plot_overview"]
            ).model_dump(mode="json")
        return result


class ControlContinuityContext(BaseModel):
    schema_version: str = "1.0"
    book_id: str
    source_chapter_id: int = Field(ge=0)
    source_chapter_title: str = ""
    ending_excerpt: str
    excerpt_strategy: Literal["chapter_tail"] = "chapter_tail"
    transition_requirements: list[str] = Field(
        default_factory=lambda: [
            "承接上一章末尾的时间、地点、在场人物、动作和情绪",
            "不得重复上一章已经完成的事件",
            "除非用户明确要求，不得无故跳转场景或时间",
        ]
    )


class ControlPlotOverviewContext(BaseModel):
    schema_version: str = "1.0"
    latest_chapter_id: int = Field(ge=0)
    latest_chapter_title: str = ""
    synopsis: str


class ContinuationOverview(BaseModel):
    schema_version: str = "1.0"
    book_id: str
    latest_chapter_id: int = Field(ge=0)
    next_chapter_id: int = Field(ge=1)
    latest_chapter_title: str = ""
    plot_synopsis: str
    ending_preview: str
    source_summary_count: int = Field(default=0, ge=0)


class StoryConsultReference(BaseModel):
    memory_id: str
    source_chapter: int = Field(default=0, ge=0)
    reason: str = ""


class StoryConsultResult(BaseModel):
    answer: str = Field(min_length=1)
    references: list[StoryConsultReference] = Field(default_factory=list)
    insufficient_context: bool = False


class ControlWriterResult(BaseModel):
    schema_version: str = "1.0"
    task_code: TaskCode
    operation: Literal["generate", "revise", "extend"]
    result: dict[str, Any]
    display_title: str
    display_text: str


class ControlUiPayload(BaseModel):
    schema_version: str = "1.0"
    phase: ControlPhase
    prompt: str = ""
    options: list[str] = Field(default_factory=list)
    title: str = ""
    text: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class ControlSession(BaseModel):
    schema_version: str = "1.0"
    session_id: str = Field(
        default_factory=lambda: f"session_{uuid.uuid4().hex[:16]}"
    )
    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:16]}")
    book_id: str
    task_code: TaskCode
    chapter_id: int = Field(default=0, ge=0)
    phase: ControlPhase = "setting_input"
    original_setting: str = ""
    refined_setting: str = ""
    draft_id: str = Field(default_factory=lambda: f"draft_{uuid.uuid4().hex[:16]}")
    draft_result: dict[str, Any] | None = None
    revision_count: int = Field(default=0, ge=0)
    archived: bool = False
    archive_result: dict[str, Any] | None = None
    pending_audit_result: dict[str, Any] | None = None
    continuation_overview: ContinuationOverview | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)
