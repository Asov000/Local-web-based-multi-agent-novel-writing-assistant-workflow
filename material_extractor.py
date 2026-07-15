from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from material_library import MATERIAL_SCHEMAS, MaterialCategory, MaterialInput
from rag.local_model import JsonModelClient
from rag.rag_message import RAGMessage, extract_rag_payload


class ExtractedMaterial(BaseModel):
    category: MaterialCategory
    name: str = Field(min_length=1, max_length=120)
    fields: dict[str, Any] = Field(default_factory=dict)
    evidence: str = Field(min_length=1, max_length=500)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class MaterialExtractionResult(BaseModel):
    materials: list[ExtractedMaterial] = Field(default_factory=list)


MATERIAL_EXTRACTION_CONFIG: dict[str, Any] = {
    "name": "正文素材提取",
    "agent": (
        "你是一名小说素材整理员。你的职责是从已经完成的章节正文中识别明确出现的"
        "角色、背景、势力、地点和物品，并整理成素材库记录。你不是创作模型，不得补写、"
        "推断或美化正文没有明确提供的信息。"
    ),
    "rule": (
        "只依据输入的章节标题和正文。每条素材必须给出能在正文中找到的简短evidence。"
        "同一对象只返回一条；名称必须使用正文中的标准名称，代词不能作为名称。"
        "仅填写正文明确出现的字段，未知字段保持空字符串或空数组。"
        "confidence必须是0.0到1.0之间的小数。"
        "背景只记录世界、时代、历史、社会结构、规则或核心冲突；普通环境描写应归入地点。"
        "势力必须是有共同身份或目标的组织；临时人群不能视为势力。"
        "物品只提取有名称、用途、持有关系或后续价值的对象，普通生活物件无需提取。"
    ),
}


def build_material_extraction_prompt() -> str:
    field_contract = {
        category: {
            "label": schema["label"],
            "fields": {key: field_type for key, _label, field_type in schema["fields"]},
        }
        for category, schema in MATERIAL_SCHEMAS.items()
    }
    example = {
        "materials": [
            {
                "category": "item",
                "name": "月族钥匙",
                "fields": {
                    "item_type": "钥匙",
                    "description": "银色钥匙",
                    "function": "开启青铜门",
                    "holder": "陈玥",
                    "origin": "",
                    "conditions": "",
                    "limitations": "",
                    "current_state": "由陈玥持有",
                },
                "evidence": "陈玥取出月族钥匙，打开了青铜门",
                "confidence": 0.95,
            }
        ]
    }
    return (
        f"任务名称：{MATERIAL_EXTRACTION_CONFIG['name']}\n"
        f"【角色】\n{MATERIAL_EXTRACTION_CONFIG['agent']}\n"
        f"【规则】\n{MATERIAL_EXTRACTION_CONFIG['rule']}\n"
        "【分类字段】\n"
        f"{json.dumps(field_contract, ensure_ascii=False, separators=(',', ':'))}\n"
        "【输出格式】\n"
        f"{json.dumps(example, ensure_ascii=False, separators=(',', ':'))}"
    )


class QwenMaterialExtractor:
    """Read-only Qwen adapter. It returns candidates and never writes storage."""

    def __init__(self, model_client: JsonModelClient) -> None:
        self.model_client = model_client

    def extract(
        self,
        *,
        book_id: str,
        chapter_id: int,
        title: str,
        text: str,
    ) -> list[dict[str, Any]]:
        request = RAGMessage(
            sender="material_extractor",
            receiver="qwen_model",
            action="rag.model.material.extract.request",
            book_id=book_id,
            payload={
                "chapter_id": int(chapter_id),
                "title": title.strip(),
                "text": text.strip(),
            },
        )
        if not request.payload["text"]:
            raise ValueError("章节正文不能为空")
        last_error: Exception | None = None
        for attempt in range(2):
            metadata = dict(request.metadata)
            metadata["attempt"] = attempt + 1
            if last_error is not None:
                metadata["validation_feedback"] = {
                    "error_type": type(last_error).__name__,
                    "error_message": str(last_error)[:2000],
                    "repair_instruction": "修复JSON格式和字段后重新返回完整响应",
                }
            attempt_request = request.model_copy(update={"metadata": metadata})
            prompt = build_material_extraction_prompt() + (
                "\n【统一消息要求】只返回完整的rag.message.v1 JSON响应。"
                "action必须为rag.model.material.extract.result，sender必须为qwen_model，"
                "receiver必须为material_extractor，message_type=response，status=ok，"
                "业务结果放在payload，operations必须为空数组。"
            )
            try:
                raw = self.model_client.invoke_json(prompt, attempt_request.model_dump())
                payload, _ = extract_rag_payload(
                    raw,
                    expected_action="rag.model.material.extract.result",
                )
                result = MaterialExtractionResult.model_validate(payload)
                candidates: list[dict[str, Any]] = []
                for item in result.materials:
                    validated = MaterialInput.model_validate(item.model_dump())
                    candidates.append(
                        {
                            **validated.model_dump(mode="json"),
                            "note": f"正文依据：{item.evidence}",
                            "confidence": item.confidence,
                        }
                    )
                return candidates
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"正文素材提取连续失败: {last_error}")
