from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent_schema import AgentMessage
from control_schemas import ControlWriterPayload, ControlWriterResult
from rag.chapter_memory_prompt import (
    CHAPTER_MEMORY_EXTRACTION_CONFIG,
    build_memory_task_prompt,
)
from rag.schemas import (
    ChapterMemoryExtractionPayload,
    ChapterMemoryExtractionResult,
    MemoryFact,
)


# =========================================================
# 1. 环境配置
# =========================================================

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
MODEL_ID = os.getenv("LLM_MODEL_ID")
BASE_URL = os.getenv("LLM_BASE_URL")
MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "8192"))

# =========================================================
# 2. 五类任务代码
# =========================================================

# BD：世界观创作
# CH：人物创作
# CT：文章续写
# NW：新文创作
# RV：原文修改

TaskCode = Literal["BD", "CH", "CT", "NW", "RV"]


# =========================================================
# 3. 五类任务配置
# 不再保存signals，不做意图识别
# =========================================================

TASK_CONFIG: dict[TaskCode, dict[str, Any]] = {
    "BD": {
        "name": "世界观创作",

        "agent": (
            "你是一名资深小说世界架构师，擅长设计具有完整规则、"
            "历史逻辑、社会结构和剧情潜力的虚构世界。\n"
            "你还需要为可写入记忆系统的信息评估重要度，并判断其是否属于权威记忆。\n"
            "重要度范围为0.0到1.0："
            "0.0-0.2表示装饰性或可忽略信息；"
            "0.3-0.4表示局部细节，短期有用；"
            "0.5-0.6表示一般重要信息，会影响近期内容；"
            "0.7-0.8表示重要设定、主要冲突或关键地点；"
            "0.9-1.0表示核心规则、不可轻易改变的基础设定。\n"
            "评分格式固定为“原文字 分数/T或F”，例如："
            "魔法只能通过血脉继承 0.95/T。"
            "T表示长期稳定、可进入权威记忆库；"
            "F表示可能变化、仅属于当前剧情或阶段性事实。"
            "重要度高不等于必须标T。\n"
            "示例："
            "background=\"大陆被永夜笼罩 0.9/T\"；"
            "rules=[\"亡者无法被真正复活 0.98/T\"]；"
        ),

        "rule": (
            "根据用户要求创作自洽的世界观。"
            "世界背景、核心规则、势力、地点和主要冲突必须相互一致，"
            "并能够支撑后续小说创作。"
        ),

        "format": {
            "world_name": "",
            "background": "世界背景 重要度/T或F",
            "rules": ["世界规则 重要度/T或F"],
            "factions": ["势力设定 重要度/T或F"],
            "locations": ["地点设定 重要度/T或F"],
            "conflict": "核心冲突 重要度/T或F",
        },
    },

    "CH": {
        "name": "人物创作",

        "agent": (
            "你是一名资深小说人物塑造作家，擅长设计具有真实动机、"
            "鲜明性格、成长空间和复杂关系的人物。\n"
            "你还需要为人物记忆评估重要度，并判断其是否属于权威记忆。\n"
            "重要度范围为0.0到1.0："
            "0.0-0.2表示普通外观细节或无后续作用的信息；"
            "0.3-0.4表示局部习惯、轻微偏好或短期状态；"
            "0.5-0.6表示一般背景、普通目标或次要关系；"
            "0.7-0.8表示重要身份、关键经历、主要目标或重要关系；"
            "0.9-1.0表示核心身份、核心性格、固定能力或不可改变的关键设定。\n"
            "评分格式固定为“原文字 分数/T或F”，例如："
            "流亡王族继承人 0.95/T。"
            "T表示长期稳定、可进入权威记忆库；"
            "F表示会随剧情变化的目标、关系或阶段性状态。"
            "重要度高不等于必须标T。\n"
            "示例："
            "role=\"流亡王族继承人 0.95/T\"；"
            "appearance=\"左眼有银色伤疤 0.7/T\"；"
        ),

        "rule": (
            "根据用户要求创作一个或多个人物。"
            "人物必须符合已有世界观，性格、经历、目标和行为逻辑必须一致。"
        ),

        "format": {
            "characters": [
                {
                    "name": "",
                    "role": "人物身份 重要度/T或F",
                    "appearance": "外貌特征 重要度/T或F",
                    "personality": "人物性格 重要度/T或F",
                    "background": "人物背景 重要度/T或F",
                    "goal": "人物目标 重要度/T或F",
                    "ability": "人物能力 重要度/T或F",
                    "relations": ["人物关系 重要度/T或F"],
                }
            ]
        },
    },

    "CT": {
        "name": "文章续写",

        "agent": (
            "你是一名专业长篇连载小说家，擅长承接前文、"
            "保持人物一致性、推进剧情并自然处理伏笔。正文字数应该在2500字上下\n"
            "你还需要为本章产生的剧情记忆评估重要度，并判断其是否属于权威记忆。\n"
            "重要度范围为0.0到1.0："
            "0.0-0.2表示普通动作、环境描写或无后续作用的信息；"
            "0.3-0.4表示局部事件或短期变化；"
            "0.5-0.6表示会影响近期章节的事件、状态或线索；"
            "0.7-0.8表示关键事件、明显状态变化或重要伏笔；"
            "0.9-1.0表示主线转折、重大不可逆事件或核心伏笔。\n"
            "评分格式固定为“原文字 分数/T或F”，例如："
            "陈玥打开了青铜门 0.8/F。"
            "T只用于本章正式确认的永久世界规则、永久人物身份或其他稳定事实；"
            "普通事件、状态变化和伏笔即使分数很高也通常标F。"
            "重要度高不等于必须标T。\n"
            "示例："
            "events=[\"陈玥打开了青铜门 0.8/F\"]；"
            "changes=[\"青铜门由关闭变为开启 0.7/F\","
        ),

        "rule": (
            "根据用户要求和已有上下文续写小说。"
            "不得重复前文，不得擅自改变人物设定、世界规则、"
            "时间线和既有事件。正文必须完整并推进剧情。"
        ),

        "format": {
            "chapter_title": "",
            "text": "",
            "characters": ["人物标准名称"],
            "events": ["重要事件 重要度/T或F"],
            "changes": ["状态或设定变化 重要度/T或F"],
            "hooks": ["伏笔或线索 重要度/T或F"],
            "next": [],
        },
    },

    "NW": {
        "name": "新文创作",

        "agent": (
            "你是一名非常专业的小说作家，擅长从零构建世界、"
            "人物、核心冲突和具有吸引力的小说开篇，TXT字数必须在3000字左右，TXT字数必须在3000字左右。\n"
            "你还需要为新作品中可写入记忆系统的信息评估重要度，"
            "并判断其是否属于权威记忆。\n"
            "重要度范围为0.0到1.0："
            "0.0-0.2表示普通描写或无长期作用的信息；"
            "0.3-0.4表示局部设定或短期目标；"
            "0.5-0.6表示一般背景、普通人物信息或近期剧情线索；"
            "0.7-0.8表示重要世界设定、主要人物目标或关键伏笔；"
            "0.9-1.0表示核心世界规则、主角永久身份、固定能力限制或作品根基。\n"
            "评分格式固定为“原文字 分数/T或F”，例如："
            "月族血脉可以开启青铜门 0.95/T。"
            "T表示长期稳定、可进入权威记忆库；"
            "F表示事件、当前目标、阶段性冲突或伏笔。"
            "重要度高不等于必须标T。\n"
            "示例："
            "world.background=\"大陆在百年前被永夜覆盖 0.9/T\"；"
            "world.rules=[\"月族血脉可以开启青铜门 0.95/T\"]；"
        ),

        "rule": (
            "根据用户要求创作一部新作品。"
            "优先采用用户明确提供的内容，缺失的世界观、人物和情节"
            "可以合理补充。正文需要建立场景、主要人物和初始冲突。"
        ),

        "format": {
            "book_title": "",
            "chapter_title": "",
            "world": {
                "background": "世界背景 重要度/T或F",
                "rules": ["世界规则 重要度/T或F"],
                "conflict": "核心冲突 重要度/T或F",
            },
            "characters": [
                {
                    "name": "",
                    "role": "人物身份 重要度/T或F",
                    "profile": "人物核心设定 重要度/T或F",
                    "goal": "人物目标 重要度/T或F",
                }
            ],
            "text": "",
            "hooks": ["伏笔或线索 重要度/T或F"],
            "next": [],
        },
    },

    "RV": {
        "name": "原文修改",

        "agent": (
            "你是一名资深小说编辑和改写作家，擅长在保留原文核心内容的"
            "基础上，改进语言、节奏、逻辑、描写和人物表现。\n"
            "你还需要为修改后真正产生的记忆变化评估重要度，"
            "并判断其是否属于权威记忆。\n"
            "重要度范围为0.0到1.0："
            "0.0-0.2表示纯措辞、标点或不影响内容的修改；"
            "0.3-0.4表示局部描写和轻微信息调整；"
            "0.5-0.6表示会影响近期理解的内容变化；"
            "0.7-0.8表示重要剧情、人物状态或关系变化；"
            "0.9-1.0表示核心设定、永久身份或重大不可逆剧情变化。\n"
            "评分格式固定为“原文字 分数/T或F”，例如："
            "将陈玥打开青铜门改为顾清打开青铜门 0.8/F。"
            "T仅用于修改后正式确立的永久规则、永久身份或固定设定；"
            "事件、状态、关系和文本层面的变化通常标F。"
            "重要度高不等于必须标T。\n"
            "changes只记录会影响剧情、设定、人物、状态、关系或伏笔的有效修改；"
            "纯语言润色、标点和语序调整不进入changes。\n"
            "示例："
            "changes=[\"将陈玥打开青铜门改为顾清打开青铜门 0.8/F\","
            "\"正式确认陈玥是月族唯一继承人 0.95/T\"]。"
        ),

        "rule": (
            "严格按照用户要求修改原文。"
            "除非用户明确提出，不得改变原文核心情节、人物关系和设定。"
            "必须返回修改后的完整正文。"
        ),

        "format": {
            "title": "",
            "text": "",
            "changes": ["有效修改 重要度/T或F"],
        },
    },
}


# =========================================================
# 4. 初始化模型
# =========================================================

llm: ChatOpenAI | None = None


def get_writer_llm() -> ChatOpenAI:
    global llm
    if llm is not None:
        return llm
    if not API_KEY:
        raise ValueError("未读取到 LLM_API_KEY")
    if not MODEL_ID:
        raise ValueError("未读取到 LLM_MODEL_ID")
    if not BASE_URL:
        raise ValueError("未读取到 LLM_BASE_URL")
    llm = ChatOpenAI(
        model=MODEL_ID,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.7,
        max_tokens=MAX_OUTPUT_TOKENS,
        timeout=120,
        max_retries=2,
    )
    return llm


# =========================================================
# 5. 压缩JSON
# =========================================================

def compact_json(data: Any) -> str:
    return json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
    )


# =========================================================
# 6. 校验用户选择的任务代码
# =========================================================

def validate_task_code(task_code: str) -> TaskCode:
    code = task_code.strip().upper()

    if code not in TASK_CONFIG:
        raise ValueError(
            "任务代码错误，只能输入：BD、CH、CT、NW、RV"
        )

    return code  # type: ignore[return-value]


# =========================================================
# 7. 生成当前任务System Prompt
# =========================================================

def build_system_prompt(task_code: TaskCode) -> str:
    config = TASK_CONFIG[task_code]

    agent_setting = config["agent"]
    task_rule = config["rule"]
    output_format = compact_json(config["format"])

    return (
        f"{agent_setting}\n"
        f"任务：{task_rule}\n"
        "遵守用户要求和已有上下文；"
        "不解释创作过程；"
        "只返回合法JSON；"
        "禁止Markdown和额外文字；"
        "必须包含格式中的全部顶层字段。\n"
        f"格式：{output_format}"
    )


# =========================================================
# 8. 封装Message
# =========================================================

def build_writer_messages(
    task_code: TaskCode,
    user_input: str,
    context: dict[str, Any] | None = None,
    control_prompt: str | None = None,
) -> list[SystemMessage | HumanMessage]:
    """
    task_code由用户或Control Agent直接提供，
    不再进行任何意图识别。
    """

    if not user_input.strip():
        raise ValueError("用户写作要求不能为空")

    system_prompt = build_system_prompt(task_code)
    if control_prompt:
        system_prompt = f"{system_prompt}\n\n{control_prompt.strip()}"

    payload: dict[str, Any] = {
        "u": user_input.strip(),
    }

    # 没有上下文时不发送c字段，节省Token
    if context:
        payload["c"] = context

    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=compact_json(payload)),
    ]


# =========================================================
# 9. 模型返回内容转文本
# =========================================================

def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []

        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(item))

        return "".join(parts)

    return str(content)


# =========================================================
# 10. 解析Gemini返回JSON
# =========================================================

def parse_json_response(content: Any) -> dict[str, Any]:
    text = content_to_text(content).strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

        text = re.sub(
            r"```$",
            "",
            text,
        ).strip()

    try:
        data = json.loads(text)

    except json.JSONDecodeError:
        match = re.search(
            r"\{.*\}",
            text,
            flags=re.DOTALL,
        )

        if not match:
            raise ValueError("Gemini未返回合法JSON")

        data = json.loads(match.group(0))

    if not isinstance(data, dict):
        raise ValueError("Gemini返回结果必须是JSON对象")

    return data


# =========================================================
# 11. 校验返回字段
# =========================================================

def validate_task_output(
    task_code: TaskCode,
    data: dict[str, Any],
) -> dict[str, Any]:
    expected_format = TASK_CONFIG[task_code]["format"]

    missing_fields = [
        field
        for field in expected_format
        if field not in data
    ]

    if missing_fields:
        raise ValueError(
            "Gemini返回结果缺少字段："
            + "、".join(missing_fields)
        )

    return data


# =========================================================
# 12. 调用Writer Agent
# =========================================================

def generate_writer_content(
    task_code: TaskCode,
    user_input: str,
    context: dict[str, Any] | None = None,
    control_prompt: str | None = None,
) -> dict[str, Any]:
    """
    task_code必须由调用方明确提供。

    不调用任何意图识别Agent，
    不执行正则任务判断。
    """

    messages = build_writer_messages(
        task_code=task_code,
        user_input=user_input,
        context=context,
        control_prompt=control_prompt,
    )

    task_llm = get_writer_llm().bind()
    response = task_llm.invoke(messages)
    raw_content = response.content
    first_error: Exception | None = None
    try:
        data = parse_json_response(raw_content)
        return validate_task_output(task_code=task_code, data=data)
    except Exception as exc:
        first_error = exc

    repair_messages = [
        *messages,
        AIMessage(content=content_to_text(raw_content)),
        HumanMessage(
            content=(
                "上一条响应不是合法且完整的JSON。请修复JSON语法并重新返回完整对象。"
                "不得省略原任务模板的任何顶层字段，不得使用Markdown，不得解释。"
                f"解析错误：{type(first_error).__name__}: {first_error}"
            )
        ),
    ]
    repair_response = task_llm.invoke(repair_messages)
    try:
        repaired = parse_json_response(repair_response.content)
        return validate_task_output(task_code=task_code, data=repaired)
    except Exception as exc:
        raise ValueError(
            "Write模型连续两次未返回合法完整JSON。"
            f"首次错误: {first_error}; 修复错误: {exc}"
        ) from exc


def extract_writer_chapter_facts(
    *,
    chapter_id: int,
    title: str,
    text: str,
) -> list[MemoryFact]:
    payload = ChapterMemoryExtractionPayload(
        chapter_id=chapter_id,
        title=title,
        text=text,
    )
    prompt = (
        build_memory_task_prompt(CHAPTER_MEMORY_EXTRACTION_CONFIG)
        + "\n本任务由WriteAgent执行。只返回合法JSON，不得输出Markdown、解释或代码围栏。"
    )
    messages: list[SystemMessage | HumanMessage | AIMessage] = [
        SystemMessage(content=prompt),
        HumanMessage(content=compact_json(payload.model_dump(mode="json"))),
    ]
    llm = get_writer_llm().bind()
    first_error: Exception | None = None
    for attempt in range(2):
        response: Any = None
        try:
            response = llm.invoke(messages)
            raw = parse_json_response(response.content)
            result = ChapterMemoryExtractionResult.model_validate(raw)
            if not result.facts:
                raise ValueError("修改后的章节未提取出任何有效记忆")
            missing_importance = [
                index
                for index, fact in enumerate(result.facts)
                if fact.raw_importance is None
            ]
            if missing_importance:
                indexes = ", ".join(str(index) for index in missing_importance)
                raise ValueError(f"facts中的raw_importance不能为空，位置: {indexes}")
            return result.facts
        except Exception as exc:
            first_error = exc
            if attempt == 0:
                messages.extend(
                    [
                        AIMessage(
                            content=(
                                content_to_text(response.content)
                                if response is not None
                                else ""
                            )
                        ),
                        HumanMessage(
                            content=(
                                "上一条记忆提取结果格式无效。请严格按照facts数组格式"
                                "重新返回完整JSON，不得解释。"
                                f"错误：{type(exc).__name__}: {exc}"
                            )
                        ),
                    ]
                )
    raise ValueError(f"WriteAgent连续两次未能提取章节记忆: {first_error}")


def build_control_bridge_prompt(payload: ControlWriterPayload) -> str:
    base = (
        "以下内容是Control_Agent追加的执行约束。它只补充当前操作信息，"
        "不得替换或削弱上方当前任务的角色、规则和JSON格式。"
        f"当前操作为{payload.operation}。"
    )
    if payload.operation == "generate":
        continuation_rule = ""
        if payload.task_code == "CT" and payload.context.get("continuity"):
            continuation_rule = (
                "context.continuity.ending_excerpt是上一章末尾原文。新章节开头必须直接承接"
                "其中的时间、地点、在场人物、动作和情绪，不得重新开场，不得复述已经"
                "完成的事件，不得在用户未要求时突然跳转时间或地点。"
                "context.plot_overview只负责全局剧情方向，不能替代ending_excerpt的局部衔接。"
            )
        return (
            base
            + "优先采用用户已确认的设定；RAG上下文只用于保持一致性。"
            + continuation_rule
            + "仍须严格返回当前任务原始模板规定的全部字段。"
        )
    if payload.operation == "extend":
        return (
            base
            + "这是章节末尾补写。context.append_context只提供已有章节的标题和末尾衔接片段。"
            "必须从ending_excerpt最后一句自然承接，但输出模板中的text只能填写本次新增正文，"
            "严禁重复、改写、概括或返回任何已有正文。其他结构化字段也只描述本次新增正文"
            "产生的人物、事件、变化和伏笔。仍须严格返回当前任务原始模板规定的全部字段。"
        )
    return (
        base
        + "这是对当前草稿的修改。依据revision中的用户反馈和目标片段修改，"
        "未被要求修改的内容应尽量保持。若提供RAG证据，不得与权威设定、"
        "人物当前状态和既有时间线冲突。必须返回当前任务原始模板规定的全部字段，"
        "不能因为是修改操作而改用其他任务的输出格式。"
    )


def writer_display(task_code: TaskCode, result: dict[str, Any]) -> tuple[str, str]:
    if task_code in {"CT", "NW"}:
        return (
            str(result.get("chapter_title") or "未命名章节"),
            str(result.get("text") or ""),
        )
    if task_code == "RV":
        return (
            str(result.get("title") or "修改稿"),
            str(result.get("text") or ""),
        )
    if task_code == "BD":
        body = {
            key: value
            for key, value in result.items()
            if key != "world_name"
        }
        return (
            str(result.get("world_name") or "世界观设定"),
            json.dumps(body, ensure_ascii=False, indent=2),
        )
    characters = result.get("characters")
    names = [
        str(item.get("name"))
        for item in characters or []
        if isinstance(item, dict) and item.get("name")
    ]
    return (
        "、".join(names) or "人物设定",
        json.dumps({"characters": characters or []}, ensure_ascii=False, indent=2),
    )


WriterGenerator = Callable[..., dict[str, Any]]
WriterMemoryExtractor = Callable[..., list[MemoryFact]]


class WriteAgent:
    """Internal writer endpoint. Production requests come from ControlAgent."""

    def __init__(
        self,
        generator: WriterGenerator | None = None,
        *,
        memory_extractor: WriterMemoryExtractor | None = None,
        agent_name: str = "write_agent",
    ) -> None:
        self.generator = generator or generate_writer_content
        self.memory_extractor = memory_extractor or extract_writer_chapter_facts
        self.agent_name = agent_name

    def handle_message(
        self,
        message: AgentMessage[Any],
    ) -> AgentMessage[dict[str, Any]]:
        if message.message_type != "request":
            return self._error(message, "WriteAgent只接受request消息")
        if message.receiver != self.agent_name:
            return self._error(message, f"消息接收者必须是{self.agent_name}")
        if message.sender != "control_agent":
            return self._error(message, "生产写作请求只能由control_agent发起")
        if message.action == "write.extract_memories":
            try:
                payload = ChapterMemoryExtractionPayload.model_validate(
                    message.payload or {}
                )
                facts = self.memory_extractor(
                    chapter_id=payload.chapter_id,
                    title=payload.title,
                    text=payload.text,
                )
                response_payload: Any = ChapterMemoryExtractionResult(
                    facts=facts
                ).model_dump(mode="json")
            except Exception as exc:
                return self._error(message, f"WriteAgent记忆提取失败: {exc}")
            return AgentMessage[dict[str, Any]](
                task_id=message.task_id,
                parent_message_id=message.message_id,
                sender=self.agent_name,
                receiver=message.sender,
                message_type="response",
                action=message.action,
                status="ok",
                payload=response_payload,
            )
        if message.action not in {"write.generate", "write.revise", "write.extend"}:
            return self._error(message, f"不支持的动作: {message.action}")
        try:
            payload = ControlWriterPayload.model_validate(message.payload or {})
            expected_action = {
                "generate": "write.generate",
                "revise": "write.revise",
                "extend": "write.extend",
            }[payload.operation]
            if message.action != expected_action:
                raise ValueError("action与payload.operation不一致")
            context = dict(payload.context)
            if payload.original_result is not None:
                context["original_result"] = payload.original_result
            if payload.revision:
                context["revision"] = payload.revision
            result = self.generator(
                task_code=payload.task_code,
                user_input=payload.user_input,
                context=context or None,
                control_prompt=build_control_bridge_prompt(payload),
            )
            title, text = writer_display(payload.task_code, result)
            response_payload = ControlWriterResult(
                task_code=payload.task_code,
                operation=payload.operation,
                result=result,
                display_title=title,
                display_text=text,
            )
        except Exception as exc:
            return self._error(message, f"WriteAgent执行失败: {exc}")
        return AgentMessage[dict[str, Any]](
            task_id=message.task_id,
            parent_message_id=message.message_id,
            sender=self.agent_name,
            receiver=message.sender,
            message_type="response",
            action=message.action,
            status="ok",
            payload=response_payload.model_dump(mode="json"),
        )

    def _error(
        self,
        request: AgentMessage[Any],
        error: str,
    ) -> AgentMessage[dict[str, Any]]:
        return AgentMessage[dict[str, Any]](
            task_id=request.task_id,
            parent_message_id=request.message_id,
            sender=self.agent_name,
            receiver=request.sender,
            message_type="response",
            action=request.action,
            status="error",
            error=error,
        )


# =========================================================
# 13. 用户选择任务
# =========================================================

def choose_task() -> TaskCode:
    print(
        "\n请选择任务：\n"
        "1. BD：世界观创作\n"
        "2. CH：人物创作\n"
        "3. CT：文章续写\n"
        "4. NW：新文创作\n"
        "5. RV：原文修改\n"
    )

    choice_map = {
        "1": "BD",
        "2": "CH",
        "3": "CT",
        "4": "NW",
        "5": "RV",
    }

    choice = input("请输入编号或任务代码：").strip().upper()

    task_code = choice_map.get(choice, choice)

    return validate_task_code(task_code)


# =========================================================
# 14. 测试入口
# =========================================================

def main() -> None:
    print("Writer Agent 已启动。")

    try:
        # 第一步：用户直接选择任务
        task_code = choose_task()

        task_name = TASK_CONFIG[task_code]["name"]

        print(f"\n当前任务：{task_name}")

        # 第二步：用户输入具体内容
        user_input = input("\n请输入具体写作要求：\n").strip()

        # 第三步：只调用一次Gemini
        result = generate_writer_content(
            task_code=task_code,
            user_input=user_input,
        )

        print(
            "\n"
            + json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )

    except Exception as error:
        print(
            f"\n执行失败："
            f"{type(error).__name__}: {error}"
        )


# =========================================================
# 12. 五类输出统一转换为RAG格式
# =========================================================

import hashlib


# 匹配：
# 魔法只能通过血脉继承 0.95/T
# 陈玥打开了青铜门 0.8/F。
SCORED_TEXT_PATTERN = re.compile(
    r"^(?P<text>.*?)\s+"
    r"(?P<score>0(?:\.\d+)?|1(?:\.0+)?)"
    r"\s*/\s*"
    r"(?P<flag>[TF])"
    r"\s*[。.]?$",
    flags=re.IGNORECASE | re.DOTALL,
)


def parse_scored_text(
    value: Any,
    default_importance: float = 0.5,
    default_authoritative: bool = False,
) -> tuple[str, float, bool]:
    """
    将：
        魔法只能通过血脉继承 0.95/T

    转换为：
        (
            "魔法只能通过血脉继承",
            0.95,
            True
        )

    如果模型没有正确输出评分格式，则使用保守默认值：
        importance = 0.5
        authoritative = False
    """

    text = str(value or "").strip()

    if not text:
        return "", default_importance, default_authoritative

    match = SCORED_TEXT_PATTERN.match(text)

    if not match:
        return text, default_importance, default_authoritative

    clean_text = match.group("text").strip()
    importance = float(match.group("score"))
    authoritative = match.group("flag").upper() == "T"

    return clean_text, importance, authoritative


def clean_scored_text(value: Any) -> str:
    """
    只提取评分字段中的原始内容。
    """

    text, _, _ = parse_scored_text(value)
    return text


def build_rag_title(
    task_code: TaskCode,
    data: dict[str, Any],
) -> str:
    """
    为不同任务生成统一文档标题。
    """

    if task_code == "BD":
        return str(
            data.get("world_name")
            or "未命名世界"
        ).strip()

    if task_code == "CH":
        names = [
            str(character.get("name") or "").strip()
            for character in data.get("characters", [])
            if isinstance(character, dict)
        ]

        names = [
            name
            for name in names
            if name
        ]

        return "、".join(names) or "人物设定"

    if task_code == "CT":
        return str(
            data.get("chapter_title")
            or "未命名章节"
        ).strip()

    if task_code == "NW":
        book_title = str(
            data.get("book_title")
            or ""
        ).strip()

        chapter_title = str(
            data.get("chapter_title")
            or ""
        ).strip()

        titles = [
            value
            for value in [
                book_title,
                chapter_title,
            ]
            if value
        ]

        return "｜".join(titles) or "新文创作"

    if task_code == "RV":
        return str(
            data.get("title")
            or "修改稿"
        ).strip()

    return "未命名文档"


def build_main_rag_content(
    task_code: TaskCode,
    data: dict[str, Any],
) -> str:
    """
    生成每个任务的主文档内容。

    CT、NW、RV直接使用正文。
    BD、CH没有统一正文，因此组装成可检索文本。
    """

    if task_code == "BD":
        lines: list[str] = []

        world_name = str(
            data.get("world_name")
            or ""
        ).strip()

        if world_name:
            lines.append(
                f"世界名称：{world_name}"
            )

        background = clean_scored_text(
            data.get("background")
        )

        if background:
            lines.append(
                f"世界背景：{background}"
            )

        rules = [
            clean_scored_text(item)
            for item in data.get("rules", [])
        ]

        rules = [
            item
            for item in rules
            if item
        ]

        if rules:
            lines.append(
                "世界规则：" + "；".join(rules)
            )

        factions = [
            clean_scored_text(item)
            for item in data.get("factions", [])
        ]

        factions = [
            item
            for item in factions
            if item
        ]

        if factions:
            lines.append(
                "势力设定：" + "；".join(factions)
            )

        locations = [
            clean_scored_text(item)
            for item in data.get("locations", [])
        ]

        locations = [
            item
            for item in locations
            if item
        ]

        if locations:
            lines.append(
                "地点设定：" + "；".join(locations)
            )

        conflict = clean_scored_text(
            data.get("conflict")
        )

        if conflict:
            lines.append(
                f"核心冲突：{conflict}"
            )

        return "\n".join(lines)

    if task_code == "CH":
        character_blocks: list[str] = []

        for character in data.get(
            "characters",
            [],
        ):
            if not isinstance(character, dict):
                continue

            name = str(
                character.get("name")
                or ""
            ).strip()

            lines = [
                f"人物：{name}"
            ]

            field_mapping = {
                "role": "身份",
                "appearance": "外貌",
                "personality": "性格",
                "background": "背景",
                "goal": "目标",
                "ability": "能力",
            }

            for field, label in field_mapping.items():
                value = clean_scored_text(
                    character.get(field)
                )

                if value:
                    lines.append(
                        f"{label}：{value}"
                    )

            relations = [
                clean_scored_text(item)
                for item in character.get(
                    "relations",
                    [],
                )
            ]

            relations = [
                item
                for item in relations
                if item
            ]

            if relations:
                lines.append(
                    "关系：" + "；".join(relations)
                )

            character_blocks.append(
                "\n".join(lines)
            )

        return "\n\n".join(character_blocks)

    # CT、NW、RV都有text字段
    return str(
        data.get("text")
        or ""
    ).strip()


def convert_to_rag_format(
    task_code: str,
    data: dict[str, Any],
    source_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    将BD、CH、CT、NW、RV五类输出转换成统一RAG文档格式。

    返回格式：

    [
        {
            "id": "记录唯一ID",
            "page_content": "用于向量化的文本",
            "metadata": {
                "source_id": "来源文档ID",
                "task_code": "BD",
                "task_name": "世界观创作",
                "title": "永夜大陆",
                "record_type": "world_rule",
                "source_field": "rules[0]",
                "scope": "world",
                "status": "fact",
                "entity": "",
                "importance": 0.95,
                "authoritative": True
            }
        }
    ]

    source_id建议由上层传入：
    - 小说ID
    - 章节ID
    - 世界观版本ID

    如果没有传入，则根据当前输出自动生成。
    """

    code = validate_task_code(task_code)

    validate_task_output(
        task_code=code,
        data=data,
    )

    task_name = TASK_CONFIG[code]["name"]

    title = build_rag_title(
        task_code=code,
        data=data,
    )

    main_content = build_main_rag_content(
        task_code=code,
        data=data,
    )

    # 未传source_id时，根据当前输出生成稳定ID
    if not source_id:
        source_seed = compact_json({
            "task_code": code,
            "title": title,
            "data": data,
        })

        source_id = hashlib.sha1(
            source_seed.encode("utf-8")
        ).hexdigest()[:20]

    common_metadata: dict[str, Any] = {
        "source_id": source_id,
        "task_code": code,
        "task_name": task_name,
        "title": title,
    }

    # 补充通用作品信息
    for field in [
        "world_name",
        "book_title",
        "chapter_title",
    ]:
        value = data.get(field)

        if value:
            common_metadata[field] = str(value)

    if code == "RV" and data.get("title"):
        common_metadata["chapter_title"] = str(
            data["title"]
        )

    if code == "CT":
        characters = data.get(
            "characters",
            [],
        )

        if isinstance(characters, list):
            common_metadata["characters"] = ",".join(
                str(item)
                for item in characters
            )

    rag_documents: list[dict[str, Any]] = []

    def add_record(
        value: Any,
        record_type: str,
        source_field: str,
        label: str = "",
        entity: str = "",
        scope: str = "",
        status: str = "fact",
        scored: bool = True,
        importance: float = 0.5,
        authoritative: bool = False,
    ) -> None:
        """
        向统一RAG列表中增加一条记录。
        """

        if value is None:
            return

        if scored:
            (
                text,
                parsed_importance,
                parsed_authoritative,
            ) = parse_scored_text(
                value=value,
                default_importance=importance,
                default_authoritative=authoritative,
            )

        else:
            text = str(value).strip()
            parsed_importance = importance
            parsed_authoritative = authoritative

        if not text:
            return

        page_content = (
            f"{label}：{text}"
            if label
            else text
        )

        # 给人物相关记录增加人物名称，
        # 避免单独向量化后失去主体信息
        if entity:
            page_content = (
                f"{entity}｜{page_content}"
            )

        index = len(rag_documents)

        record_seed = (
            f"{source_id}|"
            f"{record_type}|"
            f"{source_field}|"
            f"{entity}|"
            f"{index}|"
            f"{text}"
        )

        record_id = hashlib.sha1(
            record_seed.encode("utf-8")
        ).hexdigest()[:24]

        metadata = {
            **common_metadata,
            "record_type": record_type,
            "source_field": source_field,
            "scope": scope,
            "status": status,
            "entity": entity,
            "importance": round(
                float(parsed_importance),
                4,
            ),
            "authoritative": bool(
                parsed_authoritative
            ),
        }

        rag_documents.append({
            "id": record_id,
            "page_content": page_content,
            "metadata": metadata,
        })

    # -----------------------------------------------------
    # 1. 通用主文档
    # -----------------------------------------------------

    if main_content:
        main_source_field = (
            "text"
            if code in {"CT", "NW", "RV"}
            else "assembled"
        )

        add_record(
            value=main_content,
            record_type="main_text",
            source_field=main_source_field,
            scope="document",
            status="content",
            scored=False,
            importance=0.5,
            authoritative=False,
        )

    # -----------------------------------------------------
    # 2. BD：世界观创作
    # -----------------------------------------------------

    if code == "BD":
        world_name = str(
            data.get("world_name")
            or ""
        ).strip()

        add_record(
            value=world_name,
            record_type="world_name",
            source_field="world_name",
            label="世界名称",
            scope="world",
            status="entity",
            scored=False,
            importance=1.0,
            authoritative=True,
        )

        add_record(
            value=data.get("background"),
            record_type="world_background",
            source_field="background",
            label="世界背景",
            scope="world",
        )

        for index, item in enumerate(
            data.get("rules", [])
        ):
            add_record(
                value=item,
                record_type="world_rule",
                source_field=f"rules[{index}]",
                label="世界规则",
                scope="world",
            )

        for index, item in enumerate(
            data.get("factions", [])
        ):
            add_record(
                value=item,
                record_type="faction",
                source_field=f"factions[{index}]",
                label="势力",
                scope="world",
            )

        for index, item in enumerate(
            data.get("locations", [])
        ):
            add_record(
                value=item,
                record_type="location",
                source_field=f"locations[{index}]",
                label="地点",
                scope="world",
            )

        add_record(
            value=data.get("conflict"),
            record_type="core_conflict",
            source_field="conflict",
            label="核心冲突",
            scope="world",
        )

    # -----------------------------------------------------
    # 3. CH：人物创作
    # -----------------------------------------------------

    elif code == "CH":
        character_fields = {
            "role": (
                "character_role",
                "身份",
                "fact",
            ),
            "appearance": (
                "character_appearance",
                "外貌",
                "fact",
            ),
            "personality": (
                "character_personality",
                "性格",
                "fact",
            ),
            "background": (
                "character_background",
                "背景",
                "fact",
            ),
            "goal": (
                "character_goal",
                "目标",
                "state",
            ),
            "ability": (
                "character_ability",
                "能力",
                "fact",
            ),
        }

        for character_index, character in enumerate(
            data.get("characters", [])
        ):
            if not isinstance(character, dict):
                continue

            name = str(
                character.get("name")
                or ""
            ).strip()

            add_record(
                value=name,
                record_type="character_name",
                source_field=(
                    f"characters[{character_index}].name"
                ),
                label="人物",
                entity=name,
                scope="character",
                status="entity",
                scored=False,
                importance=1.0,
                authoritative=True,
            )

            for field, config in character_fields.items():
                (
                    record_type,
                    label,
                    status,
                ) = config

                add_record(
                    value=character.get(field),
                    record_type=record_type,
                    source_field=(
                        f"characters[{character_index}].{field}"
                    ),
                    label=label,
                    entity=name,
                    scope="character",
                    status=status,
                )

            relations = character.get(
                "relations",
                [],
            )

            if isinstance(relations, list):
                for relation_index, relation in enumerate(
                    relations
                ):
                    add_record(
                        value=relation,
                        record_type="character_relation",
                        source_field=(
                            f"characters[{character_index}]"
                            f".relations[{relation_index}]"
                        ),
                        label="关系",
                        entity=name,
                        scope="character",
                        status="state",
                    )

    # -----------------------------------------------------
    # 4. CT：文章续写
    # -----------------------------------------------------

    elif code == "CT":
        for index, item in enumerate(
            data.get("events", [])
        ):
            add_record(
                value=item,
                record_type="event",
                source_field=f"events[{index}]",
                label="事件",
                scope="chapter",
                status="fact",
            )

        for index, item in enumerate(
            data.get("changes", [])
        ):
            add_record(
                value=item,
                record_type="state_change",
                source_field=f"changes[{index}]",
                label="状态变化",
                scope="chapter",
                status="state",
            )

        for index, item in enumerate(
            data.get("hooks", [])
        ):
            add_record(
                value=item,
                record_type="hook",
                source_field=f"hooks[{index}]",
                label="伏笔",
                scope="chapter",
                status="clue",
            )

        # next只是创作建议，不能作为已发生事实
        for index, item in enumerate(
            data.get("next", [])
        ):
            add_record(
                value=item,
                record_type="next_direction",
                source_field=f"next[{index}]",
                label="后续方向",
                scope="planning",
                status="proposal",
                scored=False,
                importance=0.2,
                authoritative=False,
            )

    # -----------------------------------------------------
    # 5. NW：新文创作
    # -----------------------------------------------------

    elif code == "NW":
        world = data.get(
            "world",
            {},
        )

        if not isinstance(world, dict):
            world = {}

        add_record(
            value=world.get("background"),
            record_type="world_background",
            source_field="world.background",
            label="世界背景",
            scope="world",
        )

        rules = world.get(
            "rules",
            [],
        )

        if isinstance(rules, list):
            for index, item in enumerate(rules):
                add_record(
                    value=item,
                    record_type="world_rule",
                    source_field=f"world.rules[{index}]",
                    label="世界规则",
                    scope="world",
                )

        add_record(
            value=world.get("conflict"),
            record_type="core_conflict",
            source_field="world.conflict",
            label="核心冲突",
            scope="world",
        )

        for character_index, character in enumerate(
            data.get("characters", [])
        ):
            if not isinstance(character, dict):
                continue

            name = str(
                character.get("name")
                or ""
            ).strip()

            add_record(
                value=name,
                record_type="character_name",
                source_field=(
                    f"characters[{character_index}].name"
                ),
                label="人物",
                entity=name,
                scope="character",
                status="entity",
                scored=False,
                importance=1.0,
                authoritative=True,
            )

            add_record(
                value=character.get("role"),
                record_type="character_role",
                source_field=(
                    f"characters[{character_index}].role"
                ),
                label="身份",
                entity=name,
                scope="character",
            )

            add_record(
                value=character.get("profile"),
                record_type="character_profile",
                source_field=(
                    f"characters[{character_index}].profile"
                ),
                label="核心设定",
                entity=name,
                scope="character",
            )

            add_record(
                value=character.get("goal"),
                record_type="character_goal",
                source_field=(
                    f"characters[{character_index}].goal"
                ),
                label="目标",
                entity=name,
                scope="character",
                status="state",
            )

        for index, item in enumerate(
            data.get("hooks", [])
        ):
            add_record(
                value=item,
                record_type="hook",
                source_field=f"hooks[{index}]",
                label="伏笔",
                scope="chapter",
                status="clue",
            )

        for index, item in enumerate(
            data.get("next", [])
        ):
            add_record(
                value=item,
                record_type="next_direction",
                source_field=f"next[{index}]",
                label="后续方向",
                scope="planning",
                status="proposal",
                scored=False,
                importance=0.2,
                authoritative=False,
            )

    # -----------------------------------------------------
    # 6. RV：原文修改
    # -----------------------------------------------------

    elif code == "RV":
        for index, item in enumerate(
            data.get("changes", [])
        ):
            add_record(
                value=item,
                record_type="revision_change",
                source_field=f"changes[{index}]",
                label="有效修改",
                scope="chapter",
                status="state",
            )

    return rag_documents

if __name__ == "__main__":
    print(
        "Write_Agent是内部服务，普通用户请运行control_agent.py。"
        "端到端测试请运行pipeline_smoke_test.py。"
    )
    raise SystemExit(2)
