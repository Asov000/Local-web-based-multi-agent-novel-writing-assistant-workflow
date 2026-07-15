from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv

from agent_schema import AgentMessage
from control_schemas import (
    ContinuationOverview,
    ControlIntent,
    ControlPlotOverviewContext,
    ControlSession,
    StoryConsultResult,
    ControlUiPayload,
    ControlWriterPayload,
    ControlWriterResult,
)
from progress_display import ConsoleProgress, SilentProgress
from rag import MemoryAgent, NovelRagSystem
from rag.control_retrieval import ControlContextBuilder
from rag.local_model import (
    HuggingFaceQwenJsonClient,
    JsonModelClient,
    LocalJsonModelClient,
)
from rag.qwen_judge import QwenMemoryJudge
from rag.schemas import (
    ChapterMemoryExtractionPayload,
    ChapterMemoryExtractionResult,
    MemoryFact,
    TaskCode,
)


INTENT_PROMPT = (
    "你是Control_Agent的唯一意图路由器。结合当前会话阶段、上一问题和草稿状态，"
    "一次性判断用户意图并提取参数。不要创作小说，不要修改数据库。"
    "在setting_input阶段，正常的小说设定或写作要求应判断为start_writing；"
    "只有用户明确要求整理、检查或重建记忆库时才判断为request_memory_audit。"
    "intent只能是refine_setting、start_writing、revise_draft、approve_draft、"
    "continue_writing、request_memory_audit、confirm_audit、confirm_audit_apply、"
    "cancel、general_question、unknown。"
    "当用户指出OOC、人设不符、时间线、世界规则或逻辑矛盾时，is_logic_error和"
    "needs_rag设为true。entities必须始终是字符串数组，不能返回对象。"
    "仅返回JSON对象，严格遵循示例："
    '{"intent":"start_writing","confidence":0.95,"feedback":"",'
    '"target_hint":"","is_logic_error":false,"needs_rag":false,'
    '"entities":[],"needs_clarification":false,"clarification":""}。'
)

REFINE_PROMPT = (
    "你是Control_Agent的小说设定整理组件。只把用户给出的简略设定扩写为清晰、"
    "可交给写作Agent执行的中文要求，不创作完整正文，不改变用户核心意图。"
    "expanded_setting最多450个中文字符。只返回JSON对象："
    '{"expanded_setting":"..."}。'
)

LOCATE_PROMPT = (
    "你是正文定位器，不负责改写。根据用户反馈和target_hint，从candidate_paragraphs"
    "中选择最对应的一段，target_excerpt必须逐字复制候选段落中的原文。"
    "无法确定时needs_clarification=true。只返回JSON对象："
    '{"target_excerpt":"","confidence":0.0,"needs_clarification":false,'
    '"clarification":""}。'
)


def validate_control_task_code(task_code: str) -> TaskCode:
    code = task_code.strip().upper()
    if code not in {"BD", "CH", "CT", "NW", "RV"}:
        raise ValueError("任务代码只能是BD、CH、CT、NW或RV")
    return code  # type: ignore[return-value]

ANSWER_PROMPT = (
    "你是小说创作流程中的Control_Agent。简洁回答用户关于当前草稿、流程或操作的"
    "问题，不虚构数据库内容，不直接修改草稿。只返回JSON对象：{\"answer\":\"...\"}。"
)

GENERAL_ASSISTANT_PROMPT = (
    "任务名称：智能写作日常对话\n"
    "【角色】你是自然、直接的小说写作助手，可以回答日常问题、通用写作问题，"
    "也可以针对用户直接提供或选中的文字给出润色和创作建议。\n"
    "【资料边界】本次没有查询小说记忆库或RAG。只能使用question和selected_text，"
    "不得声称了解未提供的小说设定、前文章节或人物状态。\n"
    "【权限】只提供回答和建议，不执行保存、归档、生成正文、修改记忆或删除操作。\n"
    "【表达】像正常对话一样自然回答，不套用固定分析模板，不强制使用标题或编号。\n"
    "【格式】只返回JSON对象：{\"answer\":\"自然中文回答\"}。"
)

STORY_CONSULT_PROMPT = (
    "任务名称：小说只读咨询\n"
    "【角色】你是智能写作助手中的小说顾问，负责依据RAG检索结果分析人物状态、"
    "设定一致性、剧情冲突、伏笔和修改方向。\n"
    "【权限】你只有读取输入资料和给出建议的权限。不得修改正文、素材或记忆，不得"
    "归档、续写、删除或要求系统执行任何操作。\n"
    "【规则】只能使用payload.memories和payload.selected_text中的信息。资料不足时"
    "必须设置insufficient_context=true并明确说明缺少什么；不能把推测写成事实。"
    "提出冲突或修改建议时要说明依据。answer应自然回答用户问题，不强制套用固定"
    "分析板块。references中的memory_id只能逐字复制输入"
    "memories中的memory_id，source_chapter必须与该记忆一致。\n"
    "【格式】只返回JSON对象："
    '{"answer":"中文回答","references":[{"memory_id":"memory_x",'
    '"source_chapter":1,"reason":"引用原因"}],"insufficient_context":false}。'
)

_STORY_CONTEXT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bRAG\b|记忆库|作品资料|小说资料",
        r"根据.{0,10}(已有|既有|前文|设定|记忆|资料)",
        r"(查询|查找|检索).{0,12}(人物|角色|物品|地点|势力|事件|设定|剧情|记忆)",
        r"(前文|前几章|上文|已有设定|既有设定|原有设定)",
        r"第[0-9一二三四五六七八九十百零两]+章",
        r"(当前|目前).{0,8}(状态|关系|目标|持有者|位置)",
        r"(检查|判断|分析|是否).{0,10}(冲突|矛盾|一致性|违背设定)",
        r"(未回收|尚未回收|没有回收).{0,8}伏笔|伏笔.{0,8}(回收|解决|遗漏)",
    )
)


def requires_story_context(question: str) -> bool:
    clean = question.strip()
    return bool(clean) and any(pattern.search(clean) for pattern in _STORY_CONTEXT_PATTERNS)

CONTINUATION_OVERVIEW_PROMPT = (
    "你是Control_Agent的续写准备组件。根据按章节排列的chapter_summaries、权威设定、"
    "人物当前状态和未回收伏笔，生成便于用户确认当前进度的中文剧情梗概。"
    "不得续写正文，不得虚构输入中不存在的事件，不得把ending_preview逐字重复进梗概。"
    "plot_synopsis不超过1000个中文字符。只返回JSON对象："
    '{"plot_synopsis":"..."}。'
)

REFINABLE_TASK_CODES = frozenset({"BD", "CH", "NW"})
CONTINUATION_MEMORY_LIMIT = 60


def supports_setting_refinement(task_code: str) -> bool:
    return task_code.strip().upper() in REFINABLE_TASK_CODES


class MessageAgent(Protocol):
    agent_name: str

    def handle_message(
        self,
        message: AgentMessage[Any],
    ) -> AgentMessage[dict[str, Any]]: ...


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def save(self, session: ControlSession) -> Path:
        folder = self.root / ".control_sessions" / session.book_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{session.session_id}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            session.model_dump_json(indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    def load(self, book_id: str, session_id: str) -> ControlSession:
        clean = session_id.strip()
        if not clean or Path(clean).name != clean:
            raise ValueError("session_id格式错误")
        path = self.root / ".control_sessions" / book_id / f"{clean}.json"
        if not path.is_file():
            raise FileNotFoundError(f"未找到Control会话: {clean}")
        return ControlSession.model_validate_json(path.read_text(encoding="utf-8"))


class ControlAgent:
    agent_name = "control_agent"

    def __init__(
        self,
        *,
        writer_agent: MessageAgent,
        rag_system: NovelRagSystem,
        control_model: JsonModelClient,
        locator_model: JsonModelClient | None = None,
        progress: ConsoleProgress | SilentProgress | None = None,
    ) -> None:
        self.writer_agent = writer_agent
        self.rag_system = rag_system
        self.control_model = control_model
        self.locator_model = locator_model
        self.progress = progress or SilentProgress()
        self.context_builder = ControlContextBuilder(rag_system.config)
        self.sessions = SessionStore(rag_system.config.root_dir)

    def create_session(
        self,
        book_id: str,
        task_code: str,
        *,
        chapter_id: int,
    ) -> ControlSession:
        with self.progress.step("control.session", "正在创建并保存写作会话"):
            session = ControlSession(
                book_id=book_id,
                task_code=validate_control_task_code(task_code),
                chapter_id=chapter_id,
            )
            self._record(session, "session.created", {"task_code": session.task_code})
        return session

    def refine_setting(self, session: ControlSession, setting: str) -> str:
        if not supports_setting_refinement(session.task_code):
            raise ValueError("续写和原文修改任务不支持设定细化")
        clean = setting.strip()
        if not clean:
            raise ValueError("小说设定不能为空")
        session.original_setting = clean
        with self.progress.step("control.refine", "正在细化小说设定"):
            last_value = ""
            for _ in range(2):
                result = self.control_model.invoke_json(
                    REFINE_PROMPT,
                    {
                        "task_code": session.task_code,
                        "original_setting": clean,
                        "previous_too_long": bool(last_value),
                    },
                )
                last_value = str(result.get("expanded_setting") or "").strip()
                if last_value and len(last_value) <= 500:
                    session.refined_setting = last_value
                    self._record(
                        session,
                        "setting.refined",
                        {"length": len(last_value)},
                    )
                    return last_value
        raise ValueError("设定细化结果为空或超过500字，请缩短原始要求后重试")

    def keep_original_setting(self, session: ControlSession, setting: str) -> str:
        clean = setting.strip()
        if not clean:
            raise ValueError("小说设定不能为空")
        with self.progress.step("control.setting", "正在确认并保存原始小说设定"):
            session.original_setting = clean
            session.refined_setting = ""
            self._record(session, "setting.accepted", {"length": len(clean)})
        return clean

    def classify_intent(self, session: ControlSession, user_text: str) -> ControlIntent:
        clean = user_text.strip()
        if not clean:
            return ControlIntent(
                intent="unknown",
                confidence=1.0,
                needs_clarification=True,
                clarification="请输入你的要求。",
            )
        explicit = self._explicit_command_intent(session.phase, clean)
        if explicit is not None:
            with self.progress.step("control.intent.local", "正在确认明确操作命令"):
                return explicit
        with self.progress.step("control.intent", "正在理解你的操作意图"):
            raw = self.control_model.invoke_json(
                INTENT_PROMPT,
                {
                    "phase": session.phase,
                    "task_code": session.task_code,
                    "has_draft": session.draft_result is not None,
                    "archived": session.archived,
                    "revision_count": session.revision_count,
                    "allowed_intents": self._allowed_intents(session.phase),
                    "user_text": clean,
                },
            )
            try:
                intent = ControlIntent.model_validate(raw)
            except Exception:
                intent = self._intent_fallback(session.phase, clean, raw)
        allowed = set(self._allowed_intents(session.phase))
        if intent.intent not in allowed:
            return ControlIntent(
                intent="unknown",
                confidence=intent.confidence,
                needs_clarification=True,
                clarification="当前阶段无法执行该操作，请根据当前提示说明你的要求。",
            )
        if intent.confidence < 0.62:
            intent.needs_clarification = True
            if not intent.clarification:
                intent.clarification = "我不太确定你的意思，请说明是修改、通过、续写还是整理记忆库。"
        return intent

    def handle_message(
        self,
        message: AgentMessage[Any],
    ) -> AgentMessage[dict[str, Any]]:
        if message.message_type != "request":
            return self._message_error(message, "ControlAgent只接受request消息")
        if message.receiver != self.agent_name:
            return self._message_error(message, f"消息接收者必须是{self.agent_name}")
        if message.sender != "user":
            return self._message_error(message, "ControlAgent是唯一用户入口，只接受user消息")
        if message.action != "control.interpret":
            return self._message_error(message, f"不支持的动作: {message.action}")
        payload = message.payload if isinstance(message.payload, dict) else {}
        try:
            book_id = str(payload.get("book_id") or "")
            session_id = str(payload.get("session_id") or "")
            user_text = str(payload.get("text") or "")
            session = self.sessions.load(book_id, session_id)
            intent = self.classify_intent(session, user_text)
        except Exception as exc:
            return self._message_error(message, f"ControlAgent意图处理失败: {exc}")
        return AgentMessage[dict[str, Any]](
            task_id=message.task_id,
            parent_message_id=message.message_id,
            sender=self.agent_name,
            receiver="user",
            message_type="response",
            action=message.action,
            status="need_user_input" if intent.needs_clarification else "ok",
            payload={
                "schema_version": "1.0",
                "phase": session.phase,
                "intent": intent.model_dump(mode="json"),
            },
            metadata={"session_id": session.session_id},
        )

    def interpret_user_message(
        self,
        session: ControlSession,
        user_text: str,
    ) -> ControlIntent:
        request = AgentMessage[dict[str, Any]](
            task_id=session.task_id,
            sender="user",
            receiver=self.agent_name,
            action="control.interpret",
            payload={
                "book_id": session.book_id,
                "session_id": session.session_id,
                "text": user_text,
            },
        )
        response = self.handle_message(request)
        if response.status not in {"ok", "need_user_input"}:
            raise RuntimeError(response.error or "ControlAgent意图处理失败")
        payload = response.payload or {}
        return ControlIntent.model_validate(payload.get("intent") or {})

    def generate_draft(self, session: ControlSession) -> ControlWriterResult:
        setting = session.refined_setting or session.original_setting
        if not setting:
            raise ValueError("尚未提供小说设定")
        include_previous = session.task_code == "CT" or session.chapter_id > 1
        memory_limit = CONTINUATION_MEMORY_LIMIT if include_previous else 20
        with self.progress.step("rag.retrieve", "正在查询并排序相关小说记忆"):
            context = self.context_builder.build(
                session.book_id,
                setting,
                current_chapter=session.chapter_id,
                top_k=memory_limit,
                include_previous_chapter=include_previous,
            )
            overview = session.continuation_overview
            if overview and overview.latest_chapter_id == session.chapter_id - 1:
                context["plot_overview"] = ControlPlotOverviewContext(
                    latest_chapter_id=overview.latest_chapter_id,
                    latest_chapter_title=overview.latest_chapter_title,
                    synopsis=overview.plot_synopsis,
                ).model_dump(mode="json")
        with self.progress.step("control.compose", "正在组装Control到Write的统一Message"):
            payload = ControlWriterPayload(
                task_code=session.task_code,
                operation="generate",
                user_input=setting,
                context=context,
                output_contract={
                    "preserve_original_task_template": True,
                    "continuity_required": bool(context.get("continuity")),
                },
            )
        result = self._send_writer(session, "write.generate", payload)
        session.draft_result = result.result
        session.phase = "draft_review"
        self._record(
            session,
            "draft.generated",
            {"title": result.display_title},
        )
        return result

    def prepare_initial_continuation(
        self,
        session: ControlSession,
    ) -> ContinuationOverview:
        with self.progress.step("continuation.load", "正在读取书籍章节与记忆数据"):
            material = self.context_builder.build_continuation_overview_material(
                session.book_id
            )
        latest = material["latest_chapter"]
        synopsis = ""
        with self.progress.step("continuation.overview", "正在生成当前剧情梗概"):
            for _ in range(2):
                try:
                    raw = self.control_model.invoke_json(
                        CONTINUATION_OVERVIEW_PROMPT,
                        material,
                    )
                    candidate = str(raw.get("plot_synopsis") or "").strip()
                    if candidate and len(candidate) <= 1000:
                        synopsis = candidate
                        break
                except Exception:
                    continue
        if not synopsis:
            synopsis = self._fallback_plot_synopsis(material)
        overview = ContinuationOverview(
            book_id=session.book_id,
            latest_chapter_id=int(latest["chapter_id"]),
            next_chapter_id=int(latest["chapter_id"]) + 1,
            latest_chapter_title=str(latest.get("title") or ""),
            plot_synopsis=synopsis,
            ending_preview=str(material.get("ending_preview") or "")[-500:],
            source_summary_count=len(material.get("chapter_summaries") or []),
        )
        session.task_code = "CT"
        session.chapter_id = overview.next_chapter_id
        session.continuation_overview = overview
        self._record(
            session,
            "continuation.overview.prepared",
            {
                "latest_chapter_id": overview.latest_chapter_id,
                "next_chapter_id": overview.next_chapter_id,
                "source_summary_count": overview.source_summary_count,
            },
        )
        return overview

    def revise_draft(
        self,
        session: ControlSession,
        user_text: str,
        intent: ControlIntent,
    ) -> ControlWriterResult:
        if session.draft_result is None:
            raise ValueError("当前没有可修改的草稿")
        title, full_text = self._draft_display(session)
        feedback = intent.feedback.strip() or user_text.strip()
        if not feedback:
            raise ValueError("修改要求不能为空")
        with self.progress.step("control.locate", "正在定位需要修改的正文位置"):
            target_excerpt = self._locate_excerpt(
                full_text,
                intent.target_hint,
                feedback,
            )

        rag_context: dict[str, Any] = {}
        if intent.is_logic_error or intent.needs_rag:
            with self.progress.step("rag.logic", "正在查询人物设定、时间线和权威记忆"):
                rag_context = self.context_builder.build(
                    session.book_id,
                    f"{feedback}\n{target_excerpt}\n{' '.join(intent.entities)}",
                    current_chapter=session.chapter_id,
                    top_k=16,
                    include_previous_chapter=True,
                )

        revision = {
            "chapter_title": title,
            "target_excerpt": target_excerpt,
            "user_feedback": feedback,
            "is_logic_error": intent.is_logic_error,
            "rag_evidence": rag_context,
        }
        payload = ControlWriterPayload(
            task_code=session.task_code,
            operation="revise",
            user_input=(
                session.refined_setting
                or session.original_setting
                or feedback
            ),
            context={"revision_number": session.revision_count + 1},
            original_result=session.draft_result,
            revision=revision,
            output_contract={
                "preserve_original_task_template": True,
                "return_complete_result": True,
            },
        )
        result = self._send_writer(session, "write.revise", payload)
        session.draft_result = result.result
        session.revision_count += 1
        session.phase = "draft_review"
        self._record(
            session,
            "draft.revised",
            {
                "target_excerpt": target_excerpt,
                "feedback": feedback,
            },
        )
        return result

    def extend_draft(
        self,
        session: ControlSession,
        user_request: str,
    ) -> ControlWriterResult:
        if session.draft_result is None:
            raise ValueError("当前没有可补写的章节")
        original_result = dict(session.draft_result)
        original_revision_count = session.revision_count
        original_title, original_text = self._draft_display(session)
        if not original_text.strip():
            raise ValueError("请先写入部分正文，再使用章节补写")
        request = user_request.strip() or "自然承接现有正文继续补写"
        instruction = (
            "这是当前章节补写任务。只生成需要接在现有正文末尾的新增内容，"
            "不要重复已有正文，也不要重新输出完整章节。"
            f"补写要求：{request}"
        )
        try:
            ending_excerpt = original_text[-1600:]
            with self.progress.step("rag.extend", "正在查询补写所需的人物、设定和时间线"):
                context = self.context_builder.build(
                    session.book_id,
                    f"{request}\n{ending_excerpt}",
                    current_chapter=session.chapter_id,
                    top_k=CONTINUATION_MEMORY_LIMIT,
                    include_previous_chapter=True,
                )
            context["append_context"] = {
                "chapter_title": original_title,
                "ending_excerpt": ending_excerpt,
                "existing_text_length": len(original_text),
            }
            payload = ControlWriterPayload(
                task_code=session.task_code,
                operation="extend",
                user_input=instruction,
                context=context,
                output_contract={
                    "preserve_existing_text": True,
                    "text_contains_only_appendix": True,
                },
            )
            result = self._send_writer(session, "write.extend", payload)
            appended_text = self._strip_repeated_extension_prefix(
                original_text,
                result.display_text,
            )
            if not appended_text:
                raise ValueError("WriteAgent没有返回可追加的新正文")
            separator = self._extension_separator(original_text)
            combined_text = f"{original_text}{separator}{appended_text}"
            result.result = self._merge_extension_result(
                original_result,
                result.result,
                original_title,
                combined_text,
            )
            result.display_title = original_title
            result.display_text = combined_text
            session.draft_result = result.result
            session.revision_count += 1
            session.phase = "draft_review"
            self.sessions.save(session)
            self._record(
                session,
                "draft.extended",
                {"added_length": len(appended_text)},
            )
            return result
        except Exception:
            session.draft_result = original_result
            session.revision_count = original_revision_count
            self.sessions.save(session)
            raise

    @staticmethod
    def _strip_repeated_extension_prefix(
        original_text: str,
        generated_text: str,
    ) -> str:
        clean = generated_text.strip()
        original_clean = original_text.strip()
        if original_clean and clean.startswith(original_clean):
            return clean[len(original_clean) :].lstrip()

        max_overlap = min(len(original_text), len(clean), 2000)
        for size in range(max_overlap, 79, -1):
            if original_text[-size:] == clean[:size]:
                return clean[size:].lstrip()
        return clean

    @staticmethod
    def _extension_separator(original_text: str) -> str:
        if original_text.endswith(("\n\n", "\r\n\r\n")):
            return ""
        if original_text.endswith(("\n", "\r")):
            return "\n"
        return "\n\n"

    @staticmethod
    def _merge_extension_result(
        original: dict[str, Any],
        extension: dict[str, Any],
        title: str,
        combined_text: str,
    ) -> dict[str, Any]:
        merged = dict(original)
        ignored = {"text", "title", "chapter_title", "book_title", "world"}
        for key, value in extension.items():
            if key in ignored:
                continue
            if key == "next" and isinstance(value, list):
                merged[key] = list(value)
                continue
            if isinstance(value, list):
                current = merged.get(key)
                if isinstance(current, list):
                    combined = list(current)
                    for item in value:
                        if item not in combined:
                            combined.append(item)
                    merged[key] = combined
                elif value:
                    merged[key] = list(value)
                continue
            if key not in merged or merged[key] in (None, "", {}, []):
                merged[key] = value

        if "chapter_title" in original or "chapter_title" in extension:
            merged["chapter_title"] = title
        else:
            merged["title"] = title
        merged["text"] = combined_text
        return merged

    def archive_draft(self, session: ControlSession) -> dict[str, Any]:
        if session.archived:
            return session.archive_result or {}
        if session.draft_result is None:
            raise ValueError("当前没有可归档草稿")
        session.phase = "archive"
        with self.progress.step("memory.archive", "正在归档正文并更新小说记忆数据库"):
            result = self.rag_system.ingest(
                session.book_id,
                session.task_code,
                session.draft_result,
                chapter_id=session.chapter_id,
            )
        session.archived = True
        session.phase = "post_archive"
        session.archive_result = result.model_dump(mode="json")
        self._record(
            session,
            "draft.archived",
            {"draft_id": session.draft_id, **session.archive_result},
        )
        return session.archive_result

    def start_continuation(
        self,
        session: ControlSession,
        user_request: str,
    ) -> ControlWriterResult:
        session.task_code = "CT"
        latest = self.context_builder.latest_chapter(session.book_id)
        session.chapter_id = (
            int(latest["chapter_id"]) + 1
            if latest
            else session.chapter_id + 1
        )
        session.phase = "writing"
        session.original_setting = user_request.strip() or "自然续写下一章"
        session.refined_setting = ""
        session.draft_id = f"draft_{uuid.uuid4().hex[:16]}"
        session.draft_result = None
        session.revision_count = 0
        session.archived = False
        session.archive_result = None
        session.continuation_overview = None
        self._record(
            session,
            "continuation.started",
            {"chapter_id": session.chapter_id},
        )
        return self.generate_draft(session)

    @staticmethod
    def _fallback_plot_synopsis(material: dict[str, Any]) -> str:
        summaries = [
            str(item.get("content") or "").strip()
            for item in material.get("chapter_summaries") or []
            if isinstance(item, dict) and item.get("content")
        ]
        if summaries:
            return "\n".join(summaries)[-1000:]
        latest = material.get("latest_chapter") or {}
        return f"当前最新归档章节为第{latest.get('chapter_id', 0)}章《{latest.get('title', '')}》。"

    def prepare_memory_audit(self, session: ControlSession) -> str:
        session.phase = "audit_confirmation"
        self._record(session, "audit.requested", {})
        return (
            "记忆整理会按所选范围分批读取数据库并调用本地模型，可能消耗较多时间、"
            "计算资源和Token。确认后只先执行dry-run，不会修改数据库。"
        )

    def run_memory_audit_dry(
        self,
        session: ControlSession,
        *,
        scope: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session.phase = "audit_dry_run"
        scope_mode = str((scope or {}).get("mode") or "book")
        scope_label = {
            "chapters": "所选章节",
            "global": "全局记忆",
            "book": "全书记忆",
        }.get(scope_mode, "所选范围")
        with self.progress.step(
            "memory.audit",
            f"正在扫描并分批审计{scope_label}",
        ):
            audit_kwargs: dict[str, Any] = {"apply": False}
            if scope is not None:
                audit_kwargs["scope"] = scope
            result = self.rag_system.audit_book_memories(
                session.book_id,
                **audit_kwargs,
            )
        payload = result.model_dump(mode="json")
        payload.setdefault("coverage", {})["complete"] = result.coverage.complete
        session.pending_audit_result = payload
        session.phase = "audit_apply_confirmation"
        self._record(session, "audit.dry_run.completed", payload)
        return payload

    def apply_memory_audit(self, session: ControlSession) -> dict[str, Any]:
        pending = session.pending_audit_result or {}
        run_id = str(pending.get("run_id") or "")
        if not run_id:
            raise ValueError("没有可应用的dry-run审计计划")
        if (
            not (pending.get("coverage") or {}).get("complete", False)
            or not pending.get("semantic_candidate_complete", True)
            or pending.get("blocking_issue_ids")
            or pending.get("validation_errors")
        ):
            raise ValueError("dry-run未达到安全应用条件，禁止修改数据库")
        with self.progress.step("memory.audit.apply", "正在创建快照并应用已审核补丁"):
            result = self.rag_system.apply_saved_memory_audit(
                session.book_id,
                run_id,
            )
        payload = result.model_dump(mode="json")
        session.phase = "post_archive" if session.archived else "draft_review"
        self._record(session, "audit.applied", payload)
        return payload

    def answer_question(self, session: ControlSession, question: str) -> str:
        with self.progress.step("control.answer", "正在回答当前流程问题"):
            result = self.control_model.invoke_json(
                ANSWER_PROMPT,
                {
                    "phase": session.phase,
                    "task_code": session.task_code,
                    "has_draft": session.draft_result is not None,
                    "question": question,
                },
            )
        return str(result.get("answer") or "暂时无法回答这个问题。")

    def consult_story(
        self,
        book_id: str,
        question: str,
        *,
        current_chapter: int = 0,
        selected_text: str = "",
    ) -> dict[str, Any]:
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("请输入需要咨询的问题")
        latest = self.context_builder.latest_chapter(book_id)
        chapter_id = int(current_chapter or latest.get("chapter_id") or 1)
        with self.progress.step("control.consult.retrieve", "正在查询小说记忆"):
            context = self.context_builder.build(
                book_id,
                f"{clean_question}\n{selected_text[:2000]}",
                current_chapter=chapter_id,
                top_k=24,
                token_budget=6000,
            )
        memories = list(context.get("memories") or [])
        allowed = {
            str(item.get("memory_id")): int(item.get("source_chapter") or 0)
            for item in memories
            if item.get("memory_id")
        }
        with self.progress.step("control.consult.answer", "正在整理只读建议"):
            raw = self.control_model.invoke_json(
                STORY_CONSULT_PROMPT,
                {
                    "book_id": book_id,
                    "current_chapter": chapter_id,
                    "question": clean_question,
                    "selected_text": selected_text.strip()[:5000],
                    "memories": memories,
                },
            )
        result = StoryConsultResult.model_validate(raw)
        references = [
            reference
            for reference in result.references
            if reference.memory_id in allowed
            and reference.source_chapter == allowed[reference.memory_id]
        ]
        return {
            "answer": result.answer,
            "references": [item.model_dump(mode="json") for item in references],
            "insufficient_context": result.insufficient_context,
            "retrieved_count": len(memories),
            "used_rag": True,
        }

    def chat_story_assistant(
        self,
        question: str,
        *,
        book_id: str = "",
        current_chapter: int = 0,
        selected_text: str = "",
        force_rag: bool = False,
    ) -> dict[str, Any]:
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("请输入需要咨询的问题")
        if force_rag or requires_story_context(clean_question):
            if not book_id.strip():
                raise ValueError("请先选择需要查询的小说")
            return self.consult_story(
                book_id.strip(),
                clean_question,
                current_chapter=current_chapter,
                selected_text=selected_text,
            )
        with self.progress.step("control.chat.answer", "AI助手正在回答"):
            raw = self.control_model.invoke_json(
                GENERAL_ASSISTANT_PROMPT,
                {
                    "question": clean_question,
                    "selected_text": selected_text.strip()[:5000],
                },
            )
        answer = str(raw.get("answer") or "").strip()
        if not answer:
            raise ValueError("AI助手没有返回有效回答")
        return {
            "answer": answer,
            "references": [],
            "insufficient_context": False,
            "retrieved_count": 0,
            "used_rag": False,
        }

    def ui_message(
        self,
        session: ControlSession,
        *,
        action: str,
        prompt: str = "",
        options: list[str] | None = None,
        title: str = "",
        text: str = "",
        details: dict[str, Any] | None = None,
    ) -> AgentMessage[dict[str, Any]]:
        payload = ControlUiPayload(
            phase=session.phase,
            prompt=prompt,
            options=options or [],
            title=title,
            text=text,
            details=details or {},
        )
        return AgentMessage[dict[str, Any]](
            task_id=session.task_id,
            sender=self.agent_name,
            receiver="user",
            message_type="response",
            action=action,
            status="need_user_input" if prompt else "ok",
            payload=payload.model_dump(mode="json"),
            metadata={
                "session_id": session.session_id,
                "draft_id": session.draft_id,
                "revision": session.revision_count,
            },
        )

    def _send_writer(
        self,
        session: ControlSession,
        action: str,
        payload: ControlWriterPayload,
    ) -> ControlWriterResult:
        with self.progress.step("write.invoke", "正在调用Write_Agent生成结构化内容"):
            request = AgentMessage[dict[str, Any]](
                task_id=session.task_id,
                sender=self.agent_name,
                receiver=self.writer_agent.agent_name,
                action=action,
                payload=payload.model_dump(mode="json"),
                metadata={
                    "book_id": session.book_id,
                    "chapter_id": session.chapter_id,
                    "session_id": session.session_id,
                    "draft_id": session.draft_id,
                    "revision": session.revision_count,
                },
            )
            response = self.writer_agent.handle_message(request)
            if response.status != "ok":
                raise RuntimeError(response.error or "Write_Agent执行失败")
        with self.progress.step("control.parse", "正在用Python解析Write_Agent返回Message"):
            return ControlWriterResult.model_validate(response.payload or {})

    def extract_chapter_memories(
        self,
        *,
        book_id: str,
        chapter_id: int,
        title: str,
        text: str,
    ) -> list[MemoryFact]:
        request = AgentMessage[dict[str, Any]](
            sender=self.agent_name,
            receiver=self.writer_agent.agent_name,
            action="write.extract_memories",
            payload=ChapterMemoryExtractionPayload(
                chapter_id=chapter_id,
                title=title,
                text=text,
            ).model_dump(mode="json"),
            metadata={"book_id": book_id, "chapter_id": chapter_id},
        )
        response = self.writer_agent.handle_message(request)
        if response.status != "ok":
            raise RuntimeError(response.error or "WriteAgent章节记忆提取失败")
        result = ChapterMemoryExtractionResult.model_validate(response.payload or {})
        if not result.facts:
            raise ValueError("WriteAgent未从修改后的章节中提取出有效记忆")
        return result.facts

    def _locate_excerpt(self, full_text: str, hint: str, feedback: str) -> str:
        clean_hint = hint.strip()
        if not clean_hint:
            return ""
        if clean_hint in full_text:
            return clean_hint
        paragraphs = [
            value.strip()
            for value in re.split(r"\n+", full_text)
            if value.strip()
        ]
        ranked = sorted(
            (
                (SequenceMatcher(None, clean_hint, paragraph).ratio(), paragraph)
                for paragraph in paragraphs
            ),
            reverse=True,
        )
        if ranked:
            best_score, best = ranked[0]
            next_score = ranked[1][0] if len(ranked) > 1 else 0.0
            if best_score >= 0.82 and best_score - next_score >= 0.08:
                return best
        if self.locator_model is None:
            return ""
        candidates = [paragraph for _, paragraph in ranked[:8]]
        try:
            raw = self.locator_model.invoke_json(
                LOCATE_PROMPT,
                {
                    "target_hint": clean_hint,
                    "feedback": feedback,
                    "candidate_paragraphs": candidates,
                },
            )
        except Exception:
            return ""
        excerpt = str(raw.get("target_excerpt") or "").strip()
        confidence = float(raw.get("confidence") or 0.0)
        if raw.get("needs_clarification") or confidence < 0.65:
            return ""
        if not excerpt or excerpt not in full_text:
            return ""
        return excerpt

    @staticmethod
    def _allowed_intents(phase: str) -> list[str]:
        if phase == "draft_review":
            return [
                "revise_draft",
                "approve_draft",
                "request_memory_audit",
                "general_question",
                "cancel",
            ]
        if phase == "post_archive":
            return [
                "continue_writing",
                "request_memory_audit",
                "general_question",
                "cancel",
            ]
        if phase == "audit_confirmation":
            return ["confirm_audit", "cancel"]
        if phase == "audit_apply_confirmation":
            return ["confirm_audit_apply", "cancel"]
        return [
            "refine_setting",
            "start_writing",
            "request_memory_audit",
            "general_question",
            "cancel",
        ]

    @staticmethod
    def _explicit_command_intent(
        phase: str,
        user_text: str,
    ) -> ControlIntent | None:
        command = re.sub(r"[\s，。！？!?、,.;；:'\"“”‘’]+", "", user_text).casefold()
        if command in {"退出", "取消", "结束", "不写了", "exit", "quit"}:
            return ControlIntent(intent="cancel", confidence=1.0)
        if phase == "draft_review" and command in {
            "通过",
            "确认通过",
            "这版通过",
            "可以通过",
            "通过并归档",
            "归档",
            "approve",
        }:
            return ControlIntent(intent="approve_draft", confidence=1.0)
        if phase == "post_archive" and command in {
            "续写",
            "继续写",
            "继续续写",
            "续写下一章",
            "继续下一章",
            "continue",
        }:
            return ControlIntent(intent="continue_writing", confidence=1.0)
        return None

    @staticmethod
    def _intent_fallback(
        phase: str,
        user_text: str,
        raw: dict[str, Any],
    ) -> ControlIntent:
        if phase == "setting_input":
            raw_intent = str(raw.get("intent") or "").casefold()
            if "audit" in raw_intent or "整理" in raw_intent or "整理" in user_text:
                return ControlIntent(
                    intent="request_memory_audit",
                    confidence=0.7,
                )
            return ControlIntent(intent="start_writing", confidence=0.7)
        return ControlIntent(
            intent="unknown",
            confidence=0.0,
            needs_clarification=True,
            clarification="我收到的意图结果格式不完整。请再说明一次是修改、通过、续写还是整理记忆库。",
        )

    @staticmethod
    def _draft_display(session: ControlSession) -> tuple[str, str]:
        result = session.draft_result or {}
        if session.task_code in {"CT", "NW"}:
            return (
                str(result.get("chapter_title") or "未命名章节"),
                str(result.get("text") or ""),
            )
        if session.task_code == "RV":
            return (
                str(result.get("title") or "修改稿"),
                str(result.get("text") or ""),
            )
        if session.task_code == "BD":
            return (
                str(result.get("world_name") or "世界观设定"),
                json.dumps(
                    {key: value for key, value in result.items() if key != "world_name"},
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        characters = result.get("characters") or []
        title = "、".join(
            str(item.get("name"))
            for item in characters
            if isinstance(item, dict) and item.get("name")
        )
        return title or "人物设定", json.dumps(
            {"characters": characters},
            ensure_ascii=False,
            indent=2,
        )

    def _record(
        self,
        session: ControlSession,
        action: str,
        details: dict[str, Any],
    ) -> None:
        session.history.append({"action": action, "details": details})
        self.sessions.save(session)

    def _message_error(
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


def build_default_control_agent(
    data_dir: Path,
    *,
    heartbeat_seconds: float = 8.0,
) -> ControlAgent:
    from write_agent import WriteAgent

    load_dotenv()
    api_key = os.getenv("LLM_API_KEY")
    model_id = os.getenv("LLM_MODEL_ID")
    base_url = os.getenv("LLM_BASE_URL")
    if not api_key or not model_id or not base_url:
        raise ValueError(
            "Control_Agent需要.env中的LLM_API_KEY、LLM_MODEL_ID和LLM_BASE_URL"
        )
    control_model = LocalJsonModelClient(
        base_url,
        model_id,
        api_key=api_key,
        timeout=120,
    )
    bundled_qwen = Path(__file__).resolve().parent / "models" / "Qwen3.5-4B"
    configured_qwen = os.getenv("QWEN_LOCAL_MODEL_PATH")
    local_qwen_path = (
        configured_qwen
        or (str(bundled_qwen) if (bundled_qwen / "config.json").is_file() else None)
    )
    qwen = HuggingFaceQwenJsonClient(
        repo_id=os.getenv("QWEN_HF_MODEL_ID", "Qwen/Qwen3.5-4B"),
        local_model_path=local_qwen_path,
        cache_dir=os.getenv("QWEN_HF_CACHE"),
        device=os.getenv("QWEN_DEVICE", "auto"),
    )
    rag = NovelRagSystem(
        data_dir,
        memory_agent=MemoryAgent(qwen),
        judge=QwenMemoryJudge(qwen),
    )
    return ControlAgent(
        writer_agent=WriteAgent(),
        rag_system=rag,
        control_model=control_model,
        locator_model=qwen,
        progress=ConsoleProgress(heartbeat_seconds=heartbeat_seconds),
    )


def ask(prompt: str) -> str:
    print(f"\n[等待输入] {prompt}", flush=True)
    return input("> ").strip()


def choose_task() -> TaskCode:
    print(
        "\n请选择写作任务：\n"
        "  BD：世界观创作\n"
        "  CH：人物创作\n"
        "  CT：文章续写\n"
        "  NW：新文创作\n"
        "  RV：原文修改\n"
    )
    while True:
        try:
            return validate_control_task_code(ask("请输入 BD、CH、CT、NW 或 RV"))
        except ValueError as exc:
            print(f"[输入无效] {exc}")


def show_draft(result: ControlWriterResult) -> None:
    print("\n" + "=" * 24)
    print(result.display_title)
    print("=" * 24)
    print(result.display_text)
    print("=" * 24)


def show_continuation_overview(overview: ContinuationOverview) -> None:
    title = f"《{overview.latest_chapter_title}》" if overview.latest_chapter_title else ""
    print(
        f"\n[已载入] 最新归档章节：第{overview.latest_chapter_id}章{title}；"
        f"本次将续写第{overview.next_chapter_id}章"
    )
    print("\n[当前剧情梗概]")
    print(overview.plot_synopsis or "暂无可用剧情梗概。")
    print("\n[最新章节末尾500字]")
    print(overview.ending_preview or "最新归档章节正文为空。")


def audit_dialog(agent: ControlAgent, session: ControlSession) -> None:
    def restore_phase() -> None:
        if session.archived:
            session.phase = "post_archive"
        elif session.draft_result is not None:
            session.phase = "draft_review"
        else:
            session.phase = "setting_input"
        agent.sessions.save(session)

    print(f"\n[重要提示] {agent.prepare_memory_audit(session)}")
    confirmed = ask("确认开始全书dry-run审计吗？输入“确认”继续")
    if confirmed != "确认":
        print("[已取消] 未启动记忆整理。")
        restore_phase()
        return
    result = agent.run_memory_audit_dry(session)
    coverage = result.get("coverage") or {}
    print(
        "\n[审计报告] "
        f"记忆={len(coverage.get('total_memory_ids') or [])}，"
        f"操作={result.get('operation_count', 0)}，"
        f"覆盖完整={coverage.get('complete', False)}，"
        f"跨索引候选={result.get('semantic_candidate_reviewed_count', 0)}/"
        f"{result.get('semantic_candidate_count', 0)}，"
        f"阻断项={result.get('blocking_issue_ids') or []}，"
        f"产物={result.get('artifact_dir')}"
    )
    safe = (
        coverage.get("complete", False)
        and result.get("semantic_candidate_complete", True)
        and not result.get("blocking_issue_ids")
        and not result.get("validation_errors")
    )
    if not safe:
        print("[停止] 审计未达到安全应用条件，数据库未被修改。")
        restore_phase()
        return
    apply_confirmed = ask("确认应用上述补丁吗？输入“应用”继续；应用前会自动创建快照")
    if apply_confirmed != "应用":
        print("[已保留] dry-run报告已保存，数据库未被修改。")
        restore_phase()
        return
    applied = agent.apply_memory_audit(session)
    print(
        f"[完成] 记忆整理已应用，snapshot_id={applied.get('snapshot_id')}"
    )


def interactive_main(args: argparse.Namespace) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("Control_Agent 已启动。用户输入只由 Control_Agent 接收。")
    task_code = choose_task()
    book_id = ask("请输入书籍ID").strip()
    if not book_id:
        raise ValueError("书籍ID不能为空")
    if task_code == "CT":
        chapter_id = 0
    else:
        default_chapter = 1 if task_code in {"NW", "RV"} else 0
        chapter_text = ask(f"请输入章节号，直接回车使用 {default_chapter}")
        chapter_id = int(chapter_text) if chapter_text else default_chapter

    print("[正在] 正在初始化Control内部服务...", flush=True)
    agent = build_default_control_agent(
        args.data_dir,
        heartbeat_seconds=args.heartbeat_seconds,
    )
    print("[完成] Control内部服务初始化完成。", flush=True)
    session = agent.create_session(book_id, task_code, chapter_id=chapter_id)
    if task_code == "CT":
        try:
            overview = agent.prepare_initial_continuation(session)
        except FileNotFoundError as exc:
            print(f"[无法续写] {exc}")
            return 1
        show_continuation_overview(overview)

    while True:
        prompt = (
            "请输入续写方向，直接回车表示自然承接上一章；也可以要求整理当前书库"
            if task_code == "CT"
            else "请输入小说设定或写作要求；也可以直接要求整理当前书库"
        )
        setting = ask(prompt)
        if task_code == "CT" and not setting:
            setting = "自然承接上一章末尾继续写作"
            initial_intent = ControlIntent(
                intent="start_writing",
                confidence=1.0,
            )
        else:
            initial_intent = agent.interpret_user_message(session, setting)
        if initial_intent.needs_clarification:
            print(f"[需要确认] {initial_intent.clarification}")
            continue
        if initial_intent.intent == "request_memory_audit":
            audit_dialog(agent, session)
            return 0
        if initial_intent.intent == "general_question":
            print(f"[Control_Agent] {agent.answer_question(session, setting)}")
            continue
        if initial_intent.intent == "cancel":
            print("[已退出] 未启动写作。")
            return 0
        if initial_intent.intent in {"start_writing", "refine_setting"}:
            break
        print("[需要确认] 请说明写作要求，或明确说需要整理记忆库。")
    if supports_setting_refinement(task_code):
        refine = ask("是否需要Control_Agent细化设定？输入“需要”或“不需要”")
        if refine == "需要":
            expanded = agent.refine_setting(session, setting)
            print(f"\n[细化结果，共{len(expanded)}字]\n{expanded}")
        else:
            agent.keep_original_setting(session, setting)
    else:
        agent.keep_original_setting(session, setting)

    while True:
        try:
            result = agent.generate_draft(session)
            break
        except Exception as exc:
            print(
                f"[操作失败] 初稿生成未完成：{exc}。"
                "设定和会话已经保存，不需要重新输入。"
            )
            retry = ask("输入“重试”再次生成，输入其他内容退出并保留会话")
            if retry != "重试":
                print(f"[已保存] session_id={session.session_id}")
                return 1
    show_draft(result)

    while True:
        prompt = (
            "草稿已归档，可以要求续写、提问、整理记忆库或退出"
            if session.archived
            else "请说明修改意见，或说“通过”；也可以提问或整理记忆库"
        )
        user_text = ask(prompt)
        try:
            intent = agent.interpret_user_message(session, user_text)
        except Exception as exc:
            print(f"[操作失败] 意图判断暂时失败：{exc}。你可以直接重试当前输入。")
            continue
        if intent.needs_clarification:
            print(f"[需要确认] {intent.clarification}")
            continue
        if intent.intent == "revise_draft":
            try:
                result = agent.revise_draft(session, user_text, intent)
            except Exception as exc:
                print(f"[操作失败] 本次修改未完成：{exc}。原草稿保持不变，可以重新提出修改。")
                continue
            show_draft(result)
            continue
        if intent.intent == "approve_draft":
            try:
                archived = agent.archive_draft(session)
            except Exception as exc:
                print(f"[操作失败] 归档未完成：{exc}。草稿仍然保留，可以重试。")
                continue
            print(
                "[归档完成] "
                f"新建记忆={len(archived.get('created_memory_ids') or [])}，"
                f"更新记忆={len(archived.get('updated_memory_ids') or [])}"
            )
            continue
        if intent.intent == "continue_writing":
            if not session.archived:
                print("[需要先通过] 当前草稿尚未归档，请先确认通过后再续写。")
                continue
            try:
                result = agent.start_continuation(session, user_text)
            except Exception as exc:
                print(f"[操作失败] 续写未完成：{exc}。可以重新发起续写。")
                continue
            show_draft(result)
            continue
        if intent.intent == "request_memory_audit":
            try:
                audit_dialog(agent, session)
            except Exception as exc:
                print(f"[操作失败] 记忆整理未完成：{exc}。数据库未被直接修改。")
            continue
        if intent.intent == "general_question":
            try:
                print(f"[Control_Agent] {agent.answer_question(session, user_text)}")
            except Exception as exc:
                print(f"[操作失败] 暂时无法回答：{exc}")
            continue
        if intent.intent == "cancel":
            session.phase = "cancelled"
            agent.sessions.save(session)
            print("[已退出] 当前会话状态已保存。")
            return 0
        print("[需要确认] 请说明是修改、通过、续写、整理记忆库还是退出。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="小说写作系统唯一用户入口Control_Agent")
    parser.add_argument("--data-dir", type=Path, default=Path("rag_data"))
    parser.add_argument("--heartbeat-seconds", type=float, default=8.0)
    return parser.parse_args()


def main() -> int:
    return interactive_main(parse_args())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[Control_Agent失败] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
