from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_schema import AgentMessage
from control_agent import ControlAgent, supports_setting_refinement
from control_schemas import ControlIntent
from progress_display import ConsoleProgress, SilentProgress
from rag.config import RagConfig
from rag.maintenance_schemas import AuditCoverage, AuditRunResult
from rag.schemas import MemoryFact
from rag.system import NovelRagSystem
from write_agent import (
    WriteAgent,
    build_system_prompt,
    build_writer_messages,
    generate_writer_content,
)


class FakeControlModel:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def invoke_json(self, system_prompt: str, payload: dict) -> dict:
        self.calls.append({"system_prompt": system_prompt, "payload": payload})
        if "expanded_setting" in system_prompt:
            return {"expanded_setting": "扩写后的设定：主角在遗迹中寻找失踪的妹妹。"}
        if "续写准备组件" in system_prompt:
            return {
                "plot_synopsis": (
                    "林舟为寻找失踪的妹妹进入古代遗迹，已经抵达青铜门后的区域；"
                    "遗迹深处仍有身份不明的脚步声，相关危险尚未解除。"
                )
            }
        if "唯一意图路由器" in system_prompt:
            return {
                "intent": "approve_draft",
                "confidence": 0.98,
                "feedback": "",
                "target_hint": "",
                "is_logic_error": False,
                "needs_rag": False,
                "entities": [],
                "needs_clarification": False,
                "clarification": "",
            }
        if "简洁回答" in system_prompt:
            return {"answer": "当前草稿尚未归档。"}
        raise AssertionError("未预期的Control模型调用")


class ObjectEntitiesControlModel(FakeControlModel):
    def invoke_json(self, system_prompt: str, payload: dict) -> dict:
        if "唯一意图路由器" in system_prompt:
            return {
                "intent": "start_writing",
                "confidence": "95%",
                "feedback": "",
                "target_hint": "",
                "is_logic_error": "否",
                "needs_rag": "false",
                "entities": {
                    "世界设定": "太阳破碎，黑日从地底升起，主角陈玥在地下求生。"
                },
                "needs_clarification": "false",
                "clarification": "",
            }
        return super().invoke_json(system_prompt, payload)


class MissingLocatorRuntime:
    def invoke_json(self, system_prompt: str, payload: dict) -> dict:
        raise RuntimeError("缺少本地模型运行依赖")


class MalformedThenValidWriterModel:
    def __init__(self) -> None:
        self.calls = 0
        self.received_messages: list[list] = []

    def bind(self):
        return self

    def invoke(self, messages):
        self.calls += 1
        self.received_messages.append(messages)
        if self.calls == 1:
            return SimpleNamespace(
                content=(
                    '{"world_name":"永夜" "background":"黑日升起 0.9/T",'
                    '"rules":[],"factions":[],"locations":[],"conflict":"求生 0.8/F"}'
                )
            )
        return SimpleNamespace(
            content=(
                '{"world_name":"永夜","background":"黑日升起 0.9/T",'
                '"rules":[],"factions":[],"locations":[],"conflict":"求生 0.8/F"}'
            )
        )


class RecordingWriter:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.agent = WriteAgent(self.generate)
        self.agent_name = self.agent.agent_name

    def generate(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        task_code = kwargs["task_code"]
        is_revision = "当前操作为revise" in kwargs["control_prompt"]
        if task_code == "NW":
            return {
                "book_title": "青铜门",
                "chapter_title": "遗迹入口",
                "world": {
                    "background": "大陆遍布古代遗迹 0.85/T",
                    "rules": ["青铜钥匙可以开启外门 0.90/T"],
                    "conflict": "各方争夺遗迹 0.75/F",
                },
                "characters": [
                    {
                        "name": "林舟",
                        "role": "遗迹测绘师 0.85/T",
                        "profile": "谨慎而执着 0.80/T",
                        "goal": "寻找妹妹 0.75/F",
                    }
                ],
                "text": (
                    "林舟谨慎地观察石门，没有贸然行动。"
                    if is_revision
                    else "林舟没有观察石门，直接冲了进去。"
                ),
                "hooks": ["门后传来熟悉的声音 0.85/F"],
                "next": [],
            }
        if task_code == "CT":
            return {
                "chapter_title": "门后",
                "text": "林舟沿着石阶继续向下。",
                "characters": ["林舟"],
                "events": ["林舟进入遗迹深层 0.75/F"],
                "changes": [],
                "hooks": ["黑暗中出现新的脚步声 0.70/F"],
                "next": [],
            }
        raise AssertionError(f"未实现测试任务: {task_code}")

    def handle_message(self, message):
        return self.agent.handle_message(message)


class AppendingWriter(RecordingWriter):
    def generate(self, **kwargs) -> dict:
        result = super().generate(**kwargs)
        if "当前操作为extend" in kwargs["control_prompt"]:
            result["text"] = "随后他停下脚步，听见门后传来回声。"
        elif "当前操作为revise" in kwargs["control_prompt"]:
            result["text"] = "林舟没有观察石门，直接冲了进去。随后他停下脚步，听见门后传来回声。"
        return result


class RepeatingAppendingWriter(RecordingWriter):
    def generate(self, **kwargs) -> dict:
        result = super().generate(**kwargs)
        if "当前操作为extend" in kwargs["control_prompt"]:
            result["text"] = (
                "林舟没有观察石门，直接冲了进去。"
                "随后他停下脚步，听见门后传来回声。"
            )
        return result


class FakeAuditSystem:
    def __init__(self, root: Path) -> None:
        self.config = RagConfig(root)
        self.audit_calls = 0
        self.apply_calls = 0

    def audit_book_memories(self, book_id: str, *, apply: bool = False):
        self.audit_calls += 1
        coverage = AuditCoverage(
            total_memory_ids=["memory_1"],
            assigned_memory_ids=["memory_1"],
            reviewed_memory_ids=["memory_1"],
        )
        return AuditRunResult(
            run_id="audit_saved",
            book_id=book_id,
            dry_run=True,
            applied=False,
            artifact_dir=str(self.config.root_dir / "audit_saved"),
            deterministic_issue_count=0,
            model_finding_count=0,
            operation_count=1,
            coverage=coverage,
        )

    def apply_saved_memory_audit(self, book_id: str, run_id: str):
        self.apply_calls += 1
        coverage = AuditCoverage(
            total_memory_ids=["memory_1"],
            assigned_memory_ids=["memory_1"],
            reviewed_memory_ids=["memory_1"],
        )
        return AuditRunResult(
            run_id=run_id,
            book_id=book_id,
            dry_run=False,
            applied=True,
            snapshot_id="snapshot_1",
            artifact_dir=str(self.config.root_dir / "audit_saved"),
            deterministic_issue_count=0,
            model_finding_count=0,
            operation_count=1,
            coverage=coverage,
        )


class ControlAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.control_model = FakeControlModel()
        self.writer = RecordingWriter()
        self.rag = NovelRagSystem(self.root)
        self.agent = ControlAgent(
            writer_agent=self.writer,
            rag_system=self.rag,
            control_model=self.control_model,
            progress=SilentProgress(),
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_control_prompt_is_appended_without_replacing_original_template(self) -> None:
        original = build_system_prompt("NW")
        messages = build_writer_messages(
            "NW",
            "创作遗迹故事",
            control_prompt="CONTROL_APPEND_MARKER",
        )
        system_content = str(messages[0].content)
        self.assertIn(original, system_content)
        self.assertIn("CONTROL_APPEND_MARKER", system_content)
        self.assertLess(system_content.index(original), system_content.index("CONTROL_APPEND_MARKER"))

    def test_writer_retries_once_when_first_json_is_malformed(self) -> None:
        model = MalformedThenValidWriterModel()
        with patch("write_agent.get_writer_llm", return_value=model):
            result = generate_writer_content("BD", "创建黑日世界")
        self.assertEqual(result["world_name"], "永夜")
        self.assertEqual(model.calls, 2)
        self.assertGreater(len(model.received_messages[1]), len(model.received_messages[0]))

    def test_write_agent_rejects_direct_user_request(self) -> None:
        request = AgentMessage[dict](
            sender="user",
            receiver="write_agent",
            action="write.generate",
            payload={},
        )
        response = self.writer.handle_message(request)
        self.assertEqual(response.status, "error")
        self.assertIn("control_agent", response.error or "")

    def test_modified_chapter_memories_are_extracted_by_write_agent(self) -> None:
        calls: list[dict] = []

        def extract_memories(**kwargs) -> list[MemoryFact]:
            calls.append(kwargs)
            return [
                MemoryFact(
                    fact_type="event",
                    content="陈玥将月纹钥匙交给林舟",
                    character_names=["陈玥", "林舟"],
                    item_names=["月纹钥匙"],
                    raw_importance=0.8,
                )
            ]

        writer = WriteAgent(
            self.writer.generate,
            memory_extractor=extract_memories,
        )
        agent = ControlAgent(
            writer_agent=writer,
            rag_system=self.rag,
            control_model=self.control_model,
            progress=SilentProgress(),
        )

        facts = agent.extract_chapter_memories(
            book_id="book_modified",
            chapter_id=3,
            title="第三章 新的约定",
            text="这是用户修改后的完整正文。",
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["chapter_id"], 3)
        self.assertEqual(calls[0]["title"], "第三章 新的约定")
        self.assertEqual(calls[0]["text"], "这是用户修改后的完整正文。")
        self.assertEqual(facts[0].raw_importance, 0.8)

    def test_draft_is_revised_with_same_template_and_archived_only_after_approval(self) -> None:
        session = self.agent.create_session("book_control", "NW", chapter_id=1)
        refined = self.agent.refine_setting(session, "主角探索遗迹")
        self.assertLessEqual(len(refined), 500)

        first = self.agent.generate_draft(session)
        self.assertIn("直接冲了进去", first.display_text)
        document = self.root / "book_control" / "documents" / "chapter_000001.json"
        self.assertFalse(document.exists(), "未通过的草稿不能归档")

        intent = ControlIntent(
            intent="revise_draft",
            confidence=0.95,
            feedback="这不符合林舟谨慎的性格",
            target_hint="林舟没有观察石门，直接冲了进去。",
            is_logic_error=True,
            needs_rag=True,
            entities=["林舟"],
        )
        revised = self.agent.revise_draft(session, intent.feedback, intent)
        self.assertIn("谨慎地观察", revised.display_text)
        self.assertEqual(self.writer.calls[-1]["task_code"], "NW")
        self.assertIn("当前操作为revise", self.writer.calls[-1]["control_prompt"])
        self.assertFalse(document.exists())

        archive = self.agent.archive_draft(session)
        self.assertTrue(document.exists())
        second_archive = self.agent.archive_draft(session)
        self.assertEqual(archive, second_archive, "重复通过不得再次入库")

    def test_extend_draft_appends_without_changing_chapter(self) -> None:
        writer = AppendingWriter()
        agent = ControlAgent(
            writer_agent=writer,
            rag_system=self.rag,
            control_model=self.control_model,
            progress=SilentProgress(),
        )
        session = agent.create_session("book_extend", "NW", chapter_id=3)
        agent.keep_original_setting(session, "林舟正在探索青铜门")
        original = agent.generate_draft(session)

        extended = agent.extend_draft(session, "补写门后的声音")

        self.assertEqual(session.chapter_id, 3)
        self.assertEqual(session.task_code, "NW")
        self.assertTrue(extended.display_text.startswith(original.display_text))
        self.assertGreater(len(extended.display_text), len(original.display_text))
        self.assertEqual(writer.calls[-1]["context"]["append_context"]["chapter_title"], "遗迹入口")
        self.assertEqual(writer.calls[-1]["context"]["retrieval_trace"]["top_k"], 60)
        self.assertIn("当前操作为extend", writer.calls[-1]["control_prompt"])

    def test_extend_loaded_draft_uses_request_when_session_has_no_setting(self) -> None:
        writer = AppendingWriter()
        agent = ControlAgent(
            writer_agent=writer,
            rag_system=self.rag,
            control_model=self.control_model,
            progress=SilentProgress(),
        )
        session = agent.create_session("book_loaded_extend", "NW", chapter_id=2)
        original_text = "林舟没有观察石门，直接冲了进去。"
        session.draft_result = {
            "book_title": "青铜门",
            "chapter_title": "遗迹入口",
            "world": {"background": "", "rules": [], "conflict": ""},
            "characters": [],
            "text": original_text,
            "hooks": [],
            "next": [],
        }
        session.phase = "draft_review"

        extended = agent.extend_draft(session, "补写门后的声音")

        self.assertTrue(extended.display_text.startswith(original_text))
        self.assertTrue(writer.calls[-1]["user_input"].strip())
        self.assertIn("补写门后的声音", writer.calls[-1]["user_input"])

    def test_extend_draft_strips_repeated_original_prefix(self) -> None:
        writer = RepeatingAppendingWriter()
        agent = ControlAgent(
            writer_agent=writer,
            rag_system=self.rag,
            control_model=self.control_model,
            progress=SilentProgress(),
        )
        session = agent.create_session("book_repeat_extend", "NW", chapter_id=2)
        original_text = "林舟没有观察石门，直接冲了进去。"
        session.draft_result = {
            "book_title": "青铜门",
            "chapter_title": "遗迹入口",
            "world": {"background": "", "rules": [], "conflict": ""},
            "characters": [],
            "text": original_text,
            "hooks": [],
            "next": [],
        }
        session.phase = "draft_review"

        extended = agent.extend_draft(session, "补写门后的声音")

        self.assertEqual(extended.display_text.count(original_text), 1)
        self.assertTrue(extended.display_text.startswith(original_text))
        self.assertIn("门后传来回声", extended.display_text)

    def test_locator_failure_falls_back_to_full_draft_revision(self) -> None:
        agent = ControlAgent(
            writer_agent=self.writer,
            rag_system=self.rag,
            control_model=self.control_model,
            locator_model=MissingLocatorRuntime(),
            progress=SilentProgress(),
        )
        session = agent.create_session("book_locator_fallback", "NW", chapter_id=1)
        agent.keep_original_setting(session, "陈玥在荧光海遇见少女")
        agent.generate_draft(session)
        intent = ControlIntent(
            intent="revise_draft",
            confidence=0.9,
            feedback="将陈玥的服装改为绣着金色花纹的和服，且赤裸双脚",
            target_hint="陈玥的服装",
        )
        revised = agent.revise_draft(session, intent.feedback, intent)
        self.assertTrue(revised.display_text)
        revision = self.writer.calls[-1]["context"]["revision"]
        self.assertEqual(revision["target_excerpt"], "")
        self.assertEqual(revision["user_feedback"], intent.feedback)

    def test_continuation_reads_previous_chapter_and_ranked_memories(self) -> None:
        session = self.agent.create_session("book_continue", "NW", chapter_id=1)
        self.agent.keep_original_setting(session, "林舟探索遗迹")
        self.agent.generate_draft(session)
        self.agent.archive_draft(session)

        continued = self.agent.start_continuation(session, "继续写林舟进入石门")
        self.assertEqual(session.task_code, "CT")
        self.assertEqual(session.chapter_id, 2)
        self.assertEqual(continued.display_title, "门后")
        context = self.writer.calls[-1]["context"]
        self.assertIn("continuity", context)
        self.assertEqual(context["continuity"]["source_chapter_id"], 1)
        self.assertIn("林舟没有观察石门", context["continuity"]["ending_excerpt"])
        self.assertEqual(context["continuity"]["excerpt_strategy"], "chapter_tail")
        self.assertTrue(context["memories"])
        self.assertIn("retrieval_trace", context)
        self.assertEqual(context["retrieval_trace"]["top_k"], 60)
        self.assertIn("ending_excerpt是上一章末尾原文", self.writer.calls[-1]["control_prompt"])

    def test_initial_continuation_loads_latest_chapter_overview_and_tail(self) -> None:
        book_id = "book_initial_continue"
        first = self.agent.create_session(book_id, "NW", chapter_id=1)
        self.agent.keep_original_setting(first, "林舟探索遗迹")
        self.agent.generate_draft(first)
        self.agent.archive_draft(first)

        long_text = "HEAD_MARKER" + ("前段内容" * 700) + "林舟停在裂谷边缘。TAIL_MARKER"
        documents = self.root / book_id / "documents"
        documents.mkdir(parents=True, exist_ok=True)
        (documents / "chapter_000003.json").write_text(
            json.dumps(
                {
                    "book_id": book_id,
                    "chapter_id": 3,
                    "task_code": "CT",
                    "result": {
                        "chapter_title": "裂谷边缘",
                        "text": long_text,
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        session = self.agent.create_session(book_id, "CT", chapter_id=0)
        before = len(self.control_model.calls)
        overview = self.agent.prepare_initial_continuation(session)

        self.assertEqual(overview.latest_chapter_id, 3)
        self.assertEqual(overview.next_chapter_id, 4)
        self.assertEqual(session.chapter_id, 4)
        self.assertEqual(overview.latest_chapter_title, "裂谷边缘")
        self.assertLessEqual(len(overview.ending_preview), 500)
        self.assertIn("TAIL_MARKER", overview.ending_preview)
        self.assertNotIn("HEAD_MARKER", overview.ending_preview)
        overview_calls = [
            call
            for call in self.control_model.calls[before:]
            if "续写准备组件" in call["system_prompt"]
        ]
        self.assertEqual(len(overview_calls), 1)

        self.agent.keep_original_setting(session, "自然承接上一章末尾继续写作")
        self.agent.generate_draft(session)
        context = self.writer.calls[-1]["context"]
        self.assertEqual(context["continuity"]["source_chapter_id"], 3)
        self.assertIn("TAIL_MARKER", context["continuity"]["ending_excerpt"])
        self.assertNotIn("HEAD_MARKER", context["continuity"]["ending_excerpt"])
        self.assertEqual(context["plot_overview"]["latest_chapter_id"], 3)
        self.assertEqual(context["plot_overview"]["synopsis"], overview.plot_synopsis)
        self.assertGreater(
            context["retrieval_trace"]["estimated_continuity_tokens"],
            0,
        )

    def test_initial_continuation_rejects_book_without_archived_chapter(self) -> None:
        session = self.agent.create_session("empty_book", "CT", chapter_id=0)
        with self.assertRaisesRegex(FileNotFoundError, "没有可续写的已归档章节"):
            self.agent.prepare_initial_continuation(session)

    def test_intent_is_classified_once_with_session_state(self) -> None:
        session = self.agent.create_session("book_intent", "NW", chapter_id=1)
        session.phase = "draft_review"
        before = len(self.control_model.calls)
        intent = self.agent.classify_intent(session, "我觉得这一版已经可以正式采用了")
        self.assertEqual(intent.intent, "approve_draft")
        self.assertEqual(len(self.control_model.calls), before + 1)
        payload = self.control_model.calls[-1]["payload"]
        self.assertEqual(payload["phase"], "draft_review")
        self.assertIn("approve_draft", payload["allowed_intents"])

    def test_exact_approval_is_local_and_survives_model_network_failure(self) -> None:
        class FailingControlModel:
            def invoke_json(self, system_prompt: str, payload: dict) -> dict:
                raise OSError("SSL connection closed")

        agent = ControlAgent(
            writer_agent=self.writer,
            rag_system=self.rag,
            control_model=FailingControlModel(),
            progress=SilentProgress(),
        )
        session = agent.create_session("book_local_approval", "NW", chapter_id=1)
        session.phase = "draft_review"

        intent = agent.classify_intent(session, "通过。")

        self.assertEqual(intent.intent, "approve_draft")
        self.assertEqual(intent.confidence, 1.0)

    def test_only_creation_tasks_support_setting_refinement(self) -> None:
        self.assertTrue(supports_setting_refinement("BD"))
        self.assertTrue(supports_setting_refinement("CH"))
        self.assertTrue(supports_setting_refinement("NW"))
        self.assertFalse(supports_setting_refinement("CT"))
        self.assertFalse(supports_setting_refinement("RV"))

        before = len(self.control_model.calls)
        for task_code in ("CT", "RV"):
            session = self.agent.create_session(
                f"book_no_refine_{task_code.lower()}",
                task_code,
                chapter_id=1,
            )
            with self.assertRaisesRegex(ValueError, "不支持设定细化"):
                self.agent.refine_setting(session, "不应被扩写的要求")
        self.assertEqual(len(self.control_model.calls), before)

    def test_intent_tolerates_model_returning_entities_as_object(self) -> None:
        agent = ControlAgent(
            writer_agent=self.writer,
            rag_system=self.rag,
            control_model=ObjectEntitiesControlModel(),
            progress=SilentProgress(),
        )
        session = agent.create_session("book_object_entities", "BD", chapter_id=1)
        intent = agent.classify_intent(
            session,
            "故事发生在太阳破碎后，主角陈玥生活在地下荧光海。",
        )
        self.assertEqual(intent.intent, "start_writing")
        self.assertEqual(intent.confidence, 0.95)
        self.assertEqual(intent.entities, [])
        self.assertFalse(intent.needs_rag)

    def test_user_intent_enters_control_through_unified_message(self) -> None:
        session = self.agent.create_session("book_message", "NW", chapter_id=1)
        session.phase = "draft_review"
        self.agent.sessions.save(session)
        request = AgentMessage[dict](
            task_id=session.task_id,
            sender="user",
            receiver="control_agent",
            action="control.interpret",
            payload={
                "book_id": session.book_id,
                "session_id": session.session_id,
                "text": "这版通过",
            },
        )
        response = self.agent.handle_message(request)
        self.assertEqual(response.status, "ok")
        self.assertEqual(response.parent_message_id, request.message_id)
        self.assertEqual(response.sender, "control_agent")
        self.assertEqual(response.payload["intent"]["intent"], "approve_draft")

    def test_saved_dry_run_is_applied_without_second_model_audit(self) -> None:
        fake_rag = FakeAuditSystem(self.root)
        agent = ControlAgent(
            writer_agent=self.writer,
            rag_system=fake_rag,  # type: ignore[arg-type]
            control_model=self.control_model,
            progress=SilentProgress(),
        )
        session = agent.create_session("book_audit", "NW", chapter_id=1)
        warning = agent.prepare_memory_audit(session)
        self.assertIn("dry-run", warning)
        dry = agent.run_memory_audit_dry(session)
        self.assertTrue(dry["coverage"]["complete"])
        applied = agent.apply_memory_audit(session)
        self.assertEqual(applied["snapshot_id"], "snapshot_1")
        self.assertEqual(fake_rag.audit_calls, 1)
        self.assertEqual(fake_rag.apply_calls, 1)

    def test_console_progress_always_shows_start_and_completion(self) -> None:
        output = io.StringIO()
        progress = ConsoleProgress(heartbeat_seconds=10)
        with redirect_stdout(output):
            with progress.step("test", "正在执行测试步骤"):
                pass
        rendered = output.getvalue()
        self.assertIn("[正在]", rendered)
        self.assertIn("[完成]", rendered)


if __name__ == "__main__":
    unittest.main()
