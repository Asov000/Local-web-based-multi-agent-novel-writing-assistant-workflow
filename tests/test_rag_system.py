from __future__ import annotations

import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from agent_schema import AgentMessage
from chapter_library import ChapterLibrary
from rag.config import RagConfig
from rag.local_model import LocalJsonModelClient, ensure_huggingface_snapshot
from rag.memory_agent import MemoryAgent
from rag.qwen_judge import QwenMemoryJudge, SemanticSimilarityJudge
from rag.repository import BookRepository
from rag.retriever import estimate_tokens
from rag.schemas import AtomicMemory, MemoryFact
from rag.system import NovelRagSystem


class BooleanJudgeClient:
    def __init__(self) -> None:
        self.invoke_count = 0
        self.embed_count = 0

    def invoke_json(self, system_prompt: str, payload: dict) -> dict:
        self.invoke_count += 1
        return {
            "results": [
                {
                    "confirmed": False,
                    "updated": False,
                    "referenced": False,
                    "conflict": False,
                    "unrelated": False,
                }
            ]
        }

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.embed_count += 1
        return [[1.0, 0.0], [0.0, 1.0]]


class FixedEmbeddingClient:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0],
            [0.96, 0.28],
            [0.88, 0.475],
            [0.80, 0.60],
        ]


class FakeCompletionClient:
    def __init__(self) -> None:
        self.last_message: dict | None = None

    def invoke_json(self, system_prompt: str, payload: dict) -> dict:
        self.last_message = payload
        payload = payload.get("payload", payload)
        return {
            "completed_fields": {
                field: 0.6
                for field in payload["missing_fields"]
                if field.endswith("raw_importance")
            }
        }


class ChapterExtractionClient:
    def __init__(self, *, invalid_first: bool = False) -> None:
        self.invalid_first = invalid_first
        self.prompts: list[str] = []
        self.requests: list[dict] = []

    def invoke_json(self, system_prompt: str, payload: dict) -> dict:
        self.prompts.append(system_prompt)
        self.requests.append(payload)
        score = 10 if self.invalid_first and len(self.requests) == 1 else 0.8
        return {
            "schema_version": "rag.message.v1",
            "sender": "qwen_model",
            "receiver": "memory_agent",
            "message_type": "response",
            "action": "rag.model.memory.extract_chapter.result",
            "status": "ok",
            "task_id": payload["task_id"],
            "parent_message_id": payload["message_id"],
            "book_id": payload["book_id"],
            "payload": {
                "facts": [
                    {
                        "fact_type": "event",
                        "content": "陈玥打开了青铜门",
                        "character_names": ["陈玥"],
                        "event_names": ["打开青铜门"],
                        "raw_importance": score,
                        "canon_candidate": False,
                        "memory_scope": "temporary",
                    }
                ]
            },
            "operations": [],
        }


class FakeHttpResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return (
            b'{"choices":[{"message":{"content":"{\\"status\\":\\"ok\\"}"}}]}'
        )


class ReplacementMemoryAgent(MemoryAgent):
    def __init__(self) -> None:
        super().__init__(model_client=object())
        self.extraction_calls = 0

    def extract_chapter_facts(self, **kwargs) -> list[MemoryFact]:
        self.extraction_calls += 1
        return [
            MemoryFact(
                fact_type="event",
                content=f"重写后的章节事件：{kwargs['text']}",
                event_names=["重写事件"],
                raw_importance=0.8,
            )
        ]


class RagSystemTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.system = NovelRagSystem(self.root)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_folder_databases_and_canon_routing(self) -> None:
        result = self.system.ingest(
            "book_001",
            "BD",
            {
                "world_name": "永夜大陆",
                "background": "大陆被永夜笼罩 0.90/T",
                "rules": ["亡者无法真正复活 0.95/T"],
                "factions": ["守夜人驻守北境 0.70/F"],
                "locations": [],
                "conflict": "人类试图终结永夜 0.85/F",
            },
        )
        self.assertEqual(result.fact_count, 4)
        for store_name in (
            "canon_memory",
            "chapter_memory",
            "state_timeline_memory",
            "relation_hook_memory",
            "index",
            "conflicts",
        ):
            path = self.root / "book_001" / store_name / f"{store_name}.sqlite3"
            self.assertTrue(path.exists(), path)

        repository = BookRepository(RagConfig(self.root), "book_001")
        canon = repository.store("canon_memory").list_memories(statuses=None)
        chapter = repository.store("chapter_memory").list_memories(statuses=None)
        self.assertEqual(len(canon), 2)
        self.assertEqual(len(chapter), 2)

    def test_state_history_keeps_only_latest_current_value(self) -> None:
        first = {
            "characters": [
                {
                    "name": "林舟",
                    "role": "流亡王族 0.95/T",
                    "appearance": "黑发 0.40/T",
                    "personality": "谨慎 0.80/T",
                    "background": "来自北境 0.75/T",
                    "goal": "找到妹妹 0.70/F",
                    "ability": "感知遗迹 0.85/T",
                    "relations": [],
                }
            ]
        }
        second = {
            "characters": [
                {
                    "name": "林舟",
                    "role": "流亡王族 0.95/T",
                    "appearance": "黑发 0.40/T",
                    "personality": "谨慎 0.80/T",
                    "background": "来自北境 0.75/T",
                    "goal": "进入青铜门 0.80/F",
                    "ability": "感知遗迹 0.85/T",
                    "relations": [],
                }
            ]
        }
        self.system.ingest("book_002", "CH", first, chapter_id=1)
        self.system.ingest("book_002", "CH", second, chapter_id=2)
        repository = BookRepository(RagConfig(self.root), "book_002")
        states = repository.store("state_timeline_memory").list_memories(statuses=None)
        goals = [memory for memory in states if memory.entity_name == "林舟" and memory.field == "goal"]
        self.assertEqual(len(goals), 2)
        self.assertEqual(sum(memory.is_current for memory in goals), 1)
        current = next(memory for memory in goals if memory.is_current)
        previous = next(memory for memory in goals if not memory.is_current)
        self.assertEqual(current.content, "进入青铜门")
        self.assertEqual(previous.status, "archived")

    def test_retrieval_builds_writer_context(self) -> None:
        self.system.ingest(
            "book_003",
            "NW",
            {
                "book_title": "青铜门",
                "chapter_title": "遗迹",
                "world": {
                    "background": "大陆存在古代遗迹 0.85/T",
                    "rules": ["王族血脉可以开启青铜门 0.95/T"],
                    "conflict": "各方争夺遗迹 0.75/F",
                },
                "characters": [
                    {
                        "name": "林舟",
                        "role": "流亡王族 0.95/T",
                        "profile": "谨慎而执着 0.80/T",
                        "goal": "寻找妹妹 0.75/F",
                    }
                ],
                "text": "林舟来到遗迹入口。",
                "hooks": ["门后传来妹妹的声音 0.90/F"],
                "next": [],
            },
            chapter_id=1,
        )
        payload = self.system.build_writer_payload(
            "book_003",
            "继续写林舟进入青铜门后的剧情",
            current_chapter=2,
        )
        self.assertEqual(payload["u"], "继续写林舟进入青铜门后的剧情")
        self.assertTrue(payload["c"]["canon"])
        self.assertTrue(payload["c"]["states"])
        self.assertTrue(payload["c"]["open_hooks"])
        self.assertTrue(payload["c"]["recent_chapters"])

        compact = self.system.retrieve_context(
            "book_003",
            "完全没有实体名称的请求",
            current_chapter=2,
            token_budget=40,
        )
        contents = [
            item["content"]
            for group in compact.values()
            for item in group
        ]
        self.assertTrue(contents)
        self.assertLessEqual(sum(estimate_tokens(text) + 12 for text in contents), 40)

    def test_memory_agent_uses_unified_message(self) -> None:
        client = FakeCompletionClient()
        agent = MemoryAgent(client)
        request = AgentMessage[dict](
            task_id="task_shared",
            sender="control_agent",
            receiver="memory_agent",
            action="memory.complete",
            payload={
                "missing_fields": ["facts[0].raw_importance"],
                "known_fields": {"facts": []},
                "text": "林舟打开了青铜门。",
            },
        )
        response = agent.handle_message(request)
        self.assertEqual(response.status, "ok")
        self.assertEqual(response.message_type, "response")
        self.assertEqual(response.task_id, request.task_id)
        self.assertEqual(response.parent_message_id, request.message_id)
        self.assertEqual(response.sender, "memory_agent")
        self.assertEqual(
            response.payload["completed_fields"]["facts[0].raw_importance"],
            0.6,
        )
        self.assertEqual(client.last_message["schema_version"], "rag.message.v1")
        self.assertEqual(
            client.last_message["action"],
            "rag.model.memory.complete.request",
        )

    def test_chapter_extraction_uses_structured_prompt_and_strict_scale(self) -> None:
        client = ChapterExtractionClient()
        facts = MemoryAgent(client).extract_chapter_facts(
            book_id="book_prompt",
            chapter_id=1,
            title="青铜门",
            text="陈玥打开了青铜门。",
        )

        self.assertEqual(facts[0].raw_importance, 0.8)
        prompt = client.prompts[0]
        self.assertIn("任务名称：章节记忆提取", prompt)
        self.assertIn("【角色】", prompt)
        self.assertIn("【规则】", prompt)
        self.assertIn("【输出格式】", prompt)
        self.assertIn("禁止使用1到10的十分制", prompt)
        self.assertIn('"raw_importance":0.8', prompt)

    def test_chapter_extraction_retries_invalid_importance(self) -> None:
        client = ChapterExtractionClient(invalid_first=True)
        facts = MemoryAgent(client).extract_chapter_facts(
            book_id="book_retry",
            chapter_id=1,
            title="青铜门",
            text="陈玥打开了青铜门。",
        )

        self.assertEqual(len(client.requests), 2)
        self.assertEqual(facts[0].raw_importance, 0.8)
        feedback = client.requests[1]["metadata"]["validation_feedback"]
        self.assertEqual(feedback["source"], "python_response_validator")
        self.assertEqual(feedback["error_type"], "ValidationError")
        self.assertIn("less than or equal to 1", feedback["error_message"])

    def test_huggingface_snapshot_downloads_when_local_cache_is_absent(self) -> None:
        calls: list[bool] = []
        snapshot = self.root / "hf_snapshot"
        snapshot.mkdir()

        def fake_download(**kwargs) -> str:
            calls.append(kwargs["local_files_only"])
            if kwargs["local_files_only"]:
                raise FileNotFoundError("not cached")
            return str(snapshot)

        resolved = ensure_huggingface_snapshot(
            "Qwen/Qwen3-0.6B",
            snapshot_download_fn=fake_download,
        )
        self.assertEqual(resolved, snapshot.resolve())
        self.assertEqual(calls, [True, False])

    def test_huggingface_snapshot_prefers_explicit_local_model(self) -> None:
        local_model = self.root / "local_qwen"
        local_model.mkdir()
        (local_model / "config.json").write_text("{}", encoding="utf-8")

        def unexpected_download(**kwargs) -> str:
            raise AssertionError("本地模型存在时不应调用Hugging Face")

        resolved = ensure_huggingface_snapshot(
            "Qwen/Qwen3-0.6B",
            local_dir=local_model,
            snapshot_download_fn=unexpected_download,
        )
        self.assertEqual(resolved, local_model.resolve())

    def test_openai_compatible_client_retries_transient_ssl_disconnect(self) -> None:
        client = LocalJsonModelClient(
            "https://example.invalid/v1",
            "test-model",
            max_retries=2,
            retry_backoff_seconds=0,
        )
        with patch(
            "rag.local_model.urllib.request.urlopen",
            side_effect=[urllib.error.URLError("SSL EOF"), FakeHttpResponse()],
        ) as mocked_urlopen:
            result = client.invoke_json("return json", {"value": 1})

        self.assertEqual(result, {"status": "ok"})
        self.assertEqual(mocked_urlopen.call_count, 2)

    def test_qwen_boolean_result_falls_back_to_semantic_similarity(self) -> None:
        fact = MemoryFact(
            fact_type="event",
            content="陈玥进入荧光海",
            raw_importance=0.8,
        )
        memory = AtomicMemory(
            memory_id="memory_old",
            book_id="book_1",
            store_type="chapter_memory",
            memory_type="event",
            content="陈玥离开地下城",
            raw_importance=0.8,
            effective_importance=0.8,
            content_hash="hash",
        )
        client = BooleanJudgeClient()
        result = QwenMemoryJudge(client).judge([fact], [memory])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].memory_id, memory.memory_id)
        self.assertEqual(result[0].status, "unrelated")
        self.assertEqual(result[0].confidence, 1.0)
        self.assertEqual(client.invoke_count, 2)
        self.assertEqual(client.embed_count, 1)
        self.assertNotIn(result[0].status, {"updated", "conflict"})

    def test_semantic_fallback_never_infers_update_or_conflict(self) -> None:
        fact = MemoryFact(
            fact_type="event",
            content="new fact",
            raw_importance=0.8,
        )
        memories = [
            AtomicMemory(
                memory_id=f"memory_{index}",
                book_id="book_1",
                store_type="chapter_memory",
                memory_type="event",
                content=f"candidate {index}",
                raw_importance=0.8,
                effective_importance=0.8,
                content_hash=f"hash_{index}",
            )
            for index in range(3)
        ]
        results = SemanticSimilarityJudge(FixedEmbeddingClient()).judge(
            [fact], memories
        )
        self.assertEqual(
            [result.status for result in results],
            ["confirmed", "referenced", "unrelated"],
        )
        self.assertFalse(
            {result.status for result in results} & {"updated", "conflict"}
        )

    def test_replacing_chapter_versions_document_and_rebuilds_owned_memories(self) -> None:
        memory_agent = ReplacementMemoryAgent()
        system = NovelRagSystem(self.root, memory_agent=memory_agent)
        system.ingest(
            "book_replace",
            "CT",
            {
                "chapter_title": "第一章 旧标题",
                "text": "旧正文",
                "characters": ["林舟"],
                "events": ["林舟进入旧城 0.8/F"],
                "changes": [],
                "hooks": [],
            },
            chapter_id=1,
        )

        replaced = system.replace_chapter(
            "book_replace",
            "RV",
            {
                "title": "第一章 新标题",
                "text": "林舟离开旧城，前往北境。",
                "changes": ["林舟不再停留旧城 0.8/F"],
            },
            chapter_id=1,
            facts_override=memory_agent.extract_chapter_facts(
                text="林舟离开旧城，前往北境。"
            ),
        )

        self.assertTrue(replaced.replaced_existing)
        self.assertEqual(replaced.revision, 2)
        self.assertTrue(replaced.retired_memory_ids)
        repository = BookRepository(RagConfig(self.root), "book_replace")
        all_memories = [
            memory
            for store in repository.stores.values()
            for memory in store.list_memories(statuses=None)
        ]
        self.assertFalse(
            any(memory.content == "林舟进入旧城" and memory.status == "active" for memory in all_memories)
        )
        self.assertTrue(
            any("林舟离开旧城" in memory.content and memory.status == "active" for memory in all_memories)
        )

        library = ChapterLibrary(self.root)
        chapter = library.get_chapter("book_replace", 1)
        self.assertEqual(chapter["title"], "第一章 新标题")
        self.assertEqual(chapter["revision"], 2)
        self.assertEqual(len(library.list_versions("book_replace", 1)), 1)

        repeated = system.replace_chapter(
            "book_replace",
            "RV",
            {
                "title": "第一章 新标题",
                "text": "林舟离开旧城，前往北境。",
                "changes": [],
            },
            chapter_id=1,
            facts_override=memory_agent.extract_chapter_facts(
                text="林舟离开旧城，前往北境。"
            ),
        )
        self.assertEqual(repeated.revision, 3)
        repository = BookRepository(RagConfig(self.root), "book_replace")
        self.assertTrue(
            any(
                "林舟离开旧城" in memory.content and memory.status == "active"
                for store in repository.stores.values()
                for memory in store.list_memories(statuses=None)
            )
        )

    def test_replace_chapter_uses_writer_result_without_qwen_extraction(self) -> None:
        memory_agent = ReplacementMemoryAgent()
        system = NovelRagSystem(self.root, memory_agent=memory_agent)

        result = system.replace_chapter(
            "book_writer_facts",
            "CT",
            {
                "chapter_title": "第一章",
                "text": "林舟进入青铜门。",
                "characters": ["林舟"],
                "events": ["林舟进入青铜门 0.8/F"],
                "changes": [],
                "hooks": [],
                "next": [],
            },
            chapter_id=1,
        )

        self.assertEqual(memory_agent.extraction_calls, 0)
        self.assertEqual(result.fact_count, 1)

    def test_local_draft_does_not_replace_archived_chapter(self) -> None:
        self.system.ingest(
            "book_draft",
            "CT",
            {
                "chapter_title": "第一章 正式稿",
                "text": "正式正文",
                "characters": [],
                "events": ["主角抵达城门 0.7/F"],
                "changes": [],
                "hooks": [],
            },
            chapter_id=1,
        )
        library = ChapterLibrary(self.root)
        draft = library.save_draft(
            "book_draft",
            1,
            "RV",
            {
                "title": "第一章 本地修改稿",
                "text": "尚未归档的新正文",
                "changes": [],
            },
        )

        self.assertTrue(draft["is_draft"])
        self.assertEqual(draft["text"], "尚未归档的新正文")
        self.assertEqual(library.latest_chapter_id("book_draft"), 1)
        self.assertTrue(library.chapter_exists("book_draft", 1))
        self.assertTrue(library.list_chapters("book_draft")[0]["is_draft"])

        active_path = self.root / "book_draft" / "documents" / "chapter_000001.json"
        active_payload = json.loads(active_path.read_text(encoding="utf-8"))
        self.assertEqual(active_payload["result"]["text"], "正式正文")

        library.delete_draft("book_draft", 1)
        restored = library.get_chapter("book_draft", 1)
        self.assertFalse(restored["is_draft"])
        self.assertEqual(restored["text"], "正式正文")


if __name__ == "__main__":
    unittest.main()
