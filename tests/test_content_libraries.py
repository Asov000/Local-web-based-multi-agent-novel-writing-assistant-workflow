from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from control_agent import ControlAgent, requires_story_context
from chapter_library import ChapterLibrary
from material_extractor import QwenMaterialExtractor
from material_library import MaterialLibrary
from memory_library import MemoryLibraryService
from progress_display import SilentProgress
from rag.config import RagConfig
from rag.repository import BookRepository
from rag.system import NovelRagSystem


class DummyWriter:
    agent_name = "write_agent"

    def handle_message(self, message):
        raise AssertionError("只读咨询不应调用Write_Agent")


class ConsultModel:
    def invoke_json(self, system_prompt: str, payload: dict) -> dict:
        if "小说只读咨询" not in system_prompt:
            raise AssertionError("未预期的模型调用")
        memories = payload["memories"]
        return {
            "answer": "青铜门只能由王族血脉开启。",
            "references": [
                {
                    "memory_id": memories[0]["memory_id"],
                    "source_chapter": memories[0]["source_chapter"],
                    "reason": "核心世界规则",
                },
                {
                    "memory_id": "memory_hallucinated",
                    "source_chapter": 99,
                    "reason": "不存在的来源",
                },
            ],
            "insufficient_context": False,
        }


class GeneralChatModel:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def invoke_json(self, system_prompt: str, payload: dict) -> dict:
        if "智能写作日常对话" not in system_prompt:
            raise AssertionError("普通对话不应使用RAG咨询提示")
        if "memories" in payload:
            raise AssertionError("普通对话不应接收RAG记忆")
        self.payloads.append(payload)
        return {"answer": "你好，我们可以直接聊写作。"}


class ExtractorClient:
    def __init__(self) -> None:
        self.calls = 0

    def invoke_json(self, system_prompt: str, payload: dict) -> dict:
        self.calls += 1
        name = "" if self.calls == 1 else "月族钥匙"
        return {
            "schema_version": "rag.message.v1",
            "sender": "qwen_model",
            "receiver": "material_extractor",
            "message_type": "response",
            "action": "rag.model.material.extract.result",
            "status": "ok",
            "task_id": payload["task_id"],
            "parent_message_id": payload["message_id"],
            "book_id": payload["book_id"],
            "payload": {
                "materials": [
                    {
                        "category": "item",
                        "name": name,
                        "fields": {
                            "item_type": "钥匙",
                            "description": "银色钥匙",
                            "function": "开启青铜门",
                            "holder": "陈玥",
                        },
                        "evidence": "陈玥用月族钥匙打开青铜门",
                        "confidence": 0.95,
                    }
                ]
            },
            "operations": [],
        }


class ContentLibraryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_user_memory_crud_updates_index_and_history(self) -> None:
        service = MemoryLibraryService(self.root)
        created = service.create(
            "book_memory",
            {
                "store_type": "canon_memory",
                "memory_type": "character_identity",
                "content": "林舟是流亡王族继承人",
                "raw_importance": 0.95,
                "character_names": ["林舟"],
                "entity_name": "林舟",
                "field": "identity",
            },
        )
        self.assertTrue(created["user_managed"])
        repository = BookRepository(RagConfig(self.root), "book_memory")
        self.assertEqual(len(repository.index.links_for_memory(created["memory_id"])), 1)

        updated = service.update(
            "book_memory",
            created["memory_id"],
            {
                "store_type": "canon_memory",
                "memory_type": "character_identity",
                "content": "林舟是月族王室唯一继承人",
                "raw_importance": 1.0,
                "character_names": ["林舟"],
                "entity_name": "林舟",
                "field": "identity",
                "note": "用户确认",
            },
        )
        self.assertEqual(updated["version"], 2)
        self.assertEqual(updated["note"], "用户确认")
        self.assertEqual(len(service.history("book_memory", created["memory_id"])), 1)

        service.delete("book_memory", created["memory_id"])
        listed = service.list("book_memory")
        self.assertFalse(any(group["memories"] for group in listed["groups"]))
        self.assertEqual(repository.index.links_for_memory(created["memory_id"]), [])

    def test_material_library_enforces_schema_and_confirms_reviews(self) -> None:
        library = MaterialLibrary(self.root)
        material = library.save(
            "book_material",
            {
                "category": "item",
                "name": "青铜钥匙",
                "fields": {
                    "description": "表面布满铜锈",
                    "function": "开启外门",
                    "holder": "林舟",
                    "unexpected": "不应保存",
                },
            },
        )
        self.assertNotIn("unexpected", material["fields"])
        self.assertEqual(material["fields"]["holder"], "林舟")

        review = library.save_review(
            "book_material",
            chapter_id=2,
            candidates=[
                {
                    "category": "location",
                    "name": "青铜门",
                    "fields": {"description": "遗迹入口"},
                }
            ],
        )
        confirmed = library.confirm_review("book_material", review["review_id"])
        self.assertEqual(len(confirmed["saved"]), 1)
        self.assertEqual(library.list_reviews("book_material"), [])
        self.assertEqual(len(library.list("book_material")["materials"]), 2)

    def test_material_review_can_merge_edited_candidate_or_skip_it(self) -> None:
        library = MaterialLibrary(self.root)
        existing = library.save(
            "book_review",
            {
                "category": "item",
                "name": "青铜钥匙",
                "fields": {
                    "description": "表面布满铜锈",
                    "function": "开启外门",
                    "holder": "林舟",
                },
            },
        )
        review = library.save_review(
            "book_review",
            chapter_id=3,
            candidates=[
                {
                    "category": "item",
                    "name": "青铜钥匙",
                    "fields": {"function": "开启遗迹内门", "holder": ""},
                },
                {
                    "category": "location",
                    "name": "遗迹内门",
                    "fields": {"description": "青铜钥匙开启的门"},
                },
            ],
        )

        confirmed = library.confirm_review(
            "book_review",
            review["review_id"],
            decisions=[
                {
                    "action": "merge",
                    "material_id": existing["material_id"],
                    "value": {
                        "category": "item",
                        "name": "青铜钥匙",
                        "fields": {"function": "开启遗迹内门", "holder": ""},
                    },
                },
                {"action": "skip"},
            ],
        )

        self.assertEqual(len(confirmed["saved"]), 1)
        merged = library.get("book_review", existing["material_id"])
        self.assertEqual(merged["fields"]["description"], "表面布满铜锈")
        self.assertEqual(merged["fields"]["function"], "开启遗迹内门")
        self.assertEqual(merged["fields"]["holder"], "林舟")
        self.assertEqual(len(library.list("book_review")["materials"]), 1)

    def test_material_sync_updates_one_linked_memory(self) -> None:
        materials = MaterialLibrary(self.root)
        memories = MemoryLibraryService(self.root)
        material = materials.save(
            "book_sync",
            {
                "category": "item",
                "name": "青铜钥匙",
                "fields": {"function": "开启外门", "holder": "林舟"},
            },
        )

        first = memories.sync_from_material("book_sync", material)
        self.assertEqual(first["action"], "created")
        linked = materials.mark_synced(
            "book_sync",
            material["material_id"],
            first["memory"]["memory_id"],
        )
        again = memories.sync_from_material(
            "book_sync",
            linked,
            linked_memory_id=first["memory"]["memory_id"],
        )
        self.assertEqual(again["action"], "unchanged")

        changed = materials.save(
            "book_sync",
            {
                "category": "item",
                "name": "青铜钥匙",
                "fields": {"function": "开启遗迹内门", "holder": "林舟"},
            },
            material_id=material["material_id"],
        )
        updated = memories.sync_from_material(
            "book_sync",
            changed,
            linked_memory_id=first["memory"]["memory_id"],
        )
        self.assertEqual(updated["action"], "updated")
        self.assertEqual(updated["memory"]["memory_id"], first["memory"]["memory_id"])
        listed = memories.list("book_sync")
        self.assertEqual(sum(len(group["memories"]) for group in listed["groups"]), 1)

    def test_material_sync_requires_confirmation_after_user_memory_edit(self) -> None:
        materials = MaterialLibrary(self.root)
        memories = MemoryLibraryService(self.root)
        material = materials.save(
            "book_sync_conflict",
            {
                "category": "item",
                "name": "青铜钥匙",
                "fields": {"function": "开启外门", "holder": "林舟"},
            },
        )
        first = memories.sync_from_material("book_sync_conflict", material)
        linked = materials.mark_synced(
            "book_sync_conflict",
            material["material_id"],
            first["memory"]["memory_id"],
        )
        memory = first["memory"]
        memories.update(
            "book_sync_conflict",
            memory["memory_id"],
            {
                "store_type": memory["store_type"],
                "memory_type": memory["memory_type"],
                "content": "青铜钥匙已经被人工改为只能开启内门",
                "raw_importance": memory["raw_importance"],
                "source_chapter": memory["source_chapter"],
                "status": memory["status"],
                "hook_status": memory["hook_status"],
                "entity_name": memory["entity_name"],
                "field": memory["field"],
                "character_names": memory["character_names"],
                "item_names": memory["item_names"],
                "event_names": memory["event_names"],
                "note": memory["note"],
                "is_current": memory["is_current"],
            },
        )

        conflict = memories.sync_from_material(
            "book_sync_conflict",
            linked,
            linked_memory_id=memory["memory_id"],
        )
        self.assertEqual(conflict["action"], "conflict")
        overwritten = memories.sync_from_material(
            "book_sync_conflict",
            linked,
            linked_memory_id=memory["memory_id"],
            overwrite_user=True,
        )
        self.assertEqual(overwritten["action"], "updated")
        self.assertEqual(overwritten["memory"]["memory_id"], memory["memory_id"])

    def test_manual_book_and_blank_chapter_creation(self) -> None:
        library = ChapterLibrary(self.root)
        book = library.create_book("雾城来信")
        self.assertEqual(book["name"], "雾城来信")
        self.assertEqual(book["chapters"], [])

        chapter = library.create_chapter(book["book_id"], "雨夜来客")
        self.assertEqual(chapter["chapter_id"], 1)
        self.assertEqual(chapter["title"], "雨夜来客")
        self.assertEqual(chapter["text"], "")
        self.assertTrue(chapter["is_draft"])
        listed = library.list_books()
        self.assertEqual(listed[0]["name"], "雾城来信")

    def test_writer_material_preview_keeps_bd_and_ch_separate(self) -> None:
        library = MaterialLibrary(self.root)
        world = library.preview_writer_result(
            "BD",
            {
                "world_name": "永夜大陆",
                "background": "大陆被永夜笼罩 0.9/T",
                "rules": ["亡者不能复活 0.98/T"],
                "factions": ["守夜人驻守北境 0.7/F"],
                "locations": ["白塔位于旧都 0.6/F"],
                "conflict": "人类试图终结永夜 0.85/F",
            },
        )
        self.assertEqual([item["category"] for item in world], ["background", "faction", "location"])
        characters = library.preview_writer_result(
            "CH",
            {
                "characters": [
                    {
                        "name": "林舟",
                        "role": "流亡王族 0.95/T",
                        "appearance": "黑发 0.4/T",
                        "personality": "谨慎 0.8/T",
                        "background": "来自北境 0.75/T",
                        "goal": "寻找妹妹 0.7/F",
                        "ability": "感知遗迹 0.85/T",
                        "relations": [],
                    }
                ]
            },
        )
        self.assertEqual(len(characters), 1)
        self.assertEqual(characters[0]["category"], "character")
        self.assertNotIn("0.95/T", characters[0]["fields"]["identity"])

    def test_qwen_material_extractor_retries_without_writing(self) -> None:
        client = ExtractorClient()
        extractor = QwenMaterialExtractor(client)
        candidates = extractor.extract(
            book_id="book_extract",
            chapter_id=1,
            title="青铜门",
            text="陈玥用月族钥匙打开青铜门。",
        )
        self.assertEqual(client.calls, 2)
        self.assertEqual(candidates[0]["name"], "月族钥匙")
        self.assertFalse((self.root / "book_extract").exists())

    def test_control_agent_story_consult_is_rag_read_only(self) -> None:
        rag = NovelRagSystem(self.root)
        rag.ingest(
            "book_consult",
            "BD",
            {
                "world_name": "遗迹大陆",
                "background": "大陆遍布古代遗迹 0.85/T",
                "rules": ["王族血脉可以开启青铜门 0.95/T"],
                "factions": [],
                "locations": [],
                "conflict": "各方争夺遗迹 0.75/F",
            },
            chapter_id=0,
        )
        agent = ControlAgent(
            writer_agent=DummyWriter(),
            rag_system=rag,
            control_model=ConsultModel(),
            progress=SilentProgress(),
        )
        result = agent.consult_story(
            "book_consult",
            "谁能打开青铜门？",
            current_chapter=1,
        )
        self.assertIn("王族血脉", result["answer"])
        self.assertEqual(len(result["references"]), 1)
        self.assertGreaterEqual(result["retrieved_count"], 1)
        self.assertTrue(result["used_rag"])

    def test_control_agent_general_chat_does_not_query_rag(self) -> None:
        model = GeneralChatModel()
        agent = ControlAgent(
            writer_agent=DummyWriter(),
            rag_system=NovelRagSystem(self.root),
            control_model=model,
            progress=SilentProgress(),
        )

        def fail_retrieval(*args, **kwargs):
            raise AssertionError("普通对话不应查询RAG")

        agent.context_builder.build = fail_retrieval
        result = agent.chat_story_assistant("你好，怎么写一场雨？")
        self.assertFalse(result["used_rag"])
        self.assertEqual(result["references"], [])
        self.assertEqual(len(model.payloads), 1)
        self.assertFalse(requires_story_context("帮我润色这段话"))
        self.assertTrue(requires_story_context("检查这一章和前文是否冲突"))

    def test_chapter_replacement_preserves_user_managed_memory(self) -> None:
        rag = NovelRagSystem(self.root)
        rag.ingest(
            "book_preserve",
            "CT",
            {
                "chapter_title": "Chapter One",
                "text": "Lin enters the old city.",
                "characters": ["Lin"],
                "events": ["Lin enters the old city 0.8/F"],
                "changes": [],
                "hooks": [],
            },
            chapter_id=1,
        )
        library = MemoryLibraryService(self.root)
        editable = next(
            memory
            for group in library.list("book_preserve")["groups"]
            for memory in group["memories"]
            if memory["source_chapter"] == 1
        )
        editable.update({"content": "User confirmed this memory", "note": "manual"})
        updated = library.update("book_preserve", editable["memory_id"], editable)

        replaced = rag.replace_chapter(
            "book_preserve",
            "RV",
            {
                "title": "Chapter One Revised",
                "text": "Lin leaves the old city.",
                "changes": ["Lin leaves the old city 0.8/F"],
            },
            chapter_id=1,
        )

        preserved = library.get("book_preserve", updated["memory_id"])
        self.assertEqual(preserved["status"], "active")
        self.assertEqual(preserved["content"], "User confirmed this memory")
        self.assertNotIn(updated["memory_id"], replaced.retired_memory_ids)


if __name__ == "__main__":
    unittest.main()
