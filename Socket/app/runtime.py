from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from dotenv import load_dotenv

ProgressCallback = Callable[[dict[str, Any]], None]


class WebProgress:
    HEARTBEAT_MESSAGES: dict[str, tuple[str, ...]] = {
        "material.extract": (
            "Qwen正在阅读章节内容",
            "正在识别角色、背景、势力、地点和物品",
            "正在整理素材字段并检查固定格式",
            "章节较长时提取会更久，任务仍在正常进行",
        ),
        "memory.audit": (
            "正在扫描所选范围并筛选相关记忆",
            "正在按一条查询记忆和最多九条候选分批比较",
            "正在去除重复比较并整理审计结果",
            "正在校验覆盖范围和可安全执行的精简建议",
        ),
        "memory.audit.apply": (
            "正在创建审计前快照",
            "正在应用已确认的记忆精简方案",
            "正在校验整理后的记忆索引",
        ),
        "memory.extract.writer": (
            "Write_Agent正在阅读修改后的完整章节",
            "正在重新提取事件、状态、关系和伏笔",
            "正在校验记忆重要度和固定格式",
        ),
        "memory.replace": (
            "正在退休本章旧记忆",
            "正在写入最终版本的章节记忆",
            "正在重建人物、事件和状态索引",
        ),
        "chapter.extend": (
            "正在读取当前章节和相关作品记忆",
            "Write_Agent正在补写当前章节",
            "正在检查已有正文是否被完整保留",
        ),
    }

    def __init__(self, callback: ProgressCallback, *, heartbeat_seconds: float = 5.0) -> None:
        self.callback = callback
        self.heartbeat_seconds = max(1.0, heartbeat_seconds)

    def emit(self, step: str, state: str, message: str, elapsed_seconds: float = 0.0) -> None:
        self.callback(
            {
                "step": step,
                "state": state,
                "message": message,
                "elapsed_seconds": round(float(elapsed_seconds), 1),
            }
        )

    @contextmanager
    def step(self, step: str, message: str) -> Iterator[None]:
        started = time.monotonic()
        stopped = threading.Event()
        heartbeat_messages = self.HEARTBEAT_MESSAGES.get(step, ())
        self.emit(step, "started", message)

        def heartbeat() -> None:
            heartbeat_index = 0
            while not stopped.wait(self.heartbeat_seconds):
                heartbeat_message = (
                    heartbeat_messages[heartbeat_index % len(heartbeat_messages)]
                    if heartbeat_messages
                    else message
                )
                self.emit(
                    step,
                    "running",
                    heartbeat_message,
                    time.monotonic() - started,
                )
                heartbeat_index += 1

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            yield
        except Exception:
            stopped.set()
            self.emit(step, "failed", message, time.monotonic() - started)
            raise
        else:
            stopped.set()
            self.emit(step, "completed", message, time.monotonic() - started)


class NovelSocketRuntime:
    """Delivery adapter between the web protocol and the novel backend."""

    def __init__(self, *, backend_root: Path, data_dir: Path) -> None:
        self.backend_root = backend_root.resolve()
        self.data_dir = data_dir.resolve()
        self._agent: Any | None = None
        self._library: Any | None = None
        self._memory_library: Any | None = None
        self._material_library: Any | None = None
        self._material_extractor: Any | None = None
        self._agent_lock = threading.Lock()
        self._operation_lock = threading.Lock()
        self._sessions: dict[str, Any] = {}
        self._session_modes: dict[str, str] = {}
        self._session_lock = threading.Lock()
        self._pending_material_generations: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_env(cls) -> "NovelSocketRuntime":
        socket_root = Path(__file__).resolve().parents[1]
        default_backend = socket_root.parent / "Novel_Agentv2"
        load_dotenv(socket_root / ".env")
        load_dotenv(default_backend / ".env", override=False)
        backend_root = Path(os.getenv("NOVEL_AGENT_ROOT", str(default_backend))).expanduser()
        data_dir = Path(os.getenv("NOVEL_AGENT_DATA_DIR", str(backend_root / "rag_data"))).expanduser()
        return cls(backend_root=backend_root, data_dir=data_dir)

    def health(self) -> dict[str, Any]:
        return {
            "ok": self.backend_root.is_dir(),
            "backend_root": str(self.backend_root),
            "backend_exists": self.backend_root.is_dir(),
            "data_dir": str(self.data_dir),
            "data_dir_exists": self.data_dir.is_dir(),
            "configured_model": bool(os.getenv("LLM_MODEL_ID")),
            "configured_base_url": bool(os.getenv("LLM_BASE_URL")),
        }

    def _ensure_import_path(self) -> None:
        if not self.backend_root.is_dir():
            raise FileNotFoundError(f"Novel_Agentv2 backend not found: {self.backend_root}")
        backend = str(self.backend_root)
        if backend not in sys.path:
            sys.path.insert(0, backend)

    def agent(self) -> Any:
        if self._agent is not None:
            return self._agent
        with self._agent_lock:
            if self._agent is not None:
                return self._agent
            self._ensure_import_path()
            from control_agent import build_default_control_agent
            from progress_display import SilentProgress

            agent = build_default_control_agent(self.data_dir, heartbeat_seconds=5.0)
            agent.progress = SilentProgress()
            self._agent = agent
            return agent

    def library_service(self) -> Any:
        if self._library is None:
            self._ensure_import_path()
            from chapter_library import ChapterLibrary

            self._library = ChapterLibrary(self.data_dir)
        return self._library

    def memory_library_service(self) -> Any:
        if self._memory_library is None:
            self._ensure_import_path()
            from memory_library import MemoryLibraryService

            self._memory_library = MemoryLibraryService(self.data_dir)
        return self._memory_library

    def material_library_service(self) -> Any:
        if self._material_library is None:
            self._ensure_import_path()
            from material_library import MaterialLibrary

            self._material_library = MaterialLibrary(self.data_dir)
        return self._material_library

    def material_extractor_service(self) -> Any:
        if self._material_extractor is None:
            self._ensure_import_path()
            from material_extractor import QwenMaterialExtractor

            model_client = self.agent().rag_system.memory_agent.model_client
            if model_client is None:
                raise RuntimeError("未配置用于正文素材提取的本地Qwen")
            self._material_extractor = QwenMaterialExtractor(model_client)
        return self._material_extractor

    def _remember(self, session: Any) -> None:
        with self._session_lock:
            self._sessions[session.session_id] = session

    def _session(self, session_id: str) -> Any:
        with self._session_lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("当前写作状态已失效，请刷新网页后重试")
        return session

    @staticmethod
    def _session_payload(session: Any) -> dict[str, Any]:
        payload = session.model_dump(mode="json")
        return {
            "session_id": payload["session_id"],
            "book_id": payload["book_id"],
            "task_code": payload["task_code"],
            "chapter_id": payload["chapter_id"],
            "phase": payload["phase"],
            "archived": payload["archived"],
        }

    @staticmethod
    def _draft_payload(result: Any, chapter_id: int) -> dict[str, Any]:
        return {
            "chapter_id": chapter_id,
            "title": result.display_title,
            "text": result.display_text,
        }

    def library(self) -> dict[str, Any]:
        return {"books": self.library_service().list_books()}

    def create_book(self, *, name: str) -> dict[str, Any]:
        book = self.library_service().create_book(name)
        return {
            "kind": "book_created",
            "book": book,
            "books": self.library_service().list_books(),
        }

    def rename_book(self, *, book_id: str, name: str) -> dict[str, Any]:
        book = self.library_service().rename_book(book_id.strip(), name)
        return {
            "kind": "book_renamed",
            "book": book,
            "books": self.library_service().list_books(),
        }

    def create_chapter(self, *, book_id: str, title: str = "") -> dict[str, Any]:
        chapter = self.library_service().create_chapter(book_id.strip(), title)
        chapter.pop("result", None)
        return {
            "kind": "chapter_created",
            "chapter": chapter,
            "books": self.library_service().list_books(),
        }

    def load_chapter(self, *, book_id: str, chapter_id: int) -> dict[str, Any]:
        chapter = self.library_service().get_chapter(book_id.strip(), int(chapter_id))
        chapter.pop("result", None)
        chapter["versions"] = self.library_service().list_versions(book_id, chapter_id)
        return {"chapter": chapter}

    def memories(self, *, book_id: str) -> dict[str, Any]:
        return self.memory_library_service().list(book_id.strip())

    def create_memory(self, *, book_id: str, value: dict[str, Any]) -> dict[str, Any]:
        memory = self.memory_library_service().create(book_id.strip(), value)
        return {"kind": "memory_saved", "memory": memory}

    def update_memory(
        self,
        *,
        book_id: str,
        memory_id: str,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        memory = self.memory_library_service().update(book_id.strip(), memory_id, value)
        return {"kind": "memory_saved", "memory": memory}

    def delete_memory(self, *, book_id: str, memory_id: str) -> dict[str, Any]:
        return {
            "kind": "memory_deleted",
            **self.memory_library_service().delete(book_id.strip(), memory_id),
        }

    def memory_history(self, *, book_id: str, memory_id: str) -> dict[str, Any]:
        return {
            "memory_id": memory_id,
            "history": self.memory_library_service().history(book_id.strip(), memory_id),
        }

    def material_schemas(self) -> dict[str, Any]:
        return {"schemas": self.material_library_service().schemas()}

    def materials(self, *, book_id: str) -> dict[str, Any]:
        result = self.material_library_service().list(book_id.strip())
        result["reviews"] = self.material_library_service().list_reviews(book_id.strip())
        return result

    def save_material(
        self,
        *,
        book_id: str,
        value: dict[str, Any],
        material_id: str = "",
    ) -> dict[str, Any]:
        material = self.material_library_service().save(
            book_id.strip(),
            value,
            material_id=material_id or None,
        )
        return {"kind": "material_saved", "material": material}

    def delete_material(self, *, book_id: str, material_id: str) -> dict[str, Any]:
        return {
            "kind": "material_deleted",
            **self.material_library_service().delete(book_id.strip(), material_id),
        }

    def sync_material(
        self,
        *,
        book_id: str,
        material_id: str,
        overwrite_user: bool = False,
    ) -> dict[str, Any]:
        clean_book = book_id.strip()
        material = self.material_library_service().get(clean_book, material_id)
        sync = material.get("sync") if isinstance(material.get("sync"), dict) else {}
        result = self.memory_library_service().sync_from_material(
            clean_book,
            material,
            linked_memory_id=str(sync.get("memory_id") or ""),
            overwrite_user=overwrite_user,
        )
        if result["action"] == "conflict":
            return {
                "kind": "material_sync_conflict",
                "material": material,
                "memory": result["memory"],
            }
        material = self.material_library_service().mark_synced(
            clean_book,
            material_id,
            result["memory"]["memory_id"],
        )
        return {
            "kind": "material_synced",
            "action": result["action"],
            "material": material,
            "memory": result["memory"],
        }

    def confirm_material_review(
        self,
        *,
        book_id: str,
        review_id: str,
        decisions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        result = self.material_library_service().confirm_review(
            book_id.strip(),
            review_id,
            decisions=decisions,
        )
        return {"kind": "material_review_confirmed", **result}

    def extract_materials(
        self,
        *,
        book_id: str,
        chapter_id: int,
        title: str = "",
        text: str = "",
        save_draft: bool = False,
        session_id: str = "",
    ) -> dict[str, Any]:
        clean_book = book_id.strip()
        clean_chapter = int(chapter_id)
        draft_saved = False
        if clean_chapter < 1:
            raise ValueError("请先选择需要提取素材的章节")
        if save_draft:
            self.save_local(
                book_id=clean_book,
                chapter_id=clean_chapter,
                title=title,
                text=text,
                session_id=session_id,
            )
            draft_saved = True
        if title.strip() and text.strip():
            chapter_title = title.strip()
            chapter_text = text.strip()
        else:
            chapter = self.library_service().get_chapter(clean_book, clean_chapter)
            chapter_title = str(chapter.get("title") or "")
            chapter_text = str(chapter.get("text") or "")
        with self.agent().progress.step(
            "material.extract",
            "正在使用Qwen分析章节并提取素材",
        ):
            candidates = self.material_extractor_service().extract(
                book_id=clean_book,
                chapter_id=clean_chapter,
                title=chapter_title,
                text=chapter_text,
            )
        review = None
        if candidates:
            review = self.material_library_service().save_review(
                clean_book,
                chapter_id=clean_chapter,
                candidates=candidates,
            )
        return {
            "kind": "material_extracted",
            "candidate_count": len(candidates),
            "review": review,
            "draft_saved": draft_saved,
            "books": self.library_service().list_books() if draft_saved else None,
        }

    def consult(
        self,
        *,
        book_id: str,
        question: str,
        current_chapter: int = 0,
        selected_text: str = "",
        force_rag: bool = False,
    ) -> dict[str, Any]:
        result = self.agent().chat_story_assistant(
            question,
            book_id=book_id.strip(),
            current_chapter=int(current_chapter or 0),
            selected_text=selected_text,
            force_rag=force_rag,
        )
        return {"kind": "consult_answer", **result}

    def generate_material(
        self,
        *,
        book_id: str,
        task_code: str,
        text: str,
        refine: bool = False,
    ) -> dict[str, Any]:
        code = task_code.strip().upper()
        if code not in {"BD", "CH"}:
            raise ValueError("素材创作只支持世界观或人物")
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("请输入素材创作要求")
        agent = self.agent()
        session = agent.create_session(book_id.strip(), code, chapter_id=0)
        if refine:
            setting = agent.refine_setting(session, clean_text)
        else:
            setting = agent.keep_original_setting(session, clean_text)
        result = agent.generate_draft(session)
        candidates = self.material_library_service().preview_writer_result(
            code,
            result.result,
        )
        generation_id = f"generation_{uuid.uuid4().hex[:16]}"
        self._pending_material_generations[generation_id] = {
            "book_id": book_id.strip(),
            "task_code": code,
            "setting": setting,
            "writer_result": result.result,
            "candidates": candidates,
        }
        return {
            "kind": "material_generation_preview",
            "generation_id": generation_id,
            "task_code": code,
            "setting": setting,
            "candidates": candidates,
        }

    def confirm_material_generation(
        self,
        *,
        generation_id: str,
        decisions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        pending = self._pending_material_generations.get(generation_id)
        if pending is None:
            raise KeyError("素材生成结果已失效，请重新生成")
        if decisions is None:
            decisions = [
                {"action": "create", "value": candidate}
                for candidate in pending["candidates"]
            ]
        saved = self.material_library_service().apply_decisions(
            pending["book_id"],
            decisions,
            origin="write_agent",
        )
        synced: list[dict[str, Any]] = []
        warnings: list[str] = []
        for material in saved:
            try:
                result = self.sync_material(
                    book_id=pending["book_id"],
                    material_id=material["material_id"],
                )
                synced.append(result)
                if result["kind"] == "material_sync_conflict":
                    warnings.append(f"{material['name']}的对应记忆已被人工修改")
            except Exception as exc:  # Material files are already safe locally.
                warnings.append(f"{material['name']}同步记忆失败：{exc}")
        self._pending_material_generations.pop(generation_id, None)
        return {
            "kind": "material_generation_saved",
            "saved": saved,
            "memory": synced,
            "warning": "；".join(warnings),
        }

    def create_session(
        self,
        *,
        book_id: str,
        task_code: str,
        chapter_id: int = 0,
    ) -> dict[str, Any]:
        clean_book = book_id.strip()
        code = task_code.strip().upper()
        selected_chapter = int(chapter_id or 0)
        library = self.library_service()
        latest = library.latest_chapter_id(clean_book)
        generation_allowed = True
        message = "写作环境已就绪"

        if code == "PW":
            target_chapter = selected_chapter
            if target_chapter < 1 or not library.chapter_available(clean_book, target_chapter):
                generation_allowed = False
                message = "请先从小说目录选择需要补写的章节"
            else:
                chapter = library.get_chapter(clean_book, target_chapter)
                if not str(chapter.get("text") or "").strip():
                    generation_allowed = False
                    message = "请先写入部分正文，再使用章节补写"
                else:
                    message = f"将在第{target_chapter}章现有正文末尾继续补写"
        elif code == "NW":
            target_chapter = 1
            if library.chapter_available(clean_book, target_chapter):
                generation_allowed = False
                message = "第一章已经存在或已保存草稿，请选择修改或归档"
        elif code == "CT":
            source_chapter = selected_chapter or latest
            if source_chapter < 1:
                generation_allowed = False
                target_chapter = 1
                message = "当前作品还没有章节，请先使用新文生成第一章"
            elif not library.chapter_exists(clean_book, source_chapter):
                generation_allowed = False
                target_chapter = source_chapter + 1
                message = f"第{source_chapter}章仍是本地草稿，请先归档后再续写"
            else:
                target_chapter = source_chapter + 1
                message = f"将根据第{source_chapter}章续写第{target_chapter}章"
        elif code == "RV":
            target_chapter = selected_chapter
            if target_chapter < 1 or not library.chapter_available(clean_book, target_chapter):
                generation_allowed = False
                message = "请先从小说目录选择需要修改的章节"
        else:
            target_chapter = 0

        agent = self.agent()
        agent_code = code
        if code == "PW":
            agent_code = "RV"
            if generation_allowed:
                chapter = library.get_chapter(clean_book, target_chapter)
                candidate_code = str(chapter.get("task_code") or "RV").upper()
                agent_code = (
                    candidate_code
                    if candidate_code in {"NW", "CT", "RV"}
                    else "RV"
                )
        session = agent.create_session(clean_book, agent_code, chapter_id=target_chapter)
        self._session_modes[session.session_id] = code
        if code in {"RV", "PW"} and generation_allowed:
            chapter = library.get_chapter(clean_book, target_chapter)
            session.draft_result = chapter["result"]
            session.phase = "draft_review"
            agent.sessions.save(session)
        self._remember(session)
        return {
            "session": self._session_payload(session),
            "generation_allowed": generation_allowed,
            "target_chapter_id": target_chapter,
            "message": message,
            "task_mode": code,
        }

    def submit_setting(
        self,
        *,
        session_id: str,
        text: str,
        refine: bool = False,
        chapter_title: str = "",
        chapter_text: str = "",
    ) -> dict[str, Any]:
        agent = self.agent()
        session = self._session(session_id)
        setting = text.strip() or "自然承接上一章继续写作"
        task_mode = self._session_modes.get(session_id, session.task_code)
        if task_mode == "PW":
            if chapter_title.strip() and chapter_text.strip():
                current = dict(session.draft_result or {})
                if "chapter_title" in current:
                    current["chapter_title"] = chapter_title.strip()
                else:
                    current["title"] = chapter_title.strip()
                current["text"] = chapter_text
                session.draft_result = current
            with agent.progress.step("chapter.extend", "正在补写当前章节"):
                result = agent.extend_draft(session, setting)
            refined = setting
        elif session.task_code == "RV":
            from control_schemas import ControlIntent

            intent = ControlIntent(
                intent="revise_draft",
                confidence=1.0,
                feedback=setting,
                needs_rag=True,
            )
            result = agent.revise_draft(session, setting, intent)
            refined = setting
        else:
            from control_agent import supports_setting_refinement

            if refine and supports_setting_refinement(session.task_code):
                refined = agent.refine_setting(session, setting)
            else:
                refined = agent.keep_original_setting(session, setting)
            result = agent.generate_draft(session)
        self._remember(session)
        return {
            "kind": "draft",
            "session": self._session_payload(session),
            "setting": refined,
            "draft": self._draft_payload(result, session.chapter_id),
            "task_mode": task_mode,
        }

    def audit_memories(
        self,
        *,
        book_id: str,
        scope_mode: str = "book",
        chapter_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        clean_book = book_id.strip()
        if not clean_book:
            raise ValueError("请先选择需要检查的小说")
        scope = {
            "mode": str(scope_mode or "book").strip(),
            "chapter_ids": list(chapter_ids or []),
        }
        agent = self.agent()
        session = agent.create_session(clean_book, "NW", chapter_id=0)
        agent.prepare_memory_audit(session)
        audit = agent.run_memory_audit_dry(session, scope=scope)
        self._remember(session)
        artifact_dir = Path(str(audit.get("artifact_dir") or ""))
        plan_path = artifact_dir / "patch_plan.json"
        operations: list[dict[str, Any]] = []
        if plan_path.is_file():
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            operations = list(plan.get("operations") or [])
        findings_by_id: dict[str, dict[str, Any]] = {}
        for name in (
            "comparison_reports.json",
            "packet_reports.json",
            "cross_reports.json",
            "reconciliation.json",
        ):
            path = artifact_dir / name
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            records = payload if isinstance(payload, list) else [payload]
            for record in records:
                for finding in record.get("findings") or []:
                    finding_id = str(finding.get("finding_id") or "")
                    key = finding_id or json.dumps(finding, ensure_ascii=False, sort_keys=True)
                    findings_by_id[key] = finding
        can_apply = bool(
            (audit.get("coverage") or {}).get("complete", False)
            and audit.get("semantic_candidate_complete", True)
            and not audit.get("blocking_issue_ids")
            and not audit.get("validation_errors")
        )
        return {
            "kind": "memory_audit_preview",
            "session_id": session.session_id,
            "audit": audit,
            "findings": list(findings_by_id.values()),
            "operations": operations,
            "can_apply": can_apply,
        }

    def apply_memory_audit(
        self,
        *,
        book_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        session = self._session(session_id)
        if session.book_id != book_id.strip():
            raise ValueError("审计计划与当前小说不匹配")
        result = self.agent().apply_memory_audit(session)
        return {"kind": "memory_audit_applied", "audit": result}

    def rollback_memory_audit(
        self,
        *,
        book_id: str,
        snapshot_id: str,
    ) -> dict[str, Any]:
        self.agent().rag_system.rollback_memory_audit(
            book_id.strip(),
            snapshot_id.strip(),
        )
        return {
            "kind": "memory_audit_rolled_back",
            "snapshot_id": snapshot_id.strip(),
        }

    def _editor_result(
        self,
        *,
        book_id: str,
        chapter_id: int,
        title: str,
        text: str,
        session_id: str = "",
    ) -> tuple[Any | None, str, dict[str, Any], bool]:
        clean_title = title.strip()
        clean_text = text.strip()
        if not clean_title or not clean_text:
            raise ValueError("章节标题和正文不能为空")
        chapter_id = int(chapter_id)
        session = self._sessions.get(session_id) if session_id else None
        if session is not None and session.book_id == book_id and session.chapter_id == chapter_id and session.draft_result:
            writer_result = dict(session.draft_result)
            task_code = session.task_code
        else:
            existing = self.library_service().get_chapter(book_id, chapter_id)
            writer_result = dict(existing["result"])
            task_code = str(existing["task_code"] or "RV")

        source_title = str(
            writer_result.get("chapter_title")
            or writer_result.get("title")
            or ""
        ).strip()
        source_text = str(writer_result.get("text") or "").strip()
        memory_facts_stale = bool(writer_result.get("_memory_facts_stale")) or (
            clean_title != source_title or clean_text != source_text
        )
        if memory_facts_stale:
            writer_result["_memory_facts_stale"] = True
            writer_result.pop("_memory_facts", None)
        if task_code == "RV" or "title" in writer_result:
            writer_result["title"] = clean_title
        else:
            writer_result["chapter_title"] = clean_title
        writer_result["text"] = clean_text
        return session, task_code, writer_result, memory_facts_stale

    def save_local(
        self,
        *,
        book_id: str,
        chapter_id: int,
        title: str,
        text: str,
        session_id: str = "",
    ) -> dict[str, Any]:
        session, task_code, writer_result, _memory_facts_stale = self._editor_result(
            book_id=book_id,
            chapter_id=chapter_id,
            title=title,
            text=text,
            session_id=session_id,
        )
        chapter = self.library_service().save_draft(
            book_id,
            int(chapter_id),
            task_code,
            writer_result,
        )
        if session is not None:
            session.draft_result = writer_result
            session.archived = False
            session.phase = "draft_review"
            self.agent().sessions.save(session)
            self._remember(session)
        chapter.pop("result", None)
        return {
            "kind": "draft_saved",
            "session": self._session_payload(session) if session is not None else None,
            "chapter": chapter,
            "books": self.library_service().list_books(),
        }

    def archive_editor(
        self,
        *,
        book_id: str,
        chapter_id: int,
        title: str,
        text: str,
        session_id: str = "",
    ) -> dict[str, Any]:
        session, task_code, writer_result, memory_facts_stale = self._editor_result(
            book_id=book_id,
            chapter_id=chapter_id,
            title=title,
            text=text,
            session_id=session_id,
        )
        self.library_service().save_draft(
            book_id,
            int(chapter_id),
            task_code,
            writer_result,
        )
        agent = self.agent()
        facts_override = None
        if memory_facts_stale:
            with agent.progress.step(
                "memory.extract.writer",
                "Write_Agent正在重新提取修改后的章节记忆",
            ):
                facts_override = agent.extract_chapter_memories(
                    book_id=book_id,
                    chapter_id=int(chapter_id),
                    title=title.strip(),
                    text=text.strip(),
                )
            writer_result["_memory_facts"] = [
                fact.model_dump(mode="json") for fact in facts_override
            ]
            writer_result.pop("_memory_facts_stale", None)
        with agent.progress.step("memory.replace", "正在归档章节并重建本章记忆"):
            replaced = agent.rag_system.replace_chapter(
                book_id,
                task_code,
                writer_result,
                chapter_id=chapter_id,
                facts_override=facts_override,
            )
        self.library_service().delete_draft(book_id, int(chapter_id))
        if session is not None:
            session.draft_result = writer_result
            session.archived = True
            session.phase = "post_archive"
            session.archive_result = replaced.model_dump(mode="json")
            agent.sessions.save(session)
            self._remember(session)
        chapter = self.library_service().get_chapter(book_id, chapter_id)
        chapter.pop("result", None)
        return {
            "kind": "archived",
            "session": self._session_payload(session) if session is not None else None,
            "chapter": chapter,
            "archive": replaced.model_dump(mode="json"),
            "books": self.library_service().list_books(),
        }

    def handle_message(self, *, session_id: str, text: str) -> dict[str, Any]:
        agent = self.agent()
        session = self._session(session_id)
        user_text = text.strip()
        if not user_text:
            return {"kind": "clarification", "message": "请输入问题或操作要求。", "session": self._session_payload(session)}
        intent = agent.interpret_user_message(session, user_text)
        if intent.needs_clarification:
            return {"kind": "clarification", "message": intent.clarification, "session": self._session_payload(session)}
        if intent.intent == "request_memory_audit":
            message = agent.prepare_memory_audit(session)
            return {"kind": "audit_confirmation", "session": self._session_payload(session), "message": message}
        if intent.intent == "confirm_audit":
            result = agent.run_memory_audit_dry(session)
            return {"kind": "audit_dry_run", "session": self._session_payload(session), "audit": result}
        if intent.intent == "confirm_audit_apply":
            result = agent.apply_memory_audit(session)
            return {"kind": "audit_applied", "session": self._session_payload(session), "audit": result}
        if intent.intent == "general_question":
            answer = agent.answer_question(session, user_text)
            return {"kind": "answer", "session": self._session_payload(session), "answer": answer}
        return {"kind": "clarification", "message": "请在左侧选择写作任务，或输入记忆整理和小说问题。", "session": self._session_payload(session)}

    def _run_sync(self, action: str, progress_callback: ProgressCallback | None, kwargs: dict[str, Any]) -> dict[str, Any]:
        with self._operation_lock:
            method = getattr(self, action)
            if progress_callback is None or action in {
                "library", "create_book", "rename_book", "create_chapter",
                "load_chapter", "save_local", "memories", "create_memory",
                "update_memory", "delete_memory", "memory_history", "material_schemas",
                "materials", "save_material", "delete_material", "sync_material",
                "confirm_material_review",
            }:
                return method(**kwargs)
            agent = self.agent()
            previous_progress = agent.progress
            agent.progress = WebProgress(progress_callback)
            try:
                return method(**kwargs)
            finally:
                agent.progress = previous_progress

    async def run(
        self,
        action: str,
        *,
        progress_callback: ProgressCallback | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self._run_sync, action, progress_callback, kwargs)
