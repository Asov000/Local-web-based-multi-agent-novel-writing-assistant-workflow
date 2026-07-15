from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag.config import RagConfig


CHAPTER_FILE = re.compile(r"chapter_(?P<chapter>\d{6})\.json$")
BOOK_METADATA_FILE = "book.json"


class ChapterLibrary:
    """Chapter catalog and local draft store used by delivery layers."""

    def __init__(self, root_dir: str | Path = "rag_data") -> None:
        self.config = RagConfig(root_dir=root_dir)

    def list_books(self) -> list[dict[str, Any]]:
        if not self.config.root_dir.is_dir():
            return []
        books: list[dict[str, Any]] = []
        for book_dir in sorted(self.config.root_dir.iterdir(), key=lambda item: item.name):
            if not book_dir.is_dir() or book_dir.name.startswith("."):
                continue
            chapters = self.list_chapters(book_dir.name)
            if not chapters and not (book_dir / "documents").is_dir():
                continue
            books.append(
                {
                    "book_id": book_dir.name,
                    "name": self._book_name(book_dir.name, chapters),
                    "chapter_count": len(chapters),
                    "chapters": chapters,
                }
            )
        return books

    def create_book(self, name: str) -> dict[str, Any]:
        clean_name = str(name or "").strip()
        if not clean_name:
            raise ValueError("小说名称不能为空")
        book_id = f"book_{uuid.uuid4().hex[:12]}"
        book_dir = self.config.book_dir(book_id)
        (book_dir / "documents").mkdir(parents=True, exist_ok=False)
        self._write_metadata(book_id, clean_name)
        return {
            "book_id": book_id,
            "name": clean_name,
            "chapter_count": 0,
            "chapters": [],
        }

    def rename_book(self, book_id: str, name: str) -> dict[str, Any]:
        clean_name = str(name or "").strip()
        if not clean_name:
            raise ValueError("小说名称不能为空")
        book_dir = self.config.book_dir(book_id)
        if not book_dir.is_dir():
            raise FileNotFoundError("小说不存在")
        self._write_metadata(book_id, clean_name)
        chapters = self.list_chapters(book_id)
        return {
            "book_id": book_id,
            "name": clean_name,
            "chapter_count": len(chapters),
            "chapters": chapters,
        }

    def create_chapter(self, book_id: str, title: str = "") -> dict[str, Any]:
        book_dir = self.config.book_dir(book_id)
        if not book_dir.is_dir():
            raise FileNotFoundError("小说不存在")
        chapters = self.list_chapters(book_id)
        chapter_id = max(
            (int(chapter["chapter_id"]) for chapter in chapters),
            default=0,
        ) + 1
        clean_title = str(title or "").strip() or f"第{chapter_id}章"
        task_code = "NW" if chapter_id == 1 else "CT"
        payload = {
            "task_code": task_code,
            "chapter_id": chapter_id,
            "revision": 0,
            "status": "draft",
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "result": {
                "book_title": self._book_name(book_id, chapters),
                "chapter_title": clean_title,
                "text": "",
            },
        }
        path = self._draft_path(book_id, chapter_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_json(path, payload)
        return self.get_chapter(book_id, chapter_id)

    def list_chapters(self, book_id: str) -> list[dict[str, Any]]:
        folder = self.config.book_dir(book_id) / "documents"
        if not folder.is_dir():
            return []
        chapters: list[dict[str, Any]] = []
        paths = list(folder.glob("chapter_*.json"))
        draft_folder = folder / "drafts"
        if draft_folder.is_dir():
            paths.extend(draft_folder.glob("chapter_*.json"))
        chapter_ids = sorted(
            {
                int(match.group("chapter"))
                for path in paths
                if (match := CHAPTER_FILE.fullmatch(path.name))
                and int(match.group("chapter")) >= 1
            }
        )
        for chapter_id in chapter_ids:
            path = self._draft_path(book_id, chapter_id)
            is_draft = path.is_file()
            if not is_draft:
                path = self._chapter_path(book_id, chapter_id)
            try:
                payload = self._read_document(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            result = payload.get("result") or {}
            chapters.append(
                {
                    "chapter_id": chapter_id,
                    "title": self._title(result, chapter_id),
                    "revision": int(payload.get("revision") or 1),
                    "task_code": str(payload.get("task_code") or "CT"),
                    "is_draft": is_draft,
                }
            )
        return sorted(chapters, key=lambda item: int(item["chapter_id"]))

    def get_chapter(self, book_id: str, chapter_id: int) -> dict[str, Any]:
        draft_path = self._draft_path(book_id, chapter_id)
        is_draft = draft_path.is_file()
        path = draft_path if is_draft else self._chapter_path(book_id, chapter_id)
        if not path.is_file():
            raise FileNotFoundError(f"第{chapter_id}章不存在")
        payload = self._read_document(path)
        result = dict(payload.get("result") or {})
        return {
            "book_id": book_id,
            "book_name": self._book_name(book_id, self.list_chapters(book_id)),
            "chapter_id": chapter_id,
            "title": self._title(result, chapter_id),
            "text": str(result.get("text") or ""),
            "task_code": str(payload.get("task_code") or "CT"),
            "revision": int(payload.get("revision") or 1),
            "is_draft": is_draft,
            "result": result,
        }

    def chapter_exists(self, book_id: str, chapter_id: int) -> bool:
        return self._chapter_path(book_id, chapter_id).is_file()

    def chapter_available(self, book_id: str, chapter_id: int) -> bool:
        return self.chapter_exists(book_id, chapter_id) or self._draft_path(
            book_id, chapter_id
        ).is_file()

    def latest_chapter_id(self, book_id: str) -> int:
        folder = self.config.book_dir(book_id) / "documents"
        chapter_ids = [
            int(match.group("chapter"))
            for path in folder.glob("chapter_*.json")
            if (match := CHAPTER_FILE.fullmatch(path.name))
            and int(match.group("chapter")) >= 1
        ] if folder.is_dir() else []
        return max(chapter_ids, default=0)

    def save_draft(
        self,
        book_id: str,
        chapter_id: int,
        task_code: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        chapter_id = int(chapter_id)
        if chapter_id < 1:
            raise ValueError("正文草稿必须指定有效章节")
        title = self._title(result, chapter_id)
        text = str(result.get("text") or "").strip()
        if not title or not text:
            raise ValueError("章节标题和正文不能为空")
        active_path = self._chapter_path(book_id, chapter_id)
        base_revision = 0
        if active_path.is_file():
            active = self._read_document(active_path)
            base_revision = int(active.get("revision") or 1)
        payload = {
            "task_code": task_code,
            "chapter_id": chapter_id,
            "revision": base_revision,
            "status": "draft",
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "result": result,
        }
        path = self._draft_path(book_id, chapter_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
        return self.get_chapter(book_id, chapter_id)

    def delete_draft(self, book_id: str, chapter_id: int) -> None:
        path = self._draft_path(book_id, chapter_id)
        if path.is_file():
            path.unlink()

    def list_versions(self, book_id: str, chapter_id: int) -> list[dict[str, Any]]:
        folder = (
            self.config.book_dir(book_id)
            / "documents"
            / "history"
            / f"chapter_{chapter_id:06d}"
        )
        if not folder.is_dir():
            return []
        versions: list[dict[str, Any]] = []
        for path in sorted(folder.glob("revision_*.json"), reverse=True):
            payload = self._read_document(path)
            result = payload.get("result") or {}
            versions.append(
                {
                    "revision": int(payload.get("revision") or 1),
                    "title": self._title(result, chapter_id),
                    "superseded_at": payload.get("superseded_at"),
                    "status": "old",
                }
            )
        return versions

    def _chapter_path(self, book_id: str, chapter_id: int) -> Path:
        if chapter_id < 1:
            raise ValueError("chapter_id必须大于0")
        return (
            self.config.book_dir(book_id)
            / "documents"
            / f"chapter_{chapter_id:06d}.json"
        )

    def _draft_path(self, book_id: str, chapter_id: int) -> Path:
        if chapter_id < 1:
            raise ValueError("chapter_id必须大于0")
        return (
            self.config.book_dir(book_id)
            / "documents"
            / "drafts"
            / f"chapter_{chapter_id:06d}.json"
        )

    @staticmethod
    def _read_document(path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"章节文档格式错误: {path}")
        return payload

    @staticmethod
    def _title(result: dict[str, Any], chapter_id: int) -> str:
        return str(
            result.get("chapter_title")
            or result.get("title")
            or f"第{chapter_id}章"
        ).strip()

    def _book_name(self, book_id: str, chapters: list[dict[str, Any]]) -> str:
        metadata_path = self.config.book_dir(book_id) / BOOK_METADATA_FILE
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                name = str(metadata.get("name") or "").strip()
                if name:
                    return name
            except (OSError, json.JSONDecodeError):
                pass
        for chapter in chapters:
            try:
                chapter_id = int(chapter["chapter_id"])
                draft = self._draft_path(book_id, chapter_id)
                document = draft if draft.is_file() else self._chapter_path(book_id, chapter_id)
                payload = self._read_document(document).get("result") or {}
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            name = str(payload.get("book_title") or "").strip()
            if name:
                return name
        return f"作品 {book_id}"

    def _write_metadata(self, book_id: str, name: str) -> None:
        path = self.config.book_dir(book_id) / BOOK_METADATA_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        existing: dict[str, Any] = {}
        if path.is_file():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
        now = datetime.now(timezone.utc).isoformat()
        self._atomic_json(
            path,
            {
                **existing,
                "book_id": book_id,
                "name": name,
                "created_at": existing.get("created_at") or now,
                "updated_at": now,
            },
        )

    @staticmethod
    def _atomic_json(path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
