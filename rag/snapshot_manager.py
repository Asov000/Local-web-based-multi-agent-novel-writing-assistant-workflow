from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .config import RagConfig
from .maintenance_schemas import maintenance_id


class SnapshotError(RuntimeError):
    pass


class SnapshotManager:
    def __init__(self, config: RagConfig) -> None:
        self.config = config

    def create(self, book_id: str) -> str:
        source = self.config.book_dir(book_id)
        if not source.is_dir():
            raise SnapshotError(f"作品数据库目录不存在: {source}")
        snapshot_id = maintenance_id("snapshot")
        snapshot_dir = self._snapshot_dir(book_id, snapshot_id)
        book_copy = snapshot_dir / "book"
        snapshot_dir.mkdir(parents=True, exist_ok=False)
        shutil.copytree(source, book_copy)
        manifest = {
            "snapshot_id": snapshot_id,
            "book_id": book_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files": self._file_hashes(book_copy),
        }
        (snapshot_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return snapshot_id

    def verify(self, book_id: str, snapshot_id: str) -> None:
        snapshot_dir = self._snapshot_dir(book_id, snapshot_id)
        manifest_path = snapshot_dir / "manifest.json"
        book_copy = snapshot_dir / "book"
        if not manifest_path.is_file() or not book_copy.is_dir():
            raise SnapshotError(f"快照不存在或不完整: {snapshot_id}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("book_id") != book_id:
            raise SnapshotError("快照book_id不匹配")
        actual = self._file_hashes(book_copy)
        if actual != manifest.get("files"):
            raise SnapshotError(f"快照校验失败: {snapshot_id}")

    def rollback(self, book_id: str, snapshot_id: str) -> None:
        self.verify(book_id, snapshot_id)
        current = self.config.book_dir(book_id)
        parent = current.parent
        snapshot_book = self._snapshot_dir(book_id, snapshot_id) / "book"
        restore_dir = parent / f".restore_{snapshot_id}"
        old_dir = parent / f".old_{snapshot_id}"
        if restore_dir.exists() or old_dir.exists():
            raise SnapshotError("发现未清理的回滚临时目录，拒绝继续")
        shutil.copytree(snapshot_book, restore_dir)
        try:
            if current.exists():
                current.rename(old_dir)
            restore_dir.rename(current)
        except Exception:
            if current.exists() and not old_dir.exists():
                shutil.rmtree(current)
            if old_dir.exists() and not current.exists():
                old_dir.rename(current)
            if restore_dir.exists():
                shutil.rmtree(restore_dir)
            raise
        if old_dir.exists():
            shutil.rmtree(old_dir)

    def _snapshot_dir(self, book_id: str, snapshot_id: str) -> Path:
        if not snapshot_id.startswith("snapshot_"):
            raise SnapshotError("snapshot_id格式错误")
        return self.config.root_dir / ".maintenance_snapshots" / book_id / snapshot_id

    @staticmethod
    def _file_hashes(root: Path) -> dict[str, str]:
        result: dict[str, str] = {}
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            result[path.relative_to(root).as_posix()] = digest.hexdigest()
        return result
