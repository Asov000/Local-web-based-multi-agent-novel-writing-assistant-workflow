from __future__ import annotations

import os
from pathlib import Path
from types import TracebackType

from .config import RagConfig


class BookLockError(RuntimeError):
    pass


class BookDatabaseLock:
    """Cross-process advisory lock shared by ingestion and maintenance."""

    def __init__(self, config: RagConfig, book_id: str) -> None:
        safe_book_dir = config.book_dir(book_id)
        self.path = (
            config.root_dir
            / ".maintenance_locks"
            / f"{safe_book_dir.name}.lock"
        )
        self._stream = None

    def __enter__(self) -> BookDatabaseLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise BookLockError(
                f"作品数据库正在被其他入库或维护任务使用: {self.path.stem}"
            ) from exc
        self._stream = stream
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._stream is None:
            return
        try:
            self._stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        finally:
            self._stream.close()
            self._stream = None
