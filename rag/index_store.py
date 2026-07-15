from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .schemas import HeadType


class AliasConflictError(ValueError):
    pass


def normalize_name(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold().strip()


class IndexStore:
    def __init__(self, path: Path, book_id: str) -> None:
        self.path = path
        self.book_id = book_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS index_head (
                    head_id TEXT PRIMARY KEY,
                    book_id TEXT NOT NULL,
                    head_type TEXT NOT NULL,
                    canonical_name TEXT NOT NULL,
                    normalized_name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_head_lookup
                    ON index_head(book_id, head_type, normalized_name);
                CREATE TABLE IF NOT EXISTS index_alias (
                    alias TEXT NOT NULL,
                    normalized_alias TEXT NOT NULL,
                    head_id TEXT NOT NULL,
                    book_id TEXT NOT NULL,
                    ambiguous INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(book_id, normalized_alias, head_id),
                    FOREIGN KEY(head_id) REFERENCES index_head(head_id)
                );
                CREATE INDEX IF NOT EXISTS idx_alias_lookup
                    ON index_alias(book_id, normalized_alias);
                CREATE TABLE IF NOT EXISTS memory_index_link (
                    head_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    store_type TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_chapter INTEGER NOT NULL,
                    UNIQUE(head_id, memory_id, role),
                    FOREIGN KEY(head_id) REFERENCES index_head(head_id)
                );
                CREATE INDEX IF NOT EXISTS idx_link_head ON memory_index_link(head_id);
                CREATE INDEX IF NOT EXISTS idx_link_memory ON memory_index_link(memory_id);
                """
            )

    def resolve(self, head_type: HeadType, name: str) -> str | None:
        normalized = normalize_name(name)
        if not normalized:
            return None
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT h.head_id
                FROM index_alias a
                JOIN index_head h ON h.head_id = a.head_id
                WHERE a.book_id = ? AND h.head_type = ?
                    AND a.normalized_alias = ? AND h.status = 'active'
                """,
                (self.book_id, head_type, normalized),
            ).fetchall()
        ids = [row["head_id"] for row in rows]
        if len(ids) > 1:
            raise AliasConflictError(f"别名 {name!r} 指向多个{head_type}实体")
        return ids[0] if ids else None

    def resolve_or_create(
        self,
        head_type: HeadType,
        name: str,
        *,
        metadata: dict[str, object] | None = None,
    ) -> str:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("实体名称不能为空")
        existing = self.resolve(head_type, clean_name)
        if existing:
            return existing
        normalized = normalize_name(clean_name)
        digest = hashlib.sha1(
            f"{self.book_id}:{head_type}:{normalized}".encode("utf-8")
        ).hexdigest()[:12]
        prefix = {"character": "char", "item": "item", "event": "event"}[head_type]
        head_id = f"{prefix}_{digest}"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO index_head
                (head_id, book_id, head_type, canonical_name, normalized_name, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    head_id,
                    self.book_id,
                    head_type,
                    clean_name,
                    normalized,
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO index_alias
                (alias, normalized_alias, head_id, book_id)
                VALUES (?, ?, ?, ?)
                """,
                (clean_name, normalized, head_id, self.book_id),
            )
        return head_id

    def add_alias(self, head_id: str, alias: str) -> None:
        normalized = normalize_name(alias)
        if not normalized:
            return
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT DISTINCT head_id FROM index_alias
                WHERE book_id = ? AND normalized_alias = ?
                """,
                (self.book_id, normalized),
            ).fetchall()
            ambiguous = bool(existing and any(row["head_id"] != head_id for row in existing))
            if ambiguous:
                connection.execute(
                    "UPDATE index_alias SET ambiguous = 1 WHERE book_id = ? AND normalized_alias = ?",
                    (self.book_id, normalized),
                )
            connection.execute(
                """
                INSERT OR IGNORE INTO index_alias
                (alias, normalized_alias, head_id, book_id, ambiguous)
                VALUES (?, ?, ?, ?, ?)
                """,
                (alias.strip(), normalized, head_id, self.book_id, int(ambiguous)),
            )

    def link(
        self,
        head_id: str,
        memory_id: str,
        store_type: str,
        role: str,
        created_chapter: int,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO memory_index_link
                (head_id, memory_id, store_type, role, created_chapter)
                VALUES (?, ?, ?, ?, ?)
                """,
                (head_id, memory_id, store_type, role, created_chapter),
            )

    def find_memory_links(self, head_ids: list[str]) -> list[tuple[str, str]]:
        if not head_ids:
            return []
        placeholders = ",".join("?" for _ in head_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT DISTINCT memory_id, store_type FROM memory_index_link
                WHERE head_id IN ({placeholders})
                """,
                head_ids,
            ).fetchall()
        return [(row["memory_id"], row["store_type"]) for row in rows]

    def heads_in_text(self, text: str) -> list[str]:
        normalized_text = normalize_name(text)
        if not normalized_text:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT head_id, normalized_alias FROM index_alias
                WHERE book_id = ? AND ambiguous = 0
                """,
                (self.book_id,),
            ).fetchall()
        return list(
            dict.fromkeys(
                row["head_id"]
                for row in rows
                if row["normalized_alias"] and row["normalized_alias"] in normalized_text
            )
        )

    def remove_memory_links(self, memory_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM memory_index_link WHERE memory_id = ?",
                (memory_id,),
            )

    def list_heads(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT head_id, book_id, head_type, canonical_name,
                       normalized_name, status, metadata_json
                FROM index_head
                WHERE book_id = ?
                ORDER BY head_type, canonical_name, head_id
                """,
                (self.book_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_links(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT head_id, memory_id, store_type, role, created_chapter
                FROM memory_index_link
                ORDER BY head_id, memory_id, role
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def head_exists(self, head_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM index_head WHERE head_id = ? AND book_id = ?",
                (head_id, self.book_id),
            ).fetchone()
        return row is not None

    def links_for_memory(self, memory_id: str) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT head_id, memory_id, store_type, role, created_chapter
                FROM memory_index_link WHERE memory_id = ?
                """,
                (memory_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def remove_link(
        self,
        head_id: str,
        memory_id: str,
        role: str | None = None,
    ) -> None:
        with self._connect() as connection:
            if role is None:
                connection.execute(
                    "DELETE FROM memory_index_link WHERE head_id = ? AND memory_id = ?",
                    (head_id, memory_id),
                )
            else:
                connection.execute(
                    """
                    DELETE FROM memory_index_link
                    WHERE head_id = ? AND memory_id = ? AND role = ?
                    """,
                    (head_id, memory_id, role),
                )

    def replace_memory_link_target(
        self,
        source_memory_id: str,
        target_memory_id: str,
        target_store_type: str,
    ) -> None:
        links = self.links_for_memory(source_memory_id)
        for link in links:
            self.link(
                str(link["head_id"]),
                target_memory_id,
                target_store_type,
                str(link["role"]),
                int(link["created_chapter"]),
            )
        self.remove_memory_links(source_memory_id)
