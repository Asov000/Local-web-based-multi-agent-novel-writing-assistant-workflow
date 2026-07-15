from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .schemas import AtomicMemory, ConflictRecord


class SQLiteMemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    book_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    raw_importance REAL NOT NULL,
                    effective_importance REAL NOT NULL,
                    status TEXT NOT NULL,
                    source_chapter INTEGER NOT NULL,
                    last_mentioned_chapter INTEGER NOT NULL,
                    entity_name TEXT,
                    field_name TEXT,
                    is_current INTEGER NOT NULL,
                    hook_status TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_chapter ON memories(source_chapter)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_state ON memories(entity_name, field_name, is_current)"
            )

    def save(self, memory: AtomicMemory) -> None:
        payload = memory.model_dump_json()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memories (
                    memory_id, book_id, memory_type, content,
                    raw_importance, effective_importance, status,
                    source_chapter, last_mentioned_chapter,
                    entity_name, field_name, is_current, hook_status, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    memory_type=excluded.memory_type,
                    content=excluded.content,
                    raw_importance=excluded.raw_importance,
                    effective_importance=excluded.effective_importance,
                    status=excluded.status,
                    source_chapter=excluded.source_chapter,
                    last_mentioned_chapter=excluded.last_mentioned_chapter,
                    entity_name=excluded.entity_name,
                    field_name=excluded.field_name,
                    is_current=excluded.is_current,
                    hook_status=excluded.hook_status,
                    payload_json=excluded.payload_json
                """,
                (
                    memory.memory_id,
                    memory.book_id,
                    memory.memory_type,
                    memory.content,
                    memory.raw_importance,
                    memory.effective_importance,
                    memory.status,
                    memory.source_chapter,
                    memory.last_mentioned_chapter,
                    memory.entity_name,
                    memory.field,
                    int(memory.is_current),
                    memory.hook_status,
                    payload,
                ),
            )

    def get(self, memory_id: str) -> AtomicMemory | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
        return AtomicMemory.model_validate_json(row["payload_json"]) if row else None

    def raw_records(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT memory_id, book_id, memory_type, content, status,
                       source_chapter, last_mentioned_chapter, payload_json
                FROM memories
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def find_by_ids(self, memory_ids: Iterable[str]) -> list[AtomicMemory]:
        ids = list(dict.fromkeys(memory_ids))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT payload_json FROM memories WHERE memory_id IN ({placeholders})",
                ids,
            ).fetchall()
        return [AtomicMemory.model_validate_json(row["payload_json"]) for row in rows]

    def list_memories(
        self,
        *,
        statuses: tuple[str, ...] | None = ("active",),
        memory_types: tuple[str, ...] | None = None,
        limit: int | None = None,
        descending_chapter: bool = False,
    ) -> list[AtomicMemory]:
        clauses: list[str] = []
        params: list[object] = []
        if statuses:
            clauses.append("status IN (%s)" % ",".join("?" for _ in statuses))
            params.extend(statuses)
        if memory_types:
            clauses.append("memory_type IN (%s)" % ",".join("?" for _ in memory_types))
            params.extend(memory_types)
        sql = "SELECT payload_json FROM memories"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        if descending_chapter:
            sql += " ORDER BY source_chapter DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [AtomicMemory.model_validate_json(row["payload_json"]) for row in rows]

    def current_state(self, entity_name: str, field_name: str) -> AtomicMemory | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM memories
                WHERE entity_name = ? AND field_name = ? AND is_current = 1
                ORDER BY source_chapter DESC LIMIT 1
                """,
                (entity_name, field_name),
            ).fetchone()
        return AtomicMemory.model_validate_json(row["payload_json"]) if row else None

    def delete_physical(self) -> int:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM memories WHERE status = 'deleted'")
            return cursor.rowcount


class ConflictStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conflicts (
                    conflict_id TEXT PRIMARY KEY,
                    book_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    fact_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def add(self, conflict: ConflictRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conflicts VALUES (?, ?, ?, ?, ?, ?)",
                (
                    conflict.conflict_id,
                    conflict.book_id,
                    conflict.memory_id,
                    conflict.fact_id,
                    conflict.status,
                    conflict.model_dump_json(),
                ),
            )

    def list_pending(self) -> list[ConflictRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM conflicts WHERE status = 'pending'"
            ).fetchall()
        return [ConflictRecord.model_validate_json(row["payload_json"]) for row in rows]
