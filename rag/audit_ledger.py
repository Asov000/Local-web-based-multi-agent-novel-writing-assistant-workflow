from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .config import RagConfig


class AuditLedger:
    def __init__(self, config: RagConfig, book_id: str, run_id: str) -> None:
        self.root = config.root_dir / ".maintenance_runs" / book_id / run_id
        self.root.mkdir(parents=True, exist_ok=False)

    def write(self, name: str, value: Any) -> Path:
        path = self.root / f"{name}.json"
        if isinstance(value, BaseModel):
            payload = value.model_dump(mode="json")
        elif isinstance(value, list):
            payload = [
                item.model_dump(mode="json") if isinstance(item, BaseModel) else item
                for item in value
            ]
        else:
            payload = value
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path
