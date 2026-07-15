from __future__ import annotations

import re
from typing import Any

from .schemas import MemoryFact


FIELD_PATH = re.compile(r"^facts\[(?P<index>\d+)]\.(?P<field>[a-z_]+)$")


def find_missing_fields(facts: list[MemoryFact]) -> list[str]:
    missing: list[str] = []
    for index, fact in enumerate(facts):
        if fact.raw_importance is None:
            missing.append(f"facts[{index}].raw_importance")
    return missing


def apply_completed_fields(
    facts: list[MemoryFact],
    completed_fields: dict[str, Any],
) -> list[MemoryFact]:
    for path, value in completed_fields.items():
        match = FIELD_PATH.match(path)
        if not match:
            continue
        index = int(match.group("index"))
        field = match.group("field")
        if index >= len(facts) or field not in MemoryFact.model_fields:
            continue
        setattr(facts[index], field, value)
    return facts
