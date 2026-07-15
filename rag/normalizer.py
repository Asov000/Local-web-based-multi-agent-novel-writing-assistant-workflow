from __future__ import annotations

import re
from typing import Any

from .schemas import MemoryFact, TaskCode


SCORED_TEXT_PATTERN = re.compile(
    r"^(?P<text>.*?)\s+(?P<score>0(?:\.\d+)?|1(?:\.0+)?)\s*/\s*"
    r"(?P<flag>[TF])\s*[。.]?$",
    flags=re.IGNORECASE | re.DOTALL,
)


def parse_scored_value(value: Any) -> tuple[str, float | None, bool]:
    if isinstance(value, dict):
        text = str(
            value.get("summary")
            or value.get("content")
            or value.get("new_value")
            or value.get("value")
            or ""
        ).strip()
        score = value.get("raw_importance", value.get("importance"))
        importance = float(score) if score is not None else None
        authoritative = bool(
            value.get("canon_candidate", value.get("authoritative", False))
        )
        return text, importance, authoritative

    text = str(value or "").strip()
    if not text:
        return "", None, False
    match = SCORED_TEXT_PATTERN.match(text)
    if not match:
        return text, None, False
    return (
        match.group("text").strip(),
        float(match.group("score")),
        match.group("flag").upper() == "T",
    )


def _names(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _fact(
    fact_type: str,
    value: Any,
    *,
    source_field: str,
    character_names: list[str] | None = None,
    item_names: list[str] | None = None,
    event_names: list[str] | None = None,
    entity_name: str | None = None,
    field: str | None = None,
    hook_status: str | None = None,
) -> MemoryFact | None:
    text, score, authoritative = parse_scored_value(value)
    if not text:
        return None
    if isinstance(value, dict):
        character_names = _names(value.get("characters")) or character_names
        item_names = _names(value.get("items")) or item_names
        event_name = str(value.get("event_name") or "").strip()
        event_names = ([event_name] if event_name else []) or event_names
        entity_name = str(value.get("entity") or entity_name or "").strip() or None
        field = str(value.get("field") or field or "").strip() or None
        hook_status = str(value.get("status") or hook_status or "").strip() or None
        if fact_type == "state_change" and not value.get("summary") and not value.get("content"):
            old_value = value.get("old_value")
            new_value = value.get("new_value")
            text = f"{entity_name or '实体'}的{field or '状态'}由{old_value}变为{new_value}"
    return MemoryFact(
        fact_type=fact_type,
        content=text,
        character_names=character_names or [],
        item_names=item_names or [],
        event_names=event_names or [],
        raw_importance=score,
        canon_candidate=authoritative,
        memory_scope="permanent" if authoritative else "temporary",
        entity_name=entity_name,
        field=field,
        old_value=value.get("old_value") if isinstance(value, dict) else None,
        new_value=value.get("new_value") if isinstance(value, dict) else None,
        hook_status=hook_status if hook_status in {"open", "resolved", "abandoned"} else None,
        source_field=source_field,
    )


def normalize_writer_result(task_code: str, data: dict[str, Any]) -> list[MemoryFact]:
    code = task_code.upper().strip()
    if code not in {"BD", "CH", "CT", "NW", "RV"}:
        raise ValueError(f"不支持的任务代码: {task_code}")
    facts: list[MemoryFact] = []

    def add(fact: MemoryFact | None) -> None:
        if fact is not None:
            facts.append(fact)

    if code == "BD":
        world_name = str(data.get("world_name") or "").strip() or None
        add(_fact("world_background", data.get("background"), source_field="background", entity_name=world_name))
        for index, value in enumerate(data.get("rules", [])):
            add(_fact("world_rule", value, source_field=f"rules[{index}]", entity_name=world_name))
        for index, value in enumerate(data.get("factions", [])):
            add(_fact("faction", value, source_field=f"factions[{index}]", entity_name=world_name))
        for index, value in enumerate(data.get("locations", [])):
            add(_fact("location", value, source_field=f"locations[{index}]", entity_name=world_name))
        add(_fact("core_conflict", data.get("conflict"), source_field="conflict", entity_name=world_name))

    elif code == "CH":
        for char_index, character in enumerate(data.get("characters", [])):
            if not isinstance(character, dict):
                continue
            name = str(character.get("name") or "").strip()
            mapping = {
                "role": "character_identity",
                "appearance": "character_profile",
                "personality": "character_profile",
                "background": "character_profile",
                "goal": "character_state",
                "ability": "character_identity",
            }
            for field, fact_type in mapping.items():
                add(
                    _fact(
                        fact_type,
                        character.get(field),
                        source_field=f"characters[{char_index}].{field}",
                        character_names=[name] if name else [],
                        entity_name=name or None,
                        field=field,
                    )
                )
            for rel_index, value in enumerate(character.get("relations", [])):
                add(
                    _fact(
                        "relation",
                        value,
                        source_field=f"characters[{char_index}].relations[{rel_index}]",
                        character_names=[name] if name else [],
                        entity_name=name or None,
                    )
                )

    elif code == "CT":
        characters = _names(data.get("characters"))
        for index, value in enumerate(data.get("events", [])):
            add(_fact("event", value, source_field=f"events[{index}]", character_names=characters))
        for index, value in enumerate(data.get("changes", [])):
            add(_fact("state_change", value, source_field=f"changes[{index}]", character_names=characters))
        for index, value in enumerate(data.get("hooks", [])):
            add(
                _fact(
                    "foreshadowing_open",
                    value,
                    source_field=f"hooks[{index}]",
                    character_names=characters,
                    hook_status="open",
                )
            )

    elif code == "NW":
        world = data.get("world") if isinstance(data.get("world"), dict) else {}
        add(_fact("world_background", world.get("background"), source_field="world.background"))
        for index, value in enumerate(world.get("rules", [])):
            add(_fact("world_rule", value, source_field=f"world.rules[{index}]"))
        add(_fact("core_conflict", world.get("conflict"), source_field="world.conflict"))
        all_characters = _names(
            [item.get("name") for item in data.get("characters", []) if isinstance(item, dict)]
        )
        for char_index, character in enumerate(data.get("characters", [])):
            if not isinstance(character, dict):
                continue
            name = str(character.get("name") or "").strip()
            for field, fact_type in {
                "role": "character_identity",
                "profile": "character_profile",
                "goal": "character_state",
            }.items():
                add(
                    _fact(
                        fact_type,
                        character.get(field),
                        source_field=f"characters[{char_index}].{field}",
                        character_names=[name] if name else [],
                        entity_name=name or None,
                        field=field,
                    )
                )
        for index, value in enumerate(data.get("hooks", [])):
            add(
                _fact(
                    "foreshadowing_open",
                    value,
                    source_field=f"hooks[{index}]",
                    character_names=all_characters,
                    hook_status="open",
                )
            )

    elif code == "RV":
        for index, value in enumerate(data.get("changes", [])):
            add(_fact("revision_change", value, source_field=f"changes[{index}]"))

    return facts
