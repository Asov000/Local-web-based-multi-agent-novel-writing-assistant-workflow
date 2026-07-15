from __future__ import annotations

from .config import RagConfig


def get_time_decay(gap: int) -> float:
    if gap < 5:
        return 1.0
    if gap < 10:
        return 0.8
    if gap < 20:
        return 0.6
    if gap < 30:
        return 0.4
    return 0.2


def effective_importance(
    raw_importance: float,
    type_weight: float,
    last_mentioned_chapter: int,
    current_chapter: int,
) -> float:
    gap = max(0, current_chapter - last_mentioned_chapter)
    value = raw_importance * get_time_decay(gap) * type_weight
    return max(0.0, min(1.0, value))


def lifecycle_status(
    score: float,
    config: RagConfig,
    *,
    open_hook: bool = False,
) -> str:
    if score >= config.active_threshold:
        return "active"
    if score >= config.delete_threshold or open_hook:
        return "archived"
    return "deleted"
