from __future__ import annotations

import hashlib
import random

from .config import RagConfig
from .schemas import AtomicMemory


def _weighted_sample_without_replacement(
    memories: list[AtomicMemory],
    count: int,
    *,
    current_chapter: int,
    randomizer: random.Random,
) -> list[AtomicMemory]:
    if count <= 0 or not memories:
        return []
    scored: list[tuple[float, AtomicMemory]] = []
    for memory in memories:
        chapters_since_sample = max(0, current_chapter - memory.last_sampled_chapter)
        weight = 1.0 + chapters_since_sample * 0.1 + (1.0 - memory.effective_importance) * 0.3
        key = randomizer.random() ** (1.0 / max(weight, 0.0001))
        scored.append((key, memory))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [memory for _, memory in scored[:count]]


def sample_candidates(
    candidates: list[AtomicMemory],
    book_id: str,
    current_chapter: int,
    config: RagConfig,
) -> list[AtomicMemory]:
    unique = list({memory.memory_id: memory for memory in candidates}.values())
    if len(unique) <= config.max_judge_memories:
        return unique

    canon = sorted(
        (memory for memory in unique if memory.store_type == "canon_memory"),
        key=lambda memory: memory.raw_importance,
        reverse=True,
    )[: config.max_canon_memories]
    canon_ids = {memory.memory_id for memory in canon}
    dynamic = [memory for memory in unique if memory.memory_id not in canon_ids]
    top = sorted(
        dynamic,
        key=lambda memory: memory.effective_importance,
        reverse=True,
    )[: config.top_importance_memories]
    fixed_ids = canon_ids | {memory.memory_id for memory in top}
    random_pool = [memory for memory in dynamic if memory.memory_id not in fixed_ids]
    remaining = min(
        config.max_judge_memories - len(canon) - len(top),
        config.random_memory_count + max(0, config.max_canon_memories - len(canon)),
    )
    seed = int.from_bytes(
        hashlib.sha256(f"{book_id}:{current_chapter}".encode("utf-8")).digest()[:8],
        "big",
    )
    exploration = _weighted_sample_without_replacement(
        random_pool,
        remaining,
        current_chapter=current_chapter,
        randomizer=random.Random(seed),
    )
    return canon + top + exploration
