from __future__ import annotations

import math
from typing import Protocol

from .local_model import JsonModelClient
from .rag_message import RAGMessage, extract_rag_payload
from .schemas import AtomicMemory, MemoryFact, MemoryMatch


class MemoryJudge(Protocol):
    def judge(
        self,
        new_facts: list[MemoryFact],
        candidates: list[AtomicMemory],
    ) -> list[MemoryMatch]: ...


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class ConservativeMemoryJudge:
    """Offline fallback: only exact normalized content is treated as confirmed."""

    def judge(
        self,
        new_facts: list[MemoryFact],
        candidates: list[AtomicMemory],
    ) -> list[MemoryMatch]:
        fact_map = {"".join(fact.content.split()): fact for fact in new_facts}
        results: list[MemoryMatch] = []
        for memory in candidates:
            fact = fact_map.get("".join(memory.content.split()))
            if fact:
                results.append(
                    MemoryMatch(
                        memory_id=memory.memory_id,
                        matched_fact_ids=[fact.fact_id],
                        status="confirmed",
                        confidence=1.0,
                    )
                )
            else:
                results.append(
                    MemoryMatch(
                        memory_id=memory.memory_id,
                        status="unrelated",
                        confidence=1.0,
                    )
                )
        return results


class SemanticSimilarityJudge:
    """Safe fallback that only infers same, related, or unrelated."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        *,
        confirm_threshold: float = 0.95,
        related_threshold: float = 0.86,
    ) -> None:
        if not 0.0 <= related_threshold <= confirm_threshold <= 1.0:
            raise ValueError("语义相似度阈值必须满足0<=related<=confirm<=1")
        self.embedding_client = embedding_client
        self.confirm_threshold = confirm_threshold
        self.related_threshold = related_threshold

    def judge(
        self,
        new_facts: list[MemoryFact],
        candidates: list[AtomicMemory],
    ) -> list[MemoryMatch]:
        if not candidates:
            return []
        if not new_facts:
            return [
                MemoryMatch(
                    memory_id=memory.memory_id,
                    status="unrelated",
                    confidence=1.0,
                )
                for memory in candidates
            ]
        texts = [fact.content for fact in new_facts] + [
            memory.content for memory in candidates
        ]
        vectors = self.embedding_client.embed_texts(texts)
        if len(vectors) != len(texts):
            raise ValueError("Embedding返回向量数量与输入文本数量不一致")
        fact_vectors = vectors[: len(new_facts)]
        memory_vectors = vectors[len(new_facts) :]
        results: list[MemoryMatch] = []
        for memory, memory_vector in zip(candidates, memory_vectors):
            scores = [
                self._cosine(memory_vector, fact_vector)
                for fact_vector in fact_vectors
            ]
            best_index = max(range(len(scores)), key=scores.__getitem__)
            similarity = max(-1.0, min(1.0, scores[best_index]))
            fact = new_facts[best_index]
            exact = "".join(memory.content.split()) == "".join(fact.content.split())
            if exact or similarity >= self.confirm_threshold:
                status = "confirmed"
                confidence = 1.0 if exact else similarity
                matched_fact_ids = [fact.fact_id]
            elif similarity >= self.related_threshold:
                status = "referenced"
                confidence = similarity
                matched_fact_ids = [fact.fact_id]
            else:
                status = "unrelated"
                confidence = max(0.0, 1.0 - max(0.0, similarity))
                matched_fact_ids = []
            results.append(
                MemoryMatch(
                    memory_id=memory.memory_id,
                    matched_fact_ids=matched_fact_ids,
                    status=status,  # type: ignore[arg-type]
                    confidence=confidence,
                )
            )
        return results

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if len(left) != len(right) or not left:
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)


class QwenMemoryJudge:
    def __init__(
        self,
        client: JsonModelClient,
        *,
        semantic_fallback: SemanticSimilarityJudge | None = None,
    ) -> None:
        self.client = client
        embedder = getattr(client, "embed_texts", None)
        self.semantic_fallback = semantic_fallback
        if self.semantic_fallback is None and callable(embedder):
            self.semantic_fallback = SemanticSimilarityJudge(client)  # type: ignore[arg-type]

    def judge(
        self,
        new_facts: list[MemoryFact],
        candidates: list[AtomicMemory],
    ) -> list[MemoryMatch]:
        payload = {
            "new_facts": [
                {"fact_id": fact.fact_id, "content": fact.content}
                for fact in new_facts
            ],
            "candidate_memories": [
                {
                    "memory_id": memory.memory_id,
                    "store_type": memory.store_type,
                    "content": memory.content,
                }
                for memory in candidates
            ],
        }
        request = RAGMessage(
            sender="qwen_judge",
            receiver="qwen_model",
            action="rag.model.relation.classify.request",
            book_id=candidates[0].book_id if candidates else None,
            payload=payload,
        )
        last_error: Exception | None = None
        for _ in range(2):
            try:
                response = self.client.invoke_json(
                    (
                        "你只判断新事实与候选旧记忆的关系。每条候选只能标记为confirmed、"
                        "updated、referenced、conflict或unrelated。不要生成数据库更新内容。"
                        "必须为每条candidate_memories返回一项，memory_id必须逐字复制输入；"
                        "status必须是上述五个字符串之一，confidence必须是0到1的小数，"
                        "matched_fact_ids只能复制new_facts中的fact_id。"
                        "返回完整的rag.message.v1 JSON响应，action必须为"
                        "rag.model.relation.classify.result，sender为qwen_model，"
                        "receiver为qwen_judge，message_type为response，status为ok。"
                        "payload严格使用以下格式："
                        "{\"results\":[{\"memory_id\":\"memory_x\"," 
                        "\"matched_fact_ids\":[\"fact_x\"],"
                        "\"status\":\"unrelated\",\"confidence\":0.8}]}。"
                    ),
                    request.model_dump(),
                )
                response_payload, _ = extract_rag_payload(
                    response,
                    expected_action="rag.model.relation.classify.result",
                )
                return self._normalize_results(
                    response_payload,
                    new_facts,
                    candidates,
                )
            except Exception as exc:
                last_error = exc
        if self.semantic_fallback is not None:
            try:
                print(
                    "[警告] Qwen结构化关系判断失败，正在降级为语义向量相似度判断。"
                )
                return self.semantic_fallback.judge(new_facts, candidates)
            except Exception as fallback_error:
                raise RuntimeError(
                    "Qwen关系判断和语义向量降级均失败: "
                    f"Qwen={last_error}; Embedding={fallback_error}"
                ) from fallback_error
        raise RuntimeError(f"本地Qwen判断连续失败: {last_error}")

    @staticmethod
    def _normalize_results(
        response: dict,
        new_facts: list[MemoryFact],
        candidates: list[AtomicMemory],
    ) -> list[MemoryMatch]:
        allowed = {"confirmed", "updated", "referenced", "conflict", "unrelated"}
        fact_ids = {fact.fact_id for fact in new_facts}
        raw_results = response.get("results", [])
        if isinstance(raw_results, dict):
            raw_results = [
                {"memory_id": memory_id, **(value if isinstance(value, dict) else {"status": value})}
                for memory_id, value in raw_results.items()
            ]
        if not isinstance(raw_results, list):
            raw_results = []

        normalized: dict[str, MemoryMatch] = {}
        structured_count = 0
        for index, candidate in enumerate(candidates):
            raw = raw_results[index] if index < len(raw_results) else {}
            if not isinstance(raw, dict):
                raw = {}
            memory_id = candidate.memory_id

            status = str(raw.get("status") or "").strip().lower()
            if status not in allowed:
                selected = [name for name in allowed if raw.get(name) is True]
                status = selected[0] if len(selected) == 1 else "unrelated"
                inferred = True
            else:
                inferred = False
                structured_count += 1

            try:
                confidence = float(raw.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))
            if inferred:
                confidence = 0.0

            matched = raw.get("matched_fact_ids", [])
            if not isinstance(matched, list):
                matched = []
            matched_ids = [
                str(fact_id)
                for fact_id in matched
                if str(fact_id) in fact_ids
            ]
            normalized[memory_id] = MemoryMatch(
                memory_id=memory_id,
                matched_fact_ids=matched_ids,
                status=status,  # type: ignore[arg-type]
                confidence=confidence,
            )

        if candidates and structured_count == 0:
            raise ValueError("Qwen result is missing every required status field")
        return [normalized[memory.memory_id] for memory in candidates]
