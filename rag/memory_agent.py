from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .local_model import JsonModelClient
from .chapter_memory_prompt import (
    CHAPTER_MEMORY_EXTRACTION_CONFIG,
    build_memory_task_prompt,
)
from .maintenance_schemas import (
    AuditAgentResult,
    AuditPacket,
    CrossAuditPacket,
    CrossAuditResult,
    MemoryComparisonPacket,
    MemoryComparisonResult,
    ReconcilePayload,
    ReconcileResult,
)
from .rag_message import RAGMessage, extract_rag_payload
from .schemas import (
    ChapterMemoryExtractionPayload,
    ChapterMemoryExtractionResult,
    MemoryCompletionPayload,
    MemoryCompletionResult,
    MemoryFact,
)


class MemoryAgent:
    """Model-assisted memory analyst. Database writes remain in RAGMessageProcessor."""

    def __init__(
        self,
        model_client: JsonModelClient | None = None,
        *,
        agent_name: str = "memory_agent",
    ) -> None:
        self.model_client = model_client
        self.agent_name = agent_name

    def handle_message(self, message: Any) -> RAGMessage:
        request = RAGMessage.from_agent_message(message)
        if request.message_type != "request":
            return self._error(request, "MemoryAgent only accepts request messages")
        if request.receiver != self.agent_name:
            return self._error(
                request,
                f"Message receiver must be {self.agent_name}",
            )
        if request.action in {"memory.ping", "rag.memory.ping"}:
            return self._ok(request, {"agent": self.agent_name, "ready": True})
        if request.action in {"memory.complete", "rag.memory.complete"}:
            return self._handle_completion(request)
        if request.action == "rag.memory.extract_chapter":
            return self._handle_chapter_extraction(request)
        if request.action in {"memory.audit.plan", "rag.memory.audit.plan"}:
            return self._handle_audit_plan(request)
        if request.action in {"memory.audit.compare", "rag.memory.audit.compare"}:
            return self._handle_memory_comparison(request)
        if request.action in {
            "memory.audit.cross_check",
            "rag.memory.audit.cross_check",
        }:
            return self._handle_cross_check(request)
        if request.action in {
            "memory.audit.reconcile",
            "rag.memory.audit.reconcile",
        }:
            return self._handle_reconcile(request)
        if request.action == "rag.memory.operations.plan":
            return self._handle_operation_plan(request)
        return self._error(request, f"Unsupported action: {request.action}")

    def extract_chapter_facts(
        self,
        *,
        book_id: str,
        chapter_id: int,
        title: str,
        text: str,
    ) -> list[MemoryFact]:
        request = RAGMessage(
            sender="rag_manager",
            receiver=self.agent_name,
            action="rag.memory.extract_chapter",
            book_id=book_id,
            payload=ChapterMemoryExtractionPayload(
                chapter_id=chapter_id,
                title=title,
                text=text,
            ).model_dump(),
        )
        response = self.handle_message(request)
        if response.status != "ok":
            raise RuntimeError(response.error or "章节记忆提取失败")
        return ChapterMemoryExtractionResult.model_validate(
            response.payload or {}
        ).facts

    def _handle_chapter_extraction(self, message: RAGMessage) -> RAGMessage:
        try:
            payload = ChapterMemoryExtractionPayload.model_validate(message.payload)
        except Exception as exc:
            return self._error(message, f"Invalid chapter extraction payload: {exc}")
        if self.model_client is None:
            return self._error(message, "No local Memory model is configured")

        raw_result, error = self._invoke_with_retry(
            build_memory_task_prompt(CHAPTER_MEMORY_EXTRACTION_CONFIG),
            payload.model_dump(),
            request_action="rag.model.memory.extract_chapter.request",
            response_action="rag.model.memory.extract_chapter.result",
            book_id=message.book_id,
            task_id=message.task_id,
            validate_result=self._validate_chapter_extraction_result,
        )
        if raw_result is None:
            return self._error(message, f"Chapter memory extraction failed twice: {error}")
        try:
            result = ChapterMemoryExtractionResult.model_validate(raw_result)
        except Exception as exc:
            return self._error(message, f"Invalid chapter extraction result: {exc}")
        if not result.facts:
            return self._error(message, "章节正文未提取出任何有效记忆，拒绝替换旧记忆")
        return self._ok(message, result.model_dump(mode="json"))

    @staticmethod
    def _validate_chapter_extraction_result(result: dict[str, Any]) -> dict[str, Any]:
        return ChapterMemoryExtractionResult.model_validate(result).model_dump(mode="json")

    def _handle_completion(self, message: RAGMessage) -> RAGMessage:
        try:
            payload = MemoryCompletionPayload.model_validate(message.payload)
        except Exception as exc:
            return self._error(message, f"Invalid completion payload: {exc}")
        if not payload.missing_fields:
            return self._ok(message, MemoryCompletionResult().model_dump())
        if self.model_client is None:
            return self._error(message, "No local Memory model is configured")

        raw_result, error = self._invoke_with_retry(
            (
                "你是小说记忆字段补全器。只补全 payload.missing_fields 明确列出的字段；"
                "不得查询数据库、修改索引或编造无关事实。raw_importance 必须在 0 到 1 "
                "之间。业务结果必须放入 RAGMessage.payload，格式严格为 "
                "{\"completed_fields\":{\"字段路径\":\"补全值\"}}。"
            ),
            payload.model_dump(),
            request_action="rag.model.memory.complete.request",
            response_action="rag.model.memory.complete.result",
            book_id=message.book_id,
            task_id=message.task_id,
            request_metadata=message.metadata,
        )
        if raw_result is None:
            return self._error(message, f"Memory completion failed twice: {error}")
        try:
            result = MemoryCompletionResult.model_validate(raw_result)
        except Exception as exc:
            return self._error(message, f"Invalid memory completion result: {exc}")
        return self._ok(message, result.model_dump())

    def _handle_cross_check(self, message: RAGMessage) -> RAGMessage:
        try:
            packet = CrossAuditPacket.model_validate(message.payload)
        except Exception as exc:
            return self._error(message, f"Invalid cross audit packet: {exc}")
        if self.model_client is None:
            return self._error(message, "No local Memory model is configured")

        required_candidate_ids = [
            candidate.candidate_id for candidate in packet.candidates
        ]
        cross_payload_template = {
            "packet_id": packet.packet_id,
            "reviewed_candidate_ids": required_candidate_ids,
            "findings": [],
            "operations": [],
        }

        raw_result, error = self._invoke_with_retry(
            (
                "你是小说 RAG 记忆的跨索引审计器。只检查输入 Message.payload 中提供的"
                "候选对，不得检查或推断候选范围以外的数据。\n"
                "【强制覆盖要求】必须逐一比较每个 candidate 的两条完整记忆，并把所有 "
                "candidate_id 按输入顺序原样复制到 reviewed_candidate_ids。不得遗漏、"
                "重复、改写或添加 ID。即使所有候选都没有问题，也必须返回完整的 "
                f"reviewed_candidate_ids={json.dumps(required_candidate_ids, ensure_ascii=False)}。\n"
                "reason_codes 只是 Python 选择候选的原因，不是冲突证据。只有确实存在"
                "语义冲突、事实矛盾、危险重复或时间线错误时才生成 findings。健康候选"
                "不得生成 finding，也不得生成 no_op。\n"
                "finding 只能包含 code、severity、summary、memory_ids、evidence。"
                "operation 必须使用 operation、reason、expected_versions；"
                "expected_versions 必须是 memory_id 到整数版本号的对象。核心设定也允许"
                "update、merge、archive、relink和restore，但merge仅限正文、类型、实体和字段"
                "完全一致的记录；update或archive必须有明确正文证据。无法判断冲突双方谁正确时"
                "只能使用flag_conflict。\n"
                "【强制返回格式】只输出一个 JSON 对象，不得输出 Markdown、解释文字、"
                "代码围栏或 JSON 之外的字符。RAGMessage.payload 的最小合法模板为："
                f"{json.dumps(cross_payload_template, ensure_ascii=False)}。"
                "如果输入 Message.metadata.validation_feedback 存在，必须先按其中的"
                "错误说明修复本次输出。"
            ),
            packet.model_dump(mode="json"),
            request_action="rag.model.audit.cross_check.request",
            response_action="rag.model.audit.cross_check.result",
            book_id=message.book_id or packet.book_id,
            task_id=message.task_id,
            validate_result=lambda raw: self._validate_cross_result(raw, packet),
            request_metadata=message.metadata,
        )
        if raw_result is None:
            return self._error(message, f"Cross memory audit failed twice: {error}")
        try:
            result = CrossAuditResult.model_validate(raw_result)
        except Exception as exc:
            return self._error(message, f"Invalid cross audit result: {exc}")
        return self._ok(message, result.model_dump())

    def _handle_audit_plan(self, message: RAGMessage) -> RAGMessage:
        try:
            packet = AuditPacket.model_validate(message.payload)
        except Exception as exc:
            return self._error(message, f"Invalid audit packet: {exc}")
        if self.model_client is None:
            return self._error(message, "No local Memory model is configured")

        audit_payload_template = {
            "packet_id": packet.packet_id,
            "reviewed_memory_ids": packet.primary_memory_ids,
            "findings": [],
            "operations": [],
        }

        raw_result, error = self._invoke_with_retry(
            (
                "你是小说 RAG 记忆审计器。你的任务是审计输入 Message.payload 中的"
                "当前记忆包，不得引用包外信息，不得直接修改数据库。\n"
                "【一、审计范围】\n"
                "1. memories 是本记忆包中唯一包含完整正文的主记忆集合，也是语义判断"
                "的主要依据。\n"
                "2. related_memory_refs 和 global_context 仅为轻量引用，只能用于辅助"
                "定位和建立关联。\n"
                "3. 当缺少完整正文时，禁止仅依据 related_memory_refs 或 global_context "
                "判断语义冲突、重复、过期或其他内容性问题。\n"
                "【二、最高优先级：覆盖要求（完整覆盖 primary_memory_ids）】\n"
                "1. 必须逐条审阅 primary_memory_ids 中的每个 ID。\n"
                "2. reviewed_memory_ids 必须按照输入顺序，原样复制 primary_memory_ids "
                "的全部内容。\n"
                "3. 不得遗漏、重复、改写、重新排序或添加任何 ID。\n"
                "4. 无论某条记忆是否健康、是否产生 finding 或 operation，"
                "reviewed_memory_ids 都必须完整返回。\n"
                "5. 本包 reviewed_memory_ids 必须严格等于："
                f"{json.dumps(packet.primary_memory_ids, ensure_ascii=False)}。\n"
                "【三、finding 生成规则】\n"
                "1. 健康记忆只需出现在 reviewed_memory_ids 中。\n"
                "2. 对健康记忆，findings 和 operations 均保持空数组；不得生成 finding，"
                "也不得生成 no_op。\n"
                "3. 只有发现有明确正文证据支持的真实错误时，才允许生成 finding。\n"
                "4. 每个 finding 只能包含 code、severity、summary、memory_ids、evidence。\n"
                "5. finding 中不得出现 status、notes 或单数形式的 memory_id。\n"
                "6. code 和 summary 必须是非空字符串。\n"
                "7. severity 只能是 info、warning、error、critical 之一。\n"
                "8. memory_ids 必须是字符串数组，并准确列出与该问题相关的记忆 ID。\n"
                "9. evidence 必须是字符串数组，即 list[str]；数组中的每一项都必须是"
                "描述正文依据的纯文本字符串。\n"
                "10. evidence 中严禁使用对象、字典、键值对、嵌套数组或其他非字符串值。\n"
                "11. evidence 合法示例："
                '["memory_001 与 memory_002 的正文内容完全重复"]。\n'
                "12. evidence 非法示例："
                '[{"memory_id":"memory_001","reason":"exact_content_match"}]。\n'
                "13. 示例只用于说明数据类型；实际输出必须使用当前记忆包中的真实 ID 和"
                "真实正文证据，不得照抄示例内容。\n"
                "【四、operation 生成规则】\n"
                "1. operation 只能是 update、merge、relink、supersede、archive、restore、"
                "update_importance、flag_conflict、no_op 之一。\n"
                "2. 字段名必须是 operation，不得使用 operation_type。\n"
                "3. 每个 operation 项都必须包含非空字符串 reason，并填写该操作类型"
                "所需的记忆 ID 字段。\n"
                "4. 所有会修改记忆状态或内容的操作都必须包含 expected_versions。\n"
                "5. expected_versions 必须是对象，格式为 memory_id 到整数版本号的映射，"
                "不得使用数组。\n"
                "6. expected_versions 中必须覆盖该操作引用的全部现有记忆 ID。\n"
                "7. 核心设定允许update、merge、archive、relink和restore。merge仅限正文、"
                "类型、实体和字段完全一致的记录；update或archive必须有明确正文证据；无法"
                "判断冲突双方谁正确时只能使用flag_conflict。\n"
                "8. 不得为了填充输出而生成 operation；没有真实问题时 operations 必须为"
                "空数组。\n"
                "【五、强制输出格式】\n"
                "1. 只输出一个完整、可解析的 JSON 对象。\n"
                "2. 不得输出 Markdown、代码围栏、解释、前缀、后缀或 JSON 之外的任何"
                "字符。\n"
                "3. 输出必须符合 RAGMessage.payload 的结构要求。\n"
                "4. 最小合法模板为："
                f"{json.dumps(audit_payload_template, ensure_ascii=False)}。\n"
                "5. packet_id 必须与模板中的 packet_id 完全一致，不得修改。\n"
                "6. 不得添加模板和规则未允许的字段。\n"
                "【六、校验失败后的修复规则】\n"
                "如果输入 Message.metadata.validation_feedback 存在，必须优先读取其中的 "
                "error_message、missing_ids 和 repair_instruction。修复时只纠正反馈指出的"
                "格式、字段或覆盖问题，同时继续遵守以上全部规则，尤其必须确保 "
                "reviewed_memory_ids 完整、顺序正确且与 primary_memory_ids 完全一致。"
            ),
            packet.model_dump(),
            request_action="rag.model.audit.plan.request",
            response_action="rag.model.audit.plan.result",
            book_id=message.book_id or packet.book_id,
            task_id=message.task_id,
            validate_result=lambda raw: self._validate_audit_result(raw, packet),
            request_metadata=message.metadata,
        )
        if raw_result is None:
            return self._error(message, f"Memory audit failed twice: {error}")
        raw_result.setdefault("packet_id", packet.packet_id)
        try:
            result = AuditAgentResult.model_validate(raw_result)
        except Exception as exc:
            return self._error(message, f"Invalid audit result: {exc}")
        return self._ok(message, result.model_dump())

    def _handle_memory_comparison(self, message: RAGMessage) -> RAGMessage:
        try:
            packet = MemoryComparisonPacket.model_validate(message.payload)
        except Exception as exc:
            return self._error(message, f"Invalid memory comparison packet: {exc}")
        if self.model_client is None:
            return self._error(message, "No local Memory model is configured")

        result_template = {
            "batch_id": packet.batch_id,
            "query_memory_id": packet.query_memory_id,
            "reviewed_candidate_ids": packet.candidate_memory_ids,
            "findings": [],
            "operations": [],
        }
        raw_result, error = self._invoke_with_retry(
            (
                "你是小说RAG记忆逐条比较审计器。本次输入只有一条query_memory和最多"
                "九条candidates。只比较query_memory与每条候选记忆，不得比较候选记忆"
                "彼此，也不得引用当前Message.payload以外的信息或直接修改数据库。\n"
                "【覆盖要求】必须按candidate_memory_ids的输入顺序逐条完成比较，并把全部"
                "ID原样复制到reviewed_candidate_ids，不得遗漏、重复、改写、排序或添加。"
                f"本批必须严格返回：{json.dumps(packet.candidate_memory_ids, ensure_ascii=False)}。\n"
                "【判断依据】query_memory和candidates中的memory字段包含完整记忆记录；"
                "reason_codes只说明Python为何选择该候选，不是冲突证据。只有正文明确支持"
                "重复、矛盾、状态替代、错误关联或明显重要度异常时才生成finding。健康比较"
                "只需确认reviewed_candidate_ids，findings和operations保持空数组，不得"
                "生成健康finding或no_op。\n"
                "【finding格式】每项只能包含code、severity、summary、memory_ids、evidence。"
                "severity只能是info、warning、error、critical；memory_ids必须是字符串数组，"
                "只能引用本批query_memory_id和candidate_memory_ids，并且必须包含"
                "query_memory_id。evidence必须是list[str]，每项必须是纯文本字符串，严禁"
                "对象、字典、键值对或嵌套数组。合法示例："
                'evidence=["memory_A与memory_B对同一人物状态的描述互相矛盾"]；非法示例：'
                'evidence=[{"memory_id":"memory_A","reason":"conflict"}]。\n'
                "【operation格式】只允许update、merge、relink、supersede、archive、restore、"
                "update_importance、flag_conflict。每项必须使用operation字段并包含非空"
                "reason；只能引用本批记忆且必须涉及query_memory_id。所有现有记忆都必须"
                "在expected_versions对象中使用输入记录的真实整数version。核心设定允许"
                "update、merge、archive、relink和restore；merge仅限正文、类型、实体和字段"
                "完全一致的记录；update或archive必须有明确正文证据。无法判断冲突双方谁正确时"
                "只能flag_conflict。没有安全操作时保持operations为空数组。\n"
                "【输出格式】只输出一个完整JSON对象，不得输出Markdown、解释、代码围栏或"
                "JSON之外的字符。最小合法payload模板为："
                f"{json.dumps(result_template, ensure_ascii=False)}。batch_id和"
                "query_memory_id必须与模板完全一致。若metadata.validation_feedback存在，"
                "必须根据其中的error_message和repair_instruction修复后重新返回完整JSON。"
            ),
            packet.model_dump(mode="json"),
            request_action="rag.model.audit.compare.request",
            response_action="rag.model.audit.compare.result",
            book_id=message.book_id or packet.book_id,
            task_id=message.task_id,
            validate_result=lambda raw: self._validate_comparison_result(raw, packet),
            request_metadata=message.metadata,
        )
        if raw_result is None:
            return self._error(message, f"Memory comparison failed twice: {error}")
        try:
            result = MemoryComparisonResult.model_validate(raw_result)
        except Exception as exc:
            return self._error(message, f"Invalid memory comparison result: {exc}")
        return self._ok(message, result.model_dump(mode="json"))

    def _handle_reconcile(self, message: RAGMessage) -> RAGMessage:
        try:
            payload = ReconcilePayload.model_validate(message.payload)
        except Exception as exc:
            return self._error(message, f"Invalid reconciliation payload: {exc}")
        if self.model_client is None:
            return self._error(message, "No local Memory model is configured")

        required_finding_ids = [finding.finding_id for finding in payload.findings]
        reconcile_payload_template = {
            "reviewed_finding_ids": required_finding_ids,
            "findings": [],
            "operations": [],
        }

        raw_result, error = self._invoke_with_retry(
            (
                "你是小说 RAG 审计结果协调器。只允许使用输入中的 global_context、"
                "findings 和 proposed_operations，不得直接修改数据库。必须逐条协调所有"
                "输入 finding，并按输入顺序把全部 finding_id 原样复制到 "
                f"reviewed_finding_ids：{json.dumps(required_finding_ids, ensure_ascii=False)}。"
                "不得遗漏、重复、改写或添加 ID。修改操作必须携带 expected_versions；"
                "核心设定允许update、merge、archive、relink和restore，但必须保留明确证据，"
                "且merge只能处理完全一致的记录；无法判断冲突双方谁正确时使用flag_conflict。"
                "健康结果和 no_op 必须省略。finding "
                "只能包含 code、severity、summary、memory_ids、evidence；operation 必须"
                "使用 operation、reason 和 expected_versions 对象，不得使用 "
                "operation_type 或版本数组。只输出一个完整 JSON 对象，不得输出任何"
                "额外文字。RAGMessage.payload 的最小合法模板为："
                f"{json.dumps(reconcile_payload_template, ensure_ascii=False)}。"
                "如果 Message.metadata.validation_feedback 存在，必须先修复其中指出的"
                "错误。"
            ),
            payload.model_dump(),
            request_action="rag.model.audit.reconcile.request",
            response_action="rag.model.audit.reconcile.result",
            book_id=message.book_id or payload.book_id,
            task_id=message.task_id,
            validate_result=lambda raw: self._validate_reconcile_result(raw, payload),
            request_metadata=message.metadata,
        )
        if raw_result is None:
            return self._error(message, f"Memory reconciliation failed twice: {error}")
        try:
            result = ReconcileResult.model_validate(raw_result)
        except Exception as exc:
            return self._error(message, f"Invalid reconciliation result: {exc}")
        return self._ok(message, result.model_dump())

    def _handle_operation_plan(self, message: RAGMessage) -> RAGMessage:
        if self.model_client is None:
            return self._error(message, "No local Memory model is configured")
        if not message.book_id:
            return self._error(message, "Operation planning requires book_id")
        if not isinstance(message.payload.get("memories"), list):
            return self._error(message, "Operation planning requires payload.memories")

        request = RAGMessage(
            task_id=message.task_id,
            sender=self.agent_name,
            receiver="qwen_model",
            action="rag.model.operations.plan.request",
            book_id=message.book_id,
            payload=message.payload,
            metadata=message.metadata,
        )
        prompt = (
            "你是小说 RAG 数据库操作规划器。只能根据输入 Message 中提供的 memories "
            "和 instruction 规划操作。只输出一个完整的 rag.message.v1 JSON 响应；"
            "action 必须为 rag.model.operations.plan.result，sender=qwen_model，"
            "receiver=memory_agent，message_type=response，status=ok，dry_run=true，"
            "approval=pending。不得输出 Markdown 或 JSON 之外的文字。每个 operations "
            "元素必须使用 operation 和 reason。允许的 operation 为 create、update、"
            "delete、compress、merge、relink、supersede、archive、restore、"
            "update_importance、flag_conflict、no_op。已有目标必须提供 "
            "expected_versions。delete 只能软删除。compress 必须提供 source_ids、"
            "created_memory_id、summary 和全部来源版本。禁止删除、合并或压缩 "
            "canon_memory；禁止删除当前状态或开放伏笔。不得执行任何操作。compress "
            "必须严格使用类似结构：{operation:compress,source_ids:[memory_a,memory_b],"
            "created_memory_id:memory_new,expected_versions:{memory_a:1,memory_b:2},"
            "summary:text,reason:text}。压缩会自动归档来源，禁止额外生成 archive。"
            "解释文字放入 payload，可执行建议只能放入顶层 operations。若输入 "
            "Message.metadata.validation_feedback 存在，必须按其中错误说明修复输出。"
        )
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                attempt_request = self._request_for_attempt(
                    request,
                    attempt=attempt,
                    last_error=last_error,
                    response_action="rag.model.operations.plan.result",
                )
                raw = self.model_client.invoke_json(
                    prompt,
                    attempt_request.model_dump(),
                )
                payload, model_message = extract_rag_payload(
                    raw,
                    expected_action="rag.model.operations.plan.result",
                )
                if model_message is None:
                    raise ValueError("Operation planner must return a RAGMessage envelope")
                if not model_message.operations:
                    raise ValueError("Operation planner returned no validated operations")
                response = message.response(
                    sender=self.agent_name,
                    action="rag.memory.operations.plan.result",
                    payload=payload,
                    operations=model_message.operations,
                    metadata={
                        "model_message_id": model_message.message_id,
                        "model_metadata": model_message.metadata,
                    },
                )
                response.approval = "pending"
                response.dry_run = True
                return response
            except Exception as exc:
                last_error = exc
        return self._error(message, f"Operation planning failed twice: {last_error}")

    def _invoke_with_retry(
        self,
        system_prompt: str,
        payload: dict[str, Any],
        *,
        request_action: str,
        response_action: str,
        book_id: str | None,
        task_id: str,
        validate_result: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, Exception | None]:
        if self.model_client is None:
            return None, RuntimeError("No model is configured")
        request = RAGMessage(
            task_id=task_id,
            sender=self.agent_name,
            receiver="qwen_model",
            action=request_action,
            book_id=book_id,
            payload=payload,
            metadata=request_metadata or {},
        )
        envelope_prompt = (
            system_prompt
            + "\n【统一消息信封要求】只能返回一个完整的 rag.message.v1 JSON 对象，"
            + "不得使用 Markdown 代码围栏，不得在 JSON 前后添加任何解释。"
            + f"顶层 action 必须为 {response_action}，sender 必须为 qwen_model，"
            + f"receiver 必须为 {self.agent_name}，message_type 必须为 response，"
            + "status 必须为 ok。全部业务结果，包括审计建议操作，都必须放在 payload "
            + "内。顶层 operations 必须是空数组，因为本请求绝不直接修改数据库。"
            + "若输入 RAGMessage.metadata.validation_feedback 存在，表示上一次输出"
            + "校验失败；必须逐项阅读错误内容并修复，仍需返回完整响应，不能只返回"
            + "修正片段。"
        )
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                attempt_prompt = envelope_prompt
                attempt_request = self._request_for_attempt(
                    request,
                    attempt=attempt,
                    last_error=last_error,
                    response_action=response_action,
                )
                if attempt:
                    attempt_prompt += (
                        "\n这是校验失败后的格式修复重试。具体错误已经封装在输入 "
                        "RAGMessage.metadata.validation_feedback 中；必须修复后重新返回"
                        "完整 JSON。"
                    )
                raw = self.model_client.invoke_json(
                    attempt_prompt,
                    attempt_request.model_dump(),
                )
                result, _ = extract_rag_payload(raw, expected_action=response_action)
                if validate_result is not None:
                    result = validate_result(result)
                return result, None
            except Exception as exc:
                last_error = exc
        return None, last_error

    @staticmethod
    def _request_for_attempt(
        request: RAGMessage,
        *,
        attempt: int,
        last_error: Exception | None,
        response_action: str,
    ) -> RAGMessage:
        metadata = dict(request.metadata)
        metadata["attempt"] = attempt + 1
        metadata["required_response_action"] = response_action
        if last_error is not None:
            previous_feedback = metadata.get("validation_feedback")
            history = list(metadata.get("validation_feedback_history") or [])
            if isinstance(previous_feedback, dict):
                history.append(previous_feedback)
            feedback = {
                "source": "python_response_validator",
                "error_type": type(last_error).__name__,
                "error_message": str(last_error)[:2000],
                "repair_instruction": (
                    "根据错误信息修复字段、类型、必填 ID 或消息信封，然后重新返回完整 JSON"
                ),
            }
            metadata["validation_feedback"] = feedback
            metadata["validation_feedback_history"] = history[-3:]
        return request.model_copy(update={"metadata": metadata})

    @staticmethod
    def _validate_comparison_result(
        raw: dict[str, Any],
        packet: MemoryComparisonPacket,
    ) -> dict[str, Any]:
        candidate = dict(raw)
        candidate.setdefault("batch_id", packet.batch_id)
        candidate.setdefault("query_memory_id", packet.query_memory_id)
        result = MemoryComparisonResult.model_validate(candidate)
        if result.batch_id != packet.batch_id:
            raise ValueError(
                f"batch_id错误：期望{packet.batch_id}，实际{result.batch_id}"
            )
        if result.query_memory_id != packet.query_memory_id:
            raise ValueError(
                "query_memory_id与当前比较批次不一致"
            )
        MemoryAgent._validate_acknowledged_ids(
            result.reviewed_candidate_ids,
            packet.candidate_memory_ids,
            "reviewed_candidate_ids",
        )
        if result.reviewed_candidate_ids != packet.candidate_memory_ids:
            missing = [
                memory_id
                for memory_id in packet.candidate_memory_ids
                if memory_id not in result.reviewed_candidate_ids
            ]
            raise ValueError(f"reviewed_candidate_ids遗漏候选记忆：{missing}")

        allowed_ids = {packet.query_memory_id, *packet.candidate_memory_ids}
        for finding in result.findings:
            finding_ids = set(finding.memory_ids)
            if packet.query_memory_id not in finding_ids:
                raise ValueError("每个finding都必须包含query_memory_id")
            if not finding_ids <= allowed_ids:
                raise ValueError("finding引用了当前比较批次以外的记忆")
        for operation in result.operations:
            referenced_ids = set(operation.referenced_memory_ids())
            if packet.query_memory_id not in referenced_ids:
                raise ValueError("每个operation都必须涉及query_memory_id")
            if not referenced_ids <= allowed_ids:
                raise ValueError("operation引用了当前比较批次以外的记忆")
        return result.model_dump(mode="json")

    @staticmethod
    def _validate_audit_result(
        raw: dict[str, Any],
        packet: AuditPacket,
    ) -> dict[str, Any]:
        candidate = dict(raw)
        candidate.setdefault("packet_id", packet.packet_id)
        result = AuditAgentResult.model_validate(candidate)
        if result.packet_id != packet.packet_id:
            raise ValueError(
                f"packet_id错误：期望{packet.packet_id}，实际{result.packet_id}"
            )
        MemoryAgent._validate_acknowledged_ids(
            result.reviewed_memory_ids,
            packet.primary_memory_ids,
            "reviewed_memory_ids",
        )
        return result.model_dump(mode="json")

    @staticmethod
    def _validate_reconcile_result(
        raw: dict[str, Any],
        payload: ReconcilePayload,
    ) -> dict[str, Any]:
        result = ReconcileResult.model_validate(raw)
        MemoryAgent._validate_acknowledged_ids(
            result.reviewed_finding_ids,
            [finding.finding_id for finding in payload.findings],
            "reviewed_finding_ids",
        )
        return result.model_dump(mode="json")

    @staticmethod
    def _validate_cross_result(
        raw: dict[str, Any],
        packet: CrossAuditPacket,
    ) -> dict[str, Any]:
        candidate = dict(raw)
        candidate.setdefault("packet_id", packet.packet_id)
        result = CrossAuditResult.model_validate(candidate)
        if result.packet_id != packet.packet_id:
            raise ValueError(
                f"packet_id错误：期望{packet.packet_id}，实际{result.packet_id}"
            )
        MemoryAgent._validate_acknowledged_ids(
            result.reviewed_candidate_ids,
            [candidate.candidate_id for candidate in packet.candidates],
            "reviewed_candidate_ids",
        )
        return result.model_dump(mode="json")

    @staticmethod
    def _validate_acknowledged_ids(
        actual_ids: list[str],
        expected_ids: list[str],
        field_name: str,
    ) -> None:
        duplicates = [
            item
            for index, item in enumerate(actual_ids)
            if item in actual_ids[:index]
        ]
        unexpected = sorted(set(actual_ids) - set(expected_ids))
        actual_set = set(actual_ids)
        expected_subsequence = [
            item for item in expected_ids if item in actual_set
        ]
        wrong_order = actual_ids != expected_subsequence
        if duplicates or unexpected or wrong_order:
            raise ValueError(
                f"{field_name}格式错误：重复ID={duplicates}，非本包ID={unexpected}，"
                f"顺序错误={wrong_order}"
            )

    def _ok(self, request: RAGMessage, payload: dict[str, Any]) -> RAGMessage:
        return request.response(
            sender=self.agent_name,
            status="ok",
            payload=payload,
        )

    def _error(self, request: RAGMessage, error: str) -> RAGMessage:
        return request.response(
            sender=self.agent_name,
            status="error",
            error=error,
        )
