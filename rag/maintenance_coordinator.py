from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .audit_ledger import AuditLedger
from .audit_scanner import AuditScanner, ScanResult
from .book_lock import BookDatabaseLock
from .config import RagConfig
from .entity_partition import EntityGraphPartitioner
from .maintenance_schemas import (
    AuditAgentResult,
    AuditCoverage,
    AuditFinding,
    AuditOperation,
    AuditPacket,
    AuditRunResult,
    AuditScope,
    CrossAuditPacket,
    CrossAuditResult,
    MemoryComparisonPacket,
    MemoryComparisonResult,
    PatchPlan,
    ReconcilePayload,
    ReconcileResult,
    maintenance_id,
)
from .memory_agent import MemoryAgent
from .rag_message import RAGMessage
from .patch_executor import PatchExecutor, PatchValidationError
from .repository import BookRepository
from .scoped_audit import ScopedComparisonPlanner


BLOCKING_CODES = {
    "invalid_memory_record",
    "cross_book_record",
    "store_type_mismatch",
    "duplicate_memory_id_across_stores",
    "missing_index_head",
    "index_store_mismatch",
}


class MaintenanceCoordinator:
    def __init__(
        self,
        config: RagConfig,
        memory_agent: MemoryAgent,
        *,
        scanner: AuditScanner | None = None,
        partitioner: EntityGraphPartitioner | None = None,
        scoped_planner: ScopedComparisonPlanner | None = None,
        executor: PatchExecutor | None = None,
    ) -> None:
        self.config = config
        self.memory_agent = memory_agent
        self.scanner = scanner or AuditScanner()
        self.partitioner = partitioner or EntityGraphPartitioner()
        self.scoped_planner = scoped_planner or ScopedComparisonPlanner()
        self.executor = executor or PatchExecutor(config)

    def run(
        self,
        book_id: str,
        *,
        apply: bool = False,
        scope: AuditScope | dict[str, Any] | None = None,
    ) -> AuditRunResult:
        with BookDatabaseLock(self.config, book_id):
            if scope is not None:
                return self._run_scoped_locked(
                    book_id,
                    AuditScope.model_validate(scope),
                    apply=apply,
                )
            return self._run_locked(book_id, apply=apply)

    def _run_locked(self, book_id: str, *, apply: bool = False) -> AuditRunResult:
        run_id = maintenance_id("audit")
        ledger = AuditLedger(self.config, book_id, run_id)
        repository = BookRepository(self.config, book_id)
        scan = self.scanner.scan(repository)
        owners = self.partitioner.assign_owners(scan)
        packets = self.partitioner.build_packets(scan, owners)
        cross_packets = self.partitioner.build_cross_check_packets(scan, owners)
        ledger.write("scan", self._scan_payload(scan))
        ledger.write("ownership", owners)
        ledger.write("packets", packets)
        ledger.write("cross_packets", cross_packets)

        reports: list[AuditAgentResult] = []
        for index, packet in enumerate(packets, start=1):
            report = self._audit_packet(packet, run_id)
            reports.append(report)
            ledger.write(f"packet_report_{index:04d}", report)
        ledger.write("packet_reports", reports)

        cross_reports: list[CrossAuditResult] = []
        for index, packet in enumerate(cross_packets, start=1):
            report = self._audit_cross_packet(packet, run_id)
            cross_reports.append(report)
            ledger.write(f"cross_report_{index:04d}", report)
        ledger.write("cross_reports", cross_reports)

        findings = [finding for report in reports for finding in report.findings] + [
            finding for report in cross_reports for finding in report.findings
        ]
        model_operations = [
            operation for report in reports for operation in report.operations
        ] + [operation for report in cross_reports for operation in report.operations]
        reconciliation = self._reconcile(
            book_id,
            packets[0].global_context if packets else {},
            findings,
            model_operations,
            run_id,
        )
        ledger.write("reconciliation", reconciliation)

        coverage = self._coverage(scan, packets, reports)
        deterministic_operations = [
            issue.deterministic_operation
            for issue in scan.issues
            if issue.deterministic_operation is not None
        ]
        combined_operations = self._combine_operations(
            deterministic_operations,
            model_operations + reconciliation.operations,
        )
        blocking_issue_ids = [
            issue.issue_id
            for issue in scan.issues
            if issue.code in BLOCKING_CODES and issue.deterministic_operation is None
        ]
        expected_candidate_ids = {
            candidate.candidate_id
            for packet in cross_packets
            for candidate in packet.candidates
        }
        reviewed_candidate_ids = {
            candidate_id
            for report in cross_reports
            for candidate_id in report.reviewed_candidate_ids
        }
        missing_candidate_ids = sorted(
            expected_candidate_ids - reviewed_candidate_ids
        )
        if missing_candidate_ids:
            blocking_issue_ids.append("cross_check_incomplete")
        ledger.write(
            "cross_coverage",
            {
                "candidate_ids": sorted(expected_candidate_ids),
                "reviewed_candidate_ids": sorted(reviewed_candidate_ids),
                "missing_candidate_ids": missing_candidate_ids,
                "complete": not missing_candidate_ids,
            },
        )
        expected_finding_ids = {finding.finding_id for finding in findings}
        reviewed_finding_ids = set(reconciliation.reviewed_finding_ids)
        if expected_finding_ids - reviewed_finding_ids:
            blocking_issue_ids.append("reconciliation_incomplete")

        plan = PatchPlan(
            book_id=book_id,
            run_id=run_id,
            operations=combined_operations,
            coverage=coverage,
            blocking_issue_ids=blocking_issue_ids,
        )
        ledger.write("coverage", coverage)

        snapshot_id: str | None = None
        applied = False
        validation_errors: list[str] = []
        try:
            self.executor.dry_run(plan)
        except PatchValidationError as exc:
            validation_errors.append(str(exc))
            if "patch_validation_failed" not in plan.blocking_issue_ids:
                plan.blocking_issue_ids.append("patch_validation_failed")
        ledger.write("patch_plan", plan)
        if apply:
            try:
                snapshot_id = self.executor.apply(plan)
                applied = True
            except Exception as exc:
                ledger.write(
                    "failure",
                    {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "snapshot_id": getattr(exc, "snapshot_id", None),
                    },
                )
                raise

        result = AuditRunResult(
            run_id=run_id,
            book_id=book_id,
            dry_run=not apply,
            applied=applied,
            snapshot_id=snapshot_id,
            artifact_dir=str(ledger.root.resolve()),
            deterministic_issue_count=len(scan.issues),
            model_finding_count=len(findings) + len(reconciliation.findings),
            operation_count=len(combined_operations),
            coverage=coverage,
            semantic_candidate_count=len(expected_candidate_ids),
            semantic_candidate_reviewed_count=len(reviewed_candidate_ids),
            semantic_candidate_complete=not missing_candidate_ids,
            blocking_issue_ids=plan.blocking_issue_ids,
            validation_errors=validation_errors,
        )
        ledger.write("result", result)
        return result

    def _run_scoped_locked(
        self,
        book_id: str,
        scope: AuditScope,
        *,
        apply: bool = False,
    ) -> AuditRunResult:
        run_id = maintenance_id("audit")
        ledger = AuditLedger(self.config, book_id, run_id)
        repository = BookRepository(self.config, book_id)
        scan = self.scanner.scan(repository)
        scoped_plan = self.scoped_planner.build(scan, scope)
        owners = self.partitioner.assign_owners(scan)
        global_context = self.partitioner.build_global_context(scan, owners)

        ledger.write("scope", scope)
        ledger.write("scan", self._scan_payload(scan))
        ledger.write("comparison_batches", scoped_plan.batches)

        reports: list[MemoryComparisonResult] = []
        failed_query_ids: set[str] = set()
        validation_errors: list[str] = []
        for index, batch in enumerate(scoped_plan.batches, start=1):
            try:
                report = self._audit_comparison_batch(batch, run_id)
            except Exception as exc:
                failed_query_ids.add(batch.query_memory_id)
                message = f"比较批次{index}/{len(scoped_plan.batches)}失败：{exc}"
                validation_errors.append(message)
                ledger.write(
                    f"comparison_failure_{index:04d}",
                    {
                        "batch_id": batch.batch_id,
                        "query_memory_id": batch.query_memory_id,
                        "candidate_memory_ids": batch.candidate_memory_ids,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                continue
            reports.append(report)
            ledger.write(f"comparison_report_{index:04d}", report)
        ledger.write("comparison_reports", reports)

        findings = [finding for report in reports for finding in report.findings]
        model_operations = [
            operation for report in reports for operation in report.operations
        ]
        reconciliation = ReconcileResult()
        reconciliation_failed = False
        if findings or model_operations:
            try:
                reconciliation = self._reconcile(
                    book_id,
                    global_context,
                    findings,
                    model_operations,
                    run_id,
                )
            except Exception as exc:
                reconciliation_failed = True
                validation_errors.append(f"审计结果协调失败：{exc}")
                ledger.write(
                    "reconciliation_failure",
                    {"error_type": type(exc).__name__, "error": str(exc)},
                )
        ledger.write("reconciliation", reconciliation)

        reviewed_query_ids = [
            memory_id
            for memory_id in scoped_plan.query_memory_ids
            if memory_id not in failed_query_ids
        ]
        coverage = AuditCoverage(
            total_memory_ids=scoped_plan.query_memory_ids,
            assigned_memory_ids=scoped_plan.query_memory_ids,
            reviewed_memory_ids=reviewed_query_ids,
            unreviewed_memory_ids=[
                memory_id
                for memory_id in scoped_plan.query_memory_ids
                if memory_id in failed_query_ids
            ],
        )
        expected_comparisons = scoped_plan.comparison_pair_count
        reviewed_comparisons = sum(
            len(report.reviewed_candidate_ids) for report in reports
        )
        comparisons_complete = (
            not failed_query_ids
            and reviewed_comparisons == expected_comparisons
        )

        deterministic_operations = [
            issue.deterministic_operation
            for issue in scoped_plan.deterministic_issues
            if issue.deterministic_operation is not None
        ]
        combined_operations = self._combine_operations(
            deterministic_operations,
            model_operations + reconciliation.operations,
        )
        blocking_issue_ids = [
            issue.issue_id
            for issue in scoped_plan.deterministic_issues
            if issue.code in BLOCKING_CODES and issue.deterministic_operation is None
        ]
        if not comparisons_complete:
            blocking_issue_ids.append("comparison_incomplete")
        if reconciliation_failed:
            blocking_issue_ids.append("reconciliation_failed")
        expected_finding_ids = {finding.finding_id for finding in findings}
        reviewed_finding_ids = set(reconciliation.reviewed_finding_ids)
        if expected_finding_ids - reviewed_finding_ids:
            blocking_issue_ids.append("reconciliation_incomplete")

        patch_plan = PatchPlan(
            book_id=book_id,
            run_id=run_id,
            scope=scope,
            operations=combined_operations,
            coverage=coverage,
            blocking_issue_ids=list(dict.fromkeys(blocking_issue_ids)),
        )
        ledger.write("coverage", coverage)

        snapshot_id: str | None = None
        applied = False
        try:
            self.executor.dry_run(patch_plan)
        except PatchValidationError as exc:
            validation_errors.append(str(exc))
            if "patch_validation_failed" not in patch_plan.blocking_issue_ids:
                patch_plan.blocking_issue_ids.append("patch_validation_failed")
        ledger.write("patch_plan", patch_plan)
        if apply:
            try:
                snapshot_id = self.executor.apply(patch_plan)
                applied = True
            except Exception as exc:
                ledger.write(
                    "failure",
                    {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "snapshot_id": getattr(exc, "snapshot_id", None),
                    },
                )
                raise

        result = AuditRunResult(
            run_id=run_id,
            book_id=book_id,
            scope=scope,
            dry_run=not apply,
            applied=applied,
            snapshot_id=snapshot_id,
            artifact_dir=str(ledger.root.resolve()),
            deterministic_issue_count=len(scoped_plan.deterministic_issues),
            model_finding_count=len(findings) + len(reconciliation.findings),
            operation_count=len(combined_operations),
            coverage=coverage,
            semantic_candidate_count=expected_comparisons,
            semantic_candidate_reviewed_count=reviewed_comparisons,
            semantic_candidate_complete=comparisons_complete,
            comparison_batch_count=len(scoped_plan.batches),
            blocking_issue_ids=patch_plan.blocking_issue_ids,
            validation_errors=validation_errors,
        )
        ledger.write("result", result)
        return result

    def rollback(self, book_id: str, snapshot_id: str) -> None:
        with BookDatabaseLock(self.config, book_id):
            self.executor.rollback(book_id, snapshot_id)

    def apply_saved_plan(self, book_id: str, run_id: str) -> AuditRunResult:
        clean_run_id = run_id.strip()
        if not clean_run_id or Path(clean_run_id).name != clean_run_id:
            raise ValueError("run_id格式错误")
        root = self.config.root_dir / ".maintenance_runs" / book_id / clean_run_id
        plan_path = root / "patch_plan.json"
        result_path = root / "result.json"
        if not plan_path.is_file() or not result_path.is_file():
            raise FileNotFoundError(f"未找到审计计划: {clean_run_id}")
        plan = PatchPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
        previous = AuditRunResult.model_validate_json(
            result_path.read_text(encoding="utf-8")
        )
        if plan.book_id != book_id or previous.book_id != book_id:
            raise ValueError("保存的审计计划与book_id不匹配")
        with BookDatabaseLock(self.config, book_id):
            snapshot_id = self.executor.apply(plan)
        result = previous.model_copy(
            update={
                "dry_run": False,
                "applied": True,
                "snapshot_id": snapshot_id,
            }
        )
        (root / "applied_result.json").write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return result

    def _audit_comparison_batch(
        self,
        batch: MemoryComparisonPacket,
        run_id: str,
    ) -> MemoryComparisonResult:
        return MemoryComparisonResult.model_validate(
            self._send(
                action="memory.audit.compare",
                payload=batch.model_dump(mode="json"),
                task_id=run_id,
            )
        )

    def _audit_packet(self, packet: AuditPacket, run_id: str) -> AuditAgentResult:
        first = self._send(
            action="memory.audit.plan",
            payload=packet.model_dump(mode="json"),
            task_id=run_id,
        )
        result = AuditAgentResult.model_validate(first)
        missing = sorted(set(packet.primary_memory_ids) - set(result.reviewed_memory_ids))
        if not missing:
            return result

        retry = packet.model_copy(
            update={
                "primary_memory_ids": missing,
                "retry_for_memory_ids": missing,
            }
        )
        second = AuditAgentResult.model_validate(
            self._send(
                action="memory.audit.plan",
                payload=retry.model_dump(mode="json"),
                task_id=run_id,
                metadata={
                    "validation_feedback": {
                        "source": "maintenance_coordinator",
                        "error_type": "audit_coverage_omission",
                        "error_message": (
                            "上一次审计响应没有确认全部 primary_memory_ids"
                        ),
                        "missing_ids": missing,
                        "repair_instruction": (
                            "逐条审阅这些 ID，并将它们按原顺序完整写入 "
                            "payload.reviewed_memory_ids"
                        ),
                    }
                },
            )
        )
        return AuditAgentResult(
            packet_id=packet.packet_id,
            reviewed_memory_ids=list(
                dict.fromkeys(result.reviewed_memory_ids + second.reviewed_memory_ids)
            ),
            findings=result.findings + second.findings,
            operations=result.operations + second.operations,
        )

    def _audit_cross_packet(
        self,
        packet: CrossAuditPacket,
        run_id: str,
    ) -> CrossAuditResult:
        first = CrossAuditResult.model_validate(
            self._send(
                action="memory.audit.cross_check",
                payload=packet.model_dump(mode="json"),
                task_id=run_id,
            )
        )
        expected = {candidate.candidate_id for candidate in packet.candidates}
        missing = sorted(expected - set(first.reviewed_candidate_ids))
        if not missing:
            return first
        missing_set = set(missing)
        retry_candidates = [
            candidate
            for candidate in packet.candidates
            if candidate.candidate_id in missing_set
        ]
        retry_memory_ids = {
            memory_id
            for candidate in retry_candidates
            for memory_id in candidate.memory_ids
        }
        retry = packet.model_copy(
            update={
                "candidates": retry_candidates,
                "memories": [
                    memory
                    for memory in packet.memories
                    if str(memory.get("memory_id") or "") in retry_memory_ids
                ],
            }
        )
        second = CrossAuditResult.model_validate(
            self._send(
                action="memory.audit.cross_check",
                payload=retry.model_dump(mode="json"),
                task_id=run_id,
                metadata={
                    "validation_feedback": {
                        "source": "maintenance_coordinator",
                        "error_type": "cross_candidate_coverage_omission",
                        "error_message": (
                            "上一次跨索引审计没有确认全部 candidate_id"
                        ),
                        "missing_ids": missing,
                        "repair_instruction": (
                            "逐一比较这些候选，并将 candidate_id 按原顺序完整写入 "
                            "payload.reviewed_candidate_ids"
                        ),
                    }
                },
            )
        )
        return CrossAuditResult(
            packet_id=packet.packet_id,
            reviewed_candidate_ids=list(
                dict.fromkeys(
                    first.reviewed_candidate_ids + second.reviewed_candidate_ids
                )
            ),
            findings=first.findings + second.findings,
            operations=first.operations + second.operations,
        )

    def _reconcile(
        self,
        book_id: str,
        global_context: dict[str, Any],
        findings: list[AuditFinding],
        operations: list[AuditOperation],
        run_id: str,
    ) -> ReconcileResult:
        if not findings and not operations:
            return ReconcileResult()
        components = self._finding_components(findings)
        if not components:
            components = [[]]
        remaining_operations = list(operations)
        results: list[ReconcileResult] = []
        for component in components:
            component_memory_ids = {
                memory_id for finding in component for memory_id in finding.memory_ids
            }
            component_operations = [
                operation
                for operation in remaining_operations
                if set(operation.referenced_memory_ids()) & component_memory_ids
            ]
            selected_ids = {operation.operation_id for operation in component_operations}
            remaining_operations = [
                operation
                for operation in remaining_operations
                if operation.operation_id not in selected_ids
            ]
            payload = ReconcilePayload(
                book_id=book_id,
                global_context=global_context,
                findings=component,
                proposed_operations=component_operations,
            )
            results.append(
                ReconcileResult.model_validate(
                    self._send(
                        action="memory.audit.reconcile",
                        payload=payload.model_dump(mode="json"),
                        task_id=run_id,
                    )
                )
            )
        if remaining_operations:
            payload = ReconcilePayload(
                book_id=book_id,
                global_context=global_context,
                findings=[],
                proposed_operations=remaining_operations,
            )
            results.append(
                ReconcileResult.model_validate(
                    self._send(
                        action="memory.audit.reconcile",
                        payload=payload.model_dump(mode="json"),
                        task_id=run_id,
                    )
                )
            )
        return ReconcileResult(
            reviewed_finding_ids=list(
                dict.fromkeys(
                    finding_id
                    for result in results
                    for finding_id in result.reviewed_finding_ids
                )
            ),
            findings=[finding for result in results for finding in result.findings],
            operations=[operation for result in results for operation in result.operations],
        )

    @staticmethod
    def _finding_components(
        findings: list[AuditFinding],
    ) -> list[list[AuditFinding]]:
        if not findings:
            return []
        parents = list(range(len(findings)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        owner_by_memory: dict[str, int] = {}
        for index, finding in enumerate(findings):
            for memory_id in finding.memory_ids:
                if memory_id in owner_by_memory:
                    union(index, owner_by_memory[memory_id])
                else:
                    owner_by_memory[memory_id] = index

        groups: dict[int, list[AuditFinding]] = {}
        for index, finding in enumerate(findings):
            groups.setdefault(find(index), []).append(finding)
        return list(groups.values())

    def _send(
        self,
        *,
        action: str,
        payload: dict[str, Any],
        task_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = RAGMessage(
            task_id=task_id,
            sender="maintenance_coordinator",
            receiver=self.memory_agent.agent_name,
            action=f"rag.{action}" if action.startswith("memory.") else action,
            book_id=str(payload.get("book_id") or "") or None,
            payload=payload,
            metadata=metadata or {},
        )
        response = self.memory_agent.handle_message(request)
        if response.status != "ok":
            raise RuntimeError(response.error or f"{action}执行失败")
        return response.payload or {}

    @staticmethod
    def _coverage(
        scan: ScanResult,
        packets: list[AuditPacket],
        reports: list[AuditAgentResult],
    ) -> AuditCoverage:
        assigned = [memory_id for packet in packets for memory_id in packet.primary_memory_ids]
        reviewed = [
            memory_id for report in reports for memory_id in report.reviewed_memory_ids
        ]
        total = set(scan.all_memory_ids)
        assigned_set = set(assigned)
        reviewed_set = set(reviewed)
        counts = Counter(assigned)
        return AuditCoverage(
            total_memory_ids=sorted(total),
            assigned_memory_ids=sorted(assigned_set),
            reviewed_memory_ids=sorted(reviewed_set),
            uncovered_memory_ids=sorted(total - assigned_set),
            unreviewed_memory_ids=sorted(total - reviewed_set),
            duplicate_primary_memory_ids=sorted(
                memory_id for memory_id, count in counts.items() if count > 1
            ),
        )

    @staticmethod
    def _combine_operations(
        deterministic: list[AuditOperation],
        model_operations: list[AuditOperation],
    ) -> list[AuditOperation]:
        result: list[AuditOperation] = []
        signatures: set[str] = set()
        locked_ids: set[str] = set()
        for operation in deterministic:
            signature_payload = operation.model_dump(
                mode="json",
                exclude={"operation_id", "reason", "confidence"},
            )
            signature = json.dumps(signature_payload, sort_keys=True, ensure_ascii=False)
            if signature in signatures:
                continue
            referenced = set(operation.referenced_memory_ids())
            result.append(operation)
            signatures.add(signature)
            if operation.operation not in {
                "no_op",
                "flag_conflict",
                "remove_orphan_link",
            }:
                locked_ids.update(referenced)
        for operation in model_operations:
            signature_payload = operation.model_dump(
                mode="json",
                exclude={"operation_id", "reason", "confidence"},
            )
            signature = json.dumps(signature_payload, sort_keys=True, ensure_ascii=False)
            referenced = set(operation.referenced_memory_ids())
            if signature in signatures or referenced & locked_ids:
                continue
            result.append(operation)
            signatures.add(signature)
        return result

    @staticmethod
    def _scan_payload(scan: ScanResult) -> dict[str, Any]:
        return {
            "book_id": scan.book_id,
            "all_memory_ids": scan.all_memory_ids,
            "heads": scan.heads,
            "links": scan.links,
            "issues": [issue.model_dump(mode="json") for issue in scan.issues],
        }
