from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from rag.audit_scanner import AuditScanner, ScanResult
from rag.config import RagConfig
from rag.entity_partition import EntityGraphPartitioner
from rag.repository import BookRepository
from rag.retriever import estimate_tokens
from rag.schemas import AtomicMemory, StoreType


@dataclass(slots=True)
class TestReporter:
    passed: int = 0
    failed: int = 0
    failures: list[str] = field(default_factory=list)

    def check(self, condition: bool, name: str, detail: str = "") -> None:
        if condition:
            self.passed += 1
            print(f"[PASS] {name}")
            return
        self.failed += 1
        message = f"{name}: {detail}" if detail else name
        self.failures.append(message)
        print(f"[FAIL] {message}")


def make_memory(
    memory_id: str,
    book_id: str,
    store_type: StoreType,
    memory_type: str,
    content: str,
    *,
    chapter: int,
    character_ids: Iterable[str] = (),
    item_ids: Iterable[str] = (),
    event_ids: Iterable[str] = (),
    entity_name: str | None = None,
    field_name: str | None = None,
    is_current: bool = True,
    hook_status: str | None = None,
    importance: float = 0.8,
) -> AtomicMemory:
    return AtomicMemory(
        memory_id=memory_id,
        book_id=book_id,
        store_type=store_type,
        memory_type=memory_type,
        content=content,
        character_ids=list(character_ids),
        item_ids=list(item_ids),
        event_ids=list(event_ids),
        source_chapter=chapter,
        last_mentioned_chapter=chapter,
        raw_importance=importance,
        effective_importance=importance,
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        entity_name=entity_name,
        field=field_name,
        is_current=is_current,
        hook_status=hook_status,
    )


def save_and_link(
    repository: BookRepository,
    memory: AtomicMemory,
    links: list[tuple[str, str]],
) -> None:
    repository.store(memory.store_type).save(memory)
    for head_id, role in links:
        repository.index.link(
            head_id,
            memory.memory_id,
            memory.store_type,
            role,
            memory.source_chapter,
        )


def build_fixture(
    root: Path,
    book_id: str,
) -> tuple[BookRepository, dict[str, set[str]], dict[str, str]]:
    repository = BookRepository(RagConfig(root), book_id)
    index = repository.index

    lin = index.resolve_or_create("character", "林舟")
    gu = index.resolve_or_create("character", "顾清")
    shen = index.resolve_or_create("character", "沈默")
    key = index.resolve_or_create("item", "青铜钥匙")
    ruins = index.resolve_or_create("event", "遗迹开启")

    index.add_alias(lin, "阿舟")
    index.add_alias(gu, "清姐")
    index.add_alias(shen, "医师")
    index.add_alias(key, "铜钥匙")
    index.add_alias(ruins, "石门开启")

    expected: dict[str, set[str]] = defaultdict(set)
    names = {
        lin: "林舟",
        gu: "顾清",
        shen: "沈默",
        key: "青铜钥匙",
        ruins: "遗迹开启",
    }

    def add(memory: AtomicMemory, links: list[tuple[str, str]]) -> None:
        save_and_link(repository, memory, links)
        for head_id, _ in links:
            expected[head_id].add(memory.memory_id)

    add(
        make_memory(
            "mem_canon_key_rule",
            book_id,
            "canon_memory",
            "world_rule",
            "青铜钥匙是开启地下遗迹石门的唯一凭证。",
            chapter=0,
            item_ids=[key],
            event_ids=[ruins],
            importance=0.95,
        ),
        [(key, "subject"), (ruins, "related")],
    )
    add(
        make_memory(
            "mem_lin_identity",
            book_id,
            "chapter_memory",
            "character_profile",
            "林舟是负责调查遗迹的年轻测绘师。",
            chapter=1,
            character_ids=[lin],
            entity_name="林舟",
        ),
        [(lin, "subject")],
    )
    add(
        make_memory(
            "mem_lin_location",
            book_id,
            "state_timeline_memory",
            "character_state",
            "林舟当前位于北部遗迹入口。",
            chapter=3,
            character_ids=[lin],
            entity_name="林舟",
            field_name="location",
        ),
        [(lin, "subject")],
    )
    add(
        make_memory(
            "mem_lin_gu_relation",
            book_id,
            "relation_hook_memory",
            "relation",
            "林舟与顾清约定共同调查遗迹。",
            chapter=2,
            character_ids=[lin, gu],
        ),
        [(lin, "subject"), (gu, "related")],
    )
    add(
        make_memory(
            "mem_key_found",
            book_id,
            "chapter_memory",
            "event",
            "林舟在旧钟楼中发现青铜钥匙。",
            chapter=2,
            character_ids=[lin],
            item_ids=[key],
        ),
        [(lin, "subject"), (key, "object")],
    )
    add(
        make_memory(
            "mem_key_transfer",
            book_id,
            "chapter_memory",
            "event",
            "林舟把青铜钥匙交给顾清保管。",
            chapter=3,
            character_ids=[lin, gu],
            item_ids=[key],
        ),
        [(lin, "source"), (gu, "target"), (key, "object")],
    )
    add(
        make_memory(
            "mem_ruins_opened",
            book_id,
            "chapter_memory",
            "event",
            "林舟和顾清使用钥匙开启了遗迹外层石门。",
            chapter=4,
            character_ids=[lin, gu],
            item_ids=[key],
            event_ids=[ruins],
        ),
        [(lin, "participant"), (gu, "participant"), (key, "tool"), (ruins, "subject")],
    )
    add(
        make_memory(
            "mem_ruins_hook",
            book_id,
            "relation_hook_memory",
            "foreshadowing_open",
            "顾清发现遗迹深处还有一扇没有钥匙孔的门。",
            chapter=4,
            character_ids=[gu],
            event_ids=[ruins],
            hook_status="open",
            importance=0.9,
        ),
        [(gu, "observer"), (ruins, "related")],
    )
    add(
        make_memory(
            "mem_shen_unrelated",
            book_id,
            "state_timeline_memory",
            "character_state",
            "沈默留在南城诊所，没有参加遗迹调查。",
            chapter=4,
            character_ids=[shen],
            entity_name="沈默",
            field_name="location",
        ),
        [(shen, "subject")],
    )
    add(
        make_memory(
            "mem_unlinked_summary",
            book_id,
            "chapter_memory",
            "chapter_summary",
            "第四章结束时，遗迹外门已经开启。",
            chapter=4,
            importance=0.7,
        ),
        [],
    )
    return repository, dict(expected), names


def memory_ids_from_index(
    repository: BookRepository,
    head_ids: list[str],
) -> tuple[set[str], set[str]]:
    links = repository.index.find_memory_links(head_ids)
    linked_ids = {memory_id for memory_id, _ in links}
    loaded_ids = {
        memory.memory_id for memory in repository.memories_from_links(links)
    }
    return linked_ids, loaded_ids


def verify_index_fixture(
    repository: BookRepository,
    expected: dict[str, set[str]],
    names: dict[str, str],
    reporter: TestReporter,
) -> None:
    print("\n=== 1. 按实体索引逐项提取 ===")
    for head_id, expected_ids in sorted(expected.items(), key=lambda item: names[item[0]]):
        linked_ids, loaded_ids = memory_ids_from_index(repository, [head_id])
        label = names[head_id]
        print(f"  {label}: {sorted(loaded_ids)}")
        reporter.check(
            linked_ids == expected_ids,
            f"索引链接完整：{label}",
            f"expected={sorted(expected_ids)}, actual={sorted(linked_ids)}",
        )
        reporter.check(
            loaded_ids == expected_ids,
            f"链接可加载真实记忆：{label}",
            f"expected={sorted(expected_ids)}, actual={sorted(loaded_ids)}",
        )

    lin = repository.index.resolve("character", "阿舟")
    key = repository.index.resolve("item", "铜钥匙")
    ruins = repository.index.resolve("event", "石门开启")
    reporter.check(lin is not None, "人物别名“阿舟”可解析")
    reporter.check(key is not None, "物品别名“铜钥匙”可解析")
    reporter.check(ruins is not None, "事件别名“石门开启”可解析")

    query = "阿舟握住铜钥匙，准备等待石门开启。"
    query_heads = repository.index.heads_in_text(query)
    expected_heads = {head_id for head_id in (lin, key, ruins) if head_id}
    reporter.check(
        set(query_heads) == expected_heads,
        "从自然语言查询识别全部别名索引",
        f"expected={sorted(expected_heads)}, actual={sorted(query_heads)}",
    )

    expected_union = set().union(*(expected[head_id] for head_id in expected_heads))
    linked_ids, loaded_ids = memory_ids_from_index(repository, query_heads)
    print(f"\n  查询：{query}")
    print(f"  命中实体：{[names[head_id] for head_id in query_heads]}")
    print(f"  提取记忆：{sorted(loaded_ids)}")
    reporter.check(
        linked_ids == expected_union,
        "多索引查询返回全部相关记忆的并集",
        f"expected={sorted(expected_union)}, actual={sorted(linked_ids)}",
    )
    reporter.check(
        loaded_ids == linked_ids,
        "多索引链接没有悬空记录",
        f"linked={sorted(linked_ids)}, loaded={sorted(loaded_ids)}",
    )
    reporter.check(
        "mem_shen_unrelated" not in loaded_ids,
        "多索引查询没有混入无关人物记忆",
    )


def verify_existing_index(
    repository: BookRepository,
    scan: ScanResult,
    reporter: TestReporter,
    verbose: bool,
) -> None:
    print("\n=== 1. 检查现有数据库的全部索引链接 ===")
    links_by_head: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for link in scan.links:
        links_by_head[str(link["head_id"])].append(
            (str(link["memory_id"]), str(link["store_type"]))
        )

    total_links = 0
    for head in scan.heads:
        head_id = str(head["head_id"])
        expected_links = links_by_head.get(head_id, [])
        public_links = repository.index.find_memory_links([head_id])
        expected_pairs = set(expected_links)
        public_pairs = set(public_links)
        loaded = repository.memories_from_links(public_links)
        loaded_ids = {memory.memory_id for memory in loaded}
        linked_ids = {memory_id for memory_id, _ in public_links}
        total_links += len(public_links)

        label = str(head.get("canonical_name", head_id))
        reporter.check(
            public_pairs == expected_pairs,
            f"索引查询结果与索引表一致：{label}",
            f"table={sorted(expected_pairs)}, api={sorted(public_pairs)}",
        )
        reporter.check(
            loaded_ids == linked_ids,
            f"索引指向的记忆均可加载：{label}",
            f"linked={sorted(linked_ids)}, loaded={sorted(loaded_ids)}",
        )
        if verbose:
            print(f"  {label}: {sorted(loaded_ids)}")

    reporter.check(bool(scan.heads), "现有书库至少包含一个实体索引头")
    reporter.check(total_links > 0, "现有书库至少包含一条记忆索引链接")
    print(f"  实体头数量：{len(scan.heads)}")
    print(f"  索引链接数量：{total_links}")


def verify_partitioning(
    scan: ScanResult,
    reporter: TestReporter,
    *,
    max_primary: int,
    max_packet_tokens: int,
    max_global_tokens: int,
    verbose: bool,
) -> list[object]:
    print("\n=== 2. 记忆分包与覆盖检查 ===")
    partitioner = EntityGraphPartitioner(
        max_primary_memories=max_primary,
        max_context_memories=36,
        max_packet_tokens=max_packet_tokens,
        max_global_tokens=max_global_tokens,
    )
    packets = partitioner.build_packets(scan)
    assigned = [
        memory_id
        for packet in packets
        for memory_id in packet.primary_memory_ids
    ]
    counts = Counter(assigned)
    expected = set(scan.all_memory_ids)
    actual = set(assigned)
    duplicates = sorted(memory_id for memory_id, count in counts.items() if count > 1)
    uncovered = sorted(expected - actual)
    unexpected = sorted(actual - expected)

    reporter.check(
        actual == expected,
        "所有数据库记忆均被分配到审计包",
        f"uncovered={uncovered}, unexpected={unexpected}",
    )
    reporter.check(
        not duplicates and len(assigned) == len(scan.all_memory_ids),
        "每条记忆恰好一次作为 primary",
        f"duplicates={duplicates}",
    )

    primary_role_errors: list[str] = []
    packet_token_errors: list[str] = []
    related_packet_count = 0
    for number, packet in enumerate(packets, start=1):
        records = {
            str(record.get("memory_id")): record
            for record in packet.memories
        }
        for memory_id in packet.primary_memory_ids:
            if records.get(memory_id, {}).get("role") != "primary":
                primary_role_errors.append(memory_id)

        serialized = json.dumps(packet.model_dump(mode="json"), ensure_ascii=False)
        token_count = estimate_tokens(serialized)
        if token_count > max_packet_tokens + 250:
            packet_token_errors.append(f"{packet.packet_id}={token_count}")
        if packet.related_heads:
            related_packet_count += 1

        focus = packet.focus_head or {}
        focus_name = focus.get("canonical_name", focus.get("head_id", "unknown"))
        if verbose or len(packets) <= 20:
            print(
                f"  包 {number:02d} | focus={focus_name} | tokens≈{token_count} | "
                f"primary={packet.primary_memory_ids} | "
                f"context={packet.context_memory_ids}"
            )

    reporter.check(
        not primary_role_errors,
        "分包内容中的 primary 角色标记正确",
        f"invalid={sorted(set(primary_role_errors))}",
    )
    reporter.check(
        not packet_token_errors,
        "每个数据包均处于允许的 token 预算误差内",
        f"oversized={packet_token_errors}",
    )
    if len(scan.heads) > 1 and any(len(memory.character_ids) + len(memory.item_ids) + len(memory.event_ids) > 1 for memory in scan.memories.values()):
        reporter.check(
            related_packet_count > 0,
            "共享记忆在分包中保留了关联实体信息",
        )

    print(f"  记忆总数：{len(scan.all_memory_ids)}")
    print(f"  数据包数量：{len(packets)}")
    print(f"  primary 分配次数：{len(assigned)}")
    print(f"  含关联实体的数据包：{related_packet_count}")
    return list(packets)


def print_scan_issues(scan: ScanResult) -> None:
    if not scan.issues:
        print("\n扫描器未发现数据库完整性问题。")
        return
    print(f"\n扫描器发现 {len(scan.issues)} 个问题：")
    counts = Counter(issue.code for issue in scan.issues)
    for code, count in sorted(counts.items()):
        print(f"  - {code}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="测试小说 RAG 数据库的实体索引提取和全书记忆分包。",
    )
    parser.add_argument(
        "--existing",
        action="store_true",
        help="只读检查已有数据库；必须同时提供 --data-root 和 --book-id。",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="已有数据库根目录；默认模拟测试不需要提供。",
    )
    parser.add_argument(
        "--book-id",
        default="partition_index_demo",
        help="书籍 ID，默认 partition_index_demo。",
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="模拟测试结束后保留生成的数据库，便于手工查看。",
    )
    parser.add_argument(
        "--max-primary",
        type=int,
        default=2,
        help="每个包最多 primary 记忆数，默认 2。",
    )
    parser.add_argument(
        "--max-packet-tokens",
        type=int,
        default=1800,
        help="单包 token 预算，默认 1800。",
    )
    parser.add_argument(
        "--max-global-tokens",
        type=int,
        default=400,
        help="全局账本 token 预算，默认 400。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="现有书库较大时仍打印每个实体和数据包的详细内容。",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if args.max_primary < 1:
        raise ValueError("--max-primary 必须大于 0")
    if args.max_packet_tokens < 1000:
        raise ValueError("--max-packet-tokens 至少为 1000")
    if args.max_global_tokens < 100:
        raise ValueError("--max-global-tokens 至少为 100")

    reporter = TestReporter()
    temporary: tempfile.TemporaryDirectory[str] | None = None

    if args.existing:
        if args.data_root is None:
            raise ValueError("--existing 模式必须提供 --data-root")
        root = args.data_root.resolve()
        book_dir = root / args.book_id
        if not book_dir.is_dir():
            raise FileNotFoundError(f"书籍数据库目录不存在：{book_dir}")
        repository = BookRepository(RagConfig(root), args.book_id)
        expected = None
        names = None
        print("=== 现有数据库只读结构检查 ===")
    else:
        if args.keep_data:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            root = Path("partition_index_test_data") / stamp
            root = root.resolve()
        else:
            temporary = tempfile.TemporaryDirectory(prefix="novel_rag_partition_")
            root = Path(temporary.name)
        repository, expected, names = build_fixture(root, args.book_id)
        print("=== 隔离模拟数据库严格测试 ===")

    print(f"数据根目录：{root}")
    print(f"书籍 ID：{args.book_id}")

    scan = AuditScanner().scan(repository)
    if expected is not None and names is not None:
        verify_index_fixture(repository, expected, names, reporter)
    else:
        verify_existing_index(repository, scan, reporter, args.verbose)

    verify_partitioning(
        scan,
        reporter,
        max_primary=args.max_primary,
        max_packet_tokens=args.max_packet_tokens,
        max_global_tokens=args.max_global_tokens,
        verbose=args.verbose,
    )
    print_scan_issues(scan)

    print("\n=== 测试结论 ===")
    print(f"通过：{reporter.passed}")
    print(f"失败：{reporter.failed}")
    if reporter.failures:
        print("失败明细：")
        for failure in reporter.failures:
            print(f"  - {failure}")
    elif args.existing:
        print("现有数据库的索引链接可读取，且分包覆盖完整。")
    else:
        print("模拟数据库的索引提取、别名解析、去除无关记忆和分包覆盖全部正确。")

    if args.keep_data and not args.existing:
        print(f"测试数据库已保留：{root}")
    elif temporary is not None:
        temporary.cleanup()
        print("临时测试数据库已自动清理。")

    return 0 if reporter.failed == 0 else 1


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
