from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


DEFAULT_PROMPT = (
    "创作一部东方奇幻小说的开篇。主角林舟是流亡王族，"
    "他来到一座古代遗迹，并发现只有王族血脉才能开启的青铜门。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="测试 write_agent 生成 -> RAG 标准化 -> SQLite 入库 -> RAG 召回。"
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="发送给 write_agent 的写作要求。",
    )
    parser.add_argument(
        "--task",
        choices=("BD", "CH", "CT", "NW", "RV"),
        default="NW",
        help="Writer 任务代码，默认使用最适合空库测试的 NW。",
    )
    parser.add_argument("--book-id", default="smoke_book_001")
    parser.add_argument("--chapter-id", type=int, default=1)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("rag_smoke_data"),
        help="测试数据库根目录。",
    )
    return parser.parse_args()


def print_json(title: str, value: object) -> None:
    print(f"\n{'=' * 18} {title} {'=' * 18}")
    print(json.dumps(value, ensure_ascii=False, indent=2))


def print_database_summary(system: object, book_id: str) -> None:
    from rag.repository import BookRepository, MEMORY_STORES

    repository = BookRepository(system.config, book_id)
    print(f"\n{'=' * 18} 入库检查 {'=' * 18}")
    for store_type in MEMORY_STORES:
        records = repository.store(store_type).list_memories(statuses=None)
        database_path = system.config.database_path(book_id, store_type)
        print(f"{store_type:28s} 记录数={len(records):2d}  {database_path.resolve()}")

    index_path = system.config.database_path(book_id, "index")
    conflict_path = system.config.database_path(book_id, "conflicts")
    print(f"{'index':28s}            {index_path.resolve()}")
    print(
        f"{'conflicts':28s} 待处理={len(repository.conflicts.list_pending()):2d}  "
        f"{conflict_path.resolve()}"
    )


def main() -> int:
    args = parse_args()

    from rag import MemoryAgent, NovelRagSystem
    from rag.local_model import HuggingFaceQwenJsonClient
    from rag.qwen_judge import QwenMemoryJudge
    from write_agent import BASE_URL, MODEL_ID, generate_writer_content

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("开始一条龙测试")
    print(f"任务: {args.task}")
    print(f"作品: {args.book_id}，章节: {args.chapter_id}")
    print(f"模型: {MODEL_ID}")
    print(f"接口: {BASE_URL}")
    print(f"测试数据目录: {args.data_dir.resolve()}")

    qwen_client = HuggingFaceQwenJsonClient(
        repo_id=os.getenv("QWEN_HF_MODEL_ID", "Qwen/Qwen3.5-4B"),
        local_model_path=os.getenv("QWEN_LOCAL_MODEL_PATH"),
        cache_dir=os.getenv("QWEN_HF_CACHE"),
        device=os.getenv("QWEN_DEVICE", "auto"),
    )

    print("\n[0/4] 检查本地 Qwen 判断模型...")
    qwen_path = qwen_client.ensure_ready()
    print(f"Qwen模型路径: {qwen_path}")

    system = NovelRagSystem(
        args.data_dir,
        memory_agent=MemoryAgent(qwen_client),
        judge=QwenMemoryJudge(qwen_client),
    )

    print("\n[1/4] 调用 write_agent 生成结构化内容...")
    writer_result = generate_writer_content(args.task, args.prompt)
    print_json("Writer 输出", writer_result)

    print("\n[2/4] 标准化并写入本地 RAG 数据库...")
    ingest_result = system.ingest(
        args.book_id,
        args.task,
        writer_result,
        chapter_id=args.chapter_id,
    )
    print_json("入库结果", ingest_result.model_dump())

    print("\n[3/4] 检查各数据库文件和记录数...")
    print_database_summary(system, args.book_id)

    print("\n[4/4] 从刚写入的数据库召回下一章上下文...")
    context = system.retrieve_context(
        args.book_id,
        "继续写林舟进入青铜门后的剧情",
        current_chapter=args.chapter_id + 1,
    )
    print_json("RAG 召回上下文", context)

    recalled = sum(len(items) for items in context.values())
    if not ingest_result.created_memory_ids and not ingest_result.updated_memory_ids:
        raise RuntimeError("生成成功，但没有任何记忆被创建或更新")
    if recalled == 0:
        raise RuntimeError("入库成功，但下一章上下文没有召回任何记忆")

    print("\n测试通过：生成、标准化、入库和召回链路均可运行。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("\n测试失败，错误如下：", file=sys.stderr)
        traceback.print_exc()
        print(
            "\n请检查 llma 环境依赖、.env 中的 LLM_API_KEY/LLM_MODEL_ID/"
            "LLM_BASE_URL、本地Writer模型服务及Hugging Face网络连接。",
            file=sys.stderr,
        )
        raise SystemExit(1)
