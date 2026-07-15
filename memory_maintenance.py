from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按实体关系分批审计、整理或回滚一本小说的RAG记忆库。"
    )
    parser.add_argument("--book-id", required=True, help="需要审计的作品ID。")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("rag_data"),
        help="RAG数据根目录。",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--apply",
        action="store_true",
        help="验证通过后正式应用补丁；默认只做dry-run。",
    )
    mode.add_argument(
        "--rollback",
        metavar="SNAPSHOT_ID",
        help="回滚到指定维护快照，不调用模型。",
    )
    parser.add_argument(
        "--allow-direct-test",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.allow_direct_test:
        print(
            "memory_maintenance.py是内部测试/维护入口。"
            "普通用户请运行control_agent.py，并在对话中要求整理记忆库。"
        )
        return 2
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from dotenv import load_dotenv

    from rag import MemoryAgent, NovelRagSystem
    from rag.local_model import HuggingFaceQwenJsonClient

    load_dotenv()
    if args.rollback:
        system = NovelRagSystem(args.data_dir)
        system.rollback_memory_audit(args.book_id, args.rollback)
        print(f"回滚完成: book_id={args.book_id}, snapshot_id={args.rollback}")
        return 0

    qwen_client = HuggingFaceQwenJsonClient(
        repo_id=os.getenv("QWEN_HF_MODEL_ID", "Qwen/Qwen3.5-4B"),
        local_model_path=os.getenv("QWEN_LOCAL_MODEL_PATH"),
        cache_dir=os.getenv("QWEN_HF_CACHE"),
        device=os.getenv("QWEN_DEVICE", "auto"),
    )
    system = NovelRagSystem(
        args.data_dir,
        memory_agent=MemoryAgent(qwen_client),
    )
    mode = "正式应用" if args.apply else "dry-run"
    print(f"开始维护: book_id={args.book_id}, mode={mode}")
    result = system.audit_book_memories(args.book_id, apply=args.apply)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    print(f"审计产物目录: {result.artifact_dir}")
    print(
        "跨索引候选覆盖: "
        f"{result.semantic_candidate_reviewed_count}/"
        f"{result.semantic_candidate_count}，"
        f"完整={result.semantic_candidate_complete}"
    )
    if result.snapshot_id:
        print(f"回滚快照: {result.snapshot_id}")
    if (
        not result.coverage.complete
        or not result.semantic_candidate_complete
        or result.blocking_issue_ids
        or result.validation_errors
    ):
        print("审计未达到安全应用条件，请查看coverage、blocking_issue_ids和validation_errors。")
        return 2
    print("维护流程完成。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("记忆维护失败：", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
