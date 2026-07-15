from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


# ===== User-facing defaults =====
# You can edit these values directly, or override them with command-line args.
SOCKET_ROOT = Path(__file__).resolve().parent
DEFAULT_BACKEND_ROOT = SOCKET_ROOT.parent / "Novel_Agentv2"
DEFAULT_DATA_DIR = DEFAULT_BACKEND_ROOT / "rag_data"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8021
DEFAULT_RELOAD = False

# Existing Novel_Agentv2 model config. Prefer putting real secrets in .env.
DEFAULT_LLM_API_KEY = ""
DEFAULT_LLM_MODEL_ID = ""
DEFAULT_LLM_BASE_URL = ""
DEFAULT_LLM_MAX_OUTPUT_TOKENS = "8192"

# Optional local Qwen config.
DEFAULT_QWEN_LOCAL_MODEL_PATH = DEFAULT_BACKEND_ROOT / "models" / "Qwen3.5-4B"
DEFAULT_QWEN_HF_MODEL_ID = "Qwen/Qwen3.5-4B"
DEFAULT_QWEN_HF_CACHE = ""
DEFAULT_QWEN_DEVICE = "auto"


REQUIRED_MODEL_KEYS = ("LLM_API_KEY", "LLM_MODEL_ID", "LLM_BASE_URL")


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def ensure_env_file() -> Path:
    env_path = SOCKET_ROOT / ".env"
    example_path = SOCKET_ROOT / ".env.example"
    if not env_path.exists() and example_path.exists():
        shutil.copyfile(example_path, env_path)
        print(f"[init] 已创建配置文件: {env_path}")
        print("[init] 请在其中补齐 LLM_API_KEY / LLM_MODEL_ID / LLM_BASE_URL。")
    return env_path


def set_default_env(key: str, value: str | Path | None) -> None:
    if os.getenv(key):
        return
    if value is None:
        return
    text = str(value).strip()
    if text:
        os.environ[key] = text


def mask_secret(value: str) -> str:
    if not value:
        return "<empty>"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def print_config(keys: Iterable[str]) -> None:
    print("\n[config] 当前运行参数")
    for key in keys:
        value = os.getenv(key, "")
        if "KEY" in key or "TOKEN" in key:
            value = mask_secret(value)
        print(f"  {key}={value or '<empty>'}")


def validate_config(*, allow_missing_model: bool) -> None:
    backend_root = Path(os.environ["NOVEL_AGENT_ROOT"])
    data_dir = Path(os.environ["NOVEL_AGENT_DATA_DIR"])
    errors: list[str] = []
    if not backend_root.is_dir():
        errors.append(f"NOVEL_AGENT_ROOT 不存在: {backend_root}")
    if not (backend_root / "control_agent.py").is_file():
        errors.append(f"找不到 control_agent.py: {backend_root}")
    if not data_dir.is_dir():
        errors.append(f"NOVEL_AGENT_DATA_DIR 不存在: {data_dir}")
    missing_model = [key for key in REQUIRED_MODEL_KEYS if not os.getenv(key)]
    if missing_model and not allow_missing_model:
        errors.append(
            "缺少模型配置: "
            + ", ".join(missing_model)
            + "。可编辑 Socket/.env，或启动时加 --allow-missing-model 仅打开页面。"
        )
    if errors:
        print("\n[error] 配置检查未通过")
        for item in errors:
            print(f"  - {item}")
        raise SystemExit(2)


def listening_pids(port: int) -> list[int]:
    try:
        output = subprocess.check_output(
            ["netstat", "-ano"],
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return []
    pids: set[int] = set()
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        local_address = parts[1]
        state = parts[3].upper()
        if state != "LISTENING" or f":{port}" not in local_address:
            continue
        try:
            pid = int(parts[-1])
        except ValueError:
            continue
        if pid:
            pids.add(pid)
    return sorted(pids)


def stop_processes(pids: list[int]) -> None:
    for pid in pids:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            try:
                os.kill(pid, 15)
            except ProcessLookupError:
                pass
        print(f"[stop] 已停止占用端口的旧服务 PID={pid}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="启动 Novel_Agentv2 的 FastAPI/WebSocket 网页服务。"
    )
    parser.add_argument("--backend-root", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reload", action="store_true", default=DEFAULT_RELOAD)
    parser.add_argument("--check-only", action="store_true", help="只检查配置，不启动服务。")
    parser.add_argument(
        "--stop-existing",
        action="store_true",
        help="启动前自动停止占用当前端口的旧服务。",
    )
    parser.add_argument(
        "--allow-missing-model",
        action="store_true",
        help="允许缺少 LLM 配置时启动页面；生成时仍会失败。",
    )
    parser.add_argument("--llm-api-key", default="")
    parser.add_argument("--llm-model-id", default="")
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--qwen-local-model-path", type=Path, default=None)
    parser.add_argument("--qwen-device", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_path = ensure_env_file()

    # Load config in increasing priority: Socket/.env, backend/.env, explicit args.
    for key, value in read_env_file(env_path).items():
        set_default_env(key, value)
    backend_root = args.backend_root or Path(os.getenv("NOVEL_AGENT_ROOT", str(DEFAULT_BACKEND_ROOT)))
    data_dir = args.data_dir or Path(os.getenv("NOVEL_AGENT_DATA_DIR", str(DEFAULT_DATA_DIR)))
    host = args.host or os.getenv("HOST", DEFAULT_HOST)
    port = args.port or int(os.getenv("PORT", DEFAULT_PORT))

    backend_env = backend_root / ".env"
    for key, value in read_env_file(backend_env).items():
        set_default_env(key, value)

    os.environ["NOVEL_AGENT_ROOT"] = str(backend_root.resolve())
    os.environ["NOVEL_AGENT_DATA_DIR"] = str(data_dir.resolve())
    os.environ["HOST"] = str(host)
    os.environ["PORT"] = str(port)
    set_default_env("LLM_API_KEY", args.llm_api_key or DEFAULT_LLM_API_KEY)
    set_default_env("LLM_MODEL_ID", args.llm_model_id or DEFAULT_LLM_MODEL_ID)
    set_default_env("LLM_BASE_URL", args.llm_base_url or DEFAULT_LLM_BASE_URL)
    set_default_env("LLM_MAX_OUTPUT_TOKENS", DEFAULT_LLM_MAX_OUTPUT_TOKENS)
    set_default_env("QWEN_LOCAL_MODEL_PATH", args.qwen_local_model_path or DEFAULT_QWEN_LOCAL_MODEL_PATH)
    set_default_env("QWEN_HF_MODEL_ID", DEFAULT_QWEN_HF_MODEL_ID)
    set_default_env("QWEN_HF_CACHE", DEFAULT_QWEN_HF_CACHE)
    set_default_env("QWEN_DEVICE", args.qwen_device or DEFAULT_QWEN_DEVICE)

    print_config(
        (
            "NOVEL_AGENT_ROOT",
            "NOVEL_AGENT_DATA_DIR",
            "HOST",
            "PORT",
            "LLM_API_KEY",
            "LLM_MODEL_ID",
            "LLM_BASE_URL",
            "QWEN_LOCAL_MODEL_PATH",
            "QWEN_DEVICE",
        )
    )
    validate_config(allow_missing_model=args.allow_missing_model)
    pids = listening_pids(port)
    if pids and args.stop_existing:
        stop_processes(pids)
        pids = listening_pids(port)
    if pids:
        print(f"\n[error] 端口 {port} 已被占用，PID={', '.join(str(pid) for pid in pids)}")
        print(f"[hint] 可运行: python run_socket.py --port {port} --stop-existing")
        print("[hint] 或换一个端口: python run_socket.py --port 8011")
        return 3
    if args.check_only:
        print("\n[ok] 配置检查通过，端口可用。")
        return 0

    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if args.reload:
        command.append("--reload")

    print(f"\n[start] 网页服务启动中: http://{host}:{port}")
    print("[start] 按 Ctrl+C 停止服务。\n")
    subprocess.run(command, cwd=SOCKET_ROOT, check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())