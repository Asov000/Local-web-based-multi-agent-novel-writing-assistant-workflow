from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterator


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    step: str
    state: str
    message: str
    elapsed_seconds: float = 0.0


ProgressCallback = Callable[[ProgressEvent], None]


def silent_progress(_: ProgressEvent) -> None:
    return


class SilentProgress:
    @contextmanager
    def step(self, step: str, message: str) -> Iterator[None]:
        yield


class ConsoleProgress:
    def __init__(self, *, heartbeat_seconds: float = 8.0) -> None:
        self.heartbeat_seconds = max(1.0, heartbeat_seconds)
        self._print_lock = threading.Lock()

    def emit(self, event: ProgressEvent) -> None:
        labels = {
            "started": "正在",
            "running": "仍在运行",
            "completed": "完成",
            "failed": "失败",
        }
        label = labels.get(event.state, event.state)
        elapsed = (
            f"，已用时 {event.elapsed_seconds:.1f} 秒"
            if event.elapsed_seconds > 0
            else ""
        )
        stamp = datetime.now().strftime("%H:%M:%S")
        with self._print_lock:
            print(
                f"[{stamp}] [{label}] {event.message}{elapsed}",
                file=sys.stdout,
                flush=True,
            )

    @contextmanager
    def step(self, step: str, message: str) -> Iterator[None]:
        started = time.monotonic()
        stopped = threading.Event()
        self.emit(ProgressEvent(step, "started", message))

        def heartbeat() -> None:
            while not stopped.wait(self.heartbeat_seconds):
                self.emit(
                    ProgressEvent(
                        step,
                        "running",
                        message,
                        time.monotonic() - started,
                    )
                )

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            yield
        except Exception:
            stopped.set()
            self.emit(
                ProgressEvent(
                    step,
                    "failed",
                    message,
                    time.monotonic() - started,
                )
            )
            raise
        else:
            stopped.set()
            self.emit(
                ProgressEvent(
                    step,
                    "completed",
                    message,
                    time.monotonic() - started,
                )
            )
