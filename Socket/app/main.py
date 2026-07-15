from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .runtime import NovelSocketRuntime


class CreateSessionRequest(BaseModel):
    book_id: str = Field(min_length=1)
    task_code: str = Field(pattern="^(BD|CH|CT|NW|RV|PW|bd|ch|ct|nw|rv|pw)$")
    chapter_id: int = 0


class SettingRequest(BaseModel):
    text: str = ""
    refine: bool = False
    chapter_title: str = ""
    chapter_text: str = ""


class EditorSaveRequest(BaseModel):
    book_id: str = Field(min_length=1)
    chapter_id: int = Field(ge=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    session_id: str = ""


class MessageRequest(BaseModel):
    text: str = Field(min_length=1)


class RecordRequest(BaseModel):
    value: dict[str, Any]


class BookRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class ChapterRequest(BaseModel):
    title: str = Field(default="", max_length=200)


class MaterialSyncRequest(BaseModel):
    overwrite_user: bool = False


app = FastAPI(title="智能写作助手", version="0.5.0")
runtime = NovelSocketRuntime.from_env()
static_dir = Path(__file__).resolve().parents[1] / "static"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return runtime.health()


@app.get("/api/library")
async def library() -> dict[str, Any]:
    return await runtime.run("library")


@app.post("/api/books")
async def create_book(request: BookRequest) -> dict[str, Any]:
    try:
        return await runtime.run("create_book", name=request.name)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/api/books/{book_id}")
async def rename_book(book_id: str, request: BookRequest) -> dict[str, Any]:
    try:
        return await runtime.run("rename_book", book_id=book_id, name=request.name)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/books/{book_id}/chapters")
async def create_chapter(book_id: str, request: ChapterRequest) -> dict[str, Any]:
    try:
        return await runtime.run(
            "create_chapter",
            book_id=book_id,
            title=request.title,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/books/{book_id}/chapters/{chapter_id}")
async def chapter(book_id: str, chapter_id: int) -> dict[str, Any]:
    try:
        return await runtime.run("load_chapter", book_id=book_id, chapter_id=chapter_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/books/{book_id}/memories")
async def memories(book_id: str) -> dict[str, Any]:
    return await runtime.run("memories", book_id=book_id)


@app.post("/api/books/{book_id}/memories")
async def create_memory(book_id: str, request: RecordRequest) -> dict[str, Any]:
    try:
        return await runtime.run("create_memory", book_id=book_id, value=request.value)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/api/books/{book_id}/memories/{memory_id}")
async def update_memory(book_id: str, memory_id: str, request: RecordRequest) -> dict[str, Any]:
    try:
        return await runtime.run(
            "update_memory", book_id=book_id, memory_id=memory_id, value=request.value
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/books/{book_id}/memories/{memory_id}")
async def delete_memory(book_id: str, memory_id: str) -> dict[str, Any]:
    try:
        return await runtime.run("delete_memory", book_id=book_id, memory_id=memory_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/books/{book_id}/memories/{memory_id}/history")
async def memory_history(book_id: str, memory_id: str) -> dict[str, Any]:
    return await runtime.run("memory_history", book_id=book_id, memory_id=memory_id)


@app.get("/api/material-schemas")
async def material_schemas() -> dict[str, Any]:
    return await runtime.run("material_schemas")


@app.get("/api/books/{book_id}/materials")
async def materials(book_id: str) -> dict[str, Any]:
    return await runtime.run("materials", book_id=book_id)


@app.post("/api/books/{book_id}/materials")
async def create_material(book_id: str, request: RecordRequest) -> dict[str, Any]:
    try:
        return await runtime.run("save_material", book_id=book_id, value=request.value)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/api/books/{book_id}/materials/{material_id}")
async def update_material(book_id: str, material_id: str, request: RecordRequest) -> dict[str, Any]:
    try:
        return await runtime.run(
            "save_material", book_id=book_id, material_id=material_id, value=request.value
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/books/{book_id}/materials/{material_id}")
async def delete_material(book_id: str, material_id: str) -> dict[str, Any]:
    try:
        return await runtime.run("delete_material", book_id=book_id, material_id=material_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/books/{book_id}/materials/{material_id}/sync")
async def sync_material(
    book_id: str,
    material_id: str,
    request: MaterialSyncRequest | None = None,
) -> dict[str, Any]:
    try:
        return await runtime.run(
            "sync_material",
            book_id=book_id,
            material_id=material_id,
            overwrite_user=request.overwrite_user if request else False,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/books/{book_id}/material-reviews/{review_id}/confirm")
async def confirm_material_review(
    book_id: str,
    review_id: str,
    request: RecordRequest | None = None,
) -> dict[str, Any]:
    try:
        value = request.value if request is not None else {}
        return await runtime.run(
            "confirm_material_review",
            book_id=book_id,
            review_id=review_id,
            decisions=value.get("decisions"),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/sessions")
async def create_session(request: CreateSessionRequest) -> dict[str, Any]:
    try:
        return await runtime.run("create_session", **request.model_dump())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/sessions/{session_id}/setting")
async def submit_setting(session_id: str, request: SettingRequest) -> dict[str, Any]:
    try:
        return await runtime.run("submit_setting", session_id=session_id, **request.model_dump())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/editor/save")
async def save_editor(request: EditorSaveRequest) -> dict[str, Any]:
    try:
        return await runtime.run("save_local", **request.model_dump())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/editor/archive")
async def archive_editor(request: EditorSaveRequest) -> dict[str, Any]:
    try:
        return await runtime.run("archive_editor", **request.model_dump())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/sessions/{session_id}/message")
async def send_message(session_id: str, request: MessageRequest) -> dict[str, Any]:
    try:
        return await runtime.run("handle_message", session_id=session_id, text=request.text)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def run_with_websocket_progress(
    websocket: WebSocket,
    action: str,
    **kwargs: Any,
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    progress_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def progress_callback(event: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(progress_queue.put_nowait, event)

    task = asyncio.create_task(runtime.run(action, progress_callback=progress_callback, **kwargs))
    await websocket.send_json({"type": "busy", "busy": True})
    try:
        while True:
            if task.done() and progress_queue.empty():
                break
            try:
                event = await asyncio.wait_for(progress_queue.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            await websocket.send_json({"type": "progress", "event": event})
        return await task
    finally:
        await websocket.send_json({"type": "busy", "busy": False})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json({"type": "connected", "health": runtime.health()})
    try:
        while True:
            data = await websocket.receive_json()
            message_type = str(data.get("type") or "").strip()
            try:
                if message_type == "library":
                    action, kwargs = "library", {}
                elif message_type == "load_chapter":
                    action = "load_chapter"
                    kwargs = {
                        "book_id": str(data.get("book_id") or ""),
                        "chapter_id": int(data.get("chapter_id") or 0),
                    }
                elif message_type == "start_session":
                    action = "create_session"
                    kwargs = {
                        "book_id": str(data.get("book_id") or ""),
                        "task_code": str(data.get("task_code") or ""),
                        "chapter_id": int(data.get("chapter_id") or 0),
                    }
                elif message_type == "submit_setting":
                    action = "submit_setting"
                    kwargs = {
                        "session_id": str(data.get("session_id") or ""),
                        "text": str(data.get("text") or ""),
                        "refine": bool(data.get("refine", False)),
                        "chapter_title": str(data.get("chapter_title") or ""),
                        "chapter_text": str(data.get("chapter_text") or ""),
                    }
                elif message_type in {"save_local", "archive_editor"}:
                    action = message_type
                    kwargs = {
                        "book_id": str(data.get("book_id") or ""),
                        "chapter_id": int(data.get("chapter_id") or 0),
                        "title": str(data.get("title") or ""),
                        "text": str(data.get("text") or ""),
                        "session_id": str(data.get("session_id") or ""),
                    }
                elif message_type == "message":
                    action = "handle_message"
                    kwargs = {
                        "session_id": str(data.get("session_id") or ""),
                        "text": str(data.get("text") or ""),
                    }
                elif message_type == "consult":
                    action = "consult"
                    kwargs = {
                        "book_id": str(data.get("book_id") or ""),
                        "question": str(data.get("question") or ""),
                        "current_chapter": int(data.get("current_chapter") or 0),
                        "selected_text": str(data.get("selected_text") or ""),
                        "force_rag": bool(data.get("force_rag", False)),
                    }
                elif message_type == "generate_material":
                    action = "generate_material"
                    kwargs = {
                        "book_id": str(data.get("book_id") or ""),
                        "task_code": str(data.get("task_code") or ""),
                        "text": str(data.get("text") or ""),
                        "refine": bool(data.get("refine", False)),
                    }
                elif message_type == "confirm_material_generation":
                    action = "confirm_material_generation"
                    kwargs = {
                        "generation_id": str(data.get("generation_id") or ""),
                        "decisions": data.get("decisions"),
                    }
                elif message_type == "extract_materials":
                    action = "extract_materials"
                    kwargs = {
                        "book_id": str(data.get("book_id") or ""),
                        "chapter_id": int(data.get("chapter_id") or 0),
                        "title": str(data.get("title") or ""),
                        "text": str(data.get("text") or ""),
                        "save_draft": bool(data.get("save_draft", False)),
                        "session_id": str(data.get("session_id") or ""),
                    }
                elif message_type == "audit_memories":
                    action = "audit_memories"
                    raw_chapter_ids = data.get("chapter_ids") or []
                    if not isinstance(raw_chapter_ids, list):
                        raise ValueError("chapter_ids必须是章节编号数组")
                    kwargs = {
                        "book_id": str(data.get("book_id") or ""),
                        "scope_mode": str(data.get("scope_mode") or "book"),
                        "chapter_ids": [int(value) for value in raw_chapter_ids],
                    }
                elif message_type == "apply_memory_audit":
                    action = "apply_memory_audit"
                    kwargs = {
                        "book_id": str(data.get("book_id") or ""),
                        "session_id": str(data.get("session_id") or ""),
                    }
                elif message_type == "rollback_memory_audit":
                    action = "rollback_memory_audit"
                    kwargs = {
                        "book_id": str(data.get("book_id") or ""),
                        "snapshot_id": str(data.get("snapshot_id") or ""),
                    }
                elif message_type == "ping":
                    await websocket.send_json({"type": "result", "request_type": "ping", "data": {"kind": "pong"}})
                    continue
                else:
                    raise ValueError(f"不支持的网页操作: {message_type}")
                await websocket.send_json({"type": "status", "message": "处理中..."})
                result = await run_with_websocket_progress(websocket, action, **kwargs)
                await websocket.send_json({"type": "result", "request_type": message_type, "data": result})
            except Exception as exc:  # noqa: BLE001
                await websocket.send_json({"type": "busy", "busy": False})
                await websocket.send_json({"type": "error", "request_type": message_type, "message": str(exc)})
    except WebSocketDisconnect:
        return
