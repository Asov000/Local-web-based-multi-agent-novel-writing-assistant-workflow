from __future__ import annotations

import uuid
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field


T = TypeVar("T")


MessageType = Literal[
    "request",
    "response",
]

MessageStatus = Literal[
    "pending",
    "ok",
    "need_user_input",
    "error",
]


class AgentMessage(BaseModel, Generic[T]):
    """
    Agent之间统一传递的消息。

    request:
        Control Agent -> 其他Agent

    response:
        其他Agent -> Control Agent
    """

    # 当前消息唯一ID
    message_id: str = Field(
        default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}"
    )

    # 同一任务链共享的任务ID
    task_id: str = Field(
        default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}"
    )

    # 响应消息对应的原始请求消息ID
    parent_message_id: str | None = None

    # 消息发送者
    sender: str

    # 消息接收者
    receiver: str

    # 请求消息还是响应消息
    message_type: MessageType = "request"

    # 具体动作，例如 write.new_book
    action: str

    # 当前消息执行状态
    status: MessageStatus = "pending"

    # 真正的业务数据
    payload: T | None = None

    # 日志、模型信息、Token统计等附加信息
    metadata: dict[str, Any] = Field(default_factory=dict)

    # 错误信息
    error: str | None = None


# =========================================================
# 保留旧接口，避免其他Agent立刻全部报错
# 后续所有Agent迁移到AgentMessage后，可以删除
# =========================================================

class AgentResponse(BaseModel, Generic[T]):
    agent: str
    task_id: str
    status: Literal["ok", "error"]
    data: T | None = None
    error: str | None = None