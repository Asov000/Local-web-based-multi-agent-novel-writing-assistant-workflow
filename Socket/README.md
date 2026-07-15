# Socket

`Socket` 是 `Novel_Agentv2` 的网页化部署外壳。它通过 FastAPI 和 WebSocket 暴露当前后端能力，不复制模型、RAG 数据或核心业务逻辑。

## 目录关系

```text
Agent/
├─ Novel_Agentv2/   # 当前小说写作后端
└─ Socket/          # 本网页化部署层
```

## 启动

```powershell
cd D:\SAM\main\Agent\Socket
copy .env.example .env
# 编辑 .env，补齐 LLM_API_KEY / LLM_MODEL_ID / LLM_BASE_URL
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8010 --reload
```

浏览器打开：`http://127.0.0.1:8010`

## 主要接口

- `GET /api/health`：检查后端路径和数据目录。
- `POST /api/sessions`：创建写作会话。
- `POST /api/sessions/{session_id}/setting`：提交设定并生成草稿。
- `POST /api/sessions/{session_id}/message`：沿用 ControlAgent 的意图路由，处理修改、通过归档、续写、提问、记忆整理。
- `WS /ws`：网页端使用的双向通道。

## WebSocket 消息

创建会话：

```json
{
  "type": "start_session",
  "book_id": "002",
  "task_code": "NW",
  "chapter_id": 1
}
```

提交设定并生成：

```json
{
  "type": "submit_setting",
  "session_id": "session_xxx",
  "text": "写一个关于遗迹测绘师的开篇",
  "refine": false
}
```

继续对话：

```json
{
  "type": "message",
  "session_id": "session_xxx",
  "text": "通过并归档"
}
```

## 设计边界

`Socket` 只负责网络接口、浏览器界面和部署入口。写作、修订、归档、续写、记忆审计仍由 `Novel_Agentv2/control_agent.py` 统一调度。
