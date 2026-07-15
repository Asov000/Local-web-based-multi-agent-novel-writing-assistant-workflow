# 运行 Socket 网页服务

推荐使用一键入口：

```powershell
cd D:\SAM\main\Agent\Socket
python run_socket.py
```

第一次运行时会自动创建 `.env`。请先编辑 `.env`，至少补齐：

```text
LLM_API_KEY=你的模型接口密钥
LLM_MODEL_ID=你的模型 ID
LLM_BASE_URL=你的 OpenAI 兼容接口地址
```

只想先打开页面、不调用模型，可以运行：

```powershell
python run_socket.py --allow-missing-model
```

如果 8010 端口被占用，可以换端口：

```powershell
python run_socket.py --port 8011 --allow-missing-model
```

只检查配置，不启动服务：

```powershell
python run_socket.py --check-only --allow-missing-model
```

启动成功后浏览器打开：

```text
http://127.0.0.1:8010
```

常用参数都集中在 `run_socket.py` 顶部，包括后端目录、RAG 数据目录、Host、Port、LLM 参数和 Qwen 本地模型路径。
## 端口被占用时

如果看到 `WinError 10048` 或提示端口已被占用，可以让启动脚本自动停止占用当前端口的旧服务：

```powershell
python run_socket.py --stop-existing
```

也可以同时换端口：

```powershell
python run_socket.py --port 8011 --stop-existing
```

当前页面如果显示“网页服务未连接”，通常表示服务已经停止、端口和浏览器地址不一致，或旧服务刚被关闭。重新运行上面的启动命令后刷新页面即可。