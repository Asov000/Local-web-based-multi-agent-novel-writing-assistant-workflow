from __future__ import annotations

import json
import os
import re
import ssl
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol


class JsonModelClient(Protocol):
    def invoke_json(
        self,
        system_prompt: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]: ...


def parse_json_object(content: Any) -> dict[str, Any]:
    text = str(content or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("模型未返回JSON对象")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("模型返回值必须是JSON对象")
    return value


def ensure_huggingface_snapshot(
    repo_id: str,
    *,
    revision: str = "main",
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    token: str | None = None,
    snapshot_download_fn: Callable[..., str] | None = None,
) -> Path:
    """Return a local snapshot, downloading from the official Hub if absent."""

    preferred_dir = Path(local_dir).expanduser() if local_dir else None
    if preferred_dir and (preferred_dir / "config.json").is_file():
        return preferred_dir.resolve()

    if snapshot_download_fn is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "缺少huggingface_hub，请在当前Conda环境安装huggingface_hub"
            ) from exc
        snapshot_download_fn = snapshot_download

    common_args: dict[str, Any] = {
        "repo_id": repo_id,
        "revision": revision,
        "token": token,
    }
    if cache_dir:
        common_args["cache_dir"] = str(Path(cache_dir).expanduser())
    if preferred_dir:
        common_args["local_dir"] = str(preferred_dir)

    try:
        snapshot = snapshot_download_fn(local_files_only=True, **common_args)
        return Path(snapshot).resolve()
    except Exception:
        print(f"本地未找到 {repo_id}，开始从 Hugging Face 官方仓库下载...")

    try:
        snapshot = snapshot_download_fn(local_files_only=False, **common_args)
    except Exception as exc:
        raise RuntimeError(f"从Hugging Face下载{repo_id}失败: {exc}") from exc
    path = Path(snapshot).resolve()
    print(f"模型下载完成: {path}")
    return path


class LocalJsonModelClient:
    """Small OpenAI-compatible client intended for a local model server."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "local",
        timeout: float = 90.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.75,
    ) -> None:
        base = base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            self.endpoint = base
        elif base.endswith("/v1"):
            self.endpoint = f"{base}/chat/completions"
        else:
            self.endpoint = f"{base}/v1/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)

    def invoke_json(self, system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json; charset=utf-8",
            },
        )
        result: dict[str, Any] | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                retryable = exc.code in {408, 429} or 500 <= exc.code < 600
                if not retryable or attempt >= self.max_retries:
                    raise
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                ssl.SSLError,
            ):
                if attempt >= self.max_retries:
                    raise
            time.sleep(self.retry_backoff_seconds * (2**attempt))
        if result is None:
            raise RuntimeError("模型接口重试结束后未返回结果")
        content = result["choices"][0]["message"]["content"]
        return self._parse_json(content)

    @staticmethod
    def _parse_json(content: Any) -> dict[str, Any]:
        return parse_json_object(content)


class HuggingFaceQwenJsonClient:
    """Lazy local Qwen client with automatic official Hub download."""

    def __init__(
        self,
        repo_id: str = "Qwen/Qwen3.5-4B",
        *,
        revision: str = "main",
        local_model_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        token: str | None = None,
        device: str = "auto",
        max_new_tokens: int = 4096,
    ) -> None:
        self.repo_id = repo_id
        self.revision = revision
        self.local_model_path = local_model_path
        self.cache_dir = cache_dir
        self.token = token or os.getenv("HF_TOKEN")
        self.requested_device = device
        self.max_new_tokens = max_new_tokens
        self._model_path: Path | None = None
        self._processor: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: Any = None
        self._is_multimodal = False
        self._load_lock = threading.Lock()

    @property
    def model_path(self) -> Path | None:
        return self._model_path

    def ensure_ready(self) -> Path:
        if self._model is not None and self._model_path is not None:
            return self._model_path
        with self._load_lock:
            if self._model is not None and self._model_path is not None:
                return self._model_path
            self._load_model()
        if self._model_path is None:
            raise RuntimeError("Qwen模型加载后未得到本地路径")
        return self._model_path

    def _load_model(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "本地Qwen推理需要torch和transformers，请在当前Conda环境安装"
            ) from exc

        self._model_path = ensure_huggingface_snapshot(
            self.repo_id,
            revision=self.revision,
            cache_dir=self.cache_dir,
            local_dir=self.local_model_path,
            token=self.token,
        )
        if self.requested_device == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_name = self.requested_device
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("QWEN_DEVICE要求使用CUDA，但当前环境未检测到可用GPU")
        self._device = torch.device(device_name)

        model_kwargs: dict[str, Any] = {"local_files_only": True}
        if self._device.type == "cuda":
            model_kwargs["dtype"] = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        else:
            model_kwargs["dtype"] = torch.float32

        config_data = json.loads(
            (self._model_path / "config.json").read_text(encoding="utf-8")
        )
        self._is_multimodal = config_data.get("model_type") == "qwen3_5"
        if self._is_multimodal:
            try:
                from transformers import AutoModelForMultimodalLM, AutoProcessor
            except ImportError as exc:
                raise RuntimeError(
                    "Qwen3.5需要最新版transformers及其AutoModelForMultimodalLM支持"
                ) from exc
            self._processor = AutoProcessor.from_pretrained(
                self._model_path,
                local_files_only=True,
            )
            self._tokenizer = getattr(self._processor, "tokenizer", self._processor)
            self._model = AutoModelForMultimodalLM.from_pretrained(
                self._model_path,
                **model_kwargs,
            )
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_path,
                local_files_only=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_path,
                **model_kwargs,
            )
        self._model.to(self._device)
        self._model.eval()
        print(f"Qwen判断模型已加载: {self._model_path} ({self._device})")

    def invoke_json(
        self,
        system_prompt: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self.ensure_ready()
        import torch

        if self._is_multimodal:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(payload, ensure_ascii=False),
                        }
                    ],
                },
            ]
            kwargs = {
                "add_generation_prompt": True,
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
            }
            try:
                inputs = self._processor.apply_chat_template(
                    messages,
                    enable_thinking=False,
                    **kwargs,
                )
            except TypeError:
                inputs = self._processor.apply_chat_template(messages, **kwargs)
            inputs = inputs.to(self._device)
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
            try:
                prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        output_ids = generated[0, inputs["input_ids"].shape[1] :]
        decoder = self._processor if self._is_multimodal else self._tokenizer
        content = decoder.decode(output_ids, skip_special_tokens=True)
        return parse_json_object(content)

    def embed_texts(
        self,
        texts: list[str],
        *,
        max_length: int = 512,
        batch_size: int = 8,
    ) -> list[list[float]]:
        """Create normalized sentence vectors from the loaded Qwen backbone."""
        if not texts:
            return []
        self.ensure_ready()
        import torch
        import torch.nn.functional as functional

        vectors: list[list[float]] = []
        model_core = getattr(self._model, "model", None)
        backbone = getattr(self._model, "language_model", None)
        if backbone is None and model_core is not None:
            backbone = getattr(model_core, "language_model", None) or model_core
        for start in range(0, len(texts), max(1, batch_size)):
            batch = texts[start : start + max(1, batch_size)]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self._device)
            with torch.inference_mode():
                if backbone is not None:
                    outputs = backbone(
                        **inputs,
                        use_cache=False,
                        return_dict=True,
                    )
                    hidden = outputs.last_hidden_state
                else:
                    outputs = self._model(
                        **inputs,
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    hidden = outputs.hidden_states[-1]
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = functional.normalize(pooled.float(), p=2, dim=1)
            vectors.extend(pooled.cpu().tolist())
        return vectors
