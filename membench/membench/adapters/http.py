"""HttpAdapter：REST 接口模板适配器。

用于接入提供 HTTP 接口的在线记忆系统。约定被测服务暴露三个端点：

- ``POST {base_url}/reset``  ：清空记忆。
- ``POST {base_url}/ingest`` ：请求体 ``{"messages": [...]}``；
  响应可带 ``{"input_tokens": int, "llm_calls": int}``（缺省为 0）。
- ``POST {base_url}/query``  ：请求体 ``{"question": "..."}``；
  响应 ``{"answer": str, "retrieved": [str], "input_tokens": int,
  "output_tokens": int, "llm_calls": int}``（除 answer 外均可缺省）。

延迟（latency_ms）由客户端本地用 ``time.perf_counter`` 实测；
token / 调用数优先取服务端返回值，缺省填 0（报告中标注 n/a）。
"""

from __future__ import annotations

import time
from typing import Any

from .base import BaseMemoryAdapter, IngestStats, QueryResult


class HttpAdapter(BaseMemoryAdapter):
    """REST 接口记忆系统适配器（依赖 requests）。"""

    name = "http"

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """初始化 HTTP 适配器。

        Args:
            base_url: 服务根地址（不带尾部斜杠亦可，内部会规范化）。
            api_key: 可选的 Bearer Token。
            timeout: 单次请求超时时间（秒）。

        Raises:
            ImportError: 未安装 requests 时抛出并提示安装方式。
        """
        try:
            import requests  # noqa: F401
        except ImportError as exc:  # pragma: no cover - 环境相关
            raise ImportError(
                "HttpAdapter 依赖 requests，请先安装：pip install requests"
            ) from exc
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """发送 POST 请求并返回解析后的 JSON（响应非 JSON 时返回空 dict）。"""
        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.post(
            f"{self.base_url}{path}",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        try:
            data = resp.json()
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    # ------------------------------------------------------------------
    # 适配器接口
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """调用服务端 ``POST /reset`` 清空记忆。"""
        self._post("/reset", {})

    def ingest(self, messages: list[dict]) -> IngestStats:
        """调用服务端 ``POST /ingest`` 写入一段对话。"""
        t0 = time.perf_counter()
        data = self._post("/ingest", {"messages": messages})
        return IngestStats(
            input_tokens=int(data.get("input_tokens", 0)),
            llm_calls=int(data.get("llm_calls", 0)),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def query(self, question: str) -> QueryResult:
        """调用服务端 ``POST /query`` 查询。"""
        t0 = time.perf_counter()
        data = self._post("/query", {"question": question})
        retrieved = data.get("retrieved") or []
        return QueryResult(
            answer=str(data.get("answer", "")),
            retrieved=[str(x) for x in retrieved],
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            llm_calls=int(data.get("llm_calls", 0)),
        )
