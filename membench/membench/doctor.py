"""接入自检：按序探测被测服务三端点是否符合 HttpAdapter 契约。

面向"Node/Electron 等非 Python 系统用 HTTP bridge 接入"的场景（接入侧反馈 6）：
在跑一次完整评测之前，先用 :func:`check_target` 确认被测服务确实实现了
``POST /reset|/ingest|/query`` 且字段类型对齐
:class:`membench.adapters.http.HttpAdapter`，避免把适配器/契约问题误判成
记忆能力问题。

探测顺序与判据：

1. ``POST /reset``：2xx 即 PASS；
2. ``POST /ingest``：请求体 ``{"messages": [{"role","content","timestamp"}]}``，
   2xx 即 PASS；响应若是 JSON dict，则额外校验可选字段 ``input_tokens`` /
   ``llm_calls`` 为数值（缺失可接受）；
3. ``POST /query``：请求体 ``{"question": str}``，须 2xx 且响应为 JSON dict 且含
   ``answer``（str）才 PASS；``retrieved`` 若出现且非 null，必须是字符串列表。

设计取舍：

- **单端点失败绝不抛异常**：URL 非法（缺少 scheme 等）、网络错误、超时、非 2xx、
  响应非 JSON 一律记为 ``FAIL``，并把可定位问题的信息写进 ``detail``（如
  ``URL 非法（缺少 scheme？）``、``HTTP 404``、``响应非 JSON``、``缺少 answer
  字段``）。
- **``/reset`` 失败不短路后续端点**：接入方常把 reset 做成可选（例如每次 ingest
  前自行清库），若因 reset 失败就跳过 ingest/query，会把"reset 未实现但读写正常"
  的可用了服务误报为不可用。故本实现始终继续探测，``SKIP`` 当前不产出（保留该
  取值以兼容将来需要短路的情形）。
- ``retrieved`` 为 ``null`` 视为缺失：HttpAdapter 用 ``data.get("retrieved") or []``
  容忍 null，doctor 与之一致，避免把兼容实现误判为失败。
- 只用标准库 ``urllib``，不引入 requests，保证在最小环境（含 skill 自带 Python）
  也能运行。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

#: 探针文案：只用于打通链路，不参与任何真实评测内容。
PROBE_TEXT = "doctor 探针"

#: 响应可被解析为 JSON 的判定依据（后续字段校验都以此为前提）。
_JSON_CONTENT_TYPE = "application/json"


def _invalid_url_detail(url: str) -> str:
    """URL 无法解析（缺少 scheme 等）时的 ``detail``：指明非法并给出修正方向。

    与「连接失败」区分开：``--target 127.0.0.1:8787`` 这类漏 scheme 的笔误很常见，
    detail 要能让人直接看出该补 ``http://``，而不是误以为服务连不上。
    """
    return f"URL 非法（缺少 scheme？应形如 http://host:port）: {url}"


def _is_unknown_url_type(reason: object) -> bool:
    """``urlopen`` 是否因无法识别的 scheme 而失败（而非真的连不上）。

    ``urllib`` 对这类 URL 未必抛 ``ValueError``：``127.0.0.1:8787`` 会被 ``Request``
    接受（``127.0.0.1`` 被当成 scheme），到分派 handler 时才抛 ``URLError``，reason
    文案为 ``unknown url type: 127.0.0.1``。
    """
    return isinstance(reason, ValueError) or "unknown url type" in str(reason)


def _request(
    url: str,
    payload: dict[str, Any],
    timeout: float,
    api_key: str | None,
) -> tuple[int | None, bytes | None, str | None]:
    """发送一次 JSON POST。

    Returns:
        ``(status, body, error)``：``error`` 为 None 表示传输层成功（``status`` /
        ``body`` 有效）；非 None 时是可直接写进 ``detail`` 的失败描述。
    """
    headers = {"Content-Type": _JSON_CONTENT_TYPE}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    try:
        # Request 构造本身就会对无 scheme 的 URL 抛 ValueError，故一并放进 try。
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), None
    except urllib.error.HTTPError as exc:  # 4xx/5xx 仍是有效响应，交由 2xx 判定
        return exc.code, exc.read(), None
    except ValueError:  # ``not-a-url`` / ``''`` / ``'://x'`` 等：URL 无法解析
        return None, None, _invalid_url_detail(url)
    except urllib.error.URLError as exc:  # 连接拒绝、DNS 失败、未知 scheme 等
        if _is_unknown_url_type(exc.reason):  # 如漏了 http:// 的 127.0.0.1:8787
            return None, None, _invalid_url_detail(url)
        return None, None, f"连接失败：{exc.reason}"
    except (TimeoutError, OSError) as exc:  # 超时（TimeoutError）等
        return None, None, f"连接失败：{exc}"


def _decode_json(raw: bytes | None) -> tuple[Any, str | None]:
    """尝试把响应体解码为 JSON，失败时返回 ``(None, "响应非 JSON")``。"""
    if raw is None:
        return None, "响应非 JSON"
    try:
        return json.loads(raw.decode("utf-8")), None
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, "响应非 JSON"


def _is_number(value: Any) -> bool:
    """数值判定（bool 是 int 子类，显式排除）。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _result(name: str, status: str, detail: str) -> dict[str, str]:
    """构造单条 check 记录。"""
    return {"name": name, "status": status, "detail": detail}


def _transport_or_status(
    name: str, status: int | None, error: str | None
) -> dict[str, str] | None:
    """传输层失败或非 2xx 时返回 FAIL 记录，否则返回 None 表示可继续校验。"""
    if error is not None:
        return _result(name, "FAIL", error)
    if status is None or not 200 <= status < 300:
        return _result(name, "FAIL", f"HTTP {status}")
    return None


def _check_reset(base_url: str, timeout: float, api_key: str | None) -> dict[str, str]:
    """``POST /reset``：2xx 即 PASS。"""
    status, _body, error = _request(f"{base_url}/reset", {}, timeout, api_key)
    failed = _transport_or_status("reset", status, error)
    return failed or _result("reset", "PASS", f"HTTP {status}")


def _check_ingest(base_url: str, timeout: float, api_key: str | None) -> dict[str, str]:
    """``POST /ingest``：2xx 即 PASS，JSON dict 则校验可选数值字段。"""
    payload = {
        "messages": [{"role": "user", "content": PROBE_TEXT, "timestamp": None}]
    }
    status, body, error = _request(f"{base_url}/ingest", payload, timeout, api_key)
    failed = _transport_or_status("ingest", status, error)
    if failed is not None:
        return failed
    data, _jerr = _decode_json(body)
    if isinstance(data, dict):
        for field in ("input_tokens", "llm_calls"):
            if field in data and not _is_number(data[field]):
                return _result("ingest", "FAIL", f"{field} 必须是数值")
    return _result("ingest", "PASS", f"HTTP {status}")


def _check_query(base_url: str, timeout: float, api_key: str | None) -> dict[str, str]:
    """``POST /query``：2xx 且含字符串 ``answer`` 才 PASS。"""
    status, body, error = _request(
        f"{base_url}/query", {"question": PROBE_TEXT}, timeout, api_key
    )
    failed = _transport_or_status("query", status, error)
    if failed is not None:
        return failed
    data, jerr = _decode_json(body)
    if jerr is not None:
        return _result("query", "FAIL", jerr)
    if not isinstance(data, dict):
        return _result("query", "FAIL", "响应不是 JSON 对象")
    if "answer" not in data:
        return _result("query", "FAIL", "缺少 answer 字段")
    if not isinstance(data["answer"], str):
        return _result("query", "FAIL", "answer 必须是字符串")
    retrieved = data.get("retrieved")
    if retrieved is not None and (
        not isinstance(retrieved, list) or not all(isinstance(x, str) for x in retrieved)
    ):
        return _result("query", "FAIL", "retrieved 必须是字符串列表")
    return _result("query", "PASS", f"HTTP {status}")


def check_target(
    base_url: str, timeout: float = 10.0, api_key: str | None = None
) -> dict:
    """探测被测服务三端点是否符合 HttpAdapter 契约（单端点失败不抛异常）。

    Args:
        base_url: 服务根地址（尾部斜杠会被规范掉）。
        timeout: 单次请求超时（秒）。
        api_key: 非空时给每个请求加 ``Authorization: Bearer <key>``。

    Returns:
        ``{"ok": bool, "base_url": str, "checks": [...]}``。``checks`` 按
        ``reset`` / ``ingest`` / ``query`` 顺序排列，每项为
        ``{"name", "status", "detail"}``；``status`` 取 ``PASS`` / ``FAIL`` /
        ``SKIP``（当前实现不产出 SKIP），``ok`` 为 True 表示没有任何 check FAIL。
    """
    base = str(base_url).rstrip("/")
    checks = [
        _check_reset(base, timeout, api_key),
        _check_ingest(base, timeout, api_key),
        _check_query(base, timeout, api_key),
    ]
    return {
        "ok": not any(c["status"] == "FAIL" for c in checks),
        "base_url": base,
        "checks": checks,
    }
