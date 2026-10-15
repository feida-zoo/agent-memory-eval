"""共享 agent tool-call 循环：ingest 与 query 复用同一套驱动逻辑。

接入方的 **Python** 适配器在实现 ``ingest``（喂会话让 LLM 调记忆写入工具
落库）与 ``query``（让 LLM 调检索工具作答）时，两段是同一套骨架：拼 prompt /
调模型 / 分发工具 / 累计 usage / 回填证据。本模块把这段循环下沉为可复用 API。

**适用边界**：仅面向"写入/读取函数与 LLM 客户端都能在 **Python 侧**调用"的
接入方。非 Python 系统（Node/Electron）的 tool-call 循环必须运行在被测进程
内，membench 无法复用——这类接入走 ``HttpAdapter`` 三端点契约（循环留在
bridge 内，见 ``examples/http_bridge_node/``）。

- :func:`run_tool_loop`：纯函数，跑一轮 tool-call 循环，返回结构化结果；
- :class:`AgentLoopAdapter`：把该循环接到 membench 三方法契约（``reset`` /
  ``ingest`` / ``query``）上，接入方只需给出 ``llm_client`` 与工具即可。

客户端三选一（与 :mod:`membench.judges.client` 同约定）：

1. 对象，带 ``chat_with_tools(messages, tools)``（**优先**，走原生 tool-call）；
2. 对象，带 ``chat(prompt)`` 或 ``complete(prompt)``（走纯文本 JSON 约定）；
3. 可调用 ``fn(prompt) -> str``；或 ``"module:ClassName"`` 类路径字符串
   （经 :func:`membench.llm_ref.import_from_path` 延迟导入）。
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

from .adapters.base import BaseMemoryAdapter, IngestStats, QueryResult, estimate_tokens
from .llm_ref import import_from_path

#: 从模型文本输出中抠首个 JSON 对象（容忍代码围栏与多余前后缀），与判分器同口径。
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

#: 纯文本路径的输出约定：模型只能输出这两种 JSON 之一。
_OUTPUT_INSTRUCTION = (
    "请只输出一个 JSON 对象，不要输出任何其他内容：\n"
    '需要调用工具时输出 {"tool_calls":[{"name":"工具名","arguments":{...}}]}；\n'
    '直接回答时输出 {"content":"回答文本"}。'
)

#: 默认检索类工具名关键词（用于 :func:`default_retrieved_from` 的证据回填）。
_RETRIEVAL_TOOL_KEYWORDS: tuple[str, ...] = ("retrieve", "search", "recall")

_DEFAULT_INGEST_PROMPT = (
    "请阅读下面的对话，调用可用工具把值得长期记忆的信息写入记忆库，"
    "写入完成后给出简短确认。"
)

_DEFAULT_QUERY_PROMPT = (
    "请基于记忆库回答用户的问题；必要时先调用检索工具获取证据，再给出最终答案。"
)


# ---------------------------------------------------------------------------
# 客户端解析与调用
# ---------------------------------------------------------------------------


def _resolve_client(client: Any) -> Any:
    """把 client 归一为对象（字符串类路径延迟导入），校验其可驱动。

    Raises:
        TypeError: client 既不带 ``chat_with_tools`` 也不带 ``chat``/``complete``
            且不可调用。
    """
    if isinstance(client, str):
        client = import_from_path(client)
    if hasattr(client, "chat_with_tools"):
        return client
    if hasattr(client, "chat") or hasattr(client, "complete") or callable(client):
        return client
    raise TypeError(
        "client 需带 .chat_with_tools(messages, tools) 或 .chat(prompt) 方法，"
        "或为可调用对象 / 'module:ClassName' 类路径字符串"
    )


def _client_label(client: Any) -> str:
    """返回用于落账的驱动 LLM 标识（类路径字符串或对象类型名）。"""
    if isinstance(client, str):
        return client
    cls = type(client)
    if cls.__module__ == "builtins":
        return str(getattr(client, "__qualname__", cls.__name__))
    return f"{cls.__module__}.{cls.__qualname__}"


def _call_text_client(client: Any, prompt: str) -> str:
    """按 ``.chat`` > ``.complete`` > 可调用 的顺序驱动纯文本客户端。"""
    if hasattr(client, "chat"):
        return str(client.chat(prompt))
    if hasattr(client, "complete"):
        return str(client.complete(prompt))
    if callable(client):
        return str(client(prompt))
    raise TypeError("client 需带 .chat(str)/.complete(str) 方法，或直接是可调用对象")


def _field(obj: Any, key: str) -> Any:
    """从 dict 或对象上取值（不存在返回 None），兼容两种响应形态。"""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _int_or_none(value: Any) -> int | None:
    """尽力转 int；不可转时返回 None。"""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _usage_tokens(usage: Any, est_input: str, content: str) -> tuple[int, int]:
    """取本轮 token 数：客户端给就取，否则用 :func:`estimate_tokens` 估算。"""
    tin = _int_or_none(_field(usage, "input_tokens"))
    if tin is None:
        tin = _int_or_none(_field(usage, "prompt_tokens"))
    tout = _int_or_none(_field(usage, "output_tokens"))
    if tout is None:
        tout = _int_or_none(_field(usage, "completion_tokens"))
    if tin is None:
        tin = estimate_tokens(est_input)
    if tout is None:
        tout = estimate_tokens(content)
    return tin, tout


# ---------------------------------------------------------------------------
# 渲染与解析
# ---------------------------------------------------------------------------


def _render_messages(messages: list[dict]) -> str:
    """把消息列表渲染为带 role（必要时带工具名/时间戳）的文本。"""
    lines: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "user"))
        name = msg.get("name")
        if name:
            role = f"{role}:{name}"
        prefix = f"[{role}]"
        ts = msg.get("timestamp")
        if ts:
            prefix += f"({ts})"
        lines.append(f"{prefix} {msg.get('content', '')}")
    return "\n".join(lines)


def _render_tools(tool_schemas: list[dict[str, Any]]) -> str:
    """把工具名与 JSON schema 渲染成纯文本说明。"""
    if not tool_schemas:
        return "（无可用工具）"
    return "\n".join(
        f"- {t['name']}: {json.dumps(t['schema'], ensure_ascii=False)}"
        for t in tool_schemas
    )


def _render_text_prompt(
    system_prompt: str | None,
    tool_schemas: list[dict[str, Any]],
    messages: list[dict],
) -> str:
    """拼装纯文本路径的 prompt：system + 工具说明 + 输出约定 + 对话。"""
    parts: list[str] = []
    if system_prompt:
        parts.append(str(system_prompt))
    parts.append("可用工具：")
    parts.append(_render_tools(tool_schemas))
    parts.append(_OUTPUT_INSTRUCTION)
    parts.append("对话：")
    parts.append(_render_messages(messages))
    return "\n".join(parts)


def _try_parse_json(text: str) -> dict | None:
    """尝试把文本解析为 JSON dict（先整体、后正则抠首个对象）。

    与 :mod:`membench.judges.client` 的 ``_parse`` 同思路；解析不出返回 None，
    由调用方当作纯文本 content 处理，绝不抛异常。
    """
    stripped = (text or "").strip()
    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            return data
    except (ValueError, TypeError):
        pass
    match = _JSON_RE.search(stripped)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _read_text_response(raw: str) -> tuple[str, list[dict]]:
    """解析纯文本响应为 ``(content, tool_calls)``；解析失败整体当 content。"""
    data = _try_parse_json(raw)
    if isinstance(data, dict):
        calls = data.get("tool_calls")
        if isinstance(calls, list) and calls:
            return str(data.get("content") or ""), [
                c for c in calls if isinstance(c, dict)
            ]
        return str(data.get("content") or ""), []
    return raw or "", []


def _read_tools_response(resp: Any) -> tuple[str, list[dict], Any]:
    """解析原生 tool-call 响应（dict 或对象）为 ``(content, tool_calls, usage)``。"""
    if isinstance(resp, str):
        return resp, [], None
    content = _field(resp, "content")
    calls = _field(resp, "tool_calls")
    usage = _field(resp, "usage")
    if not isinstance(calls, list):
        calls = []
    return ("" if content is None else str(content)), [
        c for c in calls if isinstance(c, dict)
    ], usage


def _normalize_call(call: dict) -> tuple[str, Any]:
    """归一化单个 tool_call 为 ``(name, raw_arguments)``。

    同时兼容 ``{"name":..., "arguments":...}`` 与 OpenAI 风格的
    ``{"function": {"name":..., "arguments":...}}``。
    """
    fn = call.get("function")
    if isinstance(fn, dict):
        return str(fn.get("name", "")), fn.get("arguments")
    return str(call.get("name", "")), call.get("arguments")


def _coerce_arguments(raw: Any) -> dict:
    """把 tool_call 的 arguments 归一为 dict（JSON 字符串会被解析）。"""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"arguments 需为 JSON 对象，收到 {type(data).__name__}")
        return data
    raise ValueError(f"arguments 类型不支持: {type(raw).__name__}")


def _observation(value: Any) -> str:
    """把 handler 返回值转成回填给模型的观察文本（dict/list 走 JSON）。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _run_handler(
    handlers: dict[str, Any], name: str, raw_args: Any
) -> tuple[dict, str]:
    """执行工具 handler，任何异常都折叠为观察文本，绝不让循环崩。

    Returns:
        ``(arguments, observation)``——arguments 为解析后的 dict，observation
        为回填给模型的文本（异常时为 ``错误: <类型>: <消息>``）。
    """
    handler = handlers.get(name)
    if not callable(handler):
        return {}, f"错误: ValueError: 未知工具 {name!r}"
    try:
        args = _coerce_arguments(raw_args)
    except Exception as exc:  # noqa: BLE001 — 参数解析失败也当观察回填
        return {}, f"错误: {type(exc).__name__}: {exc}"
    try:
        return args, _observation(handler(args))
    except Exception as exc:  # noqa: BLE001 — 工具异常绝不能中断整个循环
        return args, f"错误: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# 循环主体
# ---------------------------------------------------------------------------


def run_tool_loop(
    client: Any,
    messages: list[dict],
    tools: list[dict[str, Any]],
    max_rounds: int = 8,
    system_prompt: str | None = None,
) -> dict:
    """驱动一轮 agent tool-call 循环，直到模型不再请求工具或达到轮数上限。

    每轮：调一次模型 → 若有 tool_calls 则依序执行 handler，把结果作为
    ``role="tool"`` 的观察消息**追加到 messages**（原地修改，调用方传入的列表
    会被续写）；若无 tool_calls 则结束。到 ``max_rounds`` 即停。handler 抛异常
    时把 ``错误: <类型>: <消息>`` 作为观察回填，循环继续。

    Args:
        client: 见模块 docstring 的三种形态（``chat_with_tools`` 优先）。
        messages: 对话消息列表（会被续写，建议传副本）。
        tools: 工具定义列表，元素形如
            ``{"name": str, "schema": dict, "handler": callable}``；
            handler 接收解析后的 ``arguments`` dict，返回 str 或 dict。
        max_rounds: 最大轮数（每轮一次模型调用）。
        system_prompt: 可选的系统提示词（纯文本路径会拼进 prompt）。

    Returns:
        ``{"content": str, "tool_calls": [{"name", "arguments", "result"}...],
        "usage": {"input_tokens", "output_tokens", "llm_calls"}, "rounds": int}``。
        客户端不给 token 数时用 :func:`estimate_tokens` 估算；``llm_calls`` 为
        实际调用次数。
    """
    resolved = _resolve_client(client)
    tool_schemas = [
        {"name": str(t.get("name", "")), "schema": t.get("schema") or {}}
        for t in tools
    ]
    handlers = {str(t.get("name", "")): t.get("handler") for t in tools}
    use_tools_api = hasattr(resolved, "chat_with_tools")

    executed: list[dict[str, Any]] = []
    input_tokens = 0
    output_tokens = 0
    llm_calls = 0
    rounds = 0
    content = ""

    for _ in range(max(0, int(max_rounds))):
        rounds += 1
        if use_tools_api:
            resp = resolved.chat_with_tools(messages, tool_schemas)
            content, raw_calls, usage = _read_tools_response(resp)
            est_input = _render_messages(messages) + _render_tools(tool_schemas)
        else:
            prompt = _render_text_prompt(system_prompt, tool_schemas, messages)
            content, raw_calls = _read_text_response(_call_text_client(resolved, prompt))
            usage = None
            est_input = prompt
        llm_calls += 1
        tin, tout = _usage_tokens(usage, est_input, content)
        input_tokens += tin
        output_tokens += tout

        if not raw_calls:
            break

        messages.append({"role": "assistant", "content": content})
        for call in raw_calls:
            name, raw_args = _normalize_call(call)
            args, observation = _run_handler(handlers, name, raw_args)
            executed.append({"name": name, "arguments": args, "result": observation})
            messages.append({"role": "tool", "name": name, "content": observation})

    return {
        "content": content,
        "tool_calls": executed,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "llm_calls": llm_calls,
        },
        "rounds": rounds,
    }


# ---------------------------------------------------------------------------
# 证据回填
# ---------------------------------------------------------------------------


def _as_evidence_list(value: Any) -> list[str]:
    """把工具结果解析为证据条目列表（JSON 列表优先，否则按行切分）。"""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, dict):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            return [line.strip() for line in text.splitlines() if line.strip()]
        return [str(x) for x in data] if isinstance(data, list) else []
    return [str(value)]


def default_retrieved_from(question: str, loop_result: dict) -> list[str]:
    """缺省证据回填：从检索类工具调用的结果里取 ``retrieved`` 条目。

    匹配规则：工具名（小写）**包含** ``retrieve`` / ``search`` / ``recall``
    之一即视为检索类工具（因此 ``retrieve``、``retrieve_memory``、
    ``memory_search`` 都命中）。取第一个结果非空的匹配调用，其结果先按 JSON
    列表解析，解析不出再按行切分；都没有则返回 ``[]``。

    这一步决定了 ``evidence_recall`` 等证据指标是否有意义：若不提供自定义
    ``retrieved_from``，请确保检索工具名含上述关键词、且结果是被检索到的
    记忆原文（JSON 列表或一行一条）。

    Args:
        question: 当前问题（缺省实现未使用，留给自定义实现）。
        loop_result: :func:`run_tool_loop` 的返回值。
    """
    del question  # 缺省实现不依赖问题文本
    for call in loop_result.get("tool_calls", []):
        name = str(call.get("name", "")).lower()
        if not any(k in name for k in _RETRIEVAL_TOOL_KEYWORDS):
            continue
        items = _as_evidence_list(call.get("result"))
        if items:
            return items
    return []


# ---------------------------------------------------------------------------
# 适配器
# ---------------------------------------------------------------------------


class AgentLoopAdapter(BaseMemoryAdapter):
    """把「驱动 LLM + 工具」的循环接到 membench 三方法契约上。

    接入方只需给出 ``llm_client`` 与工具定义，membench 负责循环、统计
    （:class:`IngestStats` / :class:`QueryResult`）与 ``retrieved`` 回填。
    同一套循环服务 Python 适配器的 ingest 与 query，避免各写一遍 tool-call
    骨架。非 Python 系统（Node/Electron）的循环运行在其自身进程内，本类不适用。

    Attributes:
        name: 注册名（默认 ``"agent-loop"``，可由构造参数覆盖）。
    """

    name = "agent-loop"

    def __init__(
        self,
        llm_client: Any,
        ingest_tools: list[dict[str, Any]],
        query_tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        name: str = "agent-loop",
        max_rounds: int = 8,
        retrieved_from: Callable[[str, dict], list[str]] | None = None,
        ingest_prompt: str | None = None,
        query_prompt: str | None = None,
    ) -> None:
        """初始化适配器。

        Args:
            llm_client: LLM 客户端对象 / 可调用对象 / ``"module:ClassName"``
                类路径字符串（构造时即解析）。
            ingest_tools: 写入记忆的工具列表（``{"name","schema","handler"}``）。
            query_tools: 检索记忆的工具列表；``None`` 时复用 ``ingest_tools``
                （单一工具集场景）。
            system_prompt: 透传给 :func:`run_tool_loop` 的系统提示词。
            name: 适配器注册名。
            max_rounds: 单次 ingest/query 的最大循环轮数。
            retrieved_from: 自定义证据回填；缺省用 :func:`default_retrieved_from`
                （签名 ``(question, loop_result) -> list[str]``）。
            ingest_prompt: 覆盖缺省的 ingest 指令文本（对话正文会自动附加）。
            query_prompt: 覆盖缺省的 query 指令文本（问题会自动附加）。
        """
        self._llm_label = _client_label(llm_client)
        self._client = _resolve_client(llm_client)
        self.ingest_tools = list(ingest_tools or [])
        self.query_tools = list(
            query_tools if query_tools is not None else (ingest_tools or [])
        )
        self.system_prompt = system_prompt
        self.name = name
        self.max_rounds = max_rounds
        self._retrieved_from = retrieved_from or default_retrieved_from
        self.ingest_prompt = ingest_prompt
        self.query_prompt = query_prompt

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """清空记忆：调用可选 ``client.reset()``（不存在则 no-op）。"""
        reset = getattr(self._client, "reset", None)
        if callable(reset):
            reset()

    def ingest(self, messages: list[dict]) -> IngestStats:
        """把对话渲染为带 role/timestamp 的文本，驱动 LLM 调写入工具落库。"""
        t0 = time.perf_counter()
        prompt = (self.ingest_prompt or _DEFAULT_INGEST_PROMPT) + "\n\n" + _render_messages(
            messages
        )
        result = run_tool_loop(
            self._client,
            [{"role": "user", "content": prompt}],
            self.ingest_tools,
            max_rounds=self.max_rounds,
            system_prompt=self.system_prompt,
        )
        usage = result["usage"]
        return IngestStats(
            input_tokens=usage["input_tokens"],
            llm_calls=usage["llm_calls"],
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def query(self, question: str) -> QueryResult:
        """驱动 LLM 调检索工具作答，并把证据回填到 ``retrieved``。"""
        t0 = time.perf_counter()
        prompt = (self.query_prompt or _DEFAULT_QUERY_PROMPT) + "\n\n" + f"问题：{question}"
        result = run_tool_loop(
            self._client,
            [{"role": "user", "content": prompt}],
            self.query_tools,
            max_rounds=self.max_rounds,
            system_prompt=self.system_prompt,
        )
        usage = result["usage"]
        return QueryResult(
            answer=result["content"],
            retrieved=list(self._retrieved_from(question, result)),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            llm_calls=usage["llm_calls"],
        )

    def describe(self) -> dict:
        """返回驱动 LLM 标识与工具名列表（供 run 元数据落账）。"""
        names: list[str] = []
        for tool in list(self.ingest_tools) + list(self.query_tools):
            tool_name = str(tool.get("name", ""))
            if tool_name and tool_name not in names:
                names.append(tool_name)
        return {"llm_client": self._llm_label, "tools": names}


__all__ = [
    "run_tool_loop",
    "AgentLoopAdapter",
    "default_retrieved_from",
]
