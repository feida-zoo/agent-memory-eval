"""agent tool-call 循环测试（离线，无网络）。

覆盖纯文本 JSON 约定路径、原生 ``chat_with_tools`` 路径、轮数上限、handler
异常回填、``AgentLoopAdapter`` 全链路与 ``llm_ref.import_from_path`` 的接入。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

from membench.agent_loop import (
    AgentLoopAdapter,
    default_retrieved_from,
    run_tool_loop,
)
from membench.llm_ref import import_from_path

# ---------------------------------------------------------------------------
# 脚本化假客户端
# ---------------------------------------------------------------------------


class _ScriptedTextClient:
    """纯文本路径：按脚本依次返回答复，并记录收到的 prompt。"""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def chat(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.replies:
            return '{"content": "（脚本耗尽）"}'
        return self.replies.pop(0)


@dataclass
class _Resp:
    """模拟 SDK 的响应对象（非 dict），字段与原生 tool-call 响应一致。"""

    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    usage: Any = None


class _ToolsClient:
    """原生 tool-call 路径：首轮请求工具，次轮直接作答（携带 usage）。"""

    def __init__(self) -> None:
        self.calls = 0
        self.seen_tools: list[dict] = []
        self.seen_messages: list[list[dict]] = []

    def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        self.calls += 1
        self.seen_tools = tools
        self.seen_messages.append([dict(m) for m in messages])
        if self.calls == 1:
            return {
                "content": "",
                "tool_calls": [{"name": "search", "arguments": '{"q": "狗"}'}],
                "usage": {"input_tokens": 10, "output_tokens": 2},
            }
        return {
            "content": "答案是 42",
            "tool_calls": [],
            "usage": {"input_tokens": 20, "output_tokens": 5},
        }


class _AlwaysToolClient:
    """原生路径：每轮都请求工具，用于验证 max_rounds 上限能终止循环。"""

    def __init__(self) -> None:
        self.calls = 0

    def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        self.calls += 1
        return {"tool_calls": [{"name": "search", "arguments": {}}]}


class _ClassPathClient:
    """类路径延迟导入用：无参可实例化，首轮请求工具、次轮作答。"""

    def __init__(self) -> None:
        self._rounds = 0

    def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        self._rounds += 1
        if self._rounds == 1:
            return {"tool_calls": [{"name": "remember", "arguments": {"x": 1}}]}
        return {"content": "done"}


class _AgentAppClient:
    """模拟接入方 LLM：按可用工具名决定调「写入」还是「检索+作答」。"""

    def __init__(self) -> None:
        self.reset_count = 0
        self.chat_calls = 0

    def reset(self) -> None:
        self.reset_count += 1

    def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        self.chat_calls += 1
        names = {t["name"] for t in tools}
        if any(m.get("role") == "tool" for m in messages):
            return {"content": "完成了", "usage": {"input_tokens": 5, "output_tokens": 2}}
        tool_name = "retrieve" if "retrieve" in names else "remember"
        arg = {"q": "狗的品种"} if tool_name == "retrieve" else {"text": "我的狗叫旺财"}
        return {
            "tool_calls": [{"name": tool_name, "arguments": arg}],
            "usage": {"input_tokens": 8, "output_tokens": 3},
        }


def _echo_handler(text: str = "ok") -> str:
    return text


# ---------------------------------------------------------------------------
# run_tool_loop：纯文本路径
# ---------------------------------------------------------------------------


class TestRunToolLoopTextPath:
    def test_tool_call_then_answer(self) -> None:
        seen: dict = {}

        def handler(args: dict) -> dict:
            seen.update(args)
            return {"stored": True}

        tools = [
            {"name": "remember", "schema": {"type": "object"}, "handler": handler}
        ]
        client = _ScriptedTextClient(
            [
                '```json\n{"tool_calls": [{"name": "remember", '
                '"arguments": {"text": "旺财"}}]}\n```',
                '{"content": "已记住"}',
            ]
        )
        messages = [{"role": "user", "content": "我的狗叫旺财"}]
        result = run_tool_loop(client, messages, tools, max_rounds=4)

        assert seen == {"text": "旺财"}  # 代码围栏被容忍，arguments 已解析为 dict
        assert result["content"] == "已记住"
        assert result["rounds"] == 2
        assert result["usage"]["llm_calls"] == 2
        assert result["tool_calls"] == [
            {"name": "remember", "arguments": {"text": "旺财"}, "result": '{"stored": true}'}
        ]
        assert result["usage"]["input_tokens"] > 0
        assert result["usage"]["output_tokens"] > 0
        # 观察消息已作为 role="tool" 追加回 messages
        assert messages[-1]["role"] == "tool"
        assert messages[-1]["content"] == '{"stored": true}'
        # 第二轮 prompt 里能看到工具观察结果
        assert "stored" in client.prompts[1]

    def test_unparsable_output_treated_as_content(self) -> None:
        client = _ScriptedTextClient(["这就是一句大白话，不是 JSON"])
        result = run_tool_loop(client, [{"role": "user", "content": "问"}], [])
        assert result["content"] == "这就是一句大白话，不是 JSON"
        assert result["tool_calls"] == []
        assert result["rounds"] == 1

    def test_usage_estimated_when_client_gives_none(self) -> None:
        client = _ScriptedTextClient(['{"content": "答案"}'])
        result = run_tool_loop(client, [{"role": "user", "content": "问题文本"}], [])
        assert result["usage"]["input_tokens"] > 0
        assert result["usage"]["output_tokens"] > 0


# ---------------------------------------------------------------------------
# run_tool_loop：原生 tool-call 路径与上限 / 异常
# ---------------------------------------------------------------------------


class TestRunToolLoopToolsPath:
    def test_native_tool_calls_and_usage(self) -> None:
        seen: dict = {}

        def search(args: dict) -> str:
            seen.update(args)
            return json.dumps(["证据A", "证据B"], ensure_ascii=False)

        client = _ToolsClient()
        tools = [{"name": "search", "schema": {}, "handler": search}]
        messages = [{"role": "user", "content": "问题"}]
        result = run_tool_loop(client, messages, tools, max_rounds=4)

        assert seen == {"q": "狗"}  # JSON 字符串 arguments 已解析
        assert result["content"] == "答案是 42"
        assert result["rounds"] == 2
        assert result["usage"] == {
            "input_tokens": 30,
            "output_tokens": 7,
            "llm_calls": 2,
        }
        assert client.seen_tools == [{"name": "search", "schema": {}}]
        # 二轮时已带上 tool 观察消息
        assert any(m.get("role") == "tool" for m in client.seen_messages[1])

    def test_object_response_supported(self) -> None:
        class _ObjClient:
            def __init__(self) -> None:
                self.n = 0

            def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> Any:
                self.n += 1
                if self.n == 1:
                    return _Resp(tool_calls=[{"name": "search", "arguments": {}}])
                return _Resp(content="对象响应", usage=None)

        tools = [{"name": "search", "schema": {}, "handler": lambda a: "hit"}]
        result = run_tool_loop(_ObjClient(), [{"role": "user", "content": "q"}], tools)
        assert result["content"] == "对象响应"
        assert result["tool_calls"][0]["result"] == "hit"

    def test_max_rounds_cap_terminates(self) -> None:
        client = _AlwaysToolClient()
        tools = [{"name": "search", "schema": {}, "handler": lambda a: "x"}]
        result = run_tool_loop(client, [{"role": "user", "content": "q"}], tools, max_rounds=3)
        assert result["rounds"] == 3
        assert result["usage"]["llm_calls"] == 3
        assert client.calls == 3
        assert len(result["tool_calls"]) == 3

    def test_handler_exception_becomes_observation(self) -> None:
        def boom(args: dict) -> str:
            raise RuntimeError("工具炸了")

        client = _ToolsClient()
        tools = [
            {"name": "search", "schema": {}, "handler": boom},
        ]
        messages = [{"role": "user", "content": "问题"}]
        result = run_tool_loop(client, messages, tools, max_rounds=4)

        assert result["tool_calls"][0]["result"] == "错误: RuntimeError: 工具炸了"
        assert result["content"] == "答案是 42"  # 循环未崩，继续到作答
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert tool_msgs and tool_msgs[0]["content"] == "错误: RuntimeError: 工具炸了"

    def test_unknown_tool_becomes_observation(self) -> None:
        class _UnknownToolClient:
            def __init__(self) -> None:
                self.n = 0

            def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
                self.n += 1
                if self.n == 1:
                    return {"tool_calls": [{"name": "nope", "arguments": {}}]}
                return {"content": "ok"}

        result = run_tool_loop(
            _UnknownToolClient(), [{"role": "user", "content": "q"}], []
        )
        assert result["tool_calls"][0]["result"].startswith("错误:")


class TestClientResolution:
    def test_class_path_string_client(self) -> None:
        tools = [{"name": "remember", "schema": {}, "handler": lambda a: "stored"}]
        result = run_tool_loop(
            "tests.test_agent_loop:_ClassPathClient",
            [{"role": "user", "content": "q"}],
            tools,
        )
        assert result["content"] == "done"
        assert result["rounds"] == 2

    def test_invalid_client_type_raises(self) -> None:
        with pytest.raises(TypeError):
            run_tool_loop(12345, [{"role": "user", "content": "q"}], [])


# ---------------------------------------------------------------------------
# default_retrieved_from
# ---------------------------------------------------------------------------


class TestDefaultRetrievedFrom:
    def test_json_list_result(self) -> None:
        loop_result = {
            "tool_calls": [
                {"name": "retrieve_memory", "arguments": {}, "result": '["证据A", "证据B"]'}
            ]
        }
        assert default_retrieved_from("q", loop_result) == ["证据A", "证据B"]

    def test_line_split_result(self) -> None:
        loop_result = {
            "tool_calls": [{"name": "Search", "arguments": {}, "result": "证据A\n证据B\n"}]
        }
        assert default_retrieved_from("q", loop_result) == ["证据A", "证据B"]

    def test_no_retrieval_tool_returns_empty(self) -> None:
        loop_result = {"tool_calls": [{"name": "remember", "arguments": {}, "result": "x"}]}
        assert default_retrieved_from("q", loop_result) == []


# ---------------------------------------------------------------------------
# AgentLoopAdapter 全链路
# ---------------------------------------------------------------------------


class TestAgentLoopAdapter:
    def _make(self) -> AgentLoopAdapter:
        ingest_tools = [
            {"name": "remember", "schema": {}, "handler": lambda a: {"ok": True}}
        ]
        query_tools = [
            {
                "name": "retrieve",
                "schema": {},
                "handler": lambda a: json.dumps(["狗叫旺财"], ensure_ascii=False),
            }
        ]
        return AgentLoopAdapter(
            _AgentAppClient(), ingest_tools, query_tools=query_tools, system_prompt="你是记忆助手"
        )

    def test_full_chain(self) -> None:
        adapter = self._make()
        adapter.reset()
        assert adapter._client.reset_count == 1  # type: ignore[attr-defined]

        stats = adapter.ingest(
            [{"role": "user", "content": "我的狗叫旺财", "timestamp": "2024-01-01T10:00:00"}]
        )
        assert stats.llm_calls > 0
        assert stats.input_tokens > 0
        assert stats.latency_ms >= 0.0

        qr = adapter.query("我的狗叫什么？")
        assert qr.answer == "完成了"
        assert qr.retrieved == ["狗叫旺财"]
        assert qr.llm_calls > 0
        assert qr.input_tokens >= 0 and qr.output_tokens >= 0
        assert qr.latency_ms >= 0.0

    def test_reset_without_client_reset_is_noop(self) -> None:
        adapter = AgentLoopAdapter(lambda prompt: '{"content": ""}', [])
        adapter.reset()  # 客户端无 reset()，不应抛异常

    def test_describe_reports_client_and_tools(self) -> None:
        adapter = self._make()
        desc = adapter.describe()
        assert desc["tools"] == ["remember", "retrieve"]
        # 类名可能因 pytest 导入方式不同而带/不带 ``tests.`` 前缀，只看结尾
        assert desc["llm_client"].endswith("._AgentAppClient")

    def test_describe_class_path_keeps_spec(self) -> None:
        adapter = AgentLoopAdapter("tests.test_agent_loop:_ClassPathClient", [])
        assert adapter.describe()["llm_client"] == "tests.test_agent_loop:_ClassPathClient"

    def test_query_tools_default_to_ingest_tools(self) -> None:
        called: list[str] = []

        def handler(args: dict) -> str:
            called.append("hit")
            return "证据A"

        tools = [{"name": "search", "schema": {}, "handler": handler}]
        adapter = AgentLoopAdapter(_ToolsClient(), tools)
        assert adapter.query_tools == tools
        qr = adapter.query("q")
        assert called == ["hit"]
        assert qr.retrieved == ["证据A"]  # 缺省回填：search 结果按行切分


# ---------------------------------------------------------------------------
# llm_ref 接入
# ---------------------------------------------------------------------------


class TestLLMRef:
    def test_missing_colon_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="module:ClassName"):
            import_from_path("no_colon_path")

    def test_unknown_module_raises_import_error(self) -> None:
        with pytest.raises(ImportError):
            import_from_path("no_such_module_xyz:Thing")

    def test_client_judge_delegates_to_llm_ref(self) -> None:
        from membench.judges.client import ClientLLMJudge

        # 委托后仍保留原 ValueError 语义与消息
        with pytest.raises(ValueError, match="module:ClassName"):
            ClientLLMJudge("still_no_colon")
        judge = ClientLLMJudge("tests.test_client_judge:_EchoClient")
        assert judge is not None
