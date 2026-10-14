"""doctor.check_target 的离线契约测试。

用标准库 ``http.server`` 在 ``127.0.0.1`` 随机端口起线程内服务，覆盖：
合规服务全 PASS、各类契约违约（非 JSON / 缺字段 / 字段类型错 / 5xx）对应 FAIL、
连接被拒与非法 URL 均不抛异常、请求头与请求体形状符合 HttpAdapter 契约。全程不依赖外网。
"""

from __future__ import annotations

import json
import socket
import socketserver
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator

from membench.doctor import PROBE_TEXT, check_target

#: 单条路由：``(状态码, 响应体, Content-Type)``，响应体为 dict 时 JSON 序列化。
_Route = tuple[int, Any, str]


class _Server(ThreadingHTTPServer):
    """跳过 ``HTTPServer`` 的反向 DNS。

    ``HTTPServer.server_bind`` 会调 ``socket.getfqdn()`` 解析本机名，某些环境首次
    要几十秒（本机 mDNS），测试只关心 127.0.0.1，直接用 host 充当 server_name。
    """

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = host
        self.server_port = port


def _make_handler(routes: dict[str, _Route], recorder: list[dict] | None):
    """按路由表构造请求处理器；recorder 非空时记录每次请求。"""

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:  # 静音默认日志
            pass

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            if recorder is not None:
                recorder.append(
                    {"path": self.path, "headers": dict(self.headers), "body": raw}
                )
            spec = routes.get(self.path)
            if spec is None:
                self._send(404, b"{}", "application/json")
                return
            code, body, ctype = spec
            payload = body if isinstance(body, bytes) else json.dumps(body).encode()
            self._send(code, payload, ctype)

        def _send(self, code: int, payload: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return _Handler


@contextmanager
def _running(
    routes: dict[str, _Route], recorder: list[dict] | None = None
) -> Iterator[str]:
    """在线程内起服务，产出其 base_url，退出时关闭。"""
    server = _Server(("127.0.0.1", 0), _make_handler(routes, recorder))
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _ok_routes() -> dict[str, _Route]:
    """三端点全合规的基本路由表。"""
    return {
        "/reset": (200, {}, "application/json"),
        "/ingest": (200, {"input_tokens": 12, "llm_calls": 3}, "application/json"),
        "/query": (
            200,
            {"answer": "记忆内容", "retrieved": ["片段一"]},
            "application/json",
        ),
    }


def _check(result: dict, name: str) -> dict:
    """按名字取出一条 check。"""
    for item in result["checks"]:
        if item["name"] == name:
            return item
    raise AssertionError(f"结果中缺少 check：{name}")


def test_compliant_server_passes() -> None:
    """合规服务：ok 为 True 且三个 check 全 PASS。"""
    with _running(_ok_routes()) as base_url:
        result = check_target(base_url)

    assert result["ok"] is True
    assert result["base_url"] == base_url
    assert [c["name"] for c in result["checks"]] == ["reset", "ingest", "query"]
    assert all(c["status"] == "PASS" for c in result["checks"])


def test_query_non_json_fails() -> None:
    """query 返回非 JSON：query FAIL 且 ok 为 False。"""
    routes = _ok_routes()
    routes["/query"] = (200, b"not json", "text/plain")
    with _running(routes) as base_url:
        result = check_target(base_url)

    assert result["ok"] is False
    query = _check(result, "query")
    assert query["status"] == "FAIL"
    assert "响应非 JSON" in query["detail"]
    assert _check(result, "reset")["status"] == "PASS"


def test_query_missing_answer_fails() -> None:
    """query 响应缺 answer 字段：query FAIL 且 detail 指明缺字段。"""
    routes = _ok_routes()
    routes["/query"] = (200, {"retrieved": ["片段一"]}, "application/json")
    with _running(routes) as base_url:
        result = check_target(base_url)

    assert result["ok"] is False
    query = _check(result, "query")
    assert query["status"] == "FAIL"
    assert "缺少 answer 字段" in query["detail"]


def test_query_bad_retrieved_type_fails() -> None:
    """retrieved 不是字符串列表：query FAIL。"""
    routes = _ok_routes()
    routes["/query"] = (200, {"answer": "x", "retrieved": [1, 2]}, "application/json")
    with _running(routes) as base_url:
        result = check_target(base_url)

    assert result["ok"] is False
    query = _check(result, "query")
    assert query["status"] == "FAIL"
    assert "retrieved 必须是字符串列表" in query["detail"]


def test_query_non_dict_json_fails() -> None:
    """query 返回 JSON 数组（非 dict）：query FAIL。"""
    routes = _ok_routes()
    routes["/query"] = (200, [1, 2, 3], "application/json")
    with _running(routes) as base_url:
        result = check_target(base_url)

    assert result["ok"] is False
    assert "响应不是 JSON 对象" in _check(result, "query")["detail"]


def test_http_500_fails() -> None:
    """query 返回 500：query FAIL 且 detail 含 HTTP 500。"""
    routes = _ok_routes()
    routes["/query"] = (500, {"error": "boom"}, "application/json")
    with _running(routes) as base_url:
        result = check_target(base_url)

    assert result["ok"] is False
    query = _check(result, "query")
    assert query["status"] == "FAIL"
    assert "HTTP 500" in query["detail"]


def test_ingest_bad_field_type_fails() -> None:
    """ingest 的 input_tokens 非数值：ingest FAIL。"""
    routes = _ok_routes()
    routes["/ingest"] = (
        200,
        {"input_tokens": "many", "llm_calls": 3},
        "application/json",
    )
    with _running(routes) as base_url:
        result = check_target(base_url)

    assert result["ok"] is False
    ingest = _check(result, "ingest")
    assert ingest["status"] == "FAIL"
    assert "input_tokens 必须是数值" in ingest["detail"]


def test_reset_failure_does_not_short_circuit() -> None:
    """reset 500 时仍继续探测 ingest/query（reset 常被接入方做成可选）。"""
    routes = _ok_routes()
    routes["/reset"] = (500, {"error": "no reset"}, "application/json")
    with _running(routes) as base_url:
        result = check_target(base_url)

    assert result["ok"] is False
    assert _check(result, "reset")["status"] == "FAIL"
    assert _check(result, "ingest")["status"] == "PASS"
    assert _check(result, "query")["status"] == "PASS"


def test_connection_refused_is_reported_not_raised() -> None:
    """连接被拒（未监听端口）：不抛异常，ok False 且三 check 全 FAIL。"""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    result = check_target(f"http://127.0.0.1:{port}", timeout=1.0)

    assert result["ok"] is False
    assert all(c["status"] == "FAIL" for c in result["checks"])
    assert all("连接失败" in c["detail"] for c in result["checks"])


def test_invalid_url_is_reported_not_raised() -> None:
    """无 scheme / 空串 / 纯空白：不抛异常，三 check 全 FAIL 且 detail 指明 URL 非法。"""
    for target in ("not-a-url", "", "   ", "://x"):
        result = check_target(target)

        assert result["ok"] is False, target
        assert all(c["status"] == "FAIL" for c in result["checks"]), target
        for check in result["checks"]:
            assert "URL 非法" in check["detail"], (target, check)
            # 必须与「连接失败」区分开
            assert "连接失败" not in check["detail"], (target, check)


def test_missing_scheme_host_port_detail_is_actionable() -> None:
    """``127.0.0.1:8787`` 这类漏 scheme 的笔误：detail 给出修正方向（补 http://）。"""
    result = check_target("127.0.0.1:8787", timeout=1.0)

    assert result["ok"] is False
    for check in result["checks"]:
        assert "URL 非法" in check["detail"]
        assert "scheme" in check["detail"]
        assert "127.0.0.1:8787" in check["detail"]


def test_api_key_and_payload_shape() -> None:
    """api_key 转成 Bearer 头；请求体形状与 HttpAdapter 契约一致。"""
    recorder: list[dict] = []
    with _running(_ok_routes(), recorder=recorder) as base_url:
        check_target(base_url, api_key="secret")

    by_path = {entry["path"]: entry for entry in recorder}
    assert set(by_path) == {"/reset", "/ingest", "/query"}
    for entry in recorder:
        headers = {k.lower(): v for k, v in entry["headers"].items()}
        assert headers["content-type"] == "application/json"
        assert headers["authorization"] == "Bearer secret"

    ingest = json.loads(by_path["/ingest"]["body"].decode())
    assert ingest == {
        "messages": [{"role": "user", "content": PROBE_TEXT, "timestamp": None}]
    }
    assert json.loads(by_path["/query"]["body"].decode()) == {"question": PROBE_TEXT}


def test_base_url_trailing_slash_normalized() -> None:
    """base_url 尾斜杠被规范掉，避免出现 ``//reset``。"""
    with _running(_ok_routes()) as base_url:
        result = check_target(base_url + "/")

    assert result["ok"] is True
    assert result["base_url"] == base_url
