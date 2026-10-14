"""doctor 子命令的 CLI 集成测试（离线）。

用标准库 ``http.server`` 在 ``127.0.0.1`` 随机端口起线程内服务：合规服务
（三端点齐全）-> ``cli.main(["doctor", ...])`` 返回 0；不合规服务（query 缺
``answer``）-> 返回 1；非法 URL（无 scheme）-> 返回 1 且不抛 traceback。
子类化 ``ThreadingHTTPServer`` 覆写 ``server_bind`` 跳过
``HTTPServer`` 的反向 DNS，避免本机 mDNS 解析拖慢测试（同 tests/test_doctor.py）。
"""

from __future__ import annotations

import json
import socketserver
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator

import pytest

from membench import cli

#: 单条路由：``(状态码, 响应体)``，响应体序列化为 JSON。
_Route = tuple[int, Any]


class _Server(ThreadingHTTPServer):
    """跳过 ``HTTPServer.server_bind`` 的 ``socket.getfqdn()`` 反向 DNS。"""

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = host
        self.server_port = port


def _make_handler(routes: dict[str, _Route]) -> type[BaseHTTPRequestHandler]:
    """按路由表构造请求处理器（未知路径返回 404 JSON）。"""

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:  # 静音默认日志
            pass

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            spec = routes.get(self.path)
            if spec is None:
                self._send(404, b"{}")
                return
            code, body = spec
            self._send(code, json.dumps(body).encode())

        def _send(self, code: int, payload: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return _Handler


@contextmanager
def _running(routes: dict[str, _Route]) -> Iterator[str]:
    """在线程内起服务，产出其 base_url，退出时关闭。"""
    server = _Server(("127.0.0.1", 0), _make_handler(routes))
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
        "/reset": (200, {}),
        "/ingest": (200, {"input_tokens": 1, "llm_calls": 1}),
        "/query": (200, {"answer": "记忆内容", "retrieved": ["片段"]}),
    }


def test_cli_doctor_compliant_returns_0(capsys: pytest.CaptureFixture[str]) -> None:
    """合规服务：doctor 退出码 0，逐项打印三端点 PASS。"""
    with _running(_ok_routes()) as url:
        rc = cli.main(["doctor", "--target", url])

    out = capsys.readouterr().out
    assert rc == 0
    for name in ("reset", "ingest", "query"):
        assert f"[membench] doctor {name}: PASS" in out


def test_cli_doctor_missing_answer_returns_1(capsys: pytest.CaptureFixture[str]) -> None:
    """query 响应缺 answer 字段：doctor 退出码 1，query 项 FAIL。"""
    routes = _ok_routes()
    routes["/query"] = (200, {"retrieved": ["片段"]})
    with _running(routes) as url:
        rc = cli.main(["doctor", "--target", url])

    out = capsys.readouterr().out
    assert rc == 1
    assert "[membench] doctor query: FAIL" in out


def test_cli_doctor_invalid_url_returns_1_no_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """非法 URL（无 scheme）：退出码 1，不抛异常，detail 指明 URL 非法。"""
    rc = cli.main(["doctor", "--target", "not-a-url"])

    out = capsys.readouterr().out
    assert rc == 1
    for name in ("reset", "ingest", "query"):
        assert f"[membench] doctor {name}: FAIL" in out
    assert "URL 非法" in out


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
