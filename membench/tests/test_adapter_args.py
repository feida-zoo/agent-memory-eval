"""CLI 适配器构造参数（``--adapter-arg KEY=VALUE``）测试。

构造参数走通用机制，不给单个适配器开专用开关：KEY 用 ``inspect.signature``
校验为该适配器构造函数的形参名（注册表项是工厂函数时同样成立），值按形参注解
做最小类型转换（``int`` / ``float`` 转数字，其余保持字符串）。参数错误
（未知 key、``KEY=VALUE`` 缺 ``=``、值转不成数字、必填参数缺失）一律以可操作的
``SystemExit`` 暴露——尤其是 ``--adapter http`` 不带 ``base_url`` 时，提示怎么传参，
而不是让 ``TypeError`` 堆栈穿透。

``http`` 的真实用例走本地 stub 服务（stdlib ``http.server`` + 后台线程），
验证 ``run --adapter http --adapter-arg base_url=...`` 是 HTTP 接入的正常通路。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from membench import cli, layout
from membench.adapters import ADAPTERS
from membench.adapters.base import BaseMemoryAdapter, IngestStats, QueryResult
from membench.data import SAMPLE_DATASET_PATH


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """隔离持久化适配器目录与默认产物根：本机状态不得影响断言。"""
    empty = tmp_path / "adapters"
    empty.mkdir()
    monkeypatch.setenv("MEMBENCH_ADAPTERS_DIR", str(empty))
    monkeypatch.setenv("MEMBENCH_OUTPUT_DIR", str(tmp_path / "output-root"))


class _ConfiguredAdapter(BaseMemoryAdapter):
    """需要构造参数的假适配器（key 校验与类型转换的样本）。"""

    name = "configured"

    def __init__(self, endpoint: str, retries: int = 1, timeout: float = 5.0) -> None:
        self.endpoint = endpoint
        self.retries = retries
        self.timeout = timeout

    def reset(self) -> None:
        return None

    def ingest(self, messages: list[dict]) -> IngestStats:
        return IngestStats()

    def query(self, question: str) -> QueryResult:
        return QueryResult(answer="", retrieved=[])


class _NoArgAdapter(BaseMemoryAdapter):
    """不需要任何构造参数的适配器（用于断言"不接受构造参数"的报错口径）。"""

    name = "noarg"

    def reset(self) -> None:
        return None

    def ingest(self, messages: list[dict]) -> IngestStats:
        return IngestStats()

    def query(self, question: str) -> QueryResult:
        return QueryResult(answer="", retrieved=[])


@pytest.fixture()
def _configured_adapter():
    """注册 _ConfiguredAdapter 与一个零参适配器，用后撤回。"""
    ADAPTERS["configured"] = _ConfiguredAdapter
    ADAPTERS["noarg"] = _NoArgAdapter
    try:
        yield
    finally:
        ADAPTERS.pop("configured", None)
        ADAPTERS.pop("noarg", None)


# ---------------------------------------------------------------------------
# 直接测 _make_adapter：key 校验、类型转换、报错口径
# ---------------------------------------------------------------------------


def test_adapter_args_bind_to_constructor(_configured_adapter: None) -> None:
    """合法 KEY=VALUE 生效：字符串原样、int/float 注解按数字转换。"""
    adapter = cli._make_adapter(
        "configured", ["endpoint=http://127.0.0.1:9", "retries=3", "timeout=2.5"]
    )
    assert adapter.endpoint == "http://127.0.0.1:9"
    assert adapter.retries == 3 and isinstance(adapter.retries, int)
    assert adapter.timeout == 2.5 and isinstance(adapter.timeout, float)
    # 不给的可选参数用构造函数的默认值
    assert cli._make_adapter("configured", ["endpoint=x"]).retries == 1


def test_adapter_arg_value_may_contain_equals(_configured_adapter: None) -> None:
    """只按第一个 ``=`` 切分：值里的 ``=``（query string 等）原样保留。"""
    adapter = cli._make_adapter("configured", ["endpoint=http://host:9/path?a=b&c=d"])
    assert adapter.endpoint == "http://host:9/path?a=b&c=d"


def test_builtin_adapter_accepts_adapter_arg() -> None:
    """内置适配器同样走这条路（mock-weak 的 top_k / max_answer_chars）。"""
    adapter = cli._make_adapter("mock-weak", ["top_k=2", "max_answer_chars=50"])
    assert adapter.top_k == 2 and adapter.max_answer_chars == 50


def test_unknown_adapter_arg_lists_accepted_params(_configured_adapter: None) -> None:
    """不认识的 key 直接报错，并列出该适配器可接受的参数名。"""
    with pytest.raises(SystemExit) as excinfo:
        cli._make_adapter("configured", ["base_url=http://127.0.0.1:9"])
    message = str(excinfo.value)
    assert "不认识参数 base_url" in message
    for name in ("endpoint", "retries", "timeout"):
        assert name in message


def test_missing_required_adapter_arg_is_actionable(_configured_adapter: None) -> None:
    """必填参数缺失：TypeError 转成提示 --adapter-arg 的可读报错。"""
    with pytest.raises(SystemExit) as excinfo:
        cli._make_adapter("configured")
    message = str(excinfo.value)
    assert "endpoint" in message and "--adapter-arg KEY=VALUE" in message


def test_adapter_arg_without_equals_reports_format(_configured_adapter: None) -> None:
    """``KEY`` 少了 ``=``：说明 KEY=VALUE 形式，而不是当成 key 处理。"""
    with pytest.raises(SystemExit) as excinfo:
        cli._make_adapter("configured", ["endpoint"])
    message = str(excinfo.value)
    assert "KEY=VALUE" in message and "endpoint" in message


def test_numeric_adapter_arg_rejects_non_numeric(_configured_adapter: None) -> None:
    """注解是 int / float 的值转不动数字时明确报错（不静默留字符串）。"""
    with pytest.raises(SystemExit, match="retries=abc.*int"):
        cli._make_adapter("configured", ["endpoint=x", "retries=abc"])


def test_adapter_without_ctor_params_rejects_arg(_configured_adapter: None) -> None:
    """零参适配器传了参数：明确"不接受构造参数"。"""
    with pytest.raises(SystemExit, match="不接受构造参数"):
        cli._make_adapter("noarg", ["top_k=2"])


def test_factory_registered_adapter_uses_factory_signature() -> None:
    """注册表项是工厂函数（插件常用写法）时按工厂签名判参数。"""
    ADAPTERS["factory"] = lambda: _ConfiguredAdapter("http://127.0.0.1:1")  # noqa: E731
    try:
        assert cli._make_adapter("factory").endpoint == "http://127.0.0.1:1"
        with pytest.raises(SystemExit, match="不接受构造参数"):
            cli._make_adapter("factory", ["endpoint=x"])
    finally:
        ADAPTERS.pop("factory", None)


def test_unknown_adapter_name_lists_available() -> None:
    with pytest.raises(SystemExit, match="未知适配器"):
        cli._make_adapter("no-such-adapter")


# ---------------------------------------------------------------------------
# http：真实适配器的构造参数
# ---------------------------------------------------------------------------


def test_http_adapter_bound_by_adapter_arg() -> None:
    """``--adapter http --adapter-arg base_url=...`` 直接可用，无需包装适配器。"""
    pytest.importorskip("requests")
    from membench.adapters.http import HttpAdapter

    adapter = cli._make_adapter(
        "http", ["base_url=http://127.0.0.1:8799/", "timeout=10", "api_key=secret"]
    )
    assert isinstance(adapter, HttpAdapter)
    assert adapter.base_url == "http://127.0.0.1:8799"  # 尾部斜杠规范化
    assert adapter.timeout == 10.0 and adapter.api_key == "secret"


def test_http_adapter_without_base_url_explains_param() -> None:
    """``--adapter http`` 不带参数：报错必须点出缺 base_url 与怎么补。"""
    pytest.importorskip("requests")
    with pytest.raises(SystemExit) as excinfo:
        cli._make_adapter("http")
    message = str(excinfo.value)
    assert "base_url" in message
    assert "--adapter-arg KEY=VALUE" in message


class _StubBridge(BaseHTTPRequestHandler):
    """最小三端点 stub：记录收到的请求路径，按契约回 JSON。"""

    paths: list[str] = []

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 的命名约定
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        type(self).paths.append(self.path)
        payload = {"answer": "stub", "retrieved": []} if self.path == "/query" else {}
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        """静音 stdout：测试输出只留断言结果。"""


def test_cli_run_with_http_adapter_hits_stub(tmp_path: Path) -> None:
    """端到端：run --adapter http --adapter-arg base_url=<stub> 真把请求发到服务端。"""
    pytest.importorskip("requests")

    class Handler(_StubBridge):
        paths: list[str] = []

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        rc = cli.main(
            [
                "run",
                "--adapter",
                "http",
                "--adapter-arg",
                f"base_url={base_url}",
                "--dataset",
                str(SAMPLE_DATASET_PATH),
                "--judge",
                "heuristic",
                "--limit",
                "1",
                "--report-out",
                str(tmp_path),
            ]
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert rc == 0
    for path in ("/reset", "/ingest", "/query"):
        assert path in Handler.paths
    experiments = [p for p in tmp_path.glob("eval_*") if p.is_dir()]
    assert len(experiments) == 1
    metrics = json.loads(
        (
            experiments[0]
            / layout.DIR_RUNS
            / "sample_dataset"
            / layout.DIR_REPORT
            / layout.FILE_METRICS
        ).read_text(encoding="utf-8")
    )
    assert metrics["overall"]["n_questions"] == 1
