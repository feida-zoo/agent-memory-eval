"""report 模块延迟导入 matplotlib 的回归测试。

matplotlib 是可选依赖（``pip install membench[report]``）。纯查询命令
（dataset info / list-datasets / user-data status 等）只用到 report 模块中
不依赖 matplotlib 的工具函数（copy_report / _md_table / PALETTE），必须在
无 matplotlib 环境下可用；画图函数则应抛出带安装提示的 RuntimeError，
而不是 ImportError 堆栈。
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import membench.report as report


@pytest.fixture()
def no_matplotlib(monkeypatch: pytest.MonkeyPatch):
    """模拟无 matplotlib 环境：封锁导入并重新加载 membench.report。

    sys.modules["matplotlib"] = None 会让 ``import matplotlib`` 抛 ImportError，
    等价于该包未安装。
    """
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    importlib.reload(report)
    yield report
    # 先撤掉 monkeypatch（恢复 sys.modules 中的 matplotlib），再 reload 还原真实环境，
    # 否则 teardown 的 reload 仍会拿到被封锁的 matplotlib。
    monkeypatch.undo()
    importlib.reload(report)


def test_report_importable_without_matplotlib(no_matplotlib) -> None:
    """无 matplotlib 时 report 模块可导入，工具函数/常量可用。"""
    from membench.report import PALETTE, _md_table, copy_report  # noqa: F401

    assert report.plt is None
    assert PALETTE  # 非空调色板
    table = _md_table(["a", "b"], [["1", "2"]])
    assert table.startswith("| a | b |")


def test_copy_report_works_without_matplotlib(no_matplotlib, tmp_path: Path) -> None:
    """copy_report 不依赖 matplotlib，无 matplotlib 时仍应正常工作。"""
    src = tmp_path / "src"
    src.mkdir()
    (src / "fig.png").write_bytes(b"fake-png")
    md = src / "report.md"
    md.write_text("# 报告\n\n![图](fig.png)\n", encoding="utf-8")

    dest = tmp_path / "dest"
    out = report.copy_report(str(md), str(dest), "eval_report_x_260918.md", "260918")

    assert Path(out).read_text(encoding="utf-8").count("260918_fig.png") == 1
    assert (dest / "260918_fig.png").is_file()


def test_fig_functions_raise_friendly_error_without_matplotlib(
    no_matplotlib, tmp_path: Path
) -> None:
    """无 matplotlib 时画图函数抛 RuntimeError，消息含 pip install matplotlib。"""
    with pytest.raises(RuntimeError, match="pip install matplotlib"):
        report.fig_category_accuracy({"c": {"accuracy": 0.5, "n": 1}}, str(tmp_path / "a.png"))
    with pytest.raises(RuntimeError, match="pip install matplotlib"):
        report.fig_latency_hist([1.0, 2.0], 1.0, 2.0, str(tmp_path / "b.png"))
    with pytest.raises(RuntimeError, match="pip install matplotlib"):
        report.fig_metric_diff({"accuracy": 0.5}, {"accuracy": 0.6}, "A", "B", str(tmp_path / "c.png"))


def test_cli_importable_without_matplotlib() -> None:
    """子进程中封锁 matplotlib 后，CLI 入口（含整条 import 链）仍可导入。"""
    code = (
        "import sys\n"
        "sys.modules['matplotlib'] = None\n"
        "sys.modules['matplotlib.pyplot'] = None\n"
        "import membench.cli\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_fig_functions_work_with_matplotlib(tmp_path: Path) -> None:
    """有 matplotlib 时画图正常（防回归）。"""
    pytest.importorskip("matplotlib")
    out = report.fig_category_accuracy(
        {"c": {"accuracy": 0.5, "n": 2}}, str(tmp_path / "cat.png")
    )
    assert Path(out).is_file()


class TestMdTableEscaping:
    """markdown 表格转义：换行折叠为 <br>、竖线转义，防止撑裂表格。"""

    def test_newline_and_pipe_escaped(self) -> None:
        from membench.report import _md_table

        rows = [["line1\nline2", "a|b"]]
        result = _md_table(["h1", "h2"], rows)
        assert "line1<br>line2" in result
        assert "a\\|b" in result
        # 表头行 + 分隔行 + 数据行 = 3 行，单元格内换行不应产生额外线
        assert result.count("\n") == 2

    def test_crlf_normalized(self) -> None:
        from membench.report import _md_cell

        assert _md_cell("a\r\nb\rc\nd") == "a<br>b<br>c<br>d"

    def test_plain_text_unchanged(self) -> None:
        from membench.report import _md_cell

        assert _md_cell("normal text") == "normal text"
        assert _md_cell(123) == "123"
