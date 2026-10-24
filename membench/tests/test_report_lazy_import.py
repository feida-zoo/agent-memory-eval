"""report 模块延迟导入 matplotlib 的回归测试。

matplotlib 是可选依赖（``pip install membench[report]``）。纯查询命令
（dataset info / list-datasets / user-data status 等）只用到 report 模块中
不依赖 matplotlib 的工具函数（copy_report / _md_table / PALETTE），必须在
无 matplotlib 环境下可用；画图函数则应抛出带安装提示的 RuntimeError，
而不是 ImportError 堆栈。

另外覆盖报告渲染的三条硬约束（都在报告文本层面断言）：

- 对话级报告给出该对话记忆快照的**可点击相对链接**（``../memory_snapshot/``）
  与清单 ``entry`` 的条目数 / 字节数；
- 适配器不支持快照（``state_snapshot.role="unsupported"``）时渲染成一句可读
  文案，不留 ``n/a`` / ``vNone`` / ``None`` 这类空值，也不给打不开的链接；
- 分片产物（``shards > 1``）在报告顶部标注"仅覆盖部分题目"，题目数写成带
  分母的形式；fuse 的整体融合产物（``shard_index=None``）不标。
"""

from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import membench.report as report
from membench.adapters import MockStrongAdapter
from membench.adapters.base import BaseMemoryAdapter, IngestStats, QueryResult
from membench.runner import Runner


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
    """copy_report 不依赖 matplotlib：图片按原名复制到同目录并改写引用。"""
    src = tmp_path / "src"
    src.mkdir()
    (src / "fig_category_accuracy.png").write_bytes(b"fake-png")
    md = src / "report.md"
    md.write_text("# 报告\n\n![图](fig_category_accuracy.png)\n", encoding="utf-8")
    metrics = src / "metrics.json"
    metrics.write_text('{"run_id": "t"}\n', encoding="utf-8")

    dest = tmp_path / "dest"
    out = report.copy_report(
        str(md),
        str(dest),
        "report.md",
        extra_files=[str(metrics), str(src / "missing.jsonl")],
    )

    assert Path(out).name == "report.md"
    content = Path(out).read_text(encoding="utf-8")
    # 引用保持同目录相对文件名（layout 契约：报告内的图片引用不写绝对路径）
    assert content.count("](fig_category_accuracy.png)") == 1
    assert (dest / "fig_category_accuracy.png").is_file()
    # extra_files 按原文件名复制；不存在的文件跳过而不报错
    assert json.loads((dest / "metrics.json").read_text(encoding="utf-8")) == {"run_id": "t"}
    assert not (dest / "missing.jsonl").exists()


def test_copy_report_keeps_remote_image_refs(tmp_path: Path) -> None:
    """非本地引用（http/https）原样保留，不当成本地文件去复制。"""
    src = tmp_path / "src"
    src.mkdir()
    md = src / "report.md"
    md.write_text("![远程](https://example.invalid/a.png)\n", encoding="utf-8")

    dest = tmp_path / "dest"
    out = report.copy_report(str(md), str(dest), "report.md")

    assert "https://example.invalid/a.png" in Path(out).read_text(encoding="utf-8")
    assert sorted(p.name for p in dest.iterdir()) == ["report.md"]


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


class TestCompareFigureLegend:
    """对比报告图例：必须标出 A/B 身份，高亮色不冒充当身份色。"""

    @staticmethod
    def _legend_texts(fig) -> list[str]:
        return [
            t.get_text()
            for ax in fig.axes
            for leg in [ax.get_legend()]
            if leg is not None
            for t in leg.get_texts()
        ]

    def test_metric_diff_legend_marks_ab_and_short_names(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        monkeypatch.setattr(plt, "close", lambda fig=None: None)  # 留住 figure 以便断言图例
        report.fig_metric_diff(
            {"accuracy": 0.50, "token_f1": 0.20, "evidence_recall": 0.90},
            {"accuracy": 0.80, "token_f1": 0.22, "evidence_recall": 0.60},
            "mock-a@v1",
            "mock-b@v2",
            str(tmp_path / "diff.png"),
        )
        texts = self._legend_texts(plt.gcf())
        plt.close(plt.gcf())
        assert any("A（基线）· mock-a@v1" in t for t in texts)
        assert any("B（候选）· mock-b@v2" in t for t in texts)
        # B 的明显优化/劣化另立图例项（accuracy +0.30 优化、evidence_recall -0.30 劣化）
        assert any("B 明显优化" in t for t in texts)
        assert any("B 明显劣化" in t for t in texts)

    def test_category_diff_legend_marks_ab(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        monkeypatch.setattr(plt, "close", lambda fig=None: None)
        report.fig_category_diff(
            {"single_hop": {"accuracy": 0.5}},
            {"single_hop": {"accuracy": 1.0}},
            "mock-a@v1",
            "mock-b@v2",
            str(tmp_path / "cat.png"),
        )
        texts = self._legend_texts(plt.gcf())
        plt.close(plt.gcf())
        assert any("A（基线）· mock-a@v1" in t for t in texts)
        assert any("B（候选）· mock-b@v2" in t for t in texts)

    def test_ab_series_labels_prefers_adapter_and_commit(self) -> None:
        """短名首选 adapter@commit；两侧完全相同才退回 run_id 以免图例分不出。"""
        a, b = report.ab_series_labels(
            {"adapter": "mock-a", "commit": "abcdef1234", "run_id": "r1"},
            {"adapter": "mock-b", "run_id": "r2"},
        )
        assert (a, b) == ("mock-a@abcdef12", "mock-b")

        same_a, same_b = report.ab_series_labels(
            {"adapter": "mock-a", "run_id": "r1"}, {"adapter": "mock-a", "run_id": "r2"}
        )
        assert (same_a, same_b) == ("r1", "r2")


# ---------------------------------------------------------------------------
# 对话级报告：记忆快照链接 / 不支持快照的文案 / 分片标注
# ---------------------------------------------------------------------------

class _SnapshotAdapter(BaseMemoryAdapter):
    """最小可快照适配器：记忆收在内存里，``snapshot()`` 把它写成快照目录下的一个文件。"""

    name = "snap-min"

    def __init__(self) -> None:
        self.memories: list[dict] = []

    def reset(self) -> None:
        self.memories = []

    def ingest(self, messages: list[dict]) -> IngestStats:
        self.memories.extend(messages)
        return IngestStats(input_tokens=5, llm_calls=1)

    def query(self, question: str) -> QueryResult:
        return QueryResult(
            answer="周五",
            retrieved=["项目截止日是周五"],
            input_tokens=3,
            output_tokens=1,
        )

    def snapshot_capabilities(self) -> dict:
        return {"supported": True, "restore": True, "format_version": 1}

    def snapshot(self, target_dir: str, tag: str) -> dict:
        with open(Path(target_dir) / "memory.jsonl", "w", encoding="utf-8") as handle:
            for message in self.memories:
                handle.write(json.dumps(message, ensure_ascii=False) + "\n")
        return {"items": len(self.memories)}


def _write_dataset(path: Path, n_dialogues: int = 1) -> Path:
    """写一个「n 对话 × 1 题」的小数据集，每题都带 evidence。"""
    rows = [
        {
            "dialogue_id": f"d{i}",
            "sessions": [
                [
                    {
                        "role": "user",
                        "content": f"第{i}次记录：项目截止日是周五。",
                        "timestamp": None,
                    }
                ]
            ],
            "questions": [
                {
                    "qid": f"d{i}_q1",
                    "question": "项目截止日是哪天？",
                    "gold_answer": "周五",
                    "category": "single_hop",
                    "evidence": ["项目截止日是周五"],
                }
            ],
        }
        for i in range(1, n_dialogues + 1)
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _run_small(
    tmp_path: Path, adapter: BaseMemoryAdapter, *, n_dialogues: int = 1, **kwargs
) -> Path:
    """跑一次迷你评测，返回数据集产物目录 ``<实验目录>/runs/mini``。"""
    dataset = _write_dataset(tmp_path / "mini.jsonl", n_dialogues)
    out = Runner(
        adapter, str(dataset), judge="heuristic", out_dir=str(tmp_path / "exp"), **kwargs
    ).run()
    assert out is not None
    return Path(out)


def test_dialogue_report_links_memory_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """对话级报告给快照目录的同目录相对链接，并给出清单 entry 的条目数与字节数。

    「跑完 ingest 就能翻记忆文件」：链接必须能点开（指向真实存在的目录），
    条目数与体积必须取自 ``manifest.json`` 的 ``entry``，不是渲染时编的数字。
    """
    monkeypatch.setenv("MEMBENCH_OUTPUT_DIR", str(tmp_path / "output-root"))
    dataset_dir = _run_small(tmp_path, _SnapshotAdapter())
    report_dir = dataset_dir / "d1" / "report"
    md = (report_dir / "report.md").read_text(encoding="utf-8")

    manifest = json.loads(
        (dataset_dir / "d1" / "memory_snapshot" / "manifest.json").read_text(encoding="utf-8")
    )
    entry = manifest["entry"]

    # 报告里的本地链接逐个解析：每一个都要指向真实存在的路径
    links = [ref for ref in re.findall(r"\]\(([^)]+)\)", md) if "://" not in ref]
    assert "../memory_snapshot/" in links
    for ref in links:
        assert (report_dir / ref).exists(), f"报告里的相对链接打不开: {ref}"

    assert f"{entry['n_sessions']} session" in md
    assert f"{entry['n_messages']} 消息" in md
    assert f"{entry['n_questions']} 题" in md
    assert f"{entry['bytes']} 字节" in md


def test_dialogue_report_renders_unsupported_snapshot_readably(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """适配器不支持快照：角色写成可读文案，不留空的格式版本 / 清单 sha256 与死链接。"""
    monkeypatch.setenv("MEMBENCH_OUTPUT_DIR", str(tmp_path / "output-root"))
    dataset_dir = _run_small(tmp_path, MockStrongAdapter())
    md = (dataset_dir / "d1" / "report" / "report.md").read_text(encoding="utf-8")

    assert "| 快照角色 | 不支持（适配器未声明快照能力） |" in md
    # 没有快照就没有格式版本与清单指纹可给：这两行整体消失，而不是渲染成空值
    assert "格式版本" not in md
    assert "清单 sha256（逐对话清单合并）" not in md
    assert "vNone" not in md
    assert "`None`" not in md
    # 没导出快照就不给链接（报告里不留打不开的相对路径）
    assert "memory_snapshot" not in md


def test_shard_run_report_marks_partial_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """分片产物：报告顶部标注分片范围，题目数写成带分母的形式。"""
    monkeypatch.setenv("MEMBENCH_OUTPUT_DIR", str(tmp_path / "output-root"))
    dataset_dir = _run_small(
        tmp_path,
        MockStrongAdapter(),
        n_dialogues=2,
        shards=2,
        shard_index=0,
        shard_by="dialogue",
    )
    md = (dataset_dir / "d1" / "report" / "report.md").read_text(encoding="utf-8")

    # 顶部：标注必须在正文（评测配置）之前
    note = "分片 0/2（按 dialogue）——仅覆盖部分题目"
    assert note in md
    assert md.index(note) < md.index("## 评测配置")
    # 题目数是**本片**的：写清分母与完整结果的去向，避免被读成整场题数
    assert "- 题目数: 1（分片 0/2，总量见融合报告）" in md


def test_fused_metrics_are_not_marked_as_partial(tmp_path: Path) -> None:
    """fuse 的整体融合产物（``shards>1`` 但 ``shard_index=None``）覆盖全部题目，不标分片。"""
    metrics = {
        "run_id": "r1",
        "shards": 2,
        "shard_index": None,
        "shard_by": "dialogue",
        "fused": True,
        "overall": {"n_questions": 2},
    }
    out = report.write_run_report_md(str(tmp_path), [], metrics, with_figures=False)
    md = Path(out).read_text(encoding="utf-8")
    assert "仅覆盖部分题目" not in md
    assert "总量见融合报告" not in md
