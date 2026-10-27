"""报告中文化与自包含实验目录（--report-out）测试（冻结契约）。

- category / 效率统计指标名的中文显示映射（未知 category 原样显示）；
- ``run --report-out DIR`` -> ``DIR/eval_<adapter>_<platform>_<ts>/``：数据集级报告落
  ``runs/<dataset>/report/``（md + metrics + 逐题明细 + 两张图），任务级报告落同实验
  目录的 ``reports/``；
- ``compare --report-out DIR`` -> ``DIR/compare_<name>_<platform>_<ts>/``：``A/``、``B/``
  各为完整评测实验形态，对比报告落 ``reports/comparison.md``；
- ``evaluate(report_out=DIR)`` -> ``DIR/eval_<adapter>_<platform>_<ts>/``：任务级
  ``reports/`` 含 report.md / metrics.json / evaluate_summary.json / 数据集总览图；
- 不传 ``--report-out`` 时实验目录落 ``output_root()``，父目录无散落文件。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from membench import cli
from membench.adapters import MockStrongAdapter
from membench.adapters.base import BaseMemoryAdapter, IngestStats, QueryResult
from membench.data import SAMPLE_DATASET_PATH
from membench.evaluate import DatasetSpec, evaluate
from membench.report import write_run_report_md
from membench.runner import Runner


def _run(exp_dir: Path) -> Path:
    """把 ``exp_dir`` 当实验目录跑一次内置 sample 数据集，返回数据集产物目录。"""
    runner = Runner(
        MockStrongAdapter(),
        str(SAMPLE_DATASET_PATH),
        judge="heuristic",
        out_dir=str(exp_dir),
    )
    dataset_dir = runner.run()
    assert dataset_dir is not None
    return Path(dataset_dir)


def _only_subdir(parent: Path) -> Path:
    """返回 parent 下唯一的子目录（多于/少于一个即失败）。"""
    subs = [p for p in parent.iterdir() if p.is_dir()]
    assert len(subs) == 1, f"期望唯一子目录，实际 {[p.name for p in subs]}"
    return subs[0]


def _local_links(md: str) -> list[str]:
    """报告里的本地引用（相对文件名 / 相对路径），跳过 http(s) 外链。"""
    return [ref for ref in re.findall(r"\]\(([^)]+)\)", md) if "://" not in ref]


# ---------------------------------------------------------------------------
# 报告中文化
# ---------------------------------------------------------------------------


def test_report_category_and_efficiency_zh(tmp_path: Path) -> None:
    """分题型表 category 列与效率统计指标名均为中文（保留英文原词）。"""
    dataset_dir = _run(tmp_path / "exp")
    report = (dataset_dir / "report" / "report.md").read_text(encoding="utf-8")

    assert "单跳（single_hop）" in report
    assert "查询延迟均值（ms）" in report
    assert "查询延迟 p50（ms）" in report
    assert "查询延迟 p95（ms）" in report
    assert "平均输入 token / 题" in report
    assert "平均 LLM 调用次数 / 题" in report
    # 旧英文指标名不再出现
    assert "query latency mean (ms)" not in report
    assert "avg llm calls / query" not in report


def test_report_unknown_category_passthrough(tmp_path: Path) -> None:
    """未知 category（如 custom_type）显示原值，不报错。"""
    metrics = {
        "run_id": "t",
        "overall": {"n_questions": 1},
        "by_category": {"custom_type": {"accuracy": 1.0, "n": 1}},
    }
    records = [{"latency_ms": 1.0, "judge": {"correct": True}}]
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    md = Path(
        write_run_report_md(str(report_dir), records, metrics, with_figures=False)
    ).read_text(encoding="utf-8")
    assert "custom_type" in md
    assert "单跳" not in md


def test_report_known_category_mapping(tmp_path: Path) -> None:
    """已知题型全部映射为「中文（英文）」；数据层 key 不变。"""
    cats = [
        "single_hop",
        "multi_hop",
        "temporal",
        "knowledge_update",
        "abstention",
        "preference",
        "open_ended",
    ]
    metrics = {
        "run_id": "t",
        "overall": {"n_questions": len(cats)},
        "by_category": {c: {"accuracy": 1.0, "n": 1} for c in cats},
    }
    records = [{"latency_ms": 1.0, "judge": {"correct": True}}]
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    md = Path(
        write_run_report_md(str(report_dir), records, metrics, with_figures=False)
    ).read_text(encoding="utf-8")
    for zh in (
        "单跳（single_hop）",
        "多跳（multi_hop）",
        "时间推理（temporal）",
        "知识更新（knowledge_update）",
        "拒答（abstention）",
        "偏好（preference）",
        "开放问答（open_ended）",
    ):
        assert zh in md
    # 验收基线对照表为「中文 label（英文 key）」格式
    assert "正确率（accuracy）" in md
    assert "证据召回率（evidence_recall）" in md
    assert "陈旧率（stale_rate）" in md
    assert "拒答准确率（abstention_accuracy）" in md


# ---------------------------------------------------------------------------
# --report-out 自包含实验目录（冻结契约）
# ---------------------------------------------------------------------------


def test_cli_run_report_out_subdir(tmp_path: Path) -> None:
    """run --report-out：DIR/eval_mock-strong_default_<ts>/ 含数据集级与任务级报告。"""
    out = tmp_path / "out"
    rc = cli.main(
        [
            "run",
            "--adapter",
            "mock-strong",
            "--dataset",
            str(SAMPLE_DATASET_PATH),
            "--judge",
            "heuristic",
            "--report-out",
            str(out),
        ]
    )
    assert rc == 0
    exp_dir = _only_subdir(out)
    assert exp_dir.name.startswith("eval_mock-strong_default_")
    # 实验目录根层只有 runs/ / reports/ / runtime/ 三个子目录，不散落任何文件
    assert sorted(p.name for p in exp_dir.iterdir()) == ["reports", "runs", "runtime"]

    dataset_dir = exp_dir / "runs" / "sample_dataset"
    for fname in (
        "report.md",
        "metrics.json",
        "results.jsonl",
        "fig_category_accuracy.png",
        "fig_latency_hist.png",
    ):
        assert (dataset_dir / "report" / fname).is_file(), f"缺少 {fname}"
    # 数据集级 report.md 的图片引用为同目录相对文件名
    report = (dataset_dir / "report" / "report.md").read_text(encoding="utf-8")
    assert "](fig_category_accuracy.png)" in report
    assert "](fig_latency_hist.png)" in report

    # 每个对话一个目录：runtime/ 始终建出，report/ 落该对话的产物
    dlg_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir() and p.name != "report")
    assert dlg_dirs, "缺少对话级目录"
    for dlg_dir in dlg_dirs:
        assert (dlg_dir / "runtime").is_dir(), f"{dlg_dir.name} 缺少 runtime/"
        for fname in ("report.md", "metrics.json", "results.jsonl"):
            assert (dlg_dir / "report" / fname).is_file(), f"{dlg_dir.name} 缺少 {fname}"

    # 任务级报告 = 数据集级报告内容（md + 两图 + metrics）；逐题明细不重复落一份
    task_dir = exp_dir / "reports"
    for fname in (
        "report.md",
        "metrics.json",
        "fig_category_accuracy.png",
        "fig_latency_hist.png",
    ):
        assert (task_dir / fname).is_file(), f"任务级报告缺少 {fname}"
    assert not (task_dir / "results.jsonl").exists()
    assert (task_dir / "report.md").read_text(
        encoding="utf-8"
    ) == (dataset_dir / "report" / "report.md").read_text(encoding="utf-8")


def test_cli_compare_report_out_subdir(tmp_path: Path) -> None:
    """compare --report-out：DIR/compare_<name>_<platform>_<ts>/ 含 A/B 完整产物与对比报告。

    A 臂传实验目录、B 臂传数据集目录——两种「产物目录」入参形态都要能解析。
    """
    exp_a = tmp_path / "exp_a"
    _run(exp_a)
    dataset_b = _run(tmp_path / "exp_b")
    out = tmp_path / "out"
    rc = cli.main(
        [
            "compare",
            "--baseline",
            str(exp_a),
            "--candidate",
            str(dataset_b),
            "--report-out",
            str(out),
        ]
    )
    assert rc == 0
    cmp_dir = _only_subdir(out)
    assert cmp_dir.name.startswith("compare_mock-strong_vs_mock-strong_default_")
    assert sorted(p.name for p in cmp_dir.iterdir()) == ["A", "B", "reports", "runtime"]

    reports_dir = cmp_dir / "reports"
    for fname in ("comparison.md", "fig_metric_diff.png", "fig_category_diff.png"):
        assert (reports_dir / fname).is_file(), f"缺少 {fname}"
    comparison = (reports_dir / "comparison.md").read_text(encoding="utf-8")
    # 门禁结论与逐指标 diff 表都在对比报告里；两张对比图为同目录相对引用
    assert "门禁结论： PASS" in comparison
    assert "## 逐指标 diff（B - A）" in comparison
    assert "](fig_metric_diff.png)" in comparison
    assert "](fig_category_diff.png)" in comparison

    # A / B 各为完整评测实验形态：runs/<dataset>/report/... + 任务级 reports/
    for arm in ("A", "B"):
        arm_dir = cmp_dir / arm
        assert sorted(p.name for p in arm_dir.iterdir()) == ["reports", "runs"]
        for fname in ("report.md", "metrics.json", "fig_category_accuracy.png"):
            assert (arm_dir / "reports" / fname).is_file(), f"{arm} 任务级报告缺少 {fname}"
        dataset_dir = arm_dir / "runs" / "sample_dataset"
        for fname in ("report.md", "metrics.json", "results.jsonl"):
            assert (dataset_dir / "report" / fname).is_file(), f"{arm} 数据集报告缺少 {fname}"
        # 对话级产物随副本整体搬运
        assert (dataset_dir / "d001" / "report" / "metrics.json").is_file()

    # A 臂是实验目录入参：整树原样复制
    assert (cmp_dir / "A" / "reports" / "report.md").read_text(
        encoding="utf-8"
    ) == (exp_a / "reports" / "report.md").read_text(encoding="utf-8")
    # B 臂是数据集目录入参：先构造 runs/<dataset>/，再据数据集报告补任务级 reports/
    assert (cmp_dir / "B" / "reports" / "metrics.json").read_text(
        encoding="utf-8"
    ) == (dataset_b / "report" / "metrics.json").read_text(encoding="utf-8")


class _StubAdapter(BaseMemoryAdapter):
    """迷你脚本化适配器：固定回答，供 evaluate --report-out 测试使用。"""

    name = "stub"

    def reset(self) -> None:
        """无状态。"""

    def ingest(self, messages: list[dict]) -> IngestStats:
        """不做任何存储。"""
        return IngestStats()

    def query(self, question: str) -> QueryResult:
        return QueryResult(answer="周五", retrieved=[])


def _mini_dataset(tmp_path: Path) -> Path:
    """写一个单题迷你数据集并返回路径。"""
    ds = tmp_path / "mini.jsonl"
    ds.write_text(
        json.dumps(
            {
                "dialogue_id": "d1",
                "sessions": [[{"role": "user", "content": "随便聊聊。", "timestamp": None}]],
                "questions": [
                    {
                        "qid": "q1",
                        "question": "项目截止日是哪天？",
                        "gold_answer": "周五",
                        "category": "single_hop",
                        "evidence": [],
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return ds


def test_evaluate_report_out_subdir(tmp_path: Path) -> None:
    """evaluate(report_out=...)：DIR/eval_stub_default_<ts>/ 含数据集树与任务级 reports/。"""
    ds = _mini_dataset(tmp_path)
    out = tmp_path / "out"
    rc = evaluate(
        _StubAdapter(),
        "heuristic",
        specs=[
            DatasetSpec(key="mini", name="mini", path=str(ds), source="user-specified")
        ],
        report_out=str(out),
    )
    assert rc == 0
    exp_dir = _only_subdir(out)
    assert exp_dir.name.startswith("eval_stub_default_")
    assert sorted(p.name for p in exp_dir.iterdir()) == ["reports", "runs"]

    reports_dir = exp_dir / "reports"
    for fname in (
        "report.md",
        "metrics.json",
        "evaluate_summary.json",
        "fig_dataset_overview.png",
    ):
        assert (reports_dir / fname).is_file(), f"任务级报告缺少 {fname}"

    dataset_dir = exp_dir / "runs" / "mini"
    for fname in (
        "report.md",
        "metrics.json",
        "results.jsonl",
        "fig_category_accuracy.png",
        "fig_latency_hist.png",
    ):
        assert (dataset_dir / "report" / fname).is_file(), f"数据集报告缺少 {fname}"
    for fname in ("report.md", "metrics.json", "results.jsonl"):
        assert (dataset_dir / "d1" / "report" / fname).is_file(), f"对话报告缺少 {fname}"

    # 任务级清单记下各数据集的相对路径与门禁结论
    summary = json.loads((reports_dir / "evaluate_summary.json").read_text(encoding="utf-8"))
    assert summary["datasets"] == ["mini"]
    assert summary["n_datasets"] == 1
    assert summary["runs"] == {"mini": "runs/mini"}
    assert "gate" in summary and "fused" in summary
    # 任务级 report.md 的内链与同目录图片引用都能解析到真实文件
    links = _local_links((reports_dir / "report.md").read_text(encoding="utf-8"))
    assert "fig_dataset_overview.png" in links
    assert any(link.endswith("report/report.md") for link in links), "应内链到数据集报告"
    for link in links:
        assert (reports_dir / link).is_file(), f"引用指向不存在的文件: {link}"


def test_default_report_out_uses_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """不传 --report-out：实验目录落 output_root()（MEMBENCH_OUTPUT_DIR），父目录无散落文件。"""
    root = tmp_path / "output"
    monkeypatch.setenv("MEMBENCH_OUTPUT_DIR", str(root))
    rc = cli.main(
        [
            "run",
            "--adapter",
            "mock-strong",
            "--dataset",
            str(SAMPLE_DATASET_PATH),
            "--judge",
            "heuristic",
        ]
    )
    assert rc == 0
    exp_dir = _only_subdir(root)
    assert exp_dir.name.startswith("eval_mock-strong_default_")
    assert (exp_dir / "runs" / "sample_dataset" / "report" / "report.md").is_file()
    assert (exp_dir / "reports" / "report.md").is_file()
    # 产物根目录下没有散落的报告/图片/指标
    assert [p for p in root.iterdir() if p.is_file()] == []


def test_removed_flags_rejected(tmp_path: Path) -> None:
    """旧参数 --out / --snapshot-out / --snapshot 已删除：传入即被 argparse 拒绝（退出码 2）。"""
    run_prefix = [
        "run",
        "--adapter",
        "mock-strong",
        "--dataset",
        str(SAMPLE_DATASET_PATH),
        "--judge",
        "heuristic",
    ]
    for command in (
        run_prefix + ["--out", str(tmp_path)],
        run_prefix + ["--snapshot-out", str(tmp_path)],
        # 快照改为默认导出，开关只剩 --no-snapshot（不留 --snapshot 兼容别名）
        run_prefix + ["--snapshot"],
        ["evaluate", "--out", str(tmp_path)],
        ["evaluate", "--snapshot"],
    ):
        with pytest.raises(SystemExit) as excinfo:
            cli.main(command)
        assert excinfo.value.code == 2, f"{command} 应被 argparse 拒绝（参数已删除）"


# ---------------------------------------------------------------------------
# CLI 失败收口：预检失败不留空实验目录
# ---------------------------------------------------------------------------


def _subdirs(parent: Path) -> list[Path]:
    """parent 下的子目录（parent 不存在时为空列表）。"""
    return sorted(p for p in parent.iterdir() if p.is_dir()) if parent.is_dir() else []


def test_cli_run_failure_leaves_no_empty_experiment_dir(tmp_path: Path) -> None:
    """run 的失败路径不留空 eval_*（Runner 层"失败不留半成品"不能在 CLI 层失效）。"""
    out = tmp_path / "out"
    empty_ds = tmp_path / "empty.jsonl"
    empty_ds.write_text("", encoding="utf-8")
    base = [
        "run",
        "--adapter",
        "mock-strong",
        "--dataset",
        str(SAMPLE_DATASET_PATH),
        "--judge",
        "heuristic",
        "--report-out",
        str(out),
    ]

    # ① 构造期参数错误（--replay-verify-source 需与 --replay 同时给）→ 友好退出且不留空实验目录
    with pytest.raises(SystemExit, match="参数错误"):
        cli.main([*base, "--replay-verify-source", str(tmp_path / "nope")])
    assert _subdirs(out) == [], "参数错误失败后残留空实验目录"

    # ② --replay 指向不存在的产物目录（回放源校验失败）
    with pytest.raises(SystemExit, match="产物目录不存在"):
        cli.main([*base, "--replay", str(tmp_path / "nope")])
    assert _subdirs(out) == [], "坏 --replay 失败后残留空实验目录"

    # ③ 空数据集：runner 明确"不产生产物"，退出码 0 也不该留下空实验目录
    assert (
        cli.main(
            [
                "run",
                "--adapter",
                "mock-strong",
                "--dataset",
                str(empty_ds),
                "--judge",
                "heuristic",
                "--report-out",
                str(out),
            ]
        )
        == 0
    )
    assert _subdirs(out) == [], "空数据集跳过时残留空实验目录"


def test_cli_evaluate_failure_leaves_no_empty_experiment_dir(tmp_path: Path) -> None:
    """evaluate 的空数据集路径同样不留空 eval_*（目录由 evaluate 内部创建）。"""
    out = tmp_path / "out"
    empty_ds = tmp_path / "empty.jsonl"
    empty_ds.write_text("", encoding="utf-8")

    with pytest.raises(SystemExit):
        cli.main(
            [
                "evaluate",
                "--adapter",
                "mock-strong",
                "--judge",
                "heuristic",
                "--dataset",
                str(empty_ds),
                "--report-out",
                str(out),
            ]
        )
    assert _subdirs(out) == [], "空数据集失败后残留空实验目录"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
