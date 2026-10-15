"""报告中文化与 --report-out 自包含实验目录测试（冻结契约）。

- category / 效率统计指标名的中文显示映射（未知 category 原样显示）；
- run --report-out DIR -> DIR/eval_<adapter>_<platform>_<ts>/：报告+图表+run 副本；
- compare --report-out DIR -> DIR/ab_<tag>_<ts>/：A/B 各自 runs/ 副本 + 对比报告；
- evaluate(report_out=DIR) -> DIR/eval_<adapter>_<platform>_<ts>/：报告+summary+run 副本；
- 不传 --report-out 时 run 只在 run 目录内产出、runs/ 父目录无散落文件。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from membench import cli
from membench.adapters import MockStrongAdapter
from membench.adapters.base import BaseMemoryAdapter, IngestStats, QueryResult
from membench.data import SAMPLE_DATASET_PATH
from membench.evaluate import DatasetSpec, evaluate
from membench.report import write_run_report_md
from membench.runner import Runner

#: 拒答话术（与 heuristic judge 的拒答信号词对齐）。
REFUSAL = "我不知道，相关信息不在我的记忆中。"


def _run(tmp_path: Path, **kwargs) -> Path:
    """用 MockStrongAdapter 在内置 sample 数据集上跑一次，返回 run 目录。"""
    runner = Runner(
        MockStrongAdapter(),
        str(SAMPLE_DATASET_PATH),
        judge="heuristic",
        out_dir=str(tmp_path / "runs"),
        **kwargs,
    )
    return Path(runner.run())


# ---------------------------------------------------------------------------
# 报告中文化
# ---------------------------------------------------------------------------


def test_report_category_and_efficiency_zh(tmp_path: Path) -> None:
    """分题型表 category 列与效率统计指标名均为中文（保留英文原词）。"""
    run_dir = _run(tmp_path)
    report = (run_dir / "report.md").read_text(encoding="utf-8")

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
    md = Path(write_run_report_md(str(tmp_path), records, metrics)).read_text(encoding="utf-8")
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
    md = Path(write_run_report_md(str(tmp_path), records, metrics)).read_text(encoding="utf-8")
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


def _latest_subdir(parent: Path) -> Path:
    """返回 parent 下唯一的子目录（多于/少于一个即失败）。"""
    subs = [p for p in parent.iterdir() if p.is_dir()]
    assert len(subs) == 1, f"期望唯一子目录，实际 {[p.name for p in subs]}"
    return subs[0]


def test_cli_run_report_out_subdir(tmp_path: Path) -> None:
    """run --report-out：DIR/eval_mock-strong_default_<ts>/ 含报告、图表与 run 副本。"""
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
            "--out",
            str(tmp_path / "runs"),
            "--report-out",
            str(out),
        ]
    )
    assert rc == 0
    exp_dir = _latest_subdir(out)
    assert exp_dir.name.startswith("eval_mock-strong_default_")
    for fname in (
        "report.md",
        "metrics.json",
        "fig_category_accuracy.png",
        "fig_latency_hist.png",
    ):
        assert (exp_dir / fname).is_file(), f"缺少 {fname}"
    # 子目录内的 report.md 图片引用为原文件名（同目录）
    report = (exp_dir / "report.md").read_text(encoding="utf-8")
    assert "](fig_category_accuracy.png)" in report
    assert "](fig_latency_hist.png)" in report
    # run 副本自包含：runs/<run_id>/ 里有自己的报告/指标/逐题结果
    run_copies = [p for p in (exp_dir / "runs").iterdir() if p.is_dir()]
    assert len(run_copies) == 1, "缺少 runs/<run_id>/ 副本"
    for fname in ("report.md", "metrics.json", "results.jsonl"):
        assert (run_copies[0] / fname).is_file(), f"run 副本缺少 {fname}"


def test_cli_compare_report_out_subdir(tmp_path: Path) -> None:
    """compare --report-out：DIR/ab_<tag>_<ts>/ 含 A/B run 副本与对比报告。"""
    run_a = _run(tmp_path / "a")
    run_b = _run(tmp_path / "b")
    out = tmp_path / "out"
    rc = cli.main(
        [
            "compare",
            "--baseline",
            str(run_a),
            "--candidate",
            str(run_b),
            "--report-out",
            str(out),
        ]
    )
    assert rc == 0
    ab_dir = _latest_subdir(out)
    assert ab_dir.name.startswith("ab_")
    assert (ab_dir / "compare_report.md").is_file()
    assert (ab_dir / "fig_metric_diff.png").is_file()
    assert (ab_dir / "fig_category_diff.png").is_file()
    # A / B 各自带自包含的 runs/<run_id>/ 副本
    for arm in ("A", "B"):
        run_copies = [p for p in (ab_dir / arm / "runs").iterdir() if p.is_dir()]
        assert len(run_copies) == 1, f"{arm} 缺少 run 副本"
        assert (run_copies[0] / "report.md").is_file()
        assert (run_copies[0] / "metrics.json").is_file()


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
    """evaluate(report_out=...)：DIR/eval_<adapter>_<platform>_<ts>/ 含报告、summary 与 run 副本。"""
    ds = _mini_dataset(tmp_path)
    out = tmp_path / "out"
    rc = evaluate(
        _StubAdapter(),
        "heuristic",
        out_dir=str(tmp_path / "runs"),
        specs=[
            DatasetSpec(key="mini", name="mini", path=str(ds), source="user-specified")
        ],
        report_out=str(out),
    )
    assert rc == 0
    exp_dir = _latest_subdir(out)
    assert exp_dir.name.startswith("eval_stub_default_")
    assert (exp_dir / "evaluate_report.md").is_file()
    assert (exp_dir / "evaluate_summary.json").is_file()
    run_copies = [p for p in (exp_dir / "runs").iterdir() if p.is_dir()]
    assert len(run_copies) == 1, "缺少 runs/<run_id>/ 副本"
    for fname in ("report.md", "metrics.json", "results.jsonl"):
        assert (run_copies[0] / fname).is_file(), f"run 副本缺少 {fname}"


def test_no_report_out_behavior_unchanged(tmp_path: Path) -> None:
    """不传 --report-out：只在 run_dir 下生成产物，runs 父目录无散落文件。"""
    run_dir = _run(tmp_path)
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "results.jsonl").is_file()
    assert (run_dir / "metrics.json").is_file()
    # runs 父目录下只有 run 目录，无散落的报告/图片
    stray = [p for p in (tmp_path / "runs").iterdir() if p.is_file()]
    assert stray == []


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
