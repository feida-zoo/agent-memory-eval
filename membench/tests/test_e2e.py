"""端到端测试：mock-weak vs mock-strong 全流程（依赖 runner.py）。

产物结构与接口约定（见 ``membench/layout.py``）：

- ``Runner(adapter=<BaseMemoryAdapter>, judge="heuristic", out_dir=<实验目录>)``
- ``runner.run() -> <实验目录>/runs/<dataset>``（数据集产物目录，不再是 run_id 目录）
- 数据集级产物落 ``<数据集目录>/report/``：``results.jsonl`` / ``metrics.json`` /
  ``report.md`` + 分题型与延迟两张图；每个对话有自己的 ``report/`` 与 ``runtime/``。
- 任务级报告落 ``<实验目录>/reports/``（单数据集 run = 数据集级报告内容）。

runner.py 缺失时本文件自动 skip。

（路径断言用 ``layout`` 常量：完整树形的逐层字面断言集中在
``tests/test_output_tree.py``，本文件只校验与本用例相关的产物。）
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from membench import layout
from membench.adapters import MockStrongAdapter, MockWeakAdapter
from membench.data import SAMPLE_DATASET_PATH

runner_mod = pytest.importorskip("membench.runner", reason="runner.py 尚未实现")
Runner = runner_mod.Runner

#: 数据集级报告目录的必备文件（md + metrics + 逐题明细 + 两张图）。
_DATASET_REPORT_FILES = (
    layout.FILE_REPORT_MD,
    layout.FILE_METRICS,
    layout.FILE_RESULTS,
    layout.FIG_CATEGORY,
    layout.FIG_LATENCY,
)

#: metrics.json overall 的必备字段（SPEC 3.5）。
_OVERALL_KEYS = {
    "accuracy",
    "token_f1",
    "bleu1",
    "evidence_recall",
    "abstention_accuracy",
    "stale_rate",
    "latency_p50_ms",
    "latency_p95_ms",
    "avg_input_tokens",
    "avg_llm_calls",
    "n_questions",
}


def _run(adapter: object, experiment_dir: Path) -> dict:
    """执行一次完整评测并返回解析后的数据集级 metrics.json。

    顺带校验产物落位：``run()`` 返回数据集目录，数据集级报告在其 ``report/``
    下，各对话有独立报告与运行时目录，任务级报告在实验目录的 ``reports/``。
    """
    runner = Runner(
        adapter=adapter,
        dataset_path=str(SAMPLE_DATASET_PATH),
        judge="heuristic",
        out_dir=str(experiment_dir),
    )
    dataset_dir = Path(runner.run())
    assert dataset_dir == experiment_dir / layout.DIR_RUNS / "sample_dataset"
    report_dir = dataset_dir / layout.DIR_REPORT
    missing = [name for name in _DATASET_REPORT_FILES if not (report_dir / name).is_file()]
    assert not missing, f"数据集报告目录缺少 {missing}"

    dialogues = sorted(
        path for path in dataset_dir.iterdir()
        if path.is_dir() and path.name != layout.DIR_REPORT
    )
    assert dialogues, "数据集目录下没有对话目录"
    for dialogue in dialogues:
        assert (dialogue / layout.DIR_REPORT / layout.FILE_METRICS).is_file()
        assert (dialogue / layout.DIR_REPORT / layout.FILE_RESULTS).is_file()
        assert (dialogue / layout.DIR_RUNTIME).is_dir()

    task_dir = experiment_dir / layout.DIR_REPORTS
    assert (task_dir / layout.FILE_REPORT_MD).is_file(), "实验目录缺少任务级报告"
    assert (task_dir / layout.FILE_METRICS).is_file(), "实验目录缺少任务级 metrics"

    metrics_path = report_dir / layout.FILE_METRICS
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    # 产物归属元数据：读取方据此定位数据集目录，不必再猜层级
    assert metrics["experiment_dir"] == str(experiment_dir)
    assert metrics["dataset_dir"] == str(dataset_dir)
    assert metrics["dataset"] == "sample_dataset"
    return metrics


def test_e2e_strong_beats_weak(tmp_path: Path) -> None:
    """strong 的 accuracy 与 stale_rate 必须全面优于 weak（SPEC 第 8 节）。"""
    weak_metrics = _run(MockWeakAdapter(), tmp_path / "eval_weak")
    strong_metrics = _run(MockStrongAdapter(), tmp_path / "eval_strong")

    for metrics in (weak_metrics, strong_metrics):
        assert "overall" in metrics and "by_category" in metrics
        assert _OVERALL_KEYS <= set(metrics["overall"])
        assert metrics["overall"]["n_questions"] >= 24

    weak_o = weak_metrics["overall"]
    strong_o = strong_metrics["overall"]

    assert strong_o["accuracy"] > weak_o["accuracy"]
    # stale 判据升级后（旧值命中且新值未命中才判 stale）：mock-weak 的原文拼接
    # 回答同时包含新值与旧值原文，按新口径不再计为 stale，故 weak 的 stale_rate
    # 也降为 0。这里只保留 strong 不劣于 weak 的弱断言（判据升级的行为变更，
    # 详见 judges/heuristic.py 的 is_stale）。
    assert strong_o["stale_rate"] <= weak_o["stale_rate"]

    # mock-strong 设计目标：全部答对 + 全部正确拒答 + 零陈旧。
    assert strong_o["accuracy"] == pytest.approx(1.0)
    assert strong_o["stale_rate"] == pytest.approx(0.0)
    assert strong_o["abstention_accuracy"] == pytest.approx(1.0)
    # mock-weak 从不拒答 → abstention 题全错。
    assert weak_o["abstention_accuracy"] == pytest.approx(0.0)
