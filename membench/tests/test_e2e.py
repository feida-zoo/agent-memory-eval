"""端到端测试：mock-weak vs mock-strong 全流程（依赖 runner.py）。

Runner 由运行层代理实现；本文件在 runner 缺失时自动 skip。
接口约定（与运行层对齐）：

- ``Runner(adapter=<BaseMemoryAdapter>, judge="heuristic", out_dir=<目录>)``
- ``runner.run(<数据集路径>) -> run 目录路径``
- run 目录内写 ``results.jsonl`` 与 ``metrics.json``（SPEC 3.4 / 3.5）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from membench.adapters import MockStrongAdapter, MockWeakAdapter
from membench.data import SAMPLE_DATASET_PATH

runner_mod = pytest.importorskip("membench.runner", reason="runner.py 尚未实现")
Runner = runner_mod.Runner

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


def _run(adapter: object, out_dir: Path) -> dict:
    """执行一次完整评测并返回解析后的 metrics.json。"""
    runner = Runner(
        adapter=adapter,
        dataset_path=str(SAMPLE_DATASET_PATH),
        judge="heuristic",
        out_dir=str(out_dir),
    )
    run_dir = Path(runner.run())
    assert (run_dir / "results.jsonl").is_file(), "run 目录缺少 results.jsonl"
    metrics_path = run_dir / "metrics.json"
    assert metrics_path.is_file(), "run 目录缺少 metrics.json"
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def test_e2e_strong_beats_weak(tmp_path: Path) -> None:
    """strong 的 accuracy 与 stale_rate 必须全面优于 weak（SPEC 第 8 节）。"""
    weak_metrics = _run(MockWeakAdapter(), tmp_path / "weak")
    strong_metrics = _run(MockStrongAdapter(), tmp_path / "strong")

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
