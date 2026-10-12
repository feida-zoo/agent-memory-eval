"""adapter 配置元数据测试：--adapter-meta 透传与 adapter.describe() 自动记录。"""

from __future__ import annotations

import json
from pathlib import Path

from membench.adapters import MockStrongAdapter
from membench.data import SAMPLE_DATASET_PATH
from membench.runner import Runner


class _DescribingAdapter(MockStrongAdapter):
    """定义 describe() 的适配器。"""

    name = "mock-describing"

    def describe(self) -> dict:
        return {"version": "1.2.3", "index": "flat"}


class _BrokenDescribeAdapter(MockStrongAdapter):
    """describe() 抛异常的适配器。"""

    name = "mock-broken-describe"

    def describe(self) -> dict:
        raise RuntimeError("describe 不可用")


def _run(tmp_path: Path, adapter, **kwargs) -> dict:
    runner = Runner(
        adapter,
        str(SAMPLE_DATASET_PATH),
        judge="heuristic",
        out_dir=str(tmp_path / "runs"),
        **kwargs,
    )
    run_dir = Path(runner.run())
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def test_adapter_meta_passthrough(tmp_path: Path) -> None:
    """--adapter-meta 自由文本原样记入 metadata["adapter_meta"]。"""
    metrics = _run(tmp_path, MockStrongAdapter(), adapter_meta="model=v3;index=hnsw")
    assert metrics["adapter_meta"] == "model=v3;index=hnsw"


def test_adapter_meta_default_none(tmp_path: Path) -> None:
    metrics = _run(tmp_path, MockStrongAdapter())
    assert metrics["adapter_meta"] is None


def test_adapter_describe_recorded(tmp_path: Path) -> None:
    """适配器定义 describe() 时结果记入 metadata["adapter_describe"]。"""
    metrics = _run(tmp_path, _DescribingAdapter())
    assert metrics["adapter_describe"] == {"version": "1.2.3", "index": "flat"}


def test_adapter_describe_failure_recorded_none(tmp_path: Path) -> None:
    """describe() 抛异常时记 None 且评测不中断。"""
    metrics = _run(tmp_path, _BrokenDescribeAdapter())
    assert metrics["adapter_describe"] is None
    assert metrics["overall"]["n_questions"] > 0


def test_adapter_without_describe(tmp_path: Path) -> None:
    """未定义 describe() 的适配器记 None。"""
    metrics = _run(tmp_path, MockStrongAdapter())
    assert metrics["adapter_describe"] is None
