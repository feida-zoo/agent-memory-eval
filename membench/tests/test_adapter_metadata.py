"""adapter 配置元数据测试：--adapter-meta 透传与 adapter.describe() 自动记录。

metrics 读自数据集级 ``<实验目录>/runs/<dataset>/report/metrics.json``
（见 :mod:`membench.layout`），报告段读同目录的 ``report.md``。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from membench.adapters import MockStrongAdapter
from membench.data import SAMPLE_DATASET_PATH
from membench.report import ADAPTER_DESCRIBE_VALUE_MAX, adapter_describe_rows
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


class _NonDictDescribeAdapter(MockStrongAdapter):
    """describe() 返回非 dict 的适配器（违反契约）。"""

    name = "mock-nondict-describe"

    def describe(self) -> dict:
        return ["not", "a", "dict"]  # type: ignore[return-value]


class _UnserializableDescribeAdapter(MockStrongAdapter):
    """describe() 返回含不可 JSON 序列化值的 dict。"""

    name = "mock-unserializable-describe"

    def describe(self) -> dict:
        return {"bad": {1, 2, 3}}


class _PlainAdapter:
    """不继承 BaseMemoryAdapter、也没有 describe() 的 duck-typed 适配器。"""

    name = "mock-plain"

    def __init__(self) -> None:
        self._inner = MockStrongAdapter()

    def reset(self) -> None:
        self._inner.reset()

    def ingest(self, messages):
        return self._inner.ingest(messages)

    def query(self, question):
        return self._inner.query(question)


def _run(tmp_path: Path, adapter, **kwargs) -> dict:
    """把 ``tmp_path/exp`` 当实验目录跑一次，返回数据集级 metrics。"""
    runner = Runner(
        adapter,
        str(SAMPLE_DATASET_PATH),
        judge="heuristic",
        out_dir=str(tmp_path / "exp"),
        **kwargs,
    )
    dataset_dir = runner.run()
    assert dataset_dir is not None
    return json.loads(
        (Path(dataset_dir) / "report" / "metrics.json").read_text(encoding="utf-8")
    )


def _report_text(tmp_path: Path) -> str:
    """读取 ``tmp_path/exp`` 这次 run 的数据集级 report.md。"""
    return (
        tmp_path / "exp" / "runs" / SAMPLE_DATASET_PATH.stem / "report" / "report.md"
    ).read_text(encoding="utf-8")


def test_base_adapter_default_describe_empty() -> None:
    """基类默认 describe() 返回空 dict（不是什么都没声明）。"""
    assert MockStrongAdapter().describe() == {}


def test_adapter_meta_passthrough(tmp_path: Path) -> None:
    """--adapter-meta 自由文本原样记入 metadata["adapter_meta"]。"""
    metrics = _run(tmp_path, MockStrongAdapter(), adapter_meta="model=v3;index=hnsw")
    assert metrics["adapter_meta"] == "model=v3;index=hnsw"


def test_adapter_meta_default_none(tmp_path: Path) -> None:
    metrics = _run(tmp_path, MockStrongAdapter())
    assert metrics["adapter_meta"] is None


def test_adapter_describe_recorded(tmp_path: Path) -> None:
    """适配器定义 describe() 时结果记入 metadata["adapter_describe"] 并进报告。"""
    metrics = _run(tmp_path, _DescribingAdapter())
    assert metrics["adapter_describe"] == {"version": "1.2.3", "index": "flat"}
    report = _report_text(tmp_path)
    assert "### 适配器自描述" in report
    assert "| version | 1.2.3 |" in report
    assert "| index | flat |" in report


def test_adapter_describe_failure_recorded_none(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """describe() 抛异常时记 None、打印一次警告，且评测不中断。"""
    metrics = _run(tmp_path, _BrokenDescribeAdapter())
    assert metrics["adapter_describe"] is None
    assert metrics["overall"]["n_questions"] > 0
    out = capsys.readouterr().out
    assert "describe() 抛错" in out
    assert "本次不记录自描述" in out
    assert "### 适配器自描述" not in _report_text(tmp_path)


def test_adapter_describe_non_dict_recorded_none(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """describe() 返回非 dict：记 None + 警告（契约要求 dict），不中断。"""
    metrics = _run(tmp_path, _NonDictDescribeAdapter())
    assert metrics["adapter_describe"] is None
    assert metrics["overall"]["n_questions"] > 0
    assert "契约要求 dict" in capsys.readouterr().out


def test_adapter_describe_unserializable_recorded_none(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """describe() 值不可 JSON 序列化：记 None + 警告，不让收尾落盘炸掉评测。"""
    metrics = _run(tmp_path, _UnserializableDescribeAdapter())
    assert metrics["adapter_describe"] is None
    assert metrics["overall"]["n_questions"] > 0
    assert "无法 JSON 序列化" in capsys.readouterr().out


def test_adapter_default_describe_empty_not_rendered(tmp_path: Path) -> None:
    """未覆写 describe() 的适配器：metrics 记 {}，报告不出现该小节。"""
    metrics = _run(tmp_path, MockStrongAdapter())
    assert metrics["adapter_describe"] == {}
    assert "### 适配器自描述" not in _report_text(tmp_path)


def test_adapter_without_describe_attribute(tmp_path: Path) -> None:
    """duck-typed 适配器没有 describe()：记 None，不中断。"""
    metrics = _run(tmp_path, _PlainAdapter())
    assert metrics["adapter_describe"] is None


def test_adapter_describe_rows_safe_rendering() -> None:
    """自描述值：嵌套结构转 JSON、换行折叠、超长截断，空/非 dict 不出行。"""
    assert adapter_describe_rows(None) == []
    assert adapter_describe_rows({}) == []
    assert adapter_describe_rows(["x"]) == []
    rows = adapter_describe_rows(
        {"nested": {"a": 1}, "multiline": "x\ny", "long": "z" * 500}
    )
    assert rows[0] == ["nested", '{"a": 1}']
    assert rows[1] == ["multiline", "x y"]
    assert rows[2][1].endswith("…")
    assert len(rows[2][1]) == ADAPTER_DESCRIBE_VALUE_MAX
