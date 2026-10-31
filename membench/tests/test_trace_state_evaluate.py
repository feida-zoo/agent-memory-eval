"""``--trace-state`` 在 evaluate / evaluate-compare / CLI 的透传。

覆盖：
- ``run_datasets(trace_state=True)`` 让每个数据集都产出逐时点状态目录；
- ``evaluate`` / ``evaluate_compare`` 把 ``trace_state`` 透传到 ``run_datasets``；
- CLI 的 ``evaluate`` / ``evaluate-compare`` 解析 ``--trace-state`` 并传进函数。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import membench.evaluate as evaluate_mod
import membench.evaluate_compare as evaluate_compare_mod
from membench import cli, layout
from membench.adapters.base import BaseMemoryAdapter, IngestStats, QueryResult
from membench.evaluate import DatasetSpec, evaluate, run_datasets


class _StateFileAdapter(BaseMemoryAdapter):
    """把已 ingest 的消息写进 ``state.txt`` 的假适配器（声明快照能力）。"""

    def __init__(self, name: str = "state-file") -> None:
        self.name = name
        self._messages: list[dict] = []

    def reset(self) -> None:
        self._messages = []

    def ingest(self, messages: list[dict]) -> IngestStats:
        self._messages.extend(messages)
        return IngestStats()

    def query(self, question: str) -> QueryResult:
        return QueryResult(answer="标准答案", retrieved=["m1 alpha fact"])

    def snapshot_capabilities(self) -> dict:
        return {"supported": True, "restore": False, "rebuild": False, "format_version": 1}

    def snapshot(self, target_dir: str, tag: str) -> dict:
        with open(Path(target_dir) / "state.txt", "w", encoding="utf-8") as handle:
            for message in self._messages:
                handle.write(str(message.get("content", "")) + "\n")
        return {"n_messages": len(self._messages)}


def _interleaved_dataset(path: Path) -> Path:
    """2 条消息、1 道 as_of=1 的 interleaved 题（含 evidence）。"""
    dialogue = {
        "dialogue_id": "d1",
        "ingest_strategy": "interleaved",
        "sessions": [
            [
                {"role": "user", "content": "m1 alpha fact", "timestamp": None},
                {"role": "user", "content": "m2 beta fact", "timestamp": None},
            ]
        ],
        "questions": [
            {
                "qid": "q1",
                "question": "q1: 问题？",
                "gold_answer": "标准答案",
                "category": "single_hop",
                "evidence": ["alpha fact"],
                "as_of": 1,
            }
        ],
    }
    path.write_text(json.dumps(dialogue, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _spec(path: Path) -> DatasetSpec:
    return DatasetSpec(key="trace_demo", name="trace_demo", path=str(path), source="user")


def _dataset_dirs(experiment_dir: Path) -> list[Path]:
    return sorted((experiment_dir / layout.DIR_RUNS).iterdir())


def test_run_datasets_writes_trace_state(tmp_path: Path) -> None:
    """run_datasets(trace_state=True)：interleaved 的 t<as_of> 目录出现。"""
    dataset = _interleaved_dataset(tmp_path / "trace.jsonl")
    experiment_dir = tmp_path / "exp"
    results = run_datasets(
        _StateFileAdapter(),
        "heuristic",
        [_spec(dataset)],
        str(experiment_dir),
        trace_state=True,
    )
    dataset_dir = Path(results[0].run_dir)
    trace_dir = dataset_dir / "d1" / layout.DIR_TRACE_STATE / "t1"
    assert (trace_dir / "state.txt").is_file()
    assert "m1 alpha fact" in (trace_dir / "state.txt").read_text(encoding="utf-8")
    assert results[0].metrics["trace_state"]["role"] == "exported"


def test_evaluate_forwards_trace_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """evaluate(trace_state=True) 透传到 run_datasets，产物里能看到逐时点状态。"""
    dataset = _interleaved_dataset(tmp_path / "trace.jsonl")
    calls: list[bool] = []
    real = evaluate_mod.run_datasets

    def spy(*args, **kwargs):
        calls.append(bool(kwargs.get("trace_state")))
        return real(*args, **kwargs)

    monkeypatch.setattr(evaluate_mod, "run_datasets", spy)
    evaluate(
        _StateFileAdapter(),
        "heuristic",
        specs=[_spec(dataset)],
        report_out=str(tmp_path / "eval"),
        driver_llm="none",
        adapter_version="v1",
        trace_state=True,
    )
    assert calls == [True]
    experiment_dir = next((tmp_path / "eval").iterdir())
    trace_dir = experiment_dir / "runs" / "trace" / "d1" / layout.DIR_TRACE_STATE / "t1"
    assert (trace_dir / "state.txt").is_file()


def test_evaluate_compare_forwards_trace_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """evaluate_compare(trace_state=True) 两侧都透传，两组各自产出逐时点状态。"""
    dataset = _interleaved_dataset(tmp_path / "trace.jsonl")
    calls: list[bool] = []
    real = evaluate_compare_mod.run_datasets

    def spy(*args, **kwargs):
        calls.append(bool(kwargs.get("trace_state")))
        return real(*args, **kwargs)

    monkeypatch.setattr(evaluate_compare_mod, "run_datasets", spy)
    result = evaluate_compare_mod.evaluate_compare(
        _StateFileAdapter("state-file-a"),
        _StateFileAdapter("state-file-b"),
        "heuristic",
        specs=[_spec(dataset)],
        report_out=str(tmp_path / "cmp"),
        driver_llm_baseline="none",
        driver_llm_candidate="none",
        adapter_version_baseline="v1",
        adapter_version_candidate="v2",
        trace_state=True,
    )
    assert calls == [True, True]
    compare_dir = Path(result["compare_dir"])
    for group in (layout.DIR_GROUP_BASELINE, layout.DIR_GROUP_CANDIDATE):
        group_dataset = next((compare_dir / group / layout.DIR_RUNS).iterdir())
        assert (group_dataset / "d1" / layout.DIR_TRACE_STATE / "t1").is_dir()


def test_cli_evaluate_parses_trace_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI evaluate 把 --trace-state 解析成 evaluate(trace_state=True)。"""
    captured: dict = {}

    def fake_evaluate(*args, **kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(evaluate_mod, "evaluate", fake_evaluate)
    assert (
        cli.main(
            [
                "evaluate",
                "--adapter",
                "mock-strong",
                "--judge",
                "heuristic",
                "--dataset",
                str(_interleaved_dataset(tmp_path / "trace.jsonl")),
                "--report-out",
                str(tmp_path / "out"),
                "--trace-state",
            ]
        )
        == 0
    )
    assert captured["trace_state"] is True


def test_cli_evaluate_compare_parses_trace_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI evaluate-compare 把 --trace-state 解析成 evaluate_compare(trace_state=True)。"""
    captured: dict = {}

    def fake_evaluate_compare(*args, **kwargs):
        captured.update(kwargs)
        return {"verdict": "PASS"}

    monkeypatch.setattr(
        evaluate_compare_mod, "evaluate_compare", fake_evaluate_compare
    )
    assert (
        cli.main(
            [
                "evaluate-compare",
                "--baseline-adapter",
                "mock-weak",
                "--candidate-adapter",
                "mock-strong",
                "--judge",
                "heuristic",
                "--dataset",
                str(_interleaved_dataset(tmp_path / "trace.jsonl")),
                "--report-out",
                str(tmp_path / "out"),
                "--trace-state",
            ]
        )
        == 0
    )
    assert captured["trace_state"] is True
