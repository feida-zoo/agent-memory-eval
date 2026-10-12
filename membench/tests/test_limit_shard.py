"""--limit 冒烟模式与 --shards/--shard-index 分片的 Runner 测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from membench.adapters import MockStrongAdapter
from membench.data import SAMPLE_DATASET_PATH
from membench.runner import Runner


def _run(tmp_path: Path, name: str, **kwargs) -> tuple[Path, dict, list[dict]]:
    runner = Runner(
        MockStrongAdapter(),
        str(SAMPLE_DATASET_PATH),
        judge="heuristic",
        out_dir=str(tmp_path / name),
        **kwargs,
    )
    run_dir = Path(runner.run())
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return run_dir, metrics, records


def test_limit_truncates_questions(tmp_path: Path) -> None:
    """limit=3 时只评测前 3 道题，metrics 与 run_id 均体现 limit。"""
    _, metrics, records = _run(tmp_path, "limited", limit=3)
    assert len(records) == 3
    assert metrics["overall"]["n_questions"] == 3
    assert metrics["limit"] == 3
    assert "_limit3_" in metrics["run_id"]


def test_limit_none_is_full_run(tmp_path: Path) -> None:
    """不传 limit 时全量执行，metadata 记录 limit=None、run_id 无 limit 标记。"""
    _, metrics, records = _run(tmp_path, "full")
    assert metrics["limit"] is None
    assert "_limit" not in metrics["run_id"]
    assert metrics["overall"]["n_questions"] == len(records) >= 24


def test_limit_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="limit"):
        Runner(
            MockStrongAdapter(), str(SAMPLE_DATASET_PATH), out_dir=str(tmp_path), limit=0
        )


def test_shards_partition_dataset(tmp_path: Path) -> None:
    """2 分片的 qid 集合不相交且并集等于全量 qid 集合。"""
    _, m0, r0 = _run(tmp_path, "s0", shards=2, shard_index=0)
    _, m1, r1 = _run(tmp_path, "s1", shards=2, shard_index=1)
    _, _, full = _run(tmp_path, "full")

    q0 = [r["qid"] for r in r0]
    q1 = [r["qid"] for r in r1]
    assert set(q0).isdisjoint(q1)
    assert set(q0) | set(q1) == {r["qid"] for r in full}
    assert m0["shards"] == 2 and m0["shard_index"] == 0
    assert m1["shards"] == 2 and m1["shard_index"] == 1
    assert "_shard0of2_" in m0["run_id"]
    assert "_shard1of2_" in m1["run_id"]


def test_shards_deterministic_order(tmp_path: Path) -> None:
    """分片按全局题号 i % N == I 选取，保持原始题目相对顺序。"""
    _, _, full = _run(tmp_path, "full")
    _, _, r0 = _run(tmp_path, "s0", shards=3, shard_index=0)
    full_qids = [r["qid"] for r in full]
    expected = [q for i, q in enumerate(full_qids) if i % 3 == 0]
    assert [r["qid"] for r in r0] == expected


def test_shard_then_limit(tmp_path: Path) -> None:
    """与 limit 组合时先分片再截断。"""
    _, _, full = _run(tmp_path, "full")
    _, metrics, records = _run(tmp_path, "s1l2", shards=2, shard_index=1, limit=2)
    full_qids = [r["qid"] for r in full]
    expected = [q for i, q in enumerate(full_qids) if i % 2 == 1][:2]
    assert [r["qid"] for r in records] == expected
    assert metrics["limit"] == 2 and metrics["shards"] == 2 and metrics["shard_index"] == 1
    assert "_shard1of2_" in metrics["run_id"] and "_limit2_" in metrics["run_id"]


def test_shard_index_out_of_range(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shard_index"):
        Runner(
            MockStrongAdapter(),
            str(SAMPLE_DATASET_PATH),
            out_dir=str(tmp_path),
            shards=2,
            shard_index=2,
        )
