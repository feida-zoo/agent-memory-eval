"""fuse 子命令测试：分片融合校验、指标重算、stale 门禁。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from membench.adapters import MockStrongAdapter, MockWeakAdapter
from membench.adapters.base import BaseMemoryAdapter, IngestStats, QueryResult
from membench.data import SAMPLE_DATASET_PATH
from membench.fuse import fuse_runs
from membench.runner import Runner


class _StaleAdapter(BaseMemoryAdapter):
    """永远答旧值的适配器：knowledge_update 题必 stale（新值未命中）。"""

    name = "mock-stale"

    def reset(self) -> None:
        pass

    def ingest(self, messages: list[dict]) -> IngestStats:
        return IngestStats()

    def query(self, question: str) -> QueryResult:
        return QueryResult(answer="截止日是周三", retrieved=[])


def _stale_dataset(path: Path) -> Path:
    """单 dialogue 两道 knowledge_update 题（gold=周五, gold_stale=周三）。"""
    dialogue = {
        "dialogue_id": "stale_d1",
        "sessions": [[{"role": "user", "content": "截止日改到周五了。", "timestamp": None}]],
        "questions": [
            {
                "qid": f"stale_q{i}",
                "question": "现在项目截止日是哪天？",
                "gold_answer": "周五",
                "category": "knowledge_update",
                "evidence": [],
                "gold_stale": "周三",
            }
            for i in (1, 2)
        ],
    }
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(dialogue, ensure_ascii=False) + "\n")
    return path


def _shard_run(tmp_path: Path, name: str, adapter, shards: int, shard_index: int) -> str:
    runner = Runner(
        adapter,
        str(SAMPLE_DATASET_PATH),
        judge="heuristic",
        out_dir=str(tmp_path / name),
        shards=shards,
        shard_index=shard_index,
    )
    return runner.run()


def _full_run(tmp_path: Path, name: str, adapter) -> tuple[str, dict, list[dict]]:
    runner = Runner(
        adapter,
        str(SAMPLE_DATASET_PATH),
        judge="heuristic",
        out_dir=str(tmp_path / name),
    )
    run_dir = runner.run()
    metrics = json.loads((Path(run_dir) / "metrics.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (Path(run_dir) / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return run_dir, metrics, records


def test_fuse_two_shards_matches_full_run(tmp_path: Path) -> None:
    """两个分片融合后的题集与指标等同于全量 run。"""
    s0 = _shard_run(tmp_path, "s0", MockStrongAdapter(), 2, 0)
    s1 = _shard_run(tmp_path, "s1", MockStrongAdapter(), 2, 1)
    _, full_metrics, full_records = _full_run(tmp_path, "full", MockStrongAdapter())

    out_dir = str(tmp_path / "fused")
    result = fuse_runs([s0, s1], out_dir)

    out = Path(out_dir)
    assert (out / "results.jsonl").is_file()
    assert (out / "metrics.json").is_file()
    assert (out / "report.md").is_file()

    fused_records = [
        json.loads(line)
        for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {r["qid"] for r in fused_records} == {r["qid"] for r in full_records}
    assert result["n_questions"] == len(full_records)

    metrics = result["metrics"]
    assert metrics["overall"]["accuracy"] == pytest.approx(full_metrics["overall"]["accuracy"])
    assert metrics["overall"]["stale_rate"] == pytest.approx(full_metrics["overall"]["stale_rate"])
    assert metrics["overall"]["n_questions"] == full_metrics["overall"]["n_questions"]
    # metadata 标记 fused 及来源分片列表
    assert metrics["fused"] is True
    assert len(metrics["fused_from"]) == 2
    assert metrics["shards"] == 2
    assert result["gate"]["status"] == "PASS"
    assert metrics["stale_gate"]["status"] == "PASS"


def test_fuse_requires_full_shard_coverage(tmp_path: Path) -> None:
    """shard_index 未覆盖 0..N-1 时拒绝融合。"""
    s0 = _shard_run(tmp_path, "s0", MockStrongAdapter(), 3, 0)
    s1 = _shard_run(tmp_path, "s1", MockStrongAdapter(), 3, 1)
    with pytest.raises(ValueError, match="shard_index"):
        fuse_runs([s0, s1], str(tmp_path / "fused"))


def test_fuse_rejects_duplicate_shards(tmp_path: Path) -> None:
    """重复的分片（同 shard_index）会被 qid 去重校验或覆盖校验拦下。"""
    s0 = _shard_run(tmp_path, "s0", MockStrongAdapter(), 2, 0)
    s1 = _shard_run(tmp_path, "s1", MockStrongAdapter(), 2, 1)
    with pytest.raises(ValueError):
        fuse_runs([s0, s0, s1], str(tmp_path / "fused"))


def test_fuse_rejects_unsharded_runs(tmp_path: Path) -> None:
    run_dir, _, _ = _full_run(tmp_path, "full", MockStrongAdapter())
    with pytest.raises(ValueError, match="未分片"):
        fuse_runs([run_dir], str(tmp_path / "fused"))


def test_fuse_rejects_adapter_mismatch(tmp_path: Path) -> None:
    s0 = _shard_run(tmp_path, "s0", MockStrongAdapter(), 2, 0)
    s1 = _shard_run(tmp_path, "s1", MockWeakAdapter(), 2, 1)
    with pytest.raises(ValueError, match="adapter 不一致"):
        fuse_runs([s0, s1], str(tmp_path / "fused"))


def test_fuse_stale_gate_fail(tmp_path: Path) -> None:
    """融合结果 stale_rate 超阈值 → 门禁 FAIL（一票否决，复用 evaluate 阈值）。"""
    ds = _stale_dataset(tmp_path / "stale.jsonl")
    s0 = Runner(_StaleAdapter(), str(ds), judge="heuristic",
                out_dir=str(tmp_path / "s0"), shards=2, shard_index=0).run()
    s1 = Runner(_StaleAdapter(), str(ds), judge="heuristic",
                out_dir=str(tmp_path / "s1"), shards=2, shard_index=1).run()
    result = fuse_runs([s0, s1], str(tmp_path / "fused"))
    assert result["metrics"]["overall"]["stale_rate"] == pytest.approx(1.0)
    assert result["gate"]["status"] == "FAIL"
    assert result["metrics"]["stale_gate"]["status"] == "FAIL"


def test_fuse_missing_metrics_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        fuse_runs([str(tmp_path / "nonexistent")], str(tmp_path / "fused"))
