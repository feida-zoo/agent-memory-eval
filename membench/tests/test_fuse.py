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


def _shard_run(
    tmp_path: Path, name: str, adapter, shards: int, shard_index: int, shard_by: str = "question"
) -> str:
    runner = Runner(
        adapter,
        str(SAMPLE_DATASET_PATH),
        judge="heuristic",
        out_dir=str(tmp_path / name),
        shards=shards,
        shard_index=shard_index,
        shard_by=shard_by,
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


def test_fuse_dialogue_shards_matches_full_run(tmp_path: Path) -> None:
    """dialogue 分片的两个分片可融合为与全量一致的完整 run。"""
    d0 = _shard_run(tmp_path, "d0", MockStrongAdapter(), 2, 0, shard_by="dialogue")
    d1 = _shard_run(tmp_path, "d1", MockStrongAdapter(), 2, 1, shard_by="dialogue")
    _, full_metrics, full_records = _full_run(tmp_path, "full", MockStrongAdapter())

    result = fuse_runs([d0, d1], str(tmp_path / "fused"))
    assert result["n_questions"] == len(full_records)
    assert result["metrics"]["shard_by"] == "dialogue"
    assert result["metrics"]["shards"] == 2
    assert result["metrics"]["overall"]["accuracy"] == pytest.approx(
        full_metrics["overall"]["accuracy"]
    )


def test_fuse_rejects_mixed_shard_by(tmp_path: Path) -> None:
    """题目级分片与 dialogue 分片混合融合 → 明确报错（不允许混维）。"""
    q0 = _shard_run(tmp_path, "q0", MockStrongAdapter(), 2, 0, shard_by="question")
    d1 = _shard_run(tmp_path, "d1", MockStrongAdapter(), 2, 1, shard_by="dialogue")
    with pytest.raises(ValueError, match="shard_by 不一致"):
        fuse_runs([q0, d1], str(tmp_path / "fused"))


def test_fuse_passes_through_comparability_metadata(tmp_path: Path) -> None:
    """融合产物透传可比性元数据：fused run 之间必须能过 strict 可比性（回归铁证）。"""
    from membench.compare import compare_runs

    def _shard(name: str, index: int) -> str:
        runner = Runner(
            MockStrongAdapter(),
            str(SAMPLE_DATASET_PATH),
            judge="heuristic",
            out_dir=str(tmp_path / name),
            shards=2,
            shard_index=index,
            driver_llm="gpt-4o",
            adapter_version="v1",
            adapter_meta="k=v",
        )
        return runner.run()

    s0 = _shard("m0", 0)
    s1 = _shard("m1", 1)
    fused = fuse_runs([s0, s1], str(tmp_path / "fused_meta"))["out_dir"]

    src = json.loads((Path(s0) / "metrics.json").read_text(encoding="utf-8"))
    dst = json.loads((Path(fused) / "metrics.json").read_text(encoding="utf-8"))
    for key in (
        "adapter_version",
        "judge_model",
        "driver_llm",
        "adapter_meta",
        "dataset_fingerprint",
        "retrieval_match",
        "schema_version",
    ):
        assert dst.get(key) == src.get(key), key
    assert dst["fused"] is True and dst["fused_from"]

    # 回归：此前关键维度全缺失，require_comparable 必然 FAIL
    result = compare_runs(
        fused, fused, require_comparable=True, out_dir=str(tmp_path / "cmp")
    )
    assert result["comparability"]["unverifiable"] == []
    assert result["comparability"]["comparable"] is True
    assert result["verdict"] != "FAIL"


def test_fuse_rejects_mixed_metadata(tmp_path: Path) -> None:
    """判分/驱动等元数据不一致的分片拒绝融合（防止混片）。"""
    s0 = _shard_run(tmp_path, "x0", MockStrongAdapter(), 2, 0)
    s1 = _shard_run(tmp_path, "x1", MockStrongAdapter(), 2, 1)
    metrics_path = Path(s1) / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["driver_llm"] = "different-model"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="driver_llm 不一致"):
        fuse_runs([s0, s1], str(tmp_path / "bad"))
