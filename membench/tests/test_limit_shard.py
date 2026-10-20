"""--limit 冒烟模式与 --shards/--shard-index 分片的 Runner 测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from membench.adapters import MockStrongAdapter
from membench.adapters.base import BaseMemoryAdapter, IngestStats, QueryResult
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


# ---------------------------------------------------------------------------
# --shard-by dialogue：整段对话分片，分片间不重复 ingest
# ---------------------------------------------------------------------------


class _CountingAdapter(BaseMemoryAdapter):
    """记录每个对话的首个 session 内容，用于核算各分片实际 ingest 的对话数。"""

    name = "counting"

    def __init__(self) -> None:
        self.segments: list[str] = []
        self._reset = False

    def reset(self) -> None:
        self._reset = True

    def ingest(self, messages: list[dict]) -> IngestStats:
        if self._reset:
            self.segments.append(str(messages[0]["content"]) if messages else "")
            self._reset = False
        return IngestStats()

    def query(self, question: str) -> QueryResult:
        return QueryResult(answer="x", retrieved=[])


def _dense_dataset(tmp_path: Path, n_dialogues: int = 4, per: int = 3) -> Path:
    """构造"每个对话都有题"的 ingest 密集型数据集（n 对话 × 每对话 per 题）。"""
    lines = []
    for d in range(n_dialogues):
        lines.append(
            json.dumps(
                {
                    "dialogue_id": f"dd{d}",
                    "sessions": [
                        [{"role": "user", "content": f"dd{d}-content", "timestamp": None}]
                    ],
                    "questions": [
                        {
                            "qid": f"dd{d}_q{q}",
                            "question": f"dd{d} 第 {q} 题？",
                            "gold_answer": "答案",
                            "category": "single_hop",
                            "evidence": [],
                        }
                        for q in range(per)
                    ],
                },
                ensure_ascii=False,
            )
        )
    path = tmp_path / "dense.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_shards_by_dialogue_partition_dataset(tmp_path: Path) -> None:
    """dialogue 分片整段选取对话：分片间对话不相交，对话与题目并集均覆盖全量。"""
    _, m0, r0 = _run(tmp_path, "d0", shards=2, shard_index=0, shard_by="dialogue")
    _, m1, r1 = _run(tmp_path, "d1", shards=2, shard_index=1, shard_by="dialogue")
    _, _, full = _run(tmp_path, "full")

    assert m0["shard_by"] == "dialogue" and m0["shard_index"] == 0
    assert m1["shard_by"] == "dialogue" and m1["shard_index"] == 1
    assert "_dshard0of2_" in m0["run_id"] and "_dshard1of2_" in m1["run_id"]

    d0 = {r["dialogue_id"] for r in r0}
    d1 = {r["dialogue_id"] for r in r1}
    d_full = {r["dialogue_id"] for r in full}
    assert d0.isdisjoint(d1)
    assert d0 | d1 == d_full
    assert {r["qid"] for r in r0} | {r["qid"] for r in r1} == {
        r["qid"] for r in full
    }


def test_shards_by_dialogue_avoid_repeated_ingest(tmp_path: Path) -> None:
    """ingest 密集型数据集：dialogue 分片两段合计只 ingest 全部对话一次；
    题目级分片两段合计会重复 ingest（每段仍覆盖几乎全部对话）。"""
    dataset = _dense_dataset(tmp_path)
    n_dialogues = 4

    def _ingested(shard_index: int, shard_by: str) -> int:
        adapter = _CountingAdapter()
        Runner(
            adapter,
            str(dataset),
            judge="heuristic",
            out_dir=str(tmp_path / f"run-{shard_by}-{shard_index}"),
            shards=2,
            shard_index=shard_index,
            shard_by=shard_by,
        ).run()
        return len(adapter.segments)

    assert _ingested(0, "dialogue") + _ingested(1, "dialogue") == n_dialogues
    assert _ingested(0, "question") + _ingested(1, "question") > n_dialogues


def test_shard_by_dialogue_then_limit(tmp_path: Path) -> None:
    """dialogue 分片与 limit 组合：先整段选对话，再截断前 limit 道题。"""
    _, metrics, records = _run(
        tmp_path, "dl", shards=2, shard_index=0, shard_by="dialogue", limit=2
    )
    assert len(records) == 2
    assert metrics["limit"] == 2 and metrics["shard_by"] == "dialogue"
    assert metrics["shards"] == 2 and metrics["shard_index"] == 0
    assert "_dshard0of2_" in metrics["run_id"] and "_limit2_" in metrics["run_id"]


def test_shard_by_invalid(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shard_by"):
        Runner(
            MockStrongAdapter(),
            str(SAMPLE_DATASET_PATH),
            out_dir=str(tmp_path),
            shard_by="turn",
        )


def test_cli_shard_by_dialogue(tmp_path: Path) -> None:
    """CLI 透传 --shard-by dialogue，metrics.json 记录 shard_by。"""
    from membench import cli

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
            "--shards",
            "2",
            "--shard-index",
            "0",
            "--shard-by",
            "dialogue",
        ]
    )
    assert rc == 0
    run_dir = next((tmp_path / "runs").iterdir())
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["shard_by"] == "dialogue"
    assert metrics["overall"]["n_questions"] > 0


def test_shards_over_dialogue_count_rejected(tmp_path: Path) -> None:
    """--shard-by dialogue 分片数超过对话数 -> 直接报错（空分片纯浪费）。"""
    ds = _dense_dataset(tmp_path, n_dialogues=2, per=2)
    with pytest.raises(ValueError, match="对话数 2"):
        Runner(
            MockStrongAdapter(),
            str(ds),
            judge="heuristic",
            out_dir=str(tmp_path / "x"),
            shards=3,
            shard_index=0,
            shard_by="dialogue",
        ).run()


def test_shards_over_question_count_rejected(tmp_path: Path) -> None:
    """question 维度分片数超过题目数 -> 直接报错。"""
    ds = _dense_dataset(tmp_path, n_dialogues=1, per=2)
    with pytest.raises(ValueError, match="题目数 2"):
        Runner(
            MockStrongAdapter(),
            str(ds),
            judge="heuristic",
            out_dir=str(tmp_path / "x"),
            shards=3,
            shard_index=0,
        ).run()


def test_shards_at_limit_ok(tmp_path: Path) -> None:
    """分片数等于上限可用（每个分片恰有归属）。"""
    ds = _dense_dataset(tmp_path, n_dialogues=2, per=1)
    run_dir = Runner(
        MockStrongAdapter(),
        str(ds),
        judge="heuristic",
        out_dir=str(tmp_path / "ok"),
        shards=2,
        shard_index=1,
        shard_by="dialogue",
    ).run()
    assert (Path(run_dir) / "metrics.json").is_file()


def test_cli_shards_over_limit_systemexit(tmp_path: Path) -> None:
    """CLI 上分片数超限 -> SystemExit 友好报错。"""
    from membench import cli

    ds = _dense_dataset(tmp_path, n_dialogues=1, per=2)
    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "run",
                "--adapter", "mock-weak",
                "--dataset", str(ds),
                "--shards", "5",
                "--shard-by", "dialogue",
            ]
        )
    assert "超限" in str(exc_info.value)
