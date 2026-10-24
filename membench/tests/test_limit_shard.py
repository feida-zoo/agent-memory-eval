"""--limit 冒烟模式与 --shards/--shard-index 分片的 Runner 测试。

产物结构见 :mod:`membench.layout`：``out_dir`` 是实验目录，``run()`` 返回数据集目录
``<实验目录>/runs/<dataset>``；分片只影响本分片的对话/题目，树形与元数据一致。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from membench import layout
from membench.adapters import MockStrongAdapter
from membench.adapters.base import BaseMemoryAdapter, IngestStats, QueryResult
from membench.data import SAMPLE_DATASET_PATH
from membench.runner import Runner

#: 测试数据集目录名（= 数据集文件名 stem）。
DATASET = "sample_dataset"


@pytest.fixture(autouse=True)
def _isolated_output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """默认产物根指向 tmp_path：任何走 output_root() 的路径都不落进仓库。"""
    root = tmp_path / "output-root"
    monkeypatch.setenv("MEMBENCH_OUTPUT_DIR", str(root))
    return root


def _dataset_artifacts(dataset_dir: Path) -> tuple[dict, list[dict]]:
    """读数据集级 metrics 与逐题明细（都在 ``<数据集目录>/report/`` 下）。"""
    report_dir = Path(dataset_dir) / layout.DIR_REPORT
    metrics = json.loads((report_dir / layout.FILE_METRICS).read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (report_dir / layout.FILE_RESULTS).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return metrics, records


def _run(tmp_path: Path, name: str, **kwargs) -> tuple[Path, dict, list[dict]]:
    """跑一次评测，返回（数据集目录, 数据集级 metrics, 数据集级逐题明细）。"""
    runner = Runner(
        MockStrongAdapter(),
        str(SAMPLE_DATASET_PATH),
        judge="heuristic",
        out_dir=str(tmp_path / name),
        **kwargs,
    )
    dataset_dir = Path(runner.run())
    metrics, records = _dataset_artifacts(dataset_dir)
    return dataset_dir, metrics, records


def test_limit_truncates_questions(tmp_path: Path) -> None:
    """limit=3 时只评测前 3 道题，metrics 与 run_id 均体现 limit。"""
    dataset_dir, metrics, records = _run(tmp_path, "limited", limit=3)
    assert len(records) == 3
    assert metrics["overall"]["n_questions"] == 3
    assert metrics["limit"] == 3
    assert "_limit3_" in metrics["run_id"]
    # 产物归属写进 metrics：数据集目录与实验目录不必靠猜层级
    assert metrics["dataset_dir"] == str(dataset_dir)
    assert Path(metrics["experiment_dir"]) == tmp_path / "limited"


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
    """2 分片的 qid 集合不相交且并集等于全量 qid 集合（分片 run 照常产全树）。"""
    s0_dir, m0, r0 = _run(tmp_path, "s0", shards=2, shard_index=0)
    _, m1, r1 = _run(tmp_path, "s1", shards=2, shard_index=1)
    _, full_metrics, full = _run(tmp_path, "full")

    q0 = [r["qid"] for r in r0]
    q1 = [r["qid"] for r in r1]
    assert set(q0).isdisjoint(q1)
    assert set(q0) | set(q1) == {r["qid"] for r in full}
    assert len(q0) + len(q1) == full_metrics["overall"]["n_questions"]
    assert m0["shards"] == 2 and m0["shard_index"] == 0
    assert m1["shards"] == 2 and m1["shard_index"] == 1
    assert "_shard0of2_" in m0["run_id"]
    assert "_shard1of2_" in m1["run_id"]
    # 分片写的仍是同一套树：数据集报告 + 任务级 reports/ 都在位
    assert (s0_dir / layout.DIR_REPORT / layout.FILE_METRICS).is_file()
    assert (
        s0_dir.parent.parent / layout.DIR_REPORTS / layout.FILE_REPORT_MD
    ).is_file()


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
    d0_dir, m0, r0 = _run(tmp_path, "d0", shards=2, shard_index=0, shard_by="dialogue")
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
    # 本分片只落自己的对话目录（不为其它的对话留空目录）
    laid_out = {
        p.name
        for p in d0_dir.iterdir()
        if p.is_dir() and p.name != layout.DIR_REPORT
    }
    assert laid_out == {layout.tag_dir_name(d) for d in d0}


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


def _cli_experiment_dirs(root: Path) -> list[Path]:
    """CLI 产物：``--report-out`` 下的实验目录（``eval_<adapter>_<platform>_<ts>``）。"""
    return sorted(p for p in root.glob("eval_*") if p.is_dir())


def test_cli_shard_by_dialogue(tmp_path: Path) -> None:
    """CLI 透传 --shard-by dialogue；实验目录与数据集目录按新结构落位。"""
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
            "--report-out",
            str(tmp_path),
            "--shards",
            "2",
            "--shard-index",
            "0",
            "--shard-by",
            "dialogue",
        ]
    )
    assert rc == 0
    experiments = _cli_experiment_dirs(tmp_path)
    assert len(experiments) == 1
    assert experiments[0].name.startswith("eval_mock-strong_default_")
    # 目录名末段是 TS_FORMAT 秒级时间戳（格式不符时 strptime 抛错即失败）
    datetime.strptime(experiments[0].name.rsplit("_", 1)[-1], layout.TS_FORMAT)

    dataset_dir = experiments[0] / layout.DIR_RUNS / DATASET
    metrics, records = _dataset_artifacts(dataset_dir)
    assert metrics["shard_by"] == "dialogue"
    assert metrics["overall"]["n_questions"] == len(records) > 0
    assert (experiments[0] / layout.DIR_REPORTS / layout.FILE_REPORT_MD).is_file()


def test_shards_over_dialogue_count_clamped(tmp_path: Path) -> None:
    """--shards 超过对话数 -> 按上限执行（不报错），标签与元数据用有效分片数。"""
    ds = _dense_dataset(tmp_path, n_dialogues=2, per=2)
    dataset_dir = Runner(
        MockStrongAdapter(),
        str(ds),
        judge="heuristic",
        out_dir=str(tmp_path / "ok"),
        shards=5,
        shard_index=1,
        shard_by="dialogue",
    ).run()
    assert dataset_dir is not None
    metrics, records = _dataset_artifacts(Path(dataset_dir))
    assert "_dshard1of2_" in metrics["run_id"]
    assert metrics["shards"] == 2
    assert metrics["overall"]["n_questions"] == len(records) == 2


def test_shards_over_question_count_clamped(tmp_path: Path) -> None:
    """question 维度超过题目数 -> 同样按上限执行。"""
    ds = _dense_dataset(tmp_path, n_dialogues=1, per=3)
    dataset_dir = Runner(
        MockStrongAdapter(),
        str(ds),
        judge="heuristic",
        out_dir=str(tmp_path / "ok"),
        shards=9,
        shard_index=2,
    ).run()
    assert dataset_dir is not None
    metrics, records = _dataset_artifacts(Path(dataset_dir))
    assert metrics["shards"] == 3
    assert "_shard2of3_" in metrics["run_id"]
    assert len(records) == 1


def test_shards_at_limit_ok(tmp_path: Path) -> None:
    """分片数等于上限可用（每个分片恰有归属）。"""
    ds = _dense_dataset(tmp_path, n_dialogues=2, per=1)
    dataset_dir = Runner(
        MockStrongAdapter(),
        str(ds),
        judge="heuristic",
        out_dir=str(tmp_path / "ok"),
        shards=2,
        shard_index=1,
        shard_by="dialogue",
    ).run()
    assert (Path(dataset_dir) / layout.DIR_REPORT / layout.FILE_METRICS).is_file()


def test_shards_index_beyond_effective_skips(tmp_path: Path) -> None:
    """索引越过有效分片数 -> 安全跳过：返回 None 且不产生 run 产物。"""
    ds = _dense_dataset(tmp_path, n_dialogues=2, per=1)
    out_dir = tmp_path / "skip"
    result = Runner(
        MockStrongAdapter(),
        str(ds),
        judge="heuristic",
        out_dir=str(out_dir),
        shards=9,
        shard_index=5,
        shard_by="dialogue",
    ).run()
    assert result is None
    assert not out_dir.exists()


def test_shards_max_auto_and_skip(tmp_path: Path) -> None:
    """--shards max 按上限自动取最大；越界索引同样安全跳过。"""
    ds = _dense_dataset(tmp_path, n_dialogues=2, per=1)
    out_dir = tmp_path / "mx"
    dataset_dir = Runner(
        MockStrongAdapter(),
        str(ds),
        judge="heuristic",
        out_dir=str(out_dir),
        shards="max",
        shard_index=0,
        shard_by="dialogue",
    ).run()
    assert dataset_dir is not None
    metrics, records = _dataset_artifacts(Path(dataset_dir))
    assert metrics["shards"] == 2
    assert "_dshard0of2_" in metrics["run_id"]
    assert len(records) == 1

    skipped = Runner(
        MockStrongAdapter(),
        str(ds),
        judge="heuristic",
        out_dir=str(out_dir),
        shards="max",
        shard_index=2,
        shard_by="dialogue",
    ).run()
    assert skipped is None
    # 跳过的分片不动已有产物（同一实验目录里的数据集报告保持原样）
    assert _dataset_artifacts(Path(dataset_dir))[0]["run_id"] == metrics["run_id"]


def test_cli_shards_max_and_clamp(tmp_path: Path) -> None:
    """CLI：--shards max 可用；超过上限按上限执行（退出码 0）。"""
    from membench import cli

    ds = _dense_dataset(tmp_path, n_dialogues=2, per=1)
    rc = cli.main(
        [
            "run",
            "--adapter", "mock-weak",
            "--dataset", str(ds),
            "--shards", "max",
            "--shard-index", "0",
            "--shard-by", "dialogue",
            "--report-out", str(tmp_path),
        ]
    )
    assert rc == 0
    experiments = _cli_experiment_dirs(tmp_path)
    assert len(experiments) == 1
    metrics, records = _dataset_artifacts(experiments[0] / layout.DIR_RUNS / "dense")
    assert metrics["shards"] == 2
    assert "_dshard0of2_" in metrics["run_id"]
    assert len(records) == 1
