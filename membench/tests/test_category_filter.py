"""题型过滤（``--category``）：过滤口径、身份落账、与分片 / 融合 / 可比性的配合。

口径（唯一实现 ``membench.runner.filter_dialogues_by_category``）：
**先按题型过滤 → 再分片 → 再 --limit**；无匹配题的对话整段跳过（连 ingest 一起省）。
过滤后的题集是另一种"题目总体"，因此必须落进 ``metrics.categories`` 并作为 compare
的**关键维度**——否则「只跑 temporal」的分数会被当成全量分数。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from membench import layout
from membench.adapters import MockStrongAdapter
from membench.compare import compare_runs
from membench.data import SAMPLE_DATASET_PATH
from membench.data.schema import load_dataset
from membench.runner import Runner, available_categories, filter_dialogues_by_category

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
    """跑一次不分片评测，返回（数据集目录, 数据集级 metrics, 数据集级逐题明细）。"""
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


# ---------------------------------------------------------------------------
# 过滤口径本身
# ---------------------------------------------------------------------------


def test_filter_keeps_only_matching_categories() -> None:
    """只保留命中的题；无命中题的对话整段剔除；大小写与空白不敏感。"""
    dialogues = load_dataset(str(SAMPLE_DATASET_PATH))
    filtered = filter_dialogues_by_category(dialogues, ["TEMPORAL", " knowledge_update "])
    assert filtered, "sample_dataset 里有这两类题"
    kept = [q for d in filtered for q in d.questions]
    assert {q.category for q in kept} == {"temporal", "knowledge_update"}
    # 每道题的相对顺序不变（过滤不打乱题序，分片索引才稳定）
    order = {q.qid: i for i, q in enumerate(q for d in dialogues for q in d.questions)}
    assert [order[q.qid] for q in kept] == sorted(order[q.qid] for q in kept)


def test_filter_without_categories_is_noop() -> None:
    """不给题型时原样返回（不复制、不改题）。"""
    dialogues = load_dataset(str(SAMPLE_DATASET_PATH))
    assert filter_dialogues_by_category(dialogues, []) == dialogues


def test_available_categories_lists_question_categories() -> None:
    """题型清单来自题目本身（去重、保序）。"""
    categories = available_categories(load_dataset(str(SAMPLE_DATASET_PATH)))
    assert "temporal" in categories
    assert len(categories) == len(set(categories))


# ---------------------------------------------------------------------------
# Runner：落账与拒绝
# ---------------------------------------------------------------------------


def test_run_records_categories_and_reports_row(tmp_path: Path) -> None:
    """过滤后的 run：metrics 记 categories、题数只算命中题，报告有「题型过滤」行。"""
    dataset_dir, metrics, records = _run(
        tmp_path, "filtered", categories=["temporal"]
    )
    assert metrics["categories"] == ["temporal"]
    assert all(r["category"] == "temporal" for r in records)
    assert metrics["overall"]["n_questions"] == len(records) > 0
    report = (dataset_dir / layout.DIR_REPORT / layout.FILE_REPORT_MD).read_text(
        encoding="utf-8"
    )
    assert "| 题型过滤 | temporal |" in report


def test_run_without_categories_has_no_field(tmp_path: Path) -> None:
    """不过滤的产物不落 categories（不到处多一个空键）。"""
    _, metrics, _ = _run(tmp_path, "full")
    assert "categories" not in metrics


def test_unknown_category_fails_with_available_list(tmp_path: Path) -> None:
    """该数据集没有这个题型：直接拒绝并列出实际题型（不能静默产出空结果）。"""
    with pytest.raises(SystemExit, match="没有匹配题目"):
        Runner(
            MockStrongAdapter(),
            str(SAMPLE_DATASET_PATH),
            judge="heuristic",
            out_dir=str(tmp_path / "bad"),
            categories=["safety_no_memory"],
        ).run()


def test_category_then_shard_then_limit(tmp_path: Path) -> None:
    """顺序固定：先过滤 → 再分片 → 再 limit（分片索引基于过滤后的题序）。"""
    dataset_dir = Path(
        Runner(
            MockStrongAdapter(),
            str(SAMPLE_DATASET_PATH),
            judge="heuristic",
            out_dir=str(tmp_path / "order"),
            categories=["temporal"],
            shards=2,
            shard_index=0,
            limit=1,
        ).run()
    )
    records: list[dict] = []
    for dialogue_dir in sorted(dataset_dir.iterdir()):
        results = dialogue_dir / layout.DIR_REPORT / layout.FILE_RESULTS
        if results.is_file():
            records.extend(
                json.loads(line)
                for line in results.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    assert len(records) == 1
    assert records[0]["category"] == "temporal"


def test_category_run_id_and_dir_tags(tmp_path: Path) -> None:
    """run_id 带 ``cat-<题型>``、实验目录名带 ``_c<题型>``：过滤产物不伪装成全量。"""
    _, metrics, _ = _run(tmp_path, "tagged", categories=["Temporal"])
    assert "cat-temporal" in metrics["run_id"]
    # 目录名标签由 CLI 组装（Runner 只写给定的 out_dir），词表在 layout 唯一决定
    assert layout.experiment_tags(1, 0, None, ["Temporal"]) == ["ctemporal"]
    assert layout.experiment_tags(2, 1, 5, ["temporal", "safety"]) == [
        "s1",
        "l5",
        "ctemporal",
        "csafety",
    ]


# ---------------------------------------------------------------------------
# 分片账本 / 融合 / 可比性
# ---------------------------------------------------------------------------


def test_shard_ledger_records_categories(tmp_path: Path) -> None:
    """账本头部记题型口径（分片产物与不全量混在同一目录）。"""
    dataset_dir = Path(
        Runner(
            MockStrongAdapter(),
            str(SAMPLE_DATASET_PATH),
            judge="heuristic",
            out_dir=str(tmp_path / "ledger"),
            categories=["single_hop"],
            shards=2,
            shard_index=0,
            shard_by="dialogue",
        ).run()
    )
    ledger = json.loads(
        (dataset_dir / layout.FILE_SHARDS).read_text(encoding="utf-8")
    )
    assert ledger["categories"] == ["single_hop"]


def test_fuse_keeps_categories_and_covers_filtered_questions(tmp_path: Path) -> None:
    """分片 + fuse：覆盖校验按过滤后的题集复算，融合产物透传 categories。"""
    from membench.fuse import fuse_runs

    out_dir = tmp_path / "fused"
    for index in range(2):
        Runner(
            MockStrongAdapter(),
            str(SAMPLE_DATASET_PATH),
            judge="heuristic",
            out_dir=str(out_dir),
            categories=["single_hop"],
            shards=2,
            shard_index=index,
            shard_by="dialogue",
        ).run()
    result = fuse_runs([str(out_dir)], str(out_dir))
    metrics = result["metrics"]
    assert metrics["categories"] == ["single_hop"]
    assert "cat-single_hop" in metrics["run_id"]
    assert result["n_questions"] == metrics["overall"]["n_questions"] == 5


def test_compare_flags_category_mismatch_as_critical(tmp_path: Path) -> None:
    """题集口径不同：compare 判关键维度不一致（strict 模式 FAIL）。"""
    full_dir, _, _ = _run(tmp_path, "full")
    filtered_dir, _, _ = _run(tmp_path, "filtered", categories=["single_hop"])
    outcome = compare_runs(
        str(full_dir), str(filtered_dir), out_dir=str(tmp_path / "cmp1"), require_comparable=True
    )
    checks = {c["dimension"]: c for c in outcome["comparability"]["checks"]}
    assert checks["categories"]["same"] is False
    assert checks["categories"]["severity"] == "critical"
    assert outcome["verdict"] == "FAIL"


def test_compare_same_categories_passes_dimension(tmp_path: Path) -> None:
    """同样只跑 temporal 的两侧：categories 维度视为一致。"""
    left, _, _ = _run(tmp_path, "left", categories=["temporal"])
    right, _, _ = _run(tmp_path, "right", categories=["temporal"])
    outcome = compare_runs(
        str(left), str(right), out_dir=str(tmp_path / "cmp2"), require_comparable=True
    )
    checks = {c["dimension"]: c for c in outcome["comparability"]["checks"]}
    assert checks["categories"]["same"] is True
