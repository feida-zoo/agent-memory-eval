"""产物布局（membench.layout）测试：命名 helper 与实验目录 materialize。

覆盖接入侧反馈 2/7：一次实验一个自包含子目录——报告与图表落目录根，
对应 run 副本落 ``runs/<run_id>/``，不再与全局 run 混放。
另覆盖 ``copy_runs_into`` 的目标目录冲突检测（同名会静默合并）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from membench import layout


def _fake_run(root: Path, name: str) -> Path:
    """构造一个最小 run 目录：报告（引用两张图）+ metrics + results + 图。"""
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "results.jsonl").write_text('{"qid": "q1"}\n', encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps({"run_id": name, "overall": {"accuracy": 1.0}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "fig_category_accuracy.png").write_bytes(b"png-cat")
    (run_dir / "fig_latency_hist.png").write_bytes(b"png-lat")
    (run_dir / "report.md").write_text(
        "# r\n\n![a](fig_category_accuracy.png)\n\n![b](fig_latency_hist.png)\n",
        encoding="utf-8",
    )
    return run_dir


def test_naming_helpers() -> None:
    """subdir_for / ab_subdir / slug 的命名约定（含缺省 platform 与空标签回退）。"""
    assert layout.slug("mock strong/a") == "mock-strong-a"
    assert layout.slug("") == "default"
    assert layout.subdir_for("mock-strong", "wx", ts="261014_120000") == (
        "eval_mock-strong_wx_261014_120000"
    )
    assert layout.subdir_for("mock-strong", None, ts="261014_120000") == (
        "eval_mock-strong_default_261014_120000"
    )
    assert layout.ab_subdir("a1_vs_b1", ts="261014_120000") == "ab_a1_vs_b1_261014_120000"


def test_materialize_run_experiment(tmp_path: Path) -> None:
    """单次评测：报告/metrics/结果/图落实验目录根，run 副本落 runs/<run_id>/。"""
    run_dir = _fake_run(tmp_path / "runs", "run-1")
    exp_dir = layout.materialize_run_experiment(str(tmp_path / "eval_x"), str(run_dir))

    exp = Path(exp_dir)
    for fname in (
        "report.md",
        "metrics.json",
        "results.jsonl",
        "fig_category_accuracy.png",
        "fig_latency_hist.png",
    ):
        assert (exp / fname).is_file(), f"缺少 {fname}"
    assert (exp / "runs" / "run-1" / "metrics.json").is_file()
    assert (exp / "runs" / "run-1" / "results.jsonl").is_file()


def test_materialize_ab_experiment(tmp_path: Path) -> None:
    """A/B：A、B 各带 runs/<run_id>/ 与自身报告，对比报告落实验目录根。"""
    run_a = _fake_run(tmp_path / "runs", "run-a")
    run_b = _fake_run(tmp_path / "runs", "run-b")
    compare_dir = tmp_path / "staging"
    compare_dir.mkdir()
    (compare_dir / "fig_metric_diff.png").write_bytes(b"png-diff")
    (compare_dir / "compare_report.md").write_text(
        "# c\n\n![d](fig_metric_diff.png)\n", encoding="utf-8"
    )

    ab_dir = layout.materialize_ab_experiment(
        str(tmp_path / "ab_x"), str(run_a), str(run_b), str(compare_dir)
    )
    ab = Path(ab_dir)
    assert (ab / "compare_report.md").is_file()
    assert (ab / "fig_metric_diff.png").is_file()
    for arm, run_id in (("A", "run-a"), ("B", "run-b")):
        assert (ab / arm / "report.md").is_file(), f"{arm} 缺少 report.md"
        assert (ab / arm / "metrics.json").is_file(), f"{arm} 缺少 metrics.json"
        assert (ab / arm / "runs" / run_id / "metrics.json").is_file()


def test_copy_runs_into_rejects_basename_conflict(tmp_path: Path) -> None:
    """labels 缺省且两个不同源目录 basename 相同：抛 ValueError，不静默合并。"""
    a = _fake_run(tmp_path / "left", "run-1")
    b = _fake_run(tmp_path / "right", "run-1")

    with pytest.raises(ValueError, match="目标目录冲突"):
        layout.copy_runs_into(str(tmp_path / "dest"), [str(a), str(b)])

    # 冲突时先整体校验再复制：不留下半个目标目录
    assert not (tmp_path / "dest" / "run-1").exists()


def test_copy_runs_into_rejects_slug_colliding_labels(tmp_path: Path) -> None:
    """label 互不相同但 slug() 归一成同名：同样抛 ValueError。"""
    a = _fake_run(tmp_path / "runs", "run-a")
    b = _fake_run(tmp_path / "runs", "run-b")

    with pytest.raises(ValueError, match="目标目录冲突"):
        layout.copy_runs_into(
            str(tmp_path / "dest"), [str(a), str(b)], labels=["a b", "a/b"]
        )


def test_copy_runs_into_allows_distinct_targets(tmp_path: Path) -> None:
    """不冲突路径不受影响：basename 互异 / label 互异时各复制到自己的目录。"""
    a = _fake_run(tmp_path / "runs", "run-a")
    b = _fake_run(tmp_path / "runs", "run-b")

    dests = layout.copy_runs_into(str(tmp_path / "dest"), [str(a), str(b)])
    assert [Path(d).name for d in dests] == ["run-a", "run-b"]
    assert (tmp_path / "dest" / "run-a" / "metrics.json").is_file()
    assert (tmp_path / "dest" / "run-b" / "metrics.json").is_file()

    labeled = layout.copy_runs_into(
        str(tmp_path / "dest2"), [str(a), str(b)], labels=["L1", "L2"]
    )
    assert [Path(d).name for d in labeled] == ["L1", "L2"]
    assert (tmp_path / "dest2" / "L1" / "metrics.json").is_file()


def test_copy_runs_into_same_source_repeated_is_not_conflict(tmp_path: Path) -> None:
    """同一源目录重复出现（dest == 源）不算冲突，也不会自我复制。"""
    run_dir = _fake_run(tmp_path / "runs", "run-1")

    dests = layout.copy_runs_into(str(tmp_path / "runs"), [str(run_dir), str(run_dir)])

    assert dests == [str(run_dir), str(run_dir)]
    assert (run_dir / "metrics.json").is_file()
