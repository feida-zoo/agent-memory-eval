"""多 run 矩阵对比（membench.matrix）测试。

用 MockWeakAdapter / MockStrongAdapter 跑 4 个带 platform@commit 标签的 run
（2 平台 × 2 commit），验证 compare_matrix 的透视表、汇总表、两两门禁用与产物文件。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from membench.adapters import MockStrongAdapter, MockWeakAdapter
from membench.data import SAMPLE_DATASET_PATH
from membench.matrix import compare_matrix
from membench.runner import Runner

_PLATFORMS = ("wx", "fs")
_COMMITS = ("a1b2c3", "d4e5f6")


def _make_runs(tmp_path: Path) -> list[tuple[str, str]]:
    """跑 4 个带不同 platform@commit 标签的 run，返回 [(label, run_dir), ...]。"""
    labeled: list[tuple[str, str]] = []
    for i, (platform, commit) in enumerate(
        (p, c) for p in _PLATFORMS for c in _COMMITS
    ):
        adapter = MockStrongAdapter() if i % 2 == 0 else MockWeakAdapter()
        runner = Runner(
            adapter,
            str(SAMPLE_DATASET_PATH),
            judge="heuristic",
            out_dir=str(tmp_path / "runs"),
            platform=platform,
            commit=commit,
        )
        run_dir = runner.run()
        labeled.append((f"{platform}@{commit}", run_dir))
    return labeled


def test_compare_matrix_pivot_and_files(tmp_path: Path) -> None:
    labeled = _make_runs(tmp_path)
    out_dir = tmp_path / "matrix_out"
    result = compare_matrix(labeled, str(out_dir))

    # 产物文件
    assert (out_dir / "matrix_report.md").is_file()
    assert (out_dir / "matrix_result.json").is_file()
    assert (out_dir / "fig_matrix_accuracy.png").is_file()
    assert Path(result["report_path"]).is_file()
    assert Path(result["result_path"]).is_file()

    # 透视表：5 个指标，每个都是 2 平台 × 2 commit
    pivots = result["pivot_tables"]
    assert pivots is not None
    assert set(pivots) == {
        "accuracy",
        "stale_rate",
        "evidence_recall",
        "abstention_accuracy",
        "latency_p95_ms",
    }
    for table in pivots.values():
        assert sorted(table) == ["fs", "wx"]
        for row in table.values():
            assert sorted(row) == ["a1b2c3", "d4e5f6"]

    # 透视值与 metrics.json 一致
    label_metrics = {}
    for label, run_dir in labeled:
        label_metrics[label] = json.loads(
            (Path(run_dir) / "metrics.json").read_text(encoding="utf-8")
        )
    acc = pivots["accuracy"]
    for label, (platform, commit) in (
        ("wx@a1b2c3", ("wx", "a1b2c3")),
        ("fs@d4e5f6", ("fs", "d4e5f6")),
    ):
        assert acc[platform][commit] == pytest.approx(
            label_metrics[label]["overall"]["accuracy"]
        )

    # 汇总表：4 个 label，每个含 5 个指标
    assert sorted(result["summary"]) == sorted(l for l, _ in labeled)
    for label in result["summary"]:
        assert set(result["summary"][label]) == set(pivots)

    # 两两组合：C(4,2)=6 组门禁判定
    assert len(result["pairwise"]) == 6
    for pair in result["pairwise"]:
        assert pair["verdict"] in {"PASS", "WARN", "FAIL"}
        assert len(pair["gates"]) == 3

    # matrix_result.json 可回读且与返回值一致
    saved = json.loads((out_dir / "matrix_result.json").read_text(encoding="utf-8"))
    assert saved["labels"] == result["labels"]
    assert saved["pivot_tables"] == result["pivot_tables"]

    # 报告内容：含透视表、汇总表与门禁小节
    report = (out_dir / "matrix_report.md").read_text(encoding="utf-8")
    assert "platform × commit 透视表" in report
    assert "逐指标汇总" in report
    assert "两两门禁判定" in report
    for label, _ in labeled:
        assert label in report


def test_compare_matrix_non_parseable_labels(tmp_path: Path) -> None:
    """任意字符串 label：无透视表，但汇总 / 门禁 / 图 / 文件仍齐全。"""
    labeled = _make_runs(tmp_path)
    labeled = [(f"run{i}", run_dir) for i, (_, run_dir) in enumerate(labeled)]
    out_dir = tmp_path / "matrix_np"
    result = compare_matrix(labeled, str(out_dir))
    assert result["pivot_tables"] is None
    assert (out_dir / "matrix_report.md").is_file()
    assert (out_dir / "fig_matrix_accuracy.png").is_file()
    assert len(result["pairwise"]) == 6
    report = (out_dir / "matrix_report.md").read_text(encoding="utf-8")
    assert "跳过透视表" in report
