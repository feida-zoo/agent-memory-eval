"""多 run 矩阵对比（membench.matrix）测试。

用 MockWeakAdapter / MockStrongAdapter 跑 4 个带 platform@commit 标签的 run
（2 平台 × 2 commit），验证 compare_matrix 的透视表、汇总表、两两门禁用与产物文件。

产物结构（见 :mod:`membench.layout`）：``out_dir`` 是**完整实验目录**——报告落
``<out_dir>/reports/{report.md, matrix_result.json, fig_matrix_accuracy.png}``，
各 label 的数据集树副本落 ``<out_dir>/runs/<label>/``。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from membench import cli, layout
from membench.adapters import MockStrongAdapter, MockWeakAdapter
from membench.data import SAMPLE_DATASET_PATH
from membench.matrix import compare_matrix
from membench.runner import Runner

_PLATFORMS = ("wx", "fs")
_COMMITS = ("a1b2c3", "d4e5f6")

_DATASET = "sample_dataset"


def _make_runs(tmp_path: Path) -> list[tuple[str, str]]:
    """跑 4 个带不同 platform@commit 标签的 run，返回 ``[(label, 实验目录), ...]``。

    每个 run 独占一个实验目录：新布局下数据集目录固定为
    ``<实验目录>/runs/<dataset>/``，共用 out_dir 会让 4 个 run 互相覆盖。
    """
    labeled: list[tuple[str, str]] = []
    for i, (platform, commit) in enumerate(
        (p, c) for p in _PLATFORMS for c in _COMMITS
    ):
        adapter = MockStrongAdapter() if i % 2 == 0 else MockWeakAdapter()
        experiment_dir = tmp_path / f"exp{i}"
        runner = Runner(
            adapter,
            str(SAMPLE_DATASET_PATH),
            judge="heuristic",
            out_dir=str(experiment_dir),
            platform=platform,
            commit=commit,
        )
        dataset_dir = runner.run()
        assert dataset_dir == str(experiment_dir / layout.DIR_RUNS / _DATASET)
        labeled.append((f"{platform}@{commit}", str(experiment_dir)))
    return labeled


def _dataset_metrics(experiment_dir: str) -> dict:
    """读某实验目录下数据集级指标（``runs/<dataset>/report/metrics.json``）。"""
    path = Path(layout.dataset_report_dir(experiment_dir, _DATASET)) / layout.FILE_METRICS
    return json.loads(path.read_text(encoding="utf-8"))


def test_compare_matrix_pivot_and_files(tmp_path: Path) -> None:
    labeled = _make_runs(tmp_path)
    out_dir = tmp_path / "matrix_out"
    result = compare_matrix(labeled, str(out_dir))

    # 产物文件：报告与图落 <实验目录>/reports/
    reports = out_dir / layout.DIR_REPORTS
    for name in (
        layout.FILE_REPORT_MD,
        layout.FILE_MATRIX_RESULT,
        layout.FIG_MATRIX_ACCURACY,
    ):
        assert (reports / name).is_file(), f"缺少 reports/{name}"
    assert Path(result["report_path"]) == reports / layout.FILE_REPORT_MD
    assert Path(result["result_path"]) == reports / layout.FILE_MATRIX_RESULT
    assert result["out_dir"] == str(out_dir)

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

    # 透视值与各 run 的数据集级 metrics.json 一致
    label_metrics = {label: _dataset_metrics(exp) for label, exp in labeled}
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

    # 各 label 的数据集树副本：runs/<slug(label)>/，含数据集级报告与对话目录
    assert sorted(result["label_dirs"]) == sorted(l for l, _ in labeled)
    for label, _exp in labeled:
        label_dir = Path(result["label_dirs"][label])
        assert label_dir == out_dir / layout.DIR_RUNS / layout.slug(label)
        assert (label_dir / layout.DIR_REPORT / layout.FILE_METRICS).is_file()
        assert (label_dir / layout.DIR_REPORT / layout.FILE_REPORT_MD).is_file()
        # 整树复制：对话目录（runtime/ + 对话级 report/）一并带过来
        first_dialogue = sorted(
            p for p in label_dir.iterdir() if p.is_dir() and p.name != layout.DIR_REPORT
        )[0]
        assert (first_dialogue / layout.DIR_RUNTIME).is_dir()
        assert (first_dialogue / layout.DIR_REPORT / layout.FILE_METRICS).is_file()

    # matrix_result.json 可回读且与返回值一致
    saved = json.loads((reports / layout.FILE_MATRIX_RESULT).read_text(encoding="utf-8"))
    assert saved["labels"] == result["labels"]
    assert saved["pivot_tables"] == result["pivot_tables"]
    assert saved["label_dirs"] == result["label_dirs"]

    # 报告内容：含透视表、汇总表与门禁小节；图片按同目录相对文件名引用
    report = (reports / layout.FILE_REPORT_MD).read_text(encoding="utf-8")
    assert "platform × commit 透视表" in report
    assert "逐指标汇总" in report
    assert "两两门禁判定" in report
    assert f"]({layout.FIG_MATRIX_ACCURACY})" in report
    assert f"]({reports / layout.FIG_MATRIX_ACCURACY})" not in report
    for label, _ in labeled:
        assert label in report


def test_compare_matrix_non_parseable_labels(tmp_path: Path) -> None:
    """任意字符串 label：无透视表，但汇总 / 门禁 / 图 / 文件仍齐全。"""
    labeled = _make_runs(tmp_path)
    labeled = [(f"run{i}", exp) for i, (_, exp) in enumerate(labeled)]
    out_dir = tmp_path / "matrix_np"
    result = compare_matrix(labeled, str(out_dir))

    assert result["pivot_tables"] is None
    reports = out_dir / layout.DIR_REPORTS
    assert (reports / layout.FILE_REPORT_MD).is_file()
    assert (reports / layout.FIG_MATRIX_ACCURACY).is_file()
    assert len(result["pairwise"]) == 6
    report = (reports / layout.FILE_REPORT_MD).read_text(encoding="utf-8")
    assert "跳过透视表" in report
    for i in range(4):
        assert (out_dir / layout.DIR_RUNS / f"run{i}").is_dir()


def test_compare_matrix_rejects_label_dir_conflict(tmp_path: Path) -> None:
    """两个 label 归一成同一目录名 -> 报错，且不留下半个实验目录（报告未落盘）。"""
    labeled = _make_runs(tmp_path)
    out_dir = tmp_path / "matrix_conflict"
    with pytest.raises(ValueError, match="标签目录冲突"):
        compare_matrix(
            [("a b", labeled[0][1]), ("a/b", labeled[1][1])], str(out_dir)
        )
    assert not (out_dir / layout.DIR_REPORTS).exists()


def test_cli_matrix_rejects_too_few_labels_without_leftovers(tmp_path: Path) -> None:
    """CLI matrix 入参不足 -> 友好报错（不是 Python 堆栈），且不留空 matrix_<ts>/。"""
    labeled = _make_runs(tmp_path)
    out = tmp_path / "cli_out"

    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            ["matrix", "--runs", f"wx@a1={labeled[0][1]}", "--report-out", str(out)]
        )
    message = str(excinfo.value)
    assert "matrix 至少需要 2 个" in message
    assert "Traceback" not in message
    assert not out.exists() or not list(out.iterdir()), "入参不足却留下空实验目录"

    # 入参够时同一批产物正常出矩阵；报告落 <report-out>/matrix_<ts>/reports/
    rc = cli.main(
        [
            "matrix",
            "--runs",
            f"wx@a1b2c3={labeled[0][1]}",
            f"fs@a1b2c3={labeled[1][1]}",
            "--report-out",
            str(out),
        ]
    )
    assert rc == 0
    matrix_dirs = [p for p in out.iterdir() if p.is_dir()]
    assert len(matrix_dirs) == 1, "失败路径不该留下目录（否则这里会多于一个）"
    assert matrix_dirs[0].name.startswith("matrix_")
    assert (matrix_dirs[0] / layout.DIR_REPORTS / layout.FILE_REPORT_MD).is_file()
