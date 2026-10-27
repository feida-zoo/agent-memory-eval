"""skill/src/scripts 的回归测试：交付前自检 + 标准流程 dry-run。

运行::

    cd membench && python -m pytest ../skill/test_scripts.py -q
    # 或从仓库根：pytest skill/test_scripts.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent
SCRIPTS = SKILL_DIR / "src" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_task  # noqa: E402
import verify_artifacts  # noqa: E402

#: 仓库内置冒烟数据集（仅用于 dry-run 的参数解析，不会真跑）。
SAMPLE_DATASET = (
    Path(__file__).resolve().parents[1]
    / "membench"
    / "membench"
    / "data"
    / "sample_dataset.jsonl"
)


def _write(path: Path, text: str) -> None:
    """写文本文件（自动建父目录）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, data: dict) -> None:
    """写 JSON 文件。"""
    _write(path, json.dumps(data, ensure_ascii=False))


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """写 JSONL 文件。"""
    _write(path, "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records))


def _dataset_tree(
    dataset_dir: Path,
    *,
    n_questions: int = 2,
    records: list[dict] | None = None,
    shards: int = 1,
    shard_index: int = 0,
    limit: int | None = None,
    n_dialogues: int = 1,
) -> None:
    """构造一个最小的数据集产物（数据集级 + 对话级）。"""
    if records is None:
        records = [{"qid": f"q{i}"} for i in range(1, n_questions + 1)]
    _write_json(
        dataset_dir / "report" / "metrics.json",
        {
            "overall": {"n_questions": n_questions},
            "shards": shards,
            "shard_index": shard_index,
            "limit": limit,
        },
    )
    _write_jsonl(dataset_dir / "report" / "results.jsonl", records)
    _write(dataset_dir / "report" / "report.md", "# dataset report\n")

    base = n_questions // n_dialogues
    remainder = n_questions % n_dialogues
    start = 0
    for index in range(n_dialogues):
        count = base + (1 if index < remainder else 0)
        dialogue_records = records[start : start + count]
        start += count
        dialogue_dir = dataset_dir / f"d{index + 1:03d}"
        _write_json(
            dialogue_dir / "report" / "metrics.json",
            {
                "overall": {"n_questions": count},
                "dialogue_id": f"d{index + 1:03d}",
            },
        )
        _write_jsonl(dialogue_dir / "report" / "results.jsonl", dialogue_records)
        _write(dialogue_dir / "report" / "report.md", f"# dialogue {index + 1}\n")


def _eval_task(
    root: Path,
    *,
    name: str = "eval_demo_default_20260101-000000",
    **dataset_kwargs,
) -> Path:
    """构造一个最小的 ``eval_*`` 任务树。"""
    task = root / name
    _write(task / "reports" / "report.md", "# task report\n")
    _write_json(task / "reports" / "metrics.json", {"overall": {}})
    _dataset_tree(task / "runs" / "sample", **dataset_kwargs)
    return task


def _compare_task(
    root: Path,
    *,
    name: str = "compare_demo_default_20260101-000000",
    comparison: str | None = None,
) -> Path:
    """构造一个最小的 ``compare_*`` 任务树（A/B 各一个数据集）。"""
    task = root / name
    _write(task / "reports" / "report.md", "# compare container\n")
    for arm in ("A", "B"):
        _dataset_tree(task / arm / "runs" / "sample")
        _write(task / arm / "reports" / "report.md", f"# arm {arm}\n")
    if comparison is not None:
        _write(task / "reports" / "comparison.md", comparison)
    return task


def _codes(result: verify_artifacts.Verification) -> set[str]:
    """失败项的 code 集合。"""
    return {item.code for item in result.errors}


# ---------------------------------------------------------------------------
# verify_artifacts：纯函数级用例
# ---------------------------------------------------------------------------


def test_valid_eval_task_passes(tmp_path: Path) -> None:
    """正常的 eval 任务应通过，并给出数据集摘要。"""
    task = _eval_task(tmp_path)
    result = verify_artifacts.verify_task(str(task))
    assert result.ok, [item.message for item in result.errors]
    assert result.kind == "eval"
    assert len(result.datasets) == 1
    info = result.datasets[0]
    assert info.dataset == "sample"
    assert info.n_questions == 2
    assert info.n_results == 2
    assert info.n_dialogues == 1


def test_missing_metrics_fails(tmp_path: Path) -> None:
    """数据集级缺 metrics.json 必须失败。"""
    task = _eval_task(tmp_path)
    (task / "runs" / "sample" / "report" / "metrics.json").unlink()
    result = verify_artifacts.verify_task(str(task))
    assert not result.ok
    assert "missing_metrics.json" in _codes(result)


def test_results_count_mismatch_fails(tmp_path: Path) -> None:
    """数据集级 results 行数 != 各对话之和必须失败。"""
    task = _eval_task(tmp_path)
    _write_jsonl(
        task / "runs" / "sample" / "report" / "results.jsonl",
        [{"qid": "q1"}, {"qid": "q2"}, {"qid": "q3"}],
    )
    result = verify_artifacts.verify_task(str(task))
    assert not result.ok
    assert "results_count_mismatch" in _codes(result)


def test_duplicate_qid_fails(tmp_path: Path) -> None:
    """数据集级 qid 重复必须失败。"""
    task = _eval_task(tmp_path)
    _write_jsonl(
        task / "runs" / "sample" / "report" / "results.jsonl",
        [{"qid": "q1"}, {"qid": "q1"}],
    )
    result = verify_artifacts.verify_task(str(task))
    assert not result.ok
    assert "duplicate_qid" in _codes(result)


def test_dead_link_fails(tmp_path: Path) -> None:
    """报告里的相对死链必须失败。"""
    task = _eval_task(tmp_path)
    _write(
        task / "runs" / "sample" / "report" / "report.md",
        "# report\n\n![missing](missing_figure.png)\n",
    )
    result = verify_artifacts.verify_task(str(task))
    assert not result.ok
    assert "dead_link" in _codes(result)


def test_limit_requires_allow_limit(tmp_path: Path) -> None:
    """limit 冒烟产物默认不能当正式结论；--allow-limit 放行。"""
    task = _eval_task(tmp_path, limit=5)
    result = verify_artifacts.verify_task(str(task))
    assert not result.ok
    assert "limit_smoke" in _codes(result)
    allowed = verify_artifacts.verify_task(str(task), allow_limit=True)
    assert allowed.ok, [item.message for item in allowed.errors]


def test_partial_shard_requires_allow_partial(tmp_path: Path) -> None:
    """未融合分片（shards>1 且 shard_index 非 None）默认失败；--allow-partial 放行。"""
    task = _eval_task(tmp_path, shards=2, shard_index=0)
    result = verify_artifacts.verify_task(str(task))
    assert not result.ok
    assert "partial_shard" in _codes(result)
    allowed = verify_artifacts.verify_task(str(task), allow_partial=True)
    assert allowed.ok, [item.message for item in allowed.errors]


def test_compare_missing_comparison_fails(tmp_path: Path) -> None:
    """compare 任务缺 reports/comparison.md 必须失败。"""
    task = _compare_task(tmp_path)
    result = verify_artifacts.verify_task(str(task))
    assert not result.ok
    assert "missing_comparison" in _codes(result)


def test_compare_with_gate_conclusion_passes(tmp_path: Path) -> None:
    """compare 任务有带门禁结论的 comparison.md 才通过。"""
    task = _compare_task(
        tmp_path,
        comparison="# 对比\n\n**门禁结论： PASS**\n",
    )
    result = verify_artifacts.verify_task(str(task))
    assert result.ok, [item.message for item in result.errors]
    assert result.kind == "compare"


def test_json_structure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--json 输出机器可读结构，且 stdout 只有 JSON。"""
    task = _eval_task(tmp_path)
    code = verify_artifacts.main([str(task), "--json"])
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["schema_version"] == 1
    assert data["task"] == str(task.resolve())
    assert data["kind"] == "eval"
    assert data["ok"] is True
    assert data["errors"] == []
    assert data["warnings"] == []
    assert data["datasets"][0]["dataset"] == "sample"
    assert data["datasets"][0]["n_questions"] == 2


def test_bad_tree_exit_code_and_fail_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """坏目录：退出码 1，人类输出逐条 FAIL。"""
    task = _eval_task(tmp_path)
    (task / "runs" / "sample" / "report" / "report.md").unlink()
    code = verify_artifacts.main([str(task)])
    out = capsys.readouterr().out
    assert code == 1
    assert "FAIL" in out
    assert "missing_report.md" in out


# ---------------------------------------------------------------------------
# run_task --dry-run：命令序列断言（不建目录、不跑评测）
# ---------------------------------------------------------------------------


def _eval_dry_run(tmp_path: Path, *extra: str) -> int:
    """调用 run_task eval --dry-run。"""
    return run_task.main(
        [
            "eval",
            "--adapter",
            "mock-strong",
            "--dataset",
            str(SAMPLE_DATASET),
            "--judge",
            "heuristic",
            "--limit",
            "5",
            "--recall-mode",
            "substring",
            "--name",
            "smoke",
            "--report-out",
            str(tmp_path),
            "--dry-run",
            *extra,
        ]
    )


def test_eval_dry_run_shards_are_parallel(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--shards 2：两片并行 + wait，然后 fuse；不创建任务目录。"""
    assert _eval_dry_run(tmp_path, "--shards", "2") == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert "task init --arms 1 --name smoke" in out
    assert "--evidence-match substring" in out
    shard0 = next(
        line for line in lines if "--shard-index 0 --shard-by dialogue" in line
    )
    shard1 = next(
        line for line in lines if "--shard-index 1 --shard-by dialogue" in line
    )
    assert shard0.rstrip().endswith("&")
    assert shard1.rstrip().endswith("&")
    assert lines.index("wait") > max(lines.index(shard0), lines.index(shard1))
    fuse_index = next(
        index
        for index, line in enumerate(lines)
        if "fuse --task-dir $TASK/runs/sample_dataset" in line
    )
    assert fuse_index > lines.index("wait")
    assert not any(path.name.startswith(("eval_", "compare_")) for path in tmp_path.iterdir())


def test_eval_dry_run_without_shards_has_no_fuse(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """不分片：只有一条 run，不出现 fuse。"""
    assert _eval_dry_run(tmp_path) == 0
    out = capsys.readouterr().out
    assert "--task-dir $TASK" in out
    assert "fuse" not in out
    assert "wait" not in out


def test_ab_dry_run_arm_and_compare_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """ab --shards 2：A 臂两片 → A fuse → B 臂两片 → B fuse → compare strict 最后。"""
    code = run_task.main(
        [
            "ab",
            "--baseline-adapter",
            "mock-weak",
            "--candidate-adapter",
            "mock-strong",
            "--dataset",
            str(SAMPLE_DATASET),
            "--judge",
            "heuristic",
            "--limit",
            "5",
            "--recall-mode",
            "substring",
            "--shards",
            "2",
            "--name",
            "ab-smoke",
            "--baseline-version",
            "v1",
            "--candidate-version",
            "v2",
            "--driver-llm-a",
            "none",
            "--driver-llm-b",
            "none",
            "--report-out",
            str(tmp_path),
            "--dry-run",
        ]
    )
    assert code == 0
    lines = capsys.readouterr().out.splitlines()
    assert "task init --arms 2 --name ab-smoke" in "\n".join(lines)

    a_run0 = next(
        index
        for index, line in enumerate(lines)
        if "--adapter mock-weak" in line and "--shard-index 0" in line
    )
    a_run1 = next(
        index
        for index, line in enumerate(lines)
        if "--adapter mock-weak" in line and "--shard-index 1" in line
    )
    b_run0 = next(
        index
        for index, line in enumerate(lines)
        if "--adapter mock-strong" in line and "--shard-index 0" in line
    )
    b_run1 = next(
        index
        for index, line in enumerate(lines)
        if "--adapter mock-strong" in line and "--shard-index 1" in line
    )
    a_fuse = next(
        index
        for index, line in enumerate(lines)
        if "fuse --task-dir $TASK/A/runs/sample_dataset" in line
    )
    b_fuse = next(
        index
        for index, line in enumerate(lines)
        if "fuse --task-dir $TASK/B/runs/sample_dataset" in line
    )
    compare = next(
        index
        for index, line in enumerate(lines)
        if "--task-dir $TASK --require-comparable" in line
    )
    assert a_run0 < a_run1 < a_fuse < b_run0 < b_run1 < b_fuse < compare
    assert lines[a_run0].rstrip().endswith("&")
    assert lines[a_run1].rstrip().endswith("&")
    assert lines[b_run0].rstrip().endswith("&")
    assert lines[b_run1].rstrip().endswith("&")


def test_heuristic_without_limit_is_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """heuristic 未配 --limit：拒绝并说明正式评测请用 llm/client。"""
    code = run_task.main(
        [
            "eval",
            "--adapter",
            "mock-strong",
            "--dataset",
            str(SAMPLE_DATASET),
            "--judge",
            "heuristic",
            "--recall-mode",
            "substring",
            "--report-out",
            str(tmp_path),
            "--dry-run",
        ]
    )
    assert code == run_task.EXIT_FLOW_FAIL
    err = capsys.readouterr().err
    assert "只允许与 --limit" in err
    assert "llm/client" in err


def test_recall_mode_maps_to_evidence_match(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--recall-mode llm 必须映射成 --evidence-match llm。"""
    code = run_task.main(
        [
            "eval",
            "--adapter",
            "mock-strong",
            "--dataset",
            str(SAMPLE_DATASET),
            "--judge",
            "heuristic",
            "--limit",
            "3",
            "--recall-mode",
            "llm",
            "--report-out",
            str(tmp_path),
            "--dry-run",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "--evidence-match llm" in out


def test_eval_multi_dataset_uses_evaluate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """多数据集且无分片 / limit：直接走 evaluate --dataset（官方路径）。"""
    other = tmp_path / "other.jsonl"
    other.write_text("{}\n", encoding="utf-8")
    code = run_task.main(
        [
            "eval",
            "--adapter",
            "mock-strong",
            "--dataset",
            str(SAMPLE_DATASET),
            "--dataset",
            str(other),
            "--judge",
            "llm",
            "--recall-mode",
            "substring",
            "--name",
            "multi",
            "--report-out",
            str(tmp_path),
            "--dry-run",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "-m membench.cli evaluate" in out
    assert "--task-dir $TASK" in out
    assert f"--dataset {SAMPLE_DATASET.resolve()}" in out
    assert f"--dataset {other.resolve()}" in out
    assert "-m membench.cli run " not in out


def test_eval_multi_dataset_with_limit_uses_from_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """多数据集 + limit（evaluate 不支持）：逐数据集 run，再 evaluate --from-runs。"""
    other = tmp_path / "other.jsonl"
    other.write_text("{}\n", encoding="utf-8")
    code = run_task.main(
        [
            "eval",
            "--adapter",
            "mock-strong",
            "--dataset",
            str(SAMPLE_DATASET),
            "--dataset",
            str(other),
            "--judge",
            "heuristic",
            "--limit",
            "3",
            "--recall-mode",
            "substring",
            "--name",
            "multi-limit",
            "--report-out",
            str(tmp_path),
            "--dry-run",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert out.count("-m membench.cli run ") == 2
    assert "-m membench.cli evaluate --task-dir $TASK --from-runs" in out
    assert "sample_dataset=$TASK/runs/sample_dataset" in out
    assert "other=$TASK/runs/other" in out
