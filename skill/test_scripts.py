"""skill/src/scripts 的回归测试：交付前自检 + 标准流程 dry-run。

运行::

    cd membench && python -m pytest ../skill/test_scripts.py -q
    # 或从仓库根：pytest skill/test_scripts.py -q
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent
SCRIPTS = SKILL_DIR / "src" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_task  # noqa: E402
import verify_artifacts  # noqa: E402
import watch_task  # noqa: E402

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
    assert data["datasets"][0]["dataset"] == "sample"
    assert data["datasets"][0]["n_questions"] == 2
    # 测试树没有 runtime/：只给警告级提示（不影响 ok / 退出码）
    assert [item["code"] for item in data["warnings"]] == ["missing_runtime"]


def test_runtime_present_has_no_runtime_warning(tmp_path: Path) -> None:
    """任务目录带 runtime/commands.jsonl 时不再提示"没有运行时数据"。"""
    task = _eval_task(tmp_path)
    _write(task / "runtime" / "commands.jsonl", "{}\n")
    result = verify_artifacts.verify_task(str(task))
    assert result.ok, [item.message for item in result.errors]
    assert not [item for item in result.warnings if item.code == "missing_runtime"]


def test_unknown_dialogue_entry_warns_only(tmp_path: Path) -> None:
    """对话目录里的未登记目录（如历史上的 memory_snapshots/）：警告级，不影响 ok。"""
    task = _eval_task(tmp_path)
    _write(task / "runs" / "sample" / "d001" / "memory_snapshots" / "entry.json", "{}")
    result = verify_artifacts.verify_task(str(task))
    assert result.ok, [item.message for item in result.errors]
    hits = [item for item in result.warnings if item.code == "dialogue_entry"]
    assert hits, [item.message for item in result.warnings]
    assert "memory_snapshots" in hits[0].message
    assert hits[0].severity == "warning"


def test_unknown_dataset_and_root_entries_warn_only(tmp_path: Path) -> None:
    """数据集目录与任务根里的未登记条目同样只警告。"""
    task = _eval_task(tmp_path)
    _write(task / "runs" / "sample" / "debug.log", "x")
    _write(task / "stray" / "note.txt", "x")
    result = verify_artifacts.verify_task(str(task))
    assert result.ok, [item.message for item in result.errors]
    codes = {item.code for item in result.warnings}
    assert {"dataset_entry", "root_entry"} <= codes
    # 正常产物（report/、对话目录）不得被误报
    assert not [item for item in result.warnings if item.code == "dialogue_entry"]


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


def test_eval_dry_run_passes_trace_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--trace-state 出现在 dry-run 的 run 命令里（单数据集路径）。"""
    assert _eval_dry_run(tmp_path, "--trace-state") == 0
    out = capsys.readouterr().out
    run_line = next(line for line in out.splitlines() if "-m membench.cli run" in line)
    assert "--trace-state" in shlex.split(run_line)


def test_eval_dry_run_passes_no_trace_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--no-trace-state 原样透传（显式关闭自动开启）。"""
    assert _eval_dry_run(tmp_path, "--no-trace-state") == 0
    out = capsys.readouterr().out
    run_line = next(line for line in out.splitlines() if "-m membench.cli run" in line)
    argv = shlex.split(run_line)
    assert "--no-trace-state" in argv
    assert "--trace-state" not in argv


def test_eval_dry_run_omits_trace_state_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """不给参数时不拼任何 trace-state 开关：自动判定交给 membench（interleaved + safety 才开）。"""
    assert _eval_dry_run(tmp_path) == 0
    out = capsys.readouterr().out
    run_line = next(line for line in out.splitlines() if "-m membench.cli run" in line)
    argv = shlex.split(run_line)
    assert "--trace-state" not in argv
    assert "--no-trace-state" not in argv


def test_eval_common_dry_run_passes_trace_state(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--trace-state 出现在 dry-run 的 evaluate 命令里（多数据集路径）。"""
    _common_home(tmp_path, monkeypatch)
    assert run_task.main(_eval_common_argv(tmp_path, "--trace-state")) == run_task.EXIT_OK
    out = capsys.readouterr().out
    line = next(line for line in out.splitlines() if "-m membench.cli evaluate" in line)
    assert "--trace-state" in shlex.split(line)


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


def test_dry_run_redacts_llm_api_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """dry-run / 后台日志里的命令展示不出现 API key（长期留存的日志不能带密钥）。"""
    code = run_task.main(
        [
            "eval",
            "--adapter",
            "mock-strong",
            "--dataset",
            str(SAMPLE_DATASET),
            "--judge",
            "llm",
            "--recall-mode",
            "substring",
            "--llm-api-key",
            "sk-dry-run-secret",
            "--report-out",
            str(tmp_path),
            "--dry-run",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "sk-dry-run-secret" not in out
    assert "--llm-api-key" in out and "***" in out


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
    assert "-m membench.cli evaluate --task-dir $TASK --fusion weighted --from-runs" in out
    assert "sample_dataset=$TASK/runs/sample_dataset" in out
    assert "other=$TASK/runs/other" in out


#: ``--common`` 预设的四个公开基准集；``plan_datasets`` 只检查文件存在，占位内容够用。
COMMON_DATASET_KEYS = ("locomo", "longmemeval_oracle", "gatemem_office", "beam_1m")


def _common_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把 HOME 指到临时目录并造齐 --common 需要的四个数据集文件。"""
    home = tmp_path / "home"
    datasets = home / ".membench" / "datasets"
    datasets.mkdir(parents=True)
    for key in COMMON_DATASET_KEYS:
        (datasets / f"{key}.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    return datasets


def _eval_common_argv(tmp_path: Path, *extra: str) -> list[str]:
    """一条最小的 ``eval --common --dry-run`` 调用。"""
    return [
        "eval",
        "--adapter",
        "mock-strong",
        "--common",
        "--judge",
        "llm",
        "--recall-mode",
        "substring",
        "--name",
        "common",
        "--report-out",
        str(tmp_path / "out"),
        "--dry-run",
        *extra,
    ]


def test_eval_common_dry_run_uses_preset_datasets(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--common：数据集恰为公开基准常用集四个，且不含办公域槽位 office。"""
    datasets = _common_home(tmp_path, monkeypatch)
    assert run_task.main(_eval_common_argv(tmp_path)) == run_task.EXIT_OK
    out = capsys.readouterr().out
    line = next(line for line in out.splitlines() if "-m membench.cli evaluate" in line)
    argv = shlex.split(line)
    values = [argv[index + 1] for index, token in enumerate(argv) if token == "--dataset"]
    expected = [
        str((datasets / f"{key}.jsonl").resolve()) for key in COMMON_DATASET_KEYS
    ]
    assert values == expected
    assert not any(Path(value).stem == "office" for value in values)
    assert not (tmp_path / "out").exists()


def test_common_and_dataset_are_mutually_exclusive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--common 与 --dataset 互斥：同时给直接报参数错误，不解析预设。"""
    code = run_task.main(
        _eval_common_argv(tmp_path, "--dataset", str(SAMPLE_DATASET))
    )
    assert code == run_task.EXIT_FLOW_FAIL
    err = capsys.readouterr().err
    assert "互斥" in err
    assert "--common" in err and "--dataset" in err


def test_eval_without_dataset_is_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--dataset 与 --common 都不给：报参数错误并说明二选一。"""
    argv = [token for token in _eval_common_argv(tmp_path) if token != "--common"]
    assert run_task.main(argv) == run_task.EXIT_FLOW_FAIL
    assert "必须给 --dataset" in capsys.readouterr().err


def test_fusion_equal_is_forwarded_to_evaluate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--fusion equal：evaluate --dataset 与 evaluate --from-runs 都带同一口径。"""
    other = tmp_path / "other.jsonl"
    other.write_text("{}\n", encoding="utf-8")
    base = [
        "eval",
        "--adapter",
        "mock-strong",
        "--dataset",
        str(SAMPLE_DATASET),
        "--dataset",
        str(other),
        "--recall-mode",
        "substring",
        "--fusion",
        "equal",
        "--report-out",
        str(tmp_path / "out"),
        "--dry-run",
    ]
    assert run_task.main([*base, "--judge", "llm"]) == run_task.EXIT_OK
    out = capsys.readouterr().out
    line = next(line for line in out.splitlines() if "-m membench.cli evaluate" in line)
    assert "--fusion equal" in line

    assert (
        run_task.main([*base, "--judge", "heuristic", "--limit", "2"])
        == run_task.EXIT_OK
    )
    out = capsys.readouterr().out
    line = next(line for line in out.splitlines() if "-m membench.cli evaluate" in line)
    assert "--from-runs" in line and "--fusion equal" in line


def test_ab_common_dry_run_covers_both_arms(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ab --common：两臂各自跑同一批四个数据集，融合口径透传 evaluate-compare。"""
    datasets = _common_home(tmp_path, monkeypatch)
    code = run_task.main(
        [
            "ab",
            "--baseline-adapter",
            "mock-weak",
            "--candidate-adapter",
            "mock-strong",
            "--common",
            "--judge",
            "llm",
            "--recall-mode",
            "substring",
            "--fusion",
            "equal",
            "--name",
            "ab-common",
            "--baseline-version",
            "v1",
            "--candidate-version",
            "v2",
            "--driver-llm-a",
            "none",
            "--driver-llm-b",
            "none",
            "--report-out",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    assert code == run_task.EXIT_OK
    out = capsys.readouterr().out
    run_lines = [line for line in out.splitlines() if "-m membench.cli run " in line]
    for key in COMMON_DATASET_KEYS:
        path = str((datasets / f"{key}.jsonl").resolve())
        hits = [line for line in run_lines if f"--dataset {path}" in line]
        assert len(hits) == 2, f"{key}: {hits}"
        arms = {line.split("--task-dir ")[1].split()[0] for line in hits}
        assert arms == {"$TASK/A", "$TASK/B"}
    compare_line = next(
        line for line in out.splitlines() if "-m membench.cli evaluate-compare" in line
    )
    assert "--fusion equal" in compare_line


# ---------------------------------------------------------------------------
# 后台执行 + 定时监控
# ---------------------------------------------------------------------------


def _background_eval_argv(tmp_path: Path, *extra: str) -> list[str]:
    """一条最小的后台 eval 调用（mock 适配器 + 冒烟 limit）。"""
    return [
        "eval",
        "--adapter",
        "mock-strong",
        "--dataset",
        str(SAMPLE_DATASET),
        "--judge",
        "heuristic",
        "--limit",
        "2",
        "--recall-mode",
        "substring",
        "--name",
        "bg-smoke",
        "--report-out",
        str(tmp_path / "out"),
        *extra,
    ]


def _orchestrator(task: Path, **overrides: object) -> Path:
    """写一个 orchestrator.json（watch_task 用例构造各种状态）。"""
    data = {
        "schema": 1,
        "pid": 999999,
        "argv": ["eval"],
        "command": "eval",
        "started_at": "2026-01-01T00:00:00+08:00",
        "log": "runtime/logs/orchestrator-20260101-000000.log",
        "status": "running",
        "exit_code": None,
        "finished_at": None,
    }
    data.update(overrides)
    path = task / "runtime" / "orchestrator.json"
    _write_json(path, data)
    return path


def _fake_task(tmp_path: Path, *, with_dataset: bool = True) -> Path:
    """最小任务目录（可选带一个完整数据集产物），用于 watch_task。"""
    task = tmp_path / "eval_demo_default_20260101-000000"
    (task / "reports").mkdir(parents=True, exist_ok=True)
    if with_dataset:
        _dataset_tree(task / "runs" / "sample")
    else:
        (task / "runs").mkdir(parents=True, exist_ok=True)
    return task


def test_run_task_background_returns_task_dir_and_completes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--background：stdout 只有任务目录一行，状态先 running，随后 exit_code=0。"""
    # 开关放在参数中间（不是最后）：子进程参数重建不能吞掉它后面的选项
    argv = _background_eval_argv(tmp_path)
    argv.insert(1, "--background")
    code = run_task.main(argv)
    assert code == run_task.EXIT_OK
    out_lines = [line for line in capsys.readouterr().out.strip().splitlines() if line.strip()]
    assert len(out_lines) == 1, f"stdout 只能有任务目录一行，实际: {out_lines}"
    task = Path(out_lines[0])
    assert task.is_dir() and task.name.startswith("eval_bg-smoke_default_")

    state_path = task / "runtime" / "orchestrator.json"
    assert state_path.is_file()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["pid"] > 0
    assert state["status"] in ("running", "done")
    assert state["log"].startswith("runtime/logs/orchestrator-")

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["status"] != "running":
            break
        time.sleep(0.2)
    assert state["status"] == "done", f"后台编排未完成: {state}"
    assert state["exit_code"] == 0
    assert state["finished_at"]
    assert (task / "runtime" / "commands.jsonl").is_file()
    log_text = (task / state["log"]).read_text(encoding="utf-8")
    assert "自检: OK" in log_text or "任务目录" in log_text

    assert watch_task.main([str(task), "--until-done", "--interval", "0.2"]) == 0


def test_run_task_background_rejects_dry_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--background 与 --dry-run 互斥（退出码 2，不建任何目录）。"""
    code = run_task.main(_background_eval_argv(tmp_path, "--background", "--dry-run"))
    assert code == run_task.EXIT_FLOW_FAIL
    assert "互斥" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_run_task_task_dir_skips_task_init_in_dry_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--task-dir：dry-run 直接把它当 $TASK，不再打印 task init。"""
    task = tmp_path / "existing"
    (task / "runs").mkdir(parents=True)
    (task / "reports").mkdir()
    code = run_task.main(_background_eval_argv(tmp_path, "--task-dir", str(task), "--dry-run"))
    assert code == run_task.EXIT_OK
    out = capsys.readouterr().out
    assert f"TASK={task}" in out
    assert "task init" not in out


def test_run_task_task_dir_requires_skeleton(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--task-dir 缺 runs/ 时拒绝（退出码 2），不会把产物写进一个随便的目录。"""
    empty = tmp_path / "empty"
    empty.mkdir()
    code = run_task.main(_background_eval_argv(tmp_path, "--task-dir", str(empty), "--dry-run"))
    assert code == run_task.EXIT_FLOW_FAIL
    assert "不是 arms=1 的任务目录" in capsys.readouterr().err


def test_watch_task_format_and_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """watch_task：单行紧凑进度 + --json 机器可读结构（数据集产物齐备）。"""
    task = _fake_task(tmp_path)
    _orchestrator(task, status="done", exit_code=0, finished_at="2026-01-01T00:00:05+08:00")

    assert watch_task.main([str(task)]) == 0
    line = capsys.readouterr().out.strip()
    assert line.startswith("[watch] ")
    assert "单臂: sample 1/1片·产物✓" in line
    assert "orchestrator done exit=0" in line

    assert watch_task.main([str(task), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["task"] == str(task)
    assert payload["orchestrator"]["status"] == "done"
    assert payload["arms"][0]["datasets"][0]["ready"] is True
    assert payload["snapshot_pool"] is None
    assert payload["done"] is True and payload["ok"] is True


def _dead_pid() -> int:
    """一个确定已经退出的 pid（避免硬编码在别的系统上撞到活进程）。"""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_watch_task_until_done_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--until-done：成功 0 / 失败 1 / 超时 2；没有 orchestrator 也按 2 退出。"""
    done = _fake_task(tmp_path / "done")
    _orchestrator(done, status="done", exit_code=0)
    assert watch_task.main([str(done), "--until-done", "--interval", "0.05"]) == 0

    failed = _fake_task(tmp_path / "failed")
    _orchestrator(failed, status="failed", exit_code=2)
    assert watch_task.main([str(failed), "--until-done", "--interval", "0.05"]) == 1

    # running 但 pid 已不在 -> stalled（失败），不会一直等下去
    stalled = _fake_task(tmp_path / "stalled")
    _orchestrator(stalled, status="running", pid=_dead_pid())
    assert watch_task.main([str(stalled), "--until-done", "--interval", "0.05"]) == 1

    # running 且进程活着（用当前测试进程）-> 等满 max-wait 后超时
    running = _fake_task(tmp_path / "running")
    _orchestrator(running, status="running", pid=os.getpid())
    assert (
        watch_task.main(
            [str(running), "--until-done", "--interval", "0.05", "--max-wait", "0.2"]
        )
        == 2
    )

    # 非后台任务（没有 orchestrator.json）：无法判断完成
    plain = _fake_task(tmp_path / "plain")
    assert (
        watch_task.main([str(plain), "--until-done", "--interval", "0.05", "--max-wait", "2"])
        == 2
    )
    assert "没有 orchestrator.json" in capsys.readouterr().err


def test_trace_state_dir_is_registered(tmp_path: Path) -> None:
    """对话目录下的 ``trace_state/`` 是框架产物（--trace-state 逐 as_of 状态）。

    interleaved + safety 题会**自动开启** trace-state，所以这目录在正常产物里
    必然出现；不登记就会把合规产物报成"未登记条目"。
    """
    task = _eval_task(tmp_path)
    dialogue = task / "runs" / "sample" / "d001"
    _write_json(
        dialogue / "trace_state" / "t60" / "manifest.json",
        {"dialogue_id": "d001", "trace": {"as_of": 60}},
    )
    result = verify_artifacts.verify_task(str(task))
    assert result.ok, [item.message for item in result.errors]
    assert "dialogue_entry" not in _codes(result)
