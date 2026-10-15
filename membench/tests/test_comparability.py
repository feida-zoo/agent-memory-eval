"""A/B 可比性前置校验（membench.compare）测试。

覆盖可比性前置校验：关键元数据（驱动/判分模型、适配器实现、数据集指纹等）
缺失或不一致时的门禁判定、报告呈现与产物布局。conftest 不是必需的——直接
构造最小 metrics.json 写到 tmp 目录即可，全部产物落 ``out_dir=tmp_path``。
"""

from __future__ import annotations

import json
from pathlib import Path

from membench.compare import compare_runs


def _metrics(run_id: str = "run-x", **overrides) -> dict:
    """最小可用的 metrics.json 内容（含全部可比性维度与 overall 指标）。"""
    metrics = {
        "run_id": run_id,
        "adapter": "mock-strong",
        "adapter_version": "v1",
        "dataset": "sample_dataset",
        "dataset_path": "/data/sample_dataset.jsonl",
        "dataset_fingerprint": {
            "sha256": "aaaa1111",
            "bytes": 10,
            "name": "sample_dataset.jsonl",
        },
        "judge": "llm",
        "judge_model": "gpt-4o-mini",
        "driver_llm": "gpt-4o",
        "platform": "default",
        "commit": "a1b2c3",
        "limit": None,
        "shards": 1,
        "shard_index": 0,
        "adapter_meta": "k=v",
        "overall": {
            "accuracy": 0.9,
            "token_f1": 0.8,
            "bleu1": 0.7,
            "evidence_recall": 0.9,
            "abstention_accuracy": 0.8,
            "stale_rate": 0.0,
            "latency_p50_ms": 10.0,
            "latency_p95_ms": 20.0,
            "avg_input_tokens": 100.0,
            "avg_llm_calls": 2.0,
            "n_questions": 24,
        },
        "by_category": {},
    }
    metrics.update(overrides)
    return metrics


def _write_run(root: Path, dir_name: str, metrics: dict) -> str:
    """把 metrics 写成 ``root/<dir_name>/metrics.json``，返回 run 目录路径。"""
    run_dir = root / dir_name
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False), encoding="utf-8"
    )
    return str(run_dir)


def test_comparable_metadata_passes(tmp_path: Path) -> None:
    """① 元数据一致 -> verdict PASS 且 comparable is True，产物落 ab_* 实验目录。"""
    a = _write_run(tmp_path, "a", _metrics("run_a"))
    b = _write_run(tmp_path, "b", _metrics("run_b"))
    result = compare_runs(a, b, out_dir=str(tmp_path))

    assert result["verdict"] == "PASS"
    assert result["comparability"]["comparable"] is True
    gate = result["gates"][0]
    assert gate["name"] == "comparability" and gate["status"] == "PASS"

    ab = Path(result["compare_dir"])
    assert ab.name.startswith("ab_")
    assert (ab / "compare_report.md").is_file()
    assert (ab / "fig_metric_diff.png").is_file()
    assert (ab / "A" / "runs" / "a" / "metrics.json").is_file()
    assert (ab / "B" / "runs" / "b" / "metrics.json").is_file()


def test_judge_model_mismatch_warns(tmp_path: Path) -> None:
    """② 仅 judge_model 不同 -> verdict WARN、可比性门禁 WARN、报告含不可比原因。"""
    a = _write_run(tmp_path, "a", _metrics("run_a"))
    b = _write_run(tmp_path, "b", _metrics("run_b", judge_model="gpt-4o"))
    result = compare_runs(a, b, out_dir=str(tmp_path))

    assert result["verdict"] == "WARN"
    gate = result["gates"][0]
    assert gate["name"] == "comparability" and gate["status"] == "WARN"
    assert result["comparability"]["comparable"] is False

    report = (Path(result["compare_dir"]) / "compare_report.md").read_text(encoding="utf-8")
    assert "可比性前置校验" in report
    assert "不可比原因" in report
    assert "judge_model" in report


def test_strict_dataset_fingerprint_mismatch_fails(tmp_path: Path) -> None:
    """③ strict + dataset 指纹不同（critical）-> verdict FAIL。"""
    a = _write_run(tmp_path, "a", _metrics("run_a"))
    b = _write_run(
        tmp_path,
        "b",
        _metrics(
            "run_b",
            dataset_fingerprint={"sha256": "bbbb2222", "bytes": 12, "name": "x.jsonl"},
        ),
    )
    result = compare_runs(a, b, out_dir=str(tmp_path), require_comparable=True)

    assert result["verdict"] == "FAIL"
    assert result["gates"][0]["name"] == "comparability"
    assert result["gates"][0]["status"] == "FAIL"
    assert result["comparability"]["comparable"] is False


def test_legacy_run_missing_metadata(tmp_path: Path) -> None:
    """④ 旧 run（缺新元数据）-> unverifiable 非空；默认不因缺元数据变 WARN/FAIL，strict 下 FAIL。"""
    legacy = _metrics("run_x")
    for key in ("driver_llm", "judge_model", "adapter_version", "dataset_fingerprint"):
        legacy.pop(key)
    a = _write_run(tmp_path, "a", legacy)
    b = _write_run(tmp_path, "b", dict(legacy, run_id="run_y"))

    relaxed = compare_runs(a, b, out_dir=str(tmp_path))
    assert relaxed["comparability"]["unverifiable"]
    assert relaxed["verdict"] == "PASS"
    assert relaxed["gates"][0]["status"] == "PASS"

    strict = compare_runs(a, b, out_dir=str(tmp_path / "strict"), require_comparable=True)
    assert strict["verdict"] == "FAIL"
    assert strict["gates"][0]["status"] == "FAIL"


def test_commit_only_diff_is_comparable(tmp_path: Path) -> None:
    """⑤ 仅 commit 不同 -> 不判不可比（commit 是 A/B 的被测变量）。"""
    a = _write_run(tmp_path, "a", _metrics("run_a", commit="a1b2c3"))
    b = _write_run(tmp_path, "b", _metrics("run_b", commit="d4e5f6"))
    result = compare_runs(a, b, out_dir=str(tmp_path))

    assert result["comparability"]["comparable"] is True
    assert result["verdict"] == "PASS"
    commit_check = next(
        c for c in result["comparability"]["checks"] if c["dimension"] == "commit"
    )
    assert commit_check["same"] is False


def test_warn_dimension_mismatch_only_warns(tmp_path: Path) -> None:
    """次要维度（limit）不一致 -> WARN 门禁但不判不可比（comparable 仍为 True）。"""
    a = _write_run(tmp_path, "a", _metrics("run_a"))
    b = _write_run(tmp_path, "b", _metrics("run_b", limit=5))
    result = compare_runs(a, b, out_dir=str(tmp_path))

    assert result["verdict"] == "WARN"
    assert result["comparability"]["comparable"] is True
    assert result["gates"][0]["status"] == "WARN"


def test_rule_based_judge_model_not_applicable(tmp_path: Path) -> None:
    """规则判分器（heuristic）无模型标识：judge_model 记「不适用」，
    strict 模式下也不因它 FAIL（否则 heuristic 的 A/B 会无条件不可比）。"""
    a = _write_run(tmp_path, "a", _metrics("run_a", judge="heuristic", judge_model=None))
    b = _write_run(tmp_path, "b", _metrics("run_b", judge="heuristic", judge_model=None))
    result = compare_runs(a, b, out_dir=str(tmp_path), require_comparable=True)

    assert result["comparability"]["unverifiable"] == []
    check = next(
        c for c in result["comparability"]["checks"] if c["dimension"] == "judge_model"
    )
    assert check["same"] is True
    assert result["verdict"] == "PASS"
    assert result["gates"][0]["status"] == "PASS"


def test_missing_driver_llm_is_reported_as_reason(tmp_path: Path) -> None:
    """关键维度缺失（driver_llm）要出现在「不可比原因」清单里，默认模式也不静默。"""
    a = _write_run(tmp_path, "a", _metrics("run_a", driver_llm=None))
    b = _write_run(tmp_path, "b", _metrics("run_b", driver_llm=None))
    result = compare_runs(a, b, out_dir=str(tmp_path))

    assert result["comparability"]["unverifiable"]
    assert any("driver_llm" in r for r in result["comparability"]["reasons"])
