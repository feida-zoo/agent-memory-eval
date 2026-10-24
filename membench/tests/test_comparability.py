"""A/B 可比性前置校验（membench.compare）测试。

覆盖可比性前置校验：关键元数据（驱动/判分模型、适配器实现、数据集指纹等）
缺失或不一致时的门禁判定、报告呈现与产物布局。conftest 不是必需的——直接
构造最小 metrics.json 写到 tmp 目录即可。

产物结构按新布局（见 :mod:`membench.layout`）：指标固定读
``<数据集目录>/report/metrics.json``，入参既可以是**实验目录**（含 ``runs/``），
也可以是**数据集目录**（``runs/<dataset>``）；compare 实验目录
``compare_<name>_<platform>_<ts>/`` 落在 ``out_dir`` 之下，含 ``A/``、``B/``
两棵完整 eval 树与 ``reports/``。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from membench import layout
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


def _write_dataset(experiment_dir: Path, dataset: str, metrics: dict) -> str:
    """写 ``<experiment_dir>/runs/<dataset>/report/{metrics.json,report.md}``。

    ``report.md`` 一并写出：compare 会把数据集级报告复制成 A/B 副本的任务级
    ``reports/``（真实产物必有它），只给 metrics.json 不算完整数据集目录。
    返回数据集目录。
    """
    report_dir = Path(layout.dataset_report_dir(str(experiment_dir), dataset))
    report_dir.mkdir(parents=True)
    (report_dir / layout.FILE_METRICS).write_text(
        json.dumps(metrics, ensure_ascii=False), encoding="utf-8"
    )
    (report_dir / layout.FILE_REPORT_MD).write_text(
        f"# 数据集报告 {dataset}\n", encoding="utf-8"
    )
    return str(report_dir.parent)


def _arm(tmp_path: Path, arm: str, metrics: dict, dataset: str = "sample_dataset") -> str:
    """构造一侧的**实验目录**（``runs/<dataset>/report/metrics.json``）。

    compare 的入参是「产物目录」：这里给实验目录形态，另一侧/另一条用例给
    数据集目录形态，两种入参都要能解析。
    """
    _write_dataset(tmp_path / arm, dataset, metrics)
    return str(tmp_path / arm)


def test_comparable_metadata_passes(tmp_path: Path) -> None:
    """① 元数据一致 -> verdict PASS、comparable 为 True，产物落 compare_* 实验目录。"""
    a = _arm(tmp_path, "a", _metrics("run_a"))
    b = _arm(tmp_path, "b", _metrics("run_b"))
    result = compare_runs(a, b, out_dir=str(tmp_path / "out"))

    assert result["verdict"] == "PASS"
    assert result["comparability"]["comparable"] is True
    gate = result["gates"][0]
    assert gate["name"] == "comparability" and gate["status"] == "PASS"

    # 实验目录名：compare_<name>_<platform>_<ts>（name 缺省取两侧 adapter，platform 取两臂平台）
    compare_dir = Path(result["compare_dir"])
    assert compare_dir.parent == tmp_path / "out"
    assert compare_dir.name.startswith("compare_mock-strong_vs_mock-strong_default_")

    reports = compare_dir / layout.DIR_REPORTS
    assert result["report_path"] == str(reports / layout.FILE_COMPARISON_MD)
    for name in (
        layout.FILE_COMPARISON_MD,
        layout.FIG_METRIC_DIFF,
        layout.FIG_CATEGORY_DIFF,
    ):
        assert (reports / name).is_file(), f"缺少 reports/{name}"

    # A/ B/ 各是一个完整 eval 形态：数据集树 + 任务级 reports/
    for arm in ("A", "B"):
        arm_dir = compare_dir / arm
        assert arm_dir.is_dir()
        assert (arm_dir / layout.DIR_RUNS / "sample_dataset" / layout.DIR_REPORT / layout.FILE_METRICS).is_file()
        assert (arm_dir / layout.DIR_REPORTS / layout.FILE_REPORT_MD).is_file()
        assert (arm_dir / layout.DIR_REPORTS / layout.FILE_METRICS).is_file()
    assert result["arm_a"] == str(compare_dir / "A")
    assert result["arm_b"] == str(compare_dir / "B")

    # 「来源与产物」小节回指两侧入参，便于事后定位
    md = (reports / layout.FILE_COMPARISON_MD).read_text(encoding="utf-8")
    assert "来源与产物" in md
    assert a in md and b in md
    assert result["sources"]["A（基线）"]["dataset_dir"] == str(
        Path(a) / layout.DIR_RUNS / "sample_dataset"
    )


def test_judge_model_mismatch_warns(tmp_path: Path) -> None:
    """② 仅 judge_model 不同 -> verdict WARN、可比性门禁 WARN、报告含不可比原因。"""
    a = _arm(tmp_path, "a", _metrics("run_a"))
    b = _arm(tmp_path, "b", _metrics("run_b", judge_model="gpt-4o"))
    result = compare_runs(a, b, out_dir=str(tmp_path / "out"))

    assert result["verdict"] == "WARN"
    gate = result["gates"][0]
    assert gate["name"] == "comparability" and gate["status"] == "WARN"
    assert result["comparability"]["comparable"] is False

    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "可比性前置校验" in report
    assert "不可比原因" in report
    assert "judge_model" in report


def test_strict_dataset_fingerprint_mismatch_fails(tmp_path: Path) -> None:
    """③ strict + dataset 指纹不同（critical）-> verdict FAIL。"""
    a = _arm(tmp_path, "a", _metrics("run_a"))
    b = _arm(
        tmp_path,
        "b",
        _metrics(
            "run_b",
            dataset_fingerprint={"sha256": "bbbb2222", "bytes": 12, "name": "x.jsonl"},
        ),
    )
    result = compare_runs(a, b, out_dir=str(tmp_path / "out"), require_comparable=True)

    assert result["verdict"] == "FAIL"
    assert result["gates"][0]["name"] == "comparability"
    assert result["gates"][0]["status"] == "FAIL"
    assert result["comparability"]["comparable"] is False


def test_run_missing_metadata_is_unverifiable(tmp_path: Path) -> None:
    """④ 缺新元数据的 metrics（旧版本产物）-> unverifiable 非空；
    默认不因缺元数据变 WARN/FAIL，strict 下 FAIL。"""
    legacy = _metrics("run_x")
    for key in ("driver_llm", "judge_model", "adapter_version", "dataset_fingerprint"):
        legacy.pop(key)
    a = _arm(tmp_path, "a", legacy)
    b = _arm(tmp_path, "b", dict(legacy, run_id="run_y"))

    relaxed = compare_runs(a, b, out_dir=str(tmp_path / "relaxed"))
    assert relaxed["comparability"]["unverifiable"]
    assert relaxed["verdict"] == "PASS"
    assert relaxed["gates"][0]["status"] == "PASS"

    strict = compare_runs(a, b, out_dir=str(tmp_path / "strict"), require_comparable=True)
    assert strict["verdict"] == "FAIL"
    assert strict["gates"][0]["status"] == "FAIL"


def test_commit_only_diff_is_comparable(tmp_path: Path) -> None:
    """⑤ 仅 commit 不同 -> 不判不可比（commit 是 A/B 的被测变量）。

    入参用**数据集目录**形态（``runs/<dataset>``），覆盖读取侧对两种入参的解析。
    """
    a = _write_dataset(tmp_path / "exp_a", "sample_dataset", _metrics("run_a", commit="a1b2c3"))
    b = _write_dataset(tmp_path / "exp_b", "sample_dataset", _metrics("run_b", commit="d4e5f6"))
    result = compare_runs(a, b, out_dir=str(tmp_path / "out"))

    assert result["comparability"]["comparable"] is True
    assert result["verdict"] == "PASS"
    commit_check = next(
        c for c in result["comparability"]["checks"] if c["dimension"] == "commit"
    )
    assert commit_check["same"] is False
    # 数据集目录入参同样被正确解析（副本落 <compare>/A/runs/<dataset>/）
    assert (
        Path(result["arm_a"]) / layout.DIR_RUNS / "sample_dataset" / layout.DIR_REPORT / layout.FILE_METRICS
    ).is_file()


def test_warn_dimension_mismatch_only_warns(tmp_path: Path) -> None:
    """次要维度（limit）不一致 -> WARN 门禁但不判不可比（comparable 仍为 True）。"""
    a = _arm(tmp_path, "a", _metrics("run_a"))
    b = _arm(tmp_path, "b", _metrics("run_b", limit=5))
    result = compare_runs(a, b, out_dir=str(tmp_path / "out"))

    assert result["verdict"] == "WARN"
    assert result["comparability"]["comparable"] is True
    assert result["gates"][0]["status"] == "WARN"


def test_retrieval_match_mismatch_only_warns(tmp_path: Path) -> None:
    """检索命中口径不同（substring vs overlap）-> WARN 门禁但不判不可比；
    两侧同缺（旧版 run）视为一致，不影响结论。"""
    a = _arm(tmp_path, "a", _metrics("run_a", retrieval_match="substring"))
    b = _arm(tmp_path, "b", _metrics("run_b", retrieval_match="overlap"))
    result = compare_runs(a, b, out_dir=str(tmp_path / "c1"))

    assert result["verdict"] == "WARN"
    assert result["comparability"]["comparable"] is True
    check = next(
        c for c in result["comparability"]["checks"] if c["dimension"] == "retrieval_match"
    )
    assert check["same"] is False

    a2 = _arm(tmp_path, "a2", _metrics("run_a2"))
    b2 = _arm(tmp_path, "b2", _metrics("run_b2"))
    result2 = compare_runs(a2, b2, out_dir=str(tmp_path / "c2"))
    check2 = next(
        c for c in result2["comparability"]["checks"] if c["dimension"] == "retrieval_match"
    )
    assert check2["same"] is True
    assert result2["verdict"] == "PASS"


def test_rule_based_judge_model_not_applicable(tmp_path: Path) -> None:
    """规则判分器（heuristic）无模型标识：judge_model 记「不适用」，
    strict 模式下也不因它 FAIL（否则 heuristic 的 A/B 会无条件不可比）。"""
    a = _arm(tmp_path, "a", _metrics("run_a", judge="heuristic", judge_model=None))
    b = _arm(tmp_path, "b", _metrics("run_b", judge="heuristic", judge_model=None))
    result = compare_runs(a, b, out_dir=str(tmp_path / "out"), require_comparable=True)

    assert result["comparability"]["unverifiable"] == []
    check = next(
        c for c in result["comparability"]["checks"] if c["dimension"] == "judge_model"
    )
    assert check["same"] is True
    assert result["verdict"] == "PASS"
    assert result["gates"][0]["status"] == "PASS"


def test_missing_driver_llm_is_reported_as_reason(tmp_path: Path) -> None:
    """关键维度缺失（driver_llm）要出现在「不可比原因」清单里，默认模式也不静默。"""
    a = _arm(tmp_path, "a", _metrics("run_a", driver_llm=None))
    b = _arm(tmp_path, "b", _metrics("run_b", driver_llm=None))
    result = compare_runs(a, b, out_dir=str(tmp_path / "out"))

    assert result["comparability"]["unverifiable"]
    assert any("driver_llm" in r for r in result["comparability"]["reasons"])


def test_incomplete_product_rejected_before_copy(tmp_path: Path) -> None:
    """产物缺 report.md（产物不完整）→ 复制 A/B 副本之前拒绝，不留半个实验目录。"""
    a = _arm(tmp_path, "a", _metrics("run_a"))
    b_dir = Path(_arm(tmp_path, "b", _metrics("run_b")))
    dataset_dir = b_dir / layout.DIR_RUNS / "sample_dataset"
    (dataset_dir / layout.DIR_REPORT / layout.FILE_REPORT_MD).unlink()
    out = tmp_path / "out"

    with pytest.raises(SystemExit) as excinfo:
        compare_runs(a, str(b_dir), out_dir=str(out))
    message = str(excinfo.value)
    assert "产物不完整" in message
    assert str(dataset_dir) in message

    # 拒绝在建目录 / 复制之前：既不残留 compare_* 实验目录，也没有 A/B 副本
    assert not out.exists()
