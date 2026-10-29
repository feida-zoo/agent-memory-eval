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


def test_n_questions_mismatch_is_critical(tmp_path: Path) -> None:
    """两侧题目数不同 -> 该维度不一致（关键）：分数算在不同题集上，不可比。

    回归：``n_questions`` 由 ``aggregate`` 写在 ``overall`` 下，此前只读顶层
    metrics，该维度恒为 ``None == None``（死维度）——``--limit`` 冒烟产物与全量
    产物因此被当成可比。
    """
    overall_a = dict(_metrics()["overall"])
    overall_b = {**overall_a, "n_questions": 6}
    a = _arm(tmp_path, "a", _metrics("run_a", overall=overall_a))
    b = _arm(tmp_path, "b", _metrics("run_b", overall=overall_b))

    result = compare_runs(a, b, out_dir=str(tmp_path / "out"))
    check = next(
        c for c in result["comparability"]["checks"] if c["dimension"] == "n_questions"
    )
    # 取值确实来自 overall（顶层没有 n_questions）
    assert check["a"] == 24 and check["b"] == 6
    assert check["same"] is False
    assert check["severity"] == "critical"
    assert result["comparability"]["comparable"] is False
    assert any("n_questions" in reason for reason in result["comparability"]["reasons"])
    assert result["verdict"] == "WARN"

    strict = compare_runs(
        a, b, out_dir=str(tmp_path / "strict"), require_comparable=True
    )
    assert strict["verdict"] == "FAIL"
    assert strict["gates"][0]["name"] == "comparability"
    assert strict["gates"][0]["status"] == "FAIL"


# ---------------------------------------------------------------------------
# 就地模式：一次 A/B 就是一个任务目录，两臂已在 task_dir/A|B
# ---------------------------------------------------------------------------


def _tree_snapshot(root: Path) -> dict[str, str]:
    """目录内容快照（相对路径 -> 文件内容），用来断言这棵树没被复制 / 改写。"""
    return {
        str(path.relative_to(root)): path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _two_arm_task(tmp_path: Path, metrics_b: dict | None = None) -> tuple[Path, Path]:
    """建一个双臂任务目录并在两臂各写一份完整数据集产物，返回 (任务目录, 父目录)。"""
    base = tmp_path / "out"
    task = Path(layout.init_task_dir(str(base), "ab", platform="t"))
    _write_dataset(task / layout.DIR_ARM_A, "sample_dataset", _metrics("run_a"))
    _write_dataset(task / layout.DIR_ARM_B, "sample_dataset", metrics_b or _metrics("run_b"))
    return task, base


def test_in_place_compare_writes_report_into_task_dir(tmp_path: Path) -> None:
    """就地对比：两臂缺省读 task_dir/A|B，报告落 task_dir/reports/，不新建目录、不复制两臂。"""
    task, base = _two_arm_task(tmp_path)
    # 两臂各留一个哨兵文件：就地模式不得复制 / 重建这两棵树
    (task / layout.DIR_ARM_A / "keep.txt").write_text("哨兵", encoding="utf-8")
    before_a = _tree_snapshot(task / layout.DIR_ARM_A)
    before_b = _tree_snapshot(task / layout.DIR_ARM_B)

    result = compare_runs(task_dir=str(task))

    assert result["verdict"] == "PASS"
    assert result["compare_dir"] == str(task)
    assert result["report_path"] == str(task / layout.DIR_REPORTS / layout.FILE_COMPARISON_MD)
    assert result["arm_a"] == str(task / layout.DIR_ARM_A)
    assert result["arm_b"] == str(task / layout.DIR_ARM_B)
    for name in (layout.FILE_COMPARISON_MD, layout.FIG_METRIC_DIFF, layout.FIG_CATEGORY_DIFF):
        assert (task / layout.DIR_REPORTS / name).is_file(), f"缺少 reports/{name}"
    md = result["report_path"]
    assert "**门禁结论： PASS" in Path(md).read_text(encoding="utf-8")
    # 「来源与产物」小节回指两臂（就地模式下"副本"就是 A/ 与 B/ 本身）
    assert str(task / layout.DIR_ARM_A) in Path(md).read_text(encoding="utf-8")

    # 只有一个任务目录、没有复制两臂：两臂内容逐字节不变
    assert [p.name for p in base.iterdir()] == [task.name]
    assert _tree_snapshot(task / layout.DIR_ARM_A) == before_a
    assert _tree_snapshot(task / layout.DIR_ARM_B) == before_b


def test_in_place_compare_accepts_explicit_arm_dirs(tmp_path: Path) -> None:
    """就地模式也接受显式两臂（CLI 用 --baseline/--candidate 覆盖缺省），输出仍落任务目录。"""
    task, base = _two_arm_task(tmp_path)

    result = compare_runs(
        str(task / layout.DIR_ARM_A), str(task / layout.DIR_ARM_B), task_dir=str(task)
    )

    assert result["verdict"] == "PASS"
    assert result["compare_dir"] == str(task)
    assert (task / layout.DIR_REPORTS / layout.FILE_COMPARISON_MD).is_file()
    assert [p.name for p in base.iterdir()] == [task.name]


def test_in_place_compare_reports_regression_verdict(tmp_path: Path) -> None:
    """就地模式的回归门禁照样生效：B 的 accuracy 掉 > 阈值 -> FAIL，报告同步。"""
    b_metrics = _metrics("run_b")
    b_metrics["overall"] = {**_metrics()["overall"], "accuracy": 0.8}
    task, _base = _two_arm_task(tmp_path, metrics_b=b_metrics)

    result = compare_runs(task_dir=str(task))

    assert result["verdict"] == "FAIL"
    acc_gate = next(g for g in result["gates"] if g["name"] == "accuracy_regression")
    assert acc_gate["status"] == "FAIL"
    md = (task / layout.DIR_REPORTS / layout.FILE_COMPARISON_MD).read_text(encoding="utf-8")
    assert "**门禁结论： FAIL" in md


def test_in_place_compare_rejects_incomplete_arm(tmp_path: Path) -> None:
    """某一臂产物不完整 -> 当场拒绝，且不写 reports/comparison.md（就地也不留半成品报告）。"""
    task = Path(layout.init_task_dir(str(tmp_path / "out"), "ab", platform="t"))
    _write_dataset(task / layout.DIR_ARM_A, "sample_dataset", _metrics("run_a"))
    dataset_b = Path(
        _write_dataset(task / layout.DIR_ARM_B, "sample_dataset", _metrics("run_b"))
    )
    (dataset_b / layout.DIR_REPORT / layout.FILE_REPORT_MD).unlink()

    with pytest.raises(SystemExit) as excinfo:
        compare_runs(task_dir=str(task))

    assert "产物不完整" in str(excinfo.value)
    assert str(dataset_b) in str(excinfo.value)
    assert not (task / layout.DIR_REPORTS / layout.FILE_COMPARISON_MD).exists()


def test_in_place_compare_requires_existing_task_dir(tmp_path: Path) -> None:
    """task_dir 不存在 / 与 out_dir 同时给 -> 报错（不建目录、不猜路径）。"""
    with pytest.raises(SystemExit, match="任务目录不存在"):
        compare_runs(task_dir=str(tmp_path / "nope"))
    task, _base = _two_arm_task(tmp_path)
    with pytest.raises(SystemExit, match="互斥"):
        compare_runs(task_dir=str(task), out_dir=str(tmp_path / "elsewhere"))


# ---------------------------------------------------------------------------
# 摄入策略（ingest_strategy）是 A/B 的关键维度
# ---------------------------------------------------------------------------


def _ingest_strategy_check(result: dict) -> dict:
    return next(
        c for c in result["checks"] if c["dimension"] == "ingest_strategy"
    )


def test_ingest_strategy_is_critical_dimension() -> None:
    """sequence vs interleaved：问的时间语义不同，分数不可直接比。"""
    from membench.compare import check_comparability

    result = check_comparability(
        _metrics(ingest_strategy="sequence"),
        _metrics(ingest_strategy="interleaved"),
    )
    check = _ingest_strategy_check(result)
    assert check["severity"] == "critical"
    assert check["same"] is False
    assert result["comparable"] is False
    assert any("ingest_strategy" in reason for reason in result["reasons"])


def test_ingest_strategy_missing_normalizes_to_sequence() -> None:
    """旧产物没有该字段：归一为 sequence，不误伤历史 A/B。"""
    from membench.compare import check_comparability

    result = check_comparability(_metrics(), _metrics())
    check = _ingest_strategy_check(result)
    assert check["same"] is True
    assert check["a"] == check["b"] == "sequence"
    assert result["comparable"] is True


def test_ingest_strategy_mixed_is_incomparable() -> None:
    """分片/拼接产物记 mixed：与单一策略同样不可比。"""
    from membench.compare import check_comparability

    result = check_comparability(
        _metrics(ingest_strategy="interleaved"),
        _metrics(ingest_strategy="mixed"),
    )
    assert _ingest_strategy_check(result)["same"] is False
    assert result["comparable"] is False
