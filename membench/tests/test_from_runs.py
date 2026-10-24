"""汇总模式测试：evaluate --from-runs 与 evaluate-compare --baseline-run/--candidate-run。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from membench import cli, layout
from membench.adapters import MockStrongAdapter, MockWeakAdapter
from membench.data import SAMPLE_DATASET_PATH
from membench.evaluate import DatasetSpec, evaluate, parse_run_kv
from membench.evaluate_compare import evaluate_compare


def _dataset_row(dialogue_id: str, qid: str) -> dict:
    return {
        "dialogue_id": dialogue_id,
        "sessions": [
            [{"role": "user", "content": "记一下：项目截止日是周五。", "timestamp": None}]
        ],
        "questions": [
            {
                "qid": qid,
                "question": "项目截止日是哪天？",
                "gold_answer": "周五",
                "category": "single_hop",
                "evidence": ["项目截止日是周五"],
            }
        ],
    }


def _write_dataset(path: Path, dialogue_id: str, qid: str) -> Path:
    path.write_text(
        json.dumps(_dataset_row(dialogue_id, qid), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _specs(a: Path, b: Path) -> list[DatasetSpec]:
    return [
        DatasetSpec(key="a", name="a", path=str(a), source="user-specified", is_office_slot=True),
        DatasetSpec(key="b", name="b", path=str(b), source="user-specified"),
    ]


def _make_runs(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    """跑一次双数据集 evaluate，返回两数据集路径与数据集产物目录映射。"""
    a = _write_dataset(tmp_path / "a.jsonl", "d_a", "q_a")
    b = _write_dataset(tmp_path / "b.jsonl", "d_b", "q_b")
    report_out = tmp_path / "out1"
    rc = evaluate(
        MockStrongAdapter(),
        "heuristic",
        report_out=str(report_out),
        specs=_specs(a, b),
    )
    assert rc == 0
    eval_dir = next(report_out.glob("eval_*"))
    summary = json.loads(
        (eval_dir / "reports" / "evaluate_summary.json").read_text(encoding="utf-8")
    )
    runs = {k: str(eval_dir / rel) for k, rel in summary["runs"].items()}
    # 数据集产物按数据集分目录（runs/<dataset>），不再是 run_id 平铺目录
    assert set(runs) == {"a", "b"}
    for key, rel in summary["runs"].items():
        assert rel == f"runs/{key}"
        assert (eval_dir / rel / "report" / "metrics.json").is_file()
    return a, b, runs


def test_evaluate_pure_aggregate_from_runs(tmp_path: Path) -> None:
    """纯汇总：只给 from_runs，无 adapter/judge，报告头取已有 run 元数据。"""
    _, _, runs = _make_runs(tmp_path)
    report_out = tmp_path / "out2"
    rc = evaluate(
        None,
        None,
        report_out=str(report_out),
        from_runs={"a": runs["a"], "b": runs["b"]},
    )
    assert rc == 0
    eval_dir = next(report_out.glob("eval_*"))
    summary = json.loads(
        (eval_dir / "reports" / "evaluate_summary.json").read_text(encoding="utf-8")
    )
    assert summary["datasets"] == ["a", "b"]
    assert summary["adapter"] == "mock-strong"
    assert summary["judge"] == "heuristic"
    assert summary["fused"]["accuracy"]["value"] is not None
    assert (eval_dir / "reports" / "report.md").is_file()
    # 复用的产物并入本次实验目录（自包含，可整体搬移）
    assert Path(summary["experiment_dir"]).resolve() == eval_dir.resolve()
    for key, rel in summary["runs"].items():
        assert rel == f"runs/{key}"
        assert (eval_dir / rel / "report" / "metrics.json").is_file()


def test_evaluate_gap_fill_from_runs(tmp_path: Path) -> None:
    """目标清单里有缺口：缺失数据集用 adapter 补跑，已有 run 复用。"""
    a, b, runs = _make_runs(tmp_path)
    rc = evaluate(
        MockStrongAdapter(),
        "heuristic",
        report_out=str(tmp_path / "out3"),
        specs=_specs(a, b),
        from_runs={"a": runs["a"]},
    )
    assert rc == 0
    eval_dir = next((tmp_path / "out3").glob("eval_*"))
    summary = json.loads(
        (eval_dir / "reports" / "evaluate_summary.json").read_text(encoding="utf-8")
    )
    assert summary["datasets"] == ["a", "b"]
    dataset_dirs = [p for p in (eval_dir / "runs").iterdir() if p.is_dir()]
    assert sorted(p.name for p in dataset_dirs) == ["a", "b"]
    for dataset_dir in dataset_dirs:
        assert (dataset_dir / "report" / "metrics.json").is_file()

    # 有缺口且无 adapter → 明确报错
    with pytest.raises(SystemExit, match="补跑需要"):
        evaluate(
            None,
            "heuristic",
            report_out=str(tmp_path / "out4"),
            specs=[DatasetSpec(key="a", name="a", path=str(a), source="user-specified")],
        )


def test_evaluate_from_runs_unknown_key_rejected(tmp_path: Path) -> None:
    """复用 run 的 key 不在目标清单中 → 报错（不静默丢弃）。"""
    _, _, runs = _make_runs(tmp_path)
    with pytest.raises(SystemExit, match="不在目标数据集清单"):
        evaluate(
            None,
            None,
            report_out=str(tmp_path / "out5"),
            specs=[DatasetSpec(key="a", name="a", path="x", source="user-specified")],
            from_runs={"b": runs["b"]},
        )


def test_parse_run_kv_validation(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="KEY=RUN_DIR"):
        parse_run_kv(["noequals"], "--from-runs")
    with pytest.raises(SystemExit, match="重复"):
        parse_run_kv([f"a={tmp_path}", f"a={tmp_path}"], "--from-runs")
    with pytest.raises(SystemExit, match="不存在"):
        parse_run_kv([f"a={tmp_path / 'nope'}"], "--from-runs")
    assert parse_run_kv([f"a={tmp_path}"], "--from-runs") == {"a": str(tmp_path)}


def test_metadata_inconsistency_note(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """混 run 汇总：platform 这类允许差异的维度不拒绝，但要在报告与控制台显著标注。"""
    _, _, runs = _make_runs(tmp_path)
    metrics_path = Path(runs["b"]) / "report" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["platform"] = "other-platform"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")

    rc = evaluate(
        None,
        None,
        report_out=str(tmp_path / "out6"),
        from_runs={"a": runs["a"], "b": runs["b"]},
    )
    assert rc == 0
    assert "platform" in capsys.readouterr().out
    eval_dir = next((tmp_path / "out6").glob("eval_*"))
    md = (eval_dir / "reports" / "report.md").read_text(encoding="utf-8")
    assert "汇总模式" in md and "platform" in md


def test_evaluate_mixed_judge_rejected_before_run(tmp_path: Path) -> None:
    """单臂汇总：judge 混用 → 跑前拒绝，且不创建报告目录。"""
    _, _, runs = _make_runs(tmp_path)
    metrics_path = Path(runs["b"]) / "report" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["judge"] = "llm"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")

    out = tmp_path / "out_judge_mix"
    with pytest.raises(SystemExit, match="混合了多个被测对象"):
        evaluate(
            None,
            None,
            report_out=str(out),
            from_runs={"a": runs["a"], "b": runs["b"]},
        )
    assert not out.exists()


def test_evaluate_gap_fill_metadata_mismatch_rejected_before_run(tmp_path: Path) -> None:
    """补跑参数与已有 run 不一致 → 跑前拒绝（不白跑一轮才发现融合不可归因）。"""
    _, _, runs = _make_runs(tmp_path)
    out = tmp_path / "out_mismatch"
    with pytest.raises(SystemExit, match="补跑参数与已有 run 不一致"):
        evaluate(
            MockStrongAdapter(),
            "heuristic",
            report_out=str(out),
            specs=[
                DatasetSpec(key="a", name="a", path=str(tmp_path / "a.jsonl"), source="user-specified"),
                DatasetSpec(key="b", name="b", path=str(tmp_path / "b.jsonl"), source="user-specified"),
                DatasetSpec(key="c", name="c", path=str(tmp_path / "a.jsonl"), source="user-specified"),
            ],
            from_runs={"a": runs["a"], "b": runs["b"]},
            adapter_version="v2",
        )
    assert not out.exists()


def test_evaluate_fresh_run_missing_verifiable_meta_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """全新建评测缺 driver_llm/adapter_version → 启动即提示严格可比性会判无法核验。"""
    a = _write_dataset(tmp_path / "a.jsonl", "d_a", "q_a")
    b = _write_dataset(tmp_path / "b.jsonl", "d_b", "q_b")
    rc = evaluate(
        MockStrongAdapter(),
        "heuristic",
        report_out=str(tmp_path / "out_hint"),
        specs=_specs(a, b),
    )
    assert rc == 0
    assert "无法核验" in capsys.readouterr().out


def _make_compare(tmp_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """跑一次双数据集 evaluate-compare，返回两侧 run 目录映射。"""
    a = _write_dataset(tmp_path / "a.jsonl", "d_a", "q_a")
    b = _write_dataset(tmp_path / "b.jsonl", "d_b", "q_b")
    result = evaluate_compare(
        MockWeakAdapter(),
        MockStrongAdapter(),
        "heuristic",
        report_out=str(tmp_path / "ab1"),
        tag="t1",
        specs=_specs(a, b),
    )
    ab_dir = Path(result["compare_dir"])
    # A/B 两侧各是一个完整 eval 实验形态：产物在 <compare>/<arm>/runs/<dataset>/
    baseline = {k: str(ab_dir / "A" / "runs" / k) for k in ("a", "b")}
    candidate = {k: str(ab_dir / "B" / "runs" / k) for k in ("a", "b")}
    for target in (*baseline.values(), *candidate.values()):
        assert (Path(target) / "report" / "metrics.json").is_file()
    return baseline, candidate


def test_evaluate_compare_pure_aggregate(tmp_path: Path) -> None:
    """两侧全复用：无 adapter/judge，重建完整对比报告。"""
    baseline, candidate = _make_compare(tmp_path)
    result = evaluate_compare(
        None,
        None,
        None,
        report_out=str(tmp_path / "ab2"),
        tag="t2",
        baseline_runs=baseline,
        candidate_runs=candidate,
    )
    ab_dir = Path(result["compare_dir"])
    md = (ab_dir / "reports" / "comparison.md").read_text(encoding="utf-8")
    assert "两组多数据集对比报告" in md
    assert "mock-weak vs mock-strong" in md
    assert result["verdict"] in ("PASS", "WARN", "FAIL")
    # 对比实验目录的形态：A/ 与 B/ 各含 runs/ + 任务级 reports/，对比报告落根层 reports/
    for arm in ("A", "B"):
        assert (ab_dir / arm / "reports" / "report.md").is_file()
        for key in ("a", "b"):
            assert (ab_dir / arm / "runs" / key / "report" / "metrics.json").is_file()
    assert (ab_dir / "reports" / "fig_per_dataset_diff.png").is_file()
    assert (ab_dir / "reports" / "fig_fused_diff.png").is_file()
    assert "](fig_per_dataset_diff.png)" in md and "](fig_fused_diff.png)" in md


def test_evaluate_compare_gap_fill_one_side(tmp_path: Path) -> None:
    """候选侧只复用一半：缺失数据集用候选 adapter 补跑。"""
    baseline, candidate = _make_compare(tmp_path)
    result = evaluate_compare(
        None,
        MockStrongAdapter(),
        "heuristic",
        report_out=str(tmp_path / "ab3"),
        tag="t3",
        baseline_runs=baseline,
        candidate_runs={"a": candidate["a"]},
    )
    ab_dir = Path(result["compare_dir"])
    cand_dirs = [p for p in (ab_dir / "B" / "runs").iterdir() if p.is_dir()]
    assert sorted(p.name for p in cand_dirs) == ["a", "b"]
    for dataset_dir in cand_dirs:
        assert (dataset_dir / "report" / "metrics.json").is_file()
    assert (ab_dir / "reports" / "comparison.md").is_file()


def test_cli_from_runs_and_conflicts(tmp_path: Path) -> None:
    _, _, runs = _make_runs(tmp_path)
    rc = cli.main(
        [
            "evaluate",
            "--from-runs", f"a={runs['a']}",
            "--from-runs", f"b={runs['b']}",
            "--report-out", str(tmp_path / "cli1"),
        ]
    )
    assert rc == 0
    assert list((tmp_path / "cli1").glob("eval_*"))

    with pytest.raises(SystemExit, match="互斥"):
        cli.main(["evaluate", "--from-runs", f"a={runs['a']}", "--quick"])

    with pytest.raises(SystemExit, match="KEY=RUN_DIR"):
        cli.main(["evaluate", "--from-runs", "badformat"])


def test_sharded_ab_official_path(tmp_path: Path) -> None:
    """分片 A/B 官方通路：两侧各自 分片 run → fuse → evaluate-compare 汇总出对比报告。"""
    from membench.fuse import fuse_runs
    from membench.runner import Runner

    def _fused_side(side: str, adapter, *, dataset_dir_only: bool = False) -> str:  # noqa: ANN001
        shard_dirs = []
        for i in (0, 1):
            shard_dirs.append(
                Runner(
                    adapter,
                    str(SAMPLE_DATASET_PATH),
                    judge="heuristic",
                    out_dir=str(tmp_path / f"{side}{i}"),
                    shards=2,
                    shard_index=i,
                ).run()
            )
        fused = fuse_runs(shard_dirs, str(tmp_path / f"{side}_fused"))
        # 融合产物是完整实验目录；两侧分别以实验目录 / 数据集目录入参，
        # 覆盖读取侧对两种「产物目录」的解析（整树复制 vs 逐个数据集并入）。
        return fused["dataset_dir"] if dataset_dir_only else fused["out_dir"]

    baseline = _fused_side("a", MockWeakAdapter())
    candidate = _fused_side("b", MockStrongAdapter(), dataset_dir_only=True)

    result = evaluate_compare(
        None,
        None,
        None,
        report_out=str(tmp_path / "ab_sharded"),
        tag="sharded",
        baseline_runs={"sample": baseline},
        candidate_runs={"sample": candidate},
    )
    md = (Path(result["compare_dir"]) / "reports" / "comparison.md").read_text(
        encoding="utf-8"
    )
    assert "两组多数据集对比报告" in md
    assert "mock-weak vs mock-strong" in md
    assert result["verdict"] in ("PASS", "WARN", "FAIL")
    # 融合产物的数据集突进两侧各自的 runs/<dataset>/ 下，逐题明细随之带过来
    dataset_name = Path(SAMPLE_DATASET_PATH).stem
    for arm in ("A", "B"):
        dataset_dir = Path(result["compare_dir"]) / arm / "runs" / dataset_name
        assert (dataset_dir / "report" / "metrics.json").is_file()
        assert (dataset_dir / "report" / "results.jsonl").is_file()
        assert (Path(result["compare_dir"]) / arm / "reports" / "report.md").is_file()


def test_evaluate_compare_side_identity_unique(tmp_path: Path) -> None:
    """同一侧混入多个被测对象（adapter 不同）→ 直接拒绝（A/B 差异只能发生在两侧之间）。"""
    from membench.runner import Runner

    baseline, candidate = _make_compare(tmp_path)
    a = _write_dataset(tmp_path / "a.jsonl", "d_a", "q_a")
    other = Runner(
        MockStrongAdapter(), str(a), judge="heuristic", out_dir=str(tmp_path / "other")
    ).run()
    with pytest.raises(SystemExit, match="混合了多个被测对象身份"):
        evaluate_compare(
            None,
            None,
            None,
            report_out=str(tmp_path / "ab_mix"),
            tag="mix",
            baseline_runs={"a": other, "b": baseline["b"]},
            candidate_runs=candidate,
        )


def test_evaluate_mixed_adapter_version_rejected(tmp_path: Path) -> None:
    """单臂汇总：adapter_version 不一致 → 直接拒绝。"""
    from membench.runner import Runner

    a = _write_dataset(tmp_path / "a.jsonl", "d_a", "q_a")
    b = _write_dataset(tmp_path / "b.jsonl", "d_b", "q_b")
    ra = Runner(
        MockStrongAdapter(),
        str(a),
        judge="heuristic",
        out_dir=str(tmp_path / "ra"),
        adapter_version="v1",
    ).run()
    rb = Runner(
        MockStrongAdapter(),
        str(b),
        judge="heuristic",
        out_dir=str(tmp_path / "rb"),
        adapter_version="v2",
    ).run()
    with pytest.raises(SystemExit, match="混合了多个被测对象身份"):
        evaluate(
            None,
            None,
            report_out=str(tmp_path / "out_mix"),
            from_runs={"a": ra, "b": rb},
        )


def test_evaluate_compare_gap_fill_metadata_mismatch_rejected_before_run(tmp_path: Path) -> None:
    """对比侧补跑参数与已有 run 不一致 → 跑前拒绝，不产出对比报告。"""
    baseline, candidate = _make_compare(tmp_path)
    out = tmp_path / "ab_mismatch"
    with pytest.raises(SystemExit, match="补跑参数与已有 run 不一致"):
        evaluate_compare(
            MockWeakAdapter(),
            None,
            "heuristic",
            report_out=str(out),
            tag="mismatch",
            baseline_runs={"a": baseline["a"]},
            candidate_runs=candidate,
            adapter_version_baseline="v2",
        )
    # 跑前拒绝：连实验目录都不建，更不写任何对比报告
    assert not out.exists()


def test_evaluate_compare_cross_side_instrument_diff_flagged(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """跨侧仪器维度（platform / 检索口径）不一致：跑前告警 + 报告「两侧可比性」逐项可查。"""
    baseline, candidate = _make_compare(tmp_path)
    capsys.readouterr()  # 丢弃 _make_compare 内部运行的输出，只断言本次调用
    metrics_path = Path(candidate["a"]) / "report" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["platform"] = "fs"
    metrics["retrieval_match"] = "llm"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")

    result = evaluate_compare(
        None,
        None,
        None,
        report_out=str(tmp_path / "ab_inst"),
        tag="inst",
        baseline_runs={"a": baseline["a"]},
        candidate_runs={"a": candidate["a"]},
    )
    out = capsys.readouterr().out
    assert "两侧可比性问题（两侧均复用产物，跑前检查）" in out
    assert "platform: A=" in out and "B=fs" in out
    assert "[次要] retrieval_match: A=substring != B=llm" in out
    md = (Path(result["compare_dir"]) / "reports" / "comparison.md").read_text(
        encoding="utf-8"
    )
    assert "## 两侧可比性（逐数据集）" in md
    assert "| 平台 | 关键 |" in md and "| 检索匹配口径 | 次要 |" in md
    # 两侧都未标注 driver_llm 记「关键缺失」，与「关键不一致」分开计数
    assert "| 驱动 LLM | 关键缺失 |" in md or "| driver_llm | 关键缺失 |" in md
    assert "两侧可比性：关键不一致 1 项、关键缺失 1 项、次要 1 项" in md


def test_evaluate_compare_cross_side_ignores_tested_object_dims(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """跨侧被测对象不同（adapter / adapter_version）属设计变量，不作为可比性问题。"""
    baseline, candidate = _make_compare(tmp_path)
    capsys.readouterr()  # 丢弃 _make_compare 内部运行的输出，只断言本次调用
    for run_map in (baseline, candidate):
        for run_dir in run_map.values():
            metrics_path = Path(run_dir) / "report" / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["driver_llm"] = "gpt-4o"  # 仪器维度对齐后不应再有任何可比性问题
            metrics_path.write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")
    result = evaluate_compare(
        None,
        None,
        None,
        report_out=str(tmp_path / "ab_tested"),
        tag="tested",
        baseline_runs={"a": baseline["a"], "b": baseline["b"]},
        candidate_runs={"a": candidate["a"], "b": candidate["b"]},
    )
    out = capsys.readouterr().out
    assert "两侧可比性问题" not in out
    md = (Path(result["compare_dir"]) / "reports" / "comparison.md").read_text(
        encoding="utf-8"
    )
    assert "（无：各数据集两侧关键与次要维度均一致且可核验）" in md
    assert "两侧可比性：各数据集关键维度一致且可核验" in md


def test_from_runs_rejects_key_product_mismatch(tmp_path: Path) -> None:
    """KEY 与实际产物身份不符 -> 直接拒绝，不把别的数据集的分数记到该 key 名下。

    回归：``resolve_dataset_dir`` 的提示不命中回退（保留）让"key 认错"静默通过——
    此前 ``--from-runs locomo=<只含 sample_dataset 的实验目录>`` 会把 sample_dataset
    当 locomo 记账，退出码 0。
    """
    _, _, runs = _make_runs(tmp_path)  # key / 目录名都是 a / b
    out = tmp_path / "out_bad_key"

    with pytest.raises(SystemExit) as excinfo:
        evaluate(None, None, report_out=str(out), from_runs={"locomo": runs["a"]})
    message = str(excinfo.value)
    assert "locomo" in message
    assert "实际解析到 a" in message
    # 认错的 key 在开跑前就被拦下：不建实验目录，更不留空目录
    assert not out.exists()


def test_from_runs_accepts_key_dir_name_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``office``(key) ↔ ``office_dataset``(目录名) 的映射继续可用（不直接比字符串）。

    同一份 office 产物的 stale 门禁在纯汇总与 ``--dataset office`` 两条路径上结论一致
    （回归：纯汇总此前不标办公域槽位 -> 门禁 SKIP、退出码 0）。
    """
    from membench.runner import Runner
    from membench.userdata import builtin_office_dataset_path

    monkeypatch.setenv("MEMBENCH_USER_DATA_DIR", str(tmp_path / "user_dataset"))
    exp = tmp_path / "office_exp"
    Runner(
        MockWeakAdapter(),
        str(builtin_office_dataset_path()),
        judge="heuristic",
        out_dir=str(exp),
    ).run()
    assert (exp / layout.DIR_RUNS / "office_dataset").is_dir()

    rc_pure = cli.main(
        ["evaluate", "--from-runs", f"office={exp}", "--report-out", str(tmp_path / "pure")]
    )
    rc_explicit = cli.main(
        [
            "evaluate",
            "--dataset",
            "office",
            "--from-runs",
            f"office={exp}",
            "--report-out",
            str(tmp_path / "explicit"),
        ]
    )
    # mock-weak 在 office 上 stale_rate=0.1 > 阈值 0.03：两条路径都必须 FAIL（退出码 1）
    assert rc_pure == 1 and rc_explicit == 1
    for out in (tmp_path / "pure", tmp_path / "explicit"):
        eval_dir = next(out.glob("eval_*"))
        summary = json.loads(
            (eval_dir / "reports" / "evaluate_summary.json").read_text(encoding="utf-8")
        )
        assert summary["gate"]["status"] == "FAIL"


def test_from_runs_unknown_key_warns_and_keeps_fallback(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """自定义 key（不在数据集注册表里）无法核对身份：保留回退，但显著警告实际采用目录。"""
    _, _, runs = _make_runs(tmp_path)
    evaluate(
        None,
        None,
        report_out=str(tmp_path / "out_custom"),
        from_runs={"mykey": runs["a"]},
    )
    out = capsys.readouterr().out
    assert "mykey" in out and "⚠️" in out

    # key 与目录名一致时不该有任何警告
    evaluate(None, None, report_out=str(tmp_path / "out_ok"), from_runs={"a": runs["a"]})
    assert "⚠️ 汇总模式" not in capsys.readouterr().out
