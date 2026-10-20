"""汇总模式测试：evaluate --from-runs 与 evaluate-compare --baseline-run/--candidate-run。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from membench import cli
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
    """跑一次双数据集 evaluate，返回两数据集路径与 run 目录映射。"""
    a = _write_dataset(tmp_path / "a.jsonl", "d_a", "q_a")
    b = _write_dataset(tmp_path / "b.jsonl", "d_b", "q_b")
    rc = evaluate(
        MockStrongAdapter(),
        "heuristic",
        out_dir=str(tmp_path / "runs1"),
        report_out=str(tmp_path / "out1"),
        specs=_specs(a, b),
    )
    assert rc == 0
    eval_dir = next((tmp_path / "out1").glob("eval_*"))
    summary = json.loads((eval_dir / "evaluate_summary.json").read_text(encoding="utf-8"))
    runs = {k: str(eval_dir / rel) for k, rel in summary["runs"].items()}
    return a, b, runs


def test_evaluate_pure_aggregate_from_runs(tmp_path: Path) -> None:
    """纯汇总：只给 from_runs，无 adapter/judge，报告头取已有 run 元数据。"""
    _, _, runs = _make_runs(tmp_path)
    rc = evaluate(
        None,
        None,
        out_dir=str(tmp_path / "runs2"),
        report_out=str(tmp_path / "out2"),
        from_runs={"a": runs["a"], "b": runs["b"]},
    )
    assert rc == 0
    eval_dir = next((tmp_path / "out2").glob("eval_*"))
    summary = json.loads((eval_dir / "evaluate_summary.json").read_text(encoding="utf-8"))
    assert summary["datasets"] == ["a", "b"]
    assert summary["adapter"] == "mock-strong"
    assert summary["judge"] == "heuristic"
    assert summary["fused"]["accuracy"]["value"] is not None
    assert (eval_dir / "evaluate_report.md").is_file()


def test_evaluate_gap_fill_from_runs(tmp_path: Path) -> None:
    """目标清单里有缺口：缺失数据集用 adapter 补跑，已有 run 复用。"""
    a, b, runs = _make_runs(tmp_path)
    rc = evaluate(
        MockStrongAdapter(),
        "heuristic",
        out_dir=str(tmp_path / "runs3"),
        report_out=str(tmp_path / "out3"),
        specs=_specs(a, b),
        from_runs={"a": runs["a"]},
    )
    assert rc == 0
    eval_dir = next((tmp_path / "out3").glob("eval_*"))
    summary = json.loads((eval_dir / "evaluate_summary.json").read_text(encoding="utf-8"))
    assert summary["datasets"] == ["a", "b"]
    copies = [p for p in (eval_dir / "runs").iterdir() if p.is_dir()]
    assert len(copies) == 2

    # 有缺口且无 adapter → 明确报错
    with pytest.raises(SystemExit, match="补跑需要"):
        evaluate(
            None,
            "heuristic",
            out_dir=str(tmp_path / "runs4"),
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
            out_dir=str(tmp_path / "runs5"),
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


def test_metadata_inconsistency_note(tmp_path: Path) -> None:
    """混 run 汇总：跨数据集元数据不一致要在报告与控制台显著标注。"""
    _, _, runs = _make_runs(tmp_path)
    metrics_path = Path(runs["b"]) / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["judge"] = "llm"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")

    rc = evaluate(
        None,
        None,
        out_dir=str(tmp_path / "runs6"),
        report_out=str(tmp_path / "out6"),
        from_runs={"a": runs["a"], "b": runs["b"]},
    )
    assert rc == 0
    eval_dir = next((tmp_path / "out6").glob("eval_*"))
    md = (eval_dir / "evaluate_report.md").read_text(encoding="utf-8")
    assert "汇总模式" in md and "judge" in md


def _make_compare(tmp_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """跑一次双数据集 evaluate-compare，返回两侧 run 目录映射。"""
    a = _write_dataset(tmp_path / "a.jsonl", "d_a", "q_a")
    b = _write_dataset(tmp_path / "b.jsonl", "d_b", "q_b")
    result = evaluate_compare(
        MockWeakAdapter(),
        MockStrongAdapter(),
        "heuristic",
        out_dir=str(tmp_path / "runs_ab1"),
        report_out=str(tmp_path / "ab1"),
        tag="t1",
        specs=_specs(a, b),
    )
    ab_dir = Path(result["compare_dir"])
    baseline = {k: str(ab_dir / "A" / "runs" / f"A_{k}") for k in ("a", "b")}
    candidate = {k: str(ab_dir / "B" / "runs" / f"B_{k}") for k in ("a", "b")}
    return baseline, candidate


def test_evaluate_compare_pure_aggregate(tmp_path: Path) -> None:
    """两侧全复用：无 adapter/judge，重建完整对比报告。"""
    baseline, candidate = _make_compare(tmp_path)
    result = evaluate_compare(
        None,
        None,
        None,
        out_dir=str(tmp_path / "runs_ab2"),
        report_out=str(tmp_path / "ab2"),
        tag="t2",
        baseline_runs=baseline,
        candidate_runs=candidate,
    )
    ab_dir = Path(result["compare_dir"])
    md = (ab_dir / "compare_report.md").read_text(encoding="utf-8")
    assert "两组多数据集对比报告" in md
    assert "mock-weak vs mock-strong" in md
    assert result["verdict"] in ("PASS", "WARN", "FAIL")


def test_evaluate_compare_gap_fill_one_side(tmp_path: Path) -> None:
    """候选侧只复用一半：缺失数据集用候选 adapter 补跑。"""
    baseline, candidate = _make_compare(tmp_path)
    result = evaluate_compare(
        None,
        MockStrongAdapter(),
        "heuristic",
        out_dir=str(tmp_path / "runs_ab3"),
        report_out=str(tmp_path / "ab3"),
        tag="t3",
        baseline_runs=baseline,
        candidate_runs={"a": candidate["a"]},
    )
    ab_dir = Path(result["compare_dir"])
    cand_runs = [p for p in (ab_dir / "B" / "runs").iterdir() if p.is_dir()]
    assert len(cand_runs) == 2
    assert (ab_dir / "compare_report.md").is_file()


def test_cli_from_runs_and_conflicts(tmp_path: Path) -> None:
    _, _, runs = _make_runs(tmp_path)
    rc = cli.main(
        [
            "evaluate",
            "--from-runs", f"a={runs['a']}",
            "--from-runs", f"b={runs['b']}",
            "--out", str(tmp_path / "runs_cli"),
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

    def _fused_side(side: str, adapter) -> str:
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
        return fuse_runs(shard_dirs, str(tmp_path / f"{side}_fused"))["out_dir"]

    baseline = _fused_side("a", MockWeakAdapter())
    candidate = _fused_side("b", MockStrongAdapter())

    result = evaluate_compare(
        None,
        None,
        None,
        report_out=str(tmp_path / "ab_sharded"),
        tag="sharded",
        baseline_runs={"sample": baseline},
        candidate_runs={"sample": candidate},
    )
    md = (Path(result["compare_dir"]) / "compare_report.md").read_text(encoding="utf-8")
    assert "两组多数据集对比报告" in md
    assert "mock-weak vs mock-strong" in md
    assert result["verdict"] in ("PASS", "WARN", "FAIL")
