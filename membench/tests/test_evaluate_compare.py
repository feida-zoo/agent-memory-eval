"""evaluate-compare A/B 两组多数据集融合对比测试。

覆盖：
- 两组融合 diff 计算；
- 逐数据集 diff 表 + 融合 diff 表 + 门禁结论写入 ``reports/comparison.md``；
- 产物布局：``compare_<name>_<platform>_<ts>/{A,B}/`` 各为完整 eval 实验形态
  （``runs/<dataset>/`` + ``reports/``），对比报告与两张对比图落 ``reports/``；
- CLI：_cmd_evaluate_compare 参数透传 + 系统退出码映射（``--out`` 已删除）；
- 缺失融合指标时的兜底（未定义 / 跳过）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from membench import cli, layout
from membench.adapters.base import BaseMemoryAdapter, IngestStats, QueryResult
from membench.evaluate import DatasetResult, DatasetSpec, fuse_metrics
from membench.evaluate_compare import (
    FIG_FUSED_DIFF,
    FIG_PER_DATASET_DIFF,
    ArmDatasetResult,
    _fused_diff,
    _fused_gates,
    _per_dataset_diff,
    _side_fuse,
    evaluate_compare,
)


REFUSAL = "我不知道，相关信息不在我的记忆中。"


class ScriptedAdapter(BaseMemoryAdapter):
    """脚本化适配器：按问题文本返回固定 (answer, retrieved)。"""

    name = "scripted"

    def __init__(
        self,
        script: dict[str, tuple[str, list[str]]],
        name: str = "scripted",
    ) -> None:
        self._script = script
        self.name = name

    def reset(self) -> None:
        pass

    def ingest(self, messages: list[dict]) -> IngestStats:
        return IngestStats()

    def query(self, question: str) -> QueryResult:
        answer, retrieved = self._script[question]
        return QueryResult(answer=answer, retrieved=list(retrieved))


def _write_dataset(path: Path, dialogues: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as f:
        for d in dialogues:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return path


def _session() -> list[dict]:
    return [{"role": "user", "content": "随便聊聊。", "timestamp": None}]


@pytest.fixture()
def mini_datasets(tmp_path: Path) -> dict[str, Path]:
    """A（office 槽位，含 evidence/stale/abstention）、B（普通）、C（普通）。"""
    a = _write_dataset(
        tmp_path / "a.jsonl",
        [
            {
                "dialogue_id": "a_d1",
                "sessions": [_session()],
                "questions": [
                    {
                        "qid": "a_q1",
                        "question": "项目截止日是哪天？",
                        "gold_answer": "周五",
                        "category": "single_hop",
                        "evidence": ["项目截止日是周五"],
                    },
                    {
                        "qid": "a_q2",
                        "question": "现在项目截止日是哪天？",
                        "gold_answer": "周五",
                        "category": "knowledge_update",
                        "evidence": ["截止日已改为周五"],
                        "gold_stale": "周三",
                    },
                    {
                        "qid": "a_q3",
                        "question": "老板私人电话？",
                        "gold_answer": "<ABSTAIN>",
                        "category": "abstention",
                        "evidence": [],
                    },
                ],
            }
        ],
    )
    b = _write_dataset(
        tmp_path / "b.jsonl",
        [
            {
                "dialogue_id": "b_d1",
                "sessions": [_session()],
                "questions": [
                    {
                        "qid": "b_q1",
                        "question": "封面颜色？",
                        "gold_answer": "蓝色",
                        "category": "single_hop",
                        "evidence": [],
                    },
                    {
                        "qid": "b_q2",
                        "question": "logo 颜色？",
                        "gold_answer": "绿色",
                        "category": "single_hop",
                        "evidence": [],
                    },
                ],
            }
        ],
    )
    c = _write_dataset(
        tmp_path / "c.jsonl",
        [
            {
                "dialogue_id": "c_d1",
                "sessions": [_session()],
                "questions": [
                    {
                        "qid": "c_q1",
                        "question": "服务器 root 密码？",
                        "gold_answer": "<ABSTAIN>",
                        "category": "abstention",
                        "evidence": [],
                    },
                    {
                        "qid": "c_q2",
                        "question": "下午茶水果？",
                        "gold_answer": "苹果",
                        "category": "single_hop",
                        "evidence": [],
                    },
                ],
            }
        ],
    )
    return {"A": a, "B": b, "C": c}


@pytest.fixture()
def scripts() -> dict[str, dict[str, tuple[str, list[str]]]]:
    """baseline / candidate 两个 adapter 的脚本；candidate 在 A/B 上整体优于 baseline。"""
    base = {
        # A: q1 对、q2 答旧值（stale 且错）、q3 正确拒答
        "项目截止日是哪天？": ("周五", ["项目截止日是周五"]),
        "现在项目截止日是哪天？": ("截止日是周三", ["截止日已改为周五"]),
        "老板私人电话？": (REFUSAL, []),
        # B: 一对一错
        "封面颜色？": ("蓝色", ["今天的会议纪要"]),
        "logo 颜色？": ("红色", ["今天的会议纪要"]),
        # C: 拒答对 + 常规题对
        "服务器 root 密码？": (REFUSAL, []),
        "下午茶水果？": ("苹果", ["购物清单"]),
    }
    cand = {
        # A: 全对 + 正确拒答
        "项目截止日是哪天？": ("周五", ["项目截止日是周五"]),
        "现在项目截止日是哪天？": ("周五", ["截止日已改为周五"]),
        "老板私人电话？": (REFUSAL, []),
        # B: 全对
        "封面颜色？": ("蓝色", ["封面颜色"]),
        "logo 颜色？": ("绿色", ["logo 颜色"]),
        # C: 同 baseline
        "服务器 root 密码？": (REFUSAL, []),
        "下午茶水果？": ("苹果", ["购物清单"]),
    }
    return {"baseline": base, "candidate": cand}


def _specs(mini: dict[str, Path]) -> list[DatasetSpec]:
    return [
        DatasetSpec(key="a", name="A（办公）", path=str(mini["A"]), source="user-specified", is_office_slot=True),
        DatasetSpec(key="b", name="B", path=str(mini["B"]), source="user-specified"),
        DatasetSpec(key="c", name="C", path=str(mini["C"]), source="user-specified"),
    ]


def _arm_results(arms: list[ArmDatasetResult], side: str) -> list[DatasetResult]:
    """把双臂结果拆成某一侧的 DatasetResult 列表（``_side_fuse`` 的入参形态）。"""
    out: list[DatasetResult] = []
    for arm in arms:
        if side == "baseline":
            out.append(
                DatasetResult(
                    spec=arm.spec,
                    run_dir=arm.baseline_dataset_dir,
                    metrics=arm.baseline,
                    capabilities=arm.capabilities,
                )
            )
        else:
            out.append(
                DatasetResult(
                    spec=arm.spec,
                    run_dir=arm.candidate_dataset_dir,
                    metrics=arm.candidate,
                    capabilities=arm.capabilities,
                )
            )
    return out


def _build_arms(mini: dict[str, Path], scripts: dict[str, dict[str, tuple[str, list[str]]]]) -> list[ArmDatasetResult]:
    """跑两侧真实 run 构造 ArmDatasetResult；两侧各占一个实验目录。"""
    base = ScriptedAdapter(scripts["baseline"], name="base")
    cand = ScriptedAdapter(scripts["candidate"], name="cand")
    from membench.evaluate import run_datasets

    root = mini["A"].parent
    specs = _specs(mini)
    base_results = run_datasets(
        base, "heuristic", specs=specs, experiment_dir=str(root / "arm_base")
    )
    cand_results = run_datasets(
        cand, "heuristic", specs=specs, experiment_dir=str(root / "arm_cand")
    )
    arms: list[ArmDatasetResult] = []
    for b, c in zip(base_results, cand_results):
        arms.append(
            ArmDatasetResult(
                spec=b.spec,
                baseline=b.metrics,
                candidate=c.metrics,
                baseline_dataset_dir=b.run_dir,
                candidate_dataset_dir=c.run_dir,
                capabilities=b.capabilities,
            )
        )
    return arms


def test_side_fuse_matches_side_metrics(mini_datasets, scripts) -> None:
    """_side_fuse 与 evaluate.fuse_metrics 在两侧应分别得到一致的融合值。"""
    arms = _build_arms(mini_datasets, scripts)
    base_results = _arm_results(arms, "baseline")
    cand_results = _arm_results(arms, "candidate")
    fused_base, sources_base = _side_fuse(base_results)
    fused_cand, _ = _side_fuse(cand_results)

    # capabilities 已落账时，两侧融合值与直接对同一批 DatasetResult 融合一致
    assert fused_base == fuse_metrics(base_results)
    assert fused_cand == fuse_metrics(cand_results)

    # 手工比对：baseline 侧 accuracy
    assert fused_base["accuracy"]["value"] is not None
    assert fused_cand["accuracy"]["value"] is not None
    assert fused_cand["accuracy"]["value"] >= fused_base["accuracy"]["value"]
    # 新 run 的 metrics.json 顶层已落账 capabilities，来源应全部为 "metrics"
    assert sources_base == {"a": "metrics", "b": "metrics", "c": "metrics"}


def test_fused_diff_computes_per_metric_diff() -> None:
    """_fused_diff：任一侧 None 时跳过；其它指标算 b - a。"""
    fused_a = {
        "accuracy": {"value": 0.6, "participants": ["x"]},
        "evidence_recall": {"value": None, "participants": []},
    }
    fused_b = {
        "accuracy": {"value": 0.7, "participants": ["x"]},
        "evidence_recall": {"value": 0.5, "participants": ["y"]},
    }
    diffs = _fused_diff(fused_a, fused_b)
    assert set(diffs.keys()) == {"accuracy"}  # evidence_recall 双侧不全，跳过
    assert diffs["accuracy"]["diff"] == pytest.approx(0.1)


def test_fused_gates_triggers_fail_on_accuracy_regression() -> None:
    """_fused_gates：accuracy 下降超阈值 -> FAIL。"""
    fused_a = {"accuracy": {"value": 0.85, "participants": ["x"]}}
    fused_b = {"accuracy": {"value": 0.80, "participants": ["x"]}}
    gates = _fused_gates(fused_a, fused_b, acc_threshold=0.02, stale_threshold=0.02, latency_warn=0.30)
    assert any(g["status"] == "FAIL" for g in gates)


def test_fused_gates_skips_when_no_metrics() -> None:
    """_fused_gates：任一侧融合值为空 -> 返回空 list。"""
    gates = _fused_gates({}, {"accuracy": {"value": 0.5, "participants": []}}, 0.02, 0.02, 0.30)
    assert gates == []


def test_per_dataset_diff_returns_one_row_per_dataset(mini_datasets, scripts) -> None:
    arms = _build_arms(mini_datasets, scripts)
    rows = _per_dataset_diff(arms)
    assert len(rows) == 3
    keys = [r["spec"].key for r in rows]
    assert keys == ["a", "b", "c"]
    # candidate 全面优于 baseline：accuracy diff 全部 >= 0
    for row in rows:
        diff = row["diffs"]["accuracy"]["diff"]
        assert diff >= -1e-9


def test_evaluate_compare_writes_compare_layout(mini_datasets, scripts, tmp_path, monkeypatch) -> None:
    """evaluate_compare 端到端：compare_<name>_<platform>_<ts>/ 落地，
    A/B 各为完整 eval 实验形态，对比报告与两张对比图落其 reports/。"""
    monkeypatch.setenv("MEMBENCH_OUTPUT_DIR", str(tmp_path / "output"))
    base = ScriptedAdapter(scripts["baseline"], name="base")
    cand = ScriptedAdapter(scripts["candidate"], name="cand")
    report_out = tmp_path / "out"
    result = evaluate_compare(
        base,
        cand,
        "heuristic",
        report_out=str(report_out),
        tag="base_vs_cand",
        specs=_specs(mini_datasets),
        adapter_version_baseline="v1",
        adapter_version_candidate="v2",
    )
    compare_dir = Path(result["compare_dir"])
    assert compare_dir.parent == report_out
    assert compare_dir.name.startswith("compare_base_vs_cand_")
    # 实验目录父目录下只有这一个实验目录，报告不再散落在父目录根层
    assert [p.name for p in report_out.iterdir()] == [compare_dir.name]

    reports = compare_dir / layout.DIR_REPORTS
    assert (reports / layout.FILE_COMPARISON_MD).is_file()
    assert (reports / FIG_PER_DATASET_DIFF).is_file()
    assert (reports / FIG_FUSED_DIFF).is_file()

    # A/B 两侧各是完整 eval 形态：runs/<dataset>/report/... + 任务级 reports/
    for arm in ("A", "B"):
        for key in ("a", "b", "c"):
            dataset_report = compare_dir / arm / layout.DIR_RUNS / key / layout.DIR_REPORT
            assert (dataset_report / layout.FILE_METRICS).is_file()
            assert (dataset_report / layout.FILE_RESULTS).is_file()
        task_dir = compare_dir / arm / layout.DIR_REPORTS
        for name in (
            layout.FILE_REPORT_MD,
            layout.FILE_METRICS,
            layout.FILE_SUMMARY,
            layout.FIG_DATASET_OVERVIEW,
        ):
            assert (task_dir / name).is_file(), f"{arm} 缺少 reports/{name}"

    md = (reports / layout.FILE_COMPARISON_MD).read_text(encoding="utf-8")
    assert "**门禁结论：" in md
    assert "基准集融合 diff" in md
    assert "逐数据集关键指标 diff" in md
    # by_category diff 段按数据集分别渲染（数据集小标题 + 中文题型名，不跨集合并）
    assert "逐数据集题型 diff" in md
    # 题型 diff 段开头给「提升最多的题型 TOP N」概览（小样本集可能为空 → 出说明）
    assert "提升最多的题型" in md or "本次没有满足条件的提升项" in md
    assert "### A（办公）" in md
    assert "单跳（single_hop）" in md
    assert "拒答（abstention）" in md
    # 中文化指标名出现在报告中
    assert "正确率（accuracy）" in md
    assert "证据召回率（evidence_recall）" in md
    # 图片引用一律同目录相对文件名（报告与图同在 reports/，可整体搬移）
    refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md)
    assert refs, "对比报告应引用对比图"
    for ref in refs:
        assert "://" not in ref and "/" not in ref, ref
        assert (reports / ref).is_file(), ref


def test_evaluate_compare_verdict_pass_when_candidate_better(mini_datasets, scripts, tmp_path, monkeypatch) -> None:
    """candidate 全面优于 baseline -> verdict 与门禁明细仍齐全。"""
    monkeypatch.setenv("MEMBENCH_OUTPUT_DIR", str(tmp_path / "output"))
    base = ScriptedAdapter(scripts["baseline"], name="base")
    cand = ScriptedAdapter(scripts["candidate"], name="cand")
    result = evaluate_compare(
        base,
        cand,
        "heuristic",
        report_out=str(tmp_path / "out"),
        tag="base_vs_cand",
        specs=_specs(mini_datasets),
    )
    assert result["verdict"] in ("PASS", "WARN")  # A 的 stale_rate=1.0 可能触发 WARN/FAIL
    # 至少 gates 包含 accuracy_regression
    gate_names = {g["name"] for g in result["gates"]}
    assert "accuracy_regression" in gate_names


def test_evaluate_compare_precheck_failure_leaves_no_experiment_dir(
    mini_datasets, scripts, tmp_path, monkeypatch
) -> None:
    """预检失败（补跑参数与已有产物不一致）在建目录 / 复制产物之前拦下：

    ``report_out`` 下不留任何实验目录——连父目录都不建，更没有 A/B 半成品树。
    """
    monkeypatch.setenv("MEMBENCH_OUTPUT_DIR", str(tmp_path / "output"))
    arms = _build_arms(mini_datasets, scripts)
    report_out = tmp_path / "out_precheck"

    with pytest.raises(SystemExit, match="补跑参数与已有 run 不一致"):
        evaluate_compare(
            ScriptedAdapter(scripts["baseline"], name="base"),
            ScriptedAdapter(scripts["candidate"], name="cand"),
            "heuristic",
            report_out=str(report_out),
            tag="base_vs_cand",
            specs=_specs(mini_datasets),
            # 基线侧只复用 a：b / c 需补跑，补跑标注与已有产物不一致即拒绝
            baseline_runs={arms[0].spec.key: arms[0].baseline_dataset_dir},
            candidate_runs={arm.spec.key: arm.candidate_dataset_dir for arm in arms},
            adapter_version_baseline="v9",
        )

    assert not report_out.exists()
    assert not list(report_out.glob("compare_*"))


def test_cli_evaluate_compare_quick_and_beam10m_exclusive() -> None:
    """CLI 互斥参数：--quick 与 --beam-10m 同时传 -> argparse 拒绝。"""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "evaluate-compare",
                "--baseline-adapter", "mock-strong",
                "--candidate-adapter", "mock-weak",
                "--quick",
                "--beam-10m",
            ]
        )
    assert exc_info.value.code == 2


def test_cli_evaluate_compare_common_conflicts_and_pass_through(monkeypatch) -> None:
    """CLI --common：与 --dataset / --quick 互斥；合法组合透传 common=True。"""
    with pytest.raises(SystemExit, match="互斥"):
        cli.main(
            [
                "evaluate-compare",
                "--baseline-adapter", "mock-strong",
                "--candidate-adapter", "mock-weak",
                "--common",
                "--dataset", "office",
            ]
        )
    with pytest.raises(SystemExit, match="互斥"):
        cli.main(
            [
                "evaluate-compare",
                "--baseline-adapter", "mock-strong",
                "--candidate-adapter", "mock-weak",
                "--common",
                "--quick",
            ]
        )

    captured: dict[str, object] = {}

    def _stub(baseline_adapter, candidate_adapter, judge, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return {"verdict": "PASS", "gates": [], "fused_diffs": {}, "fused_baseline": {}, "fused_candidate": {}, "per_dataset": [], "compare_dir": "/tmp/compare_x"}

    import membench.evaluate_compare as ec_mod

    monkeypatch.setattr(ec_mod, "evaluate_compare", _stub)
    rc = cli.main(
        [
            "evaluate-compare",
            "--baseline-adapter", "mock-strong",
            "--candidate-adapter", "mock-weak",
            "--common",
            "--no-beam",
        ]
    )
    assert rc == 0
    assert captured["common"] is True
    assert captured["no_beam"] is True


def test_cli_evaluate_compare_pass_through(tmp_path, monkeypatch) -> None:
    """_cmd_evaluate_compare 透传到 evaluate_compare；失败时退出码 1。

    ``--out`` 已删除：CLI 只透传 ``--report-out``（实验目录父目录），不再传 out_dir。
    """
    captured: dict[str, object] = {}
    report_out = tmp_path / "out"

    def _stub(baseline_adapter, candidate_adapter, judge, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return {"verdict": "FAIL", "gates": [], "fused_diffs": {}, "fused_baseline": {}, "fused_candidate": {}, "per_dataset": [], "compare_dir": str(tmp_path / "compare_x")}

    import membench.evaluate_compare as ec_mod

    monkeypatch.setattr(ec_mod, "evaluate_compare", _stub)
    rc = cli.main(
        [
            "evaluate-compare",
            "--baseline-adapter", "mock-strong",
            "--candidate-adapter", "mock-weak",
            "--quick",
            "--acc-threshold", "0.01",
            "--stale-threshold", "0.01",
            "--latency-warn", "0.2",
            "--tag", "my_tag",
            "--report-out", str(report_out),
        ]
    )
    assert rc == 1
    assert captured["acc_threshold"] == 0.01
    assert captured["stale_threshold"] == 0.01
    assert captured["latency_warn"] == 0.2
    assert captured["tag"] == "my_tag"
    assert captured["report_out"] == str(report_out)
    assert "out_dir" not in captured
    assert captured["quick"] is True
    assert captured["beam_10m"] is False


def test_cli_evidence_match_pass_through(tmp_path, monkeypatch) -> None:
    """--evidence-match overlap 构造匹配器并透传到 evaluate_compare。"""
    captured: dict[str, object] = {}

    def _stub(baseline_adapter, candidate_adapter, judge, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return {"verdict": "PASS", "gates": [], "fused_diffs": {}, "fused_baseline": {}, "fused_candidate": {}, "per_dataset": [], "compare_dir": str(tmp_path / "compare_x")}

    import membench.evaluate_compare as ec_mod

    monkeypatch.setattr(ec_mod, "evaluate_compare", _stub)
    rc = cli.main(
        [
            "evaluate-compare",
            "--baseline-adapter", "mock-strong",
            "--candidate-adapter", "mock-weak",
            "--quick",
            "--evidence-match", "overlap",
            "--evidence-match-threshold", "0.7",
        ]
    )
    assert rc == 0
    matcher = captured["evidence_matcher"]
    assert matcher.name == "overlap"
    assert matcher.threshold == 0.7


def test_cli_evidence_match_llm_requires_config(monkeypatch) -> None:
    """--evidence-match llm 缺少 --llm-base-url 时报错退出（可操作的提示）。"""
    monkeypatch.delenv("MEMBENCH_JUDGE_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "evaluate-compare",
                "--baseline-adapter", "mock-strong",
                "--candidate-adapter", "mock-weak",
                "--quick",
                "--evidence-match", "llm",
            ]
        )
    assert "--llm-base-url" in str(exc_info.value)


def test_cli_evaluate_compare_pass_returns_zero(tmp_path, monkeypatch) -> None:
    """verdict=PASS 时退出码 0。"""
    def _stub(baseline_adapter, candidate_adapter, judge, **kwargs):  # type: ignore[no-untyped-def]
        return {"verdict": "PASS", "gates": [], "fused_diffs": {}, "fused_baseline": {}, "fused_candidate": {}, "per_dataset": [], "compare_dir": str(tmp_path / "compare_x")}

    import membench.evaluate_compare as ec_mod

    monkeypatch.setattr(ec_mod, "evaluate_compare", _stub)
    rc = cli.main(
        [
            "evaluate-compare",
            "--baseline-adapter", "mock-strong",
            "--candidate-adapter", "mock-weak",
        ]
    )
    assert rc == 0


def test_per_dataset_gates_flags_single_dataset_spike() -> None:
    """单数据集 spike：融合平均抹平后，per-dataset 门禁仍能捕获。

    A 数据集 stale_rate 0.01 → 0.10 远超阈值：逐数据集 verdict 应为 FAIL，
    且门禁明细包含 stale_rate_regression。
    """
    from membench.evaluate import DatasetSpec
    from membench.evaluate_compare import ArmDatasetResult, _per_dataset_gates

    arms = [
        ArmDatasetResult(
            spec=DatasetSpec(key="a", name="A", path="x", source="builtin"),
            baseline={"overall": {"accuracy": 0.9, "stale_rate": 0.01, "latency_p95_ms": 100.0}},
            candidate={"overall": {"accuracy": 0.9, "stale_rate": 0.10, "latency_p95_ms": 100.0}},
            baseline_dataset_dir="x",
            candidate_dataset_dir="y",
        )
    ]
    rows = _per_dataset_gates(arms, acc_threshold=0.02, stale_threshold=0.02, latency_warn=0.30)
    assert len(rows) == 1
    assert rows[0]["verdict"] == "FAIL"
    assert any(g["name"] == "stale_rate_regression" for g in rows[0]["gates"])


def test_resolve_capabilities_reads_top_level_and_falls_back(tmp_path) -> None:
    """capabilities 解析：顶层落账优先直读；旧版全 False 时回退重算并核对。"""
    from membench.evaluate_compare import _resolve_capabilities

    # 顶层已落账（任一 True）→ 直读，来源 metrics
    caps, src = _resolve_capabilities(
        {"has_evidence": True, "has_stale": False, "has_abstention": True},
        dataset_dir=str(tmp_path),
        spec_path=str(tmp_path / "missing.jsonl"),
    )
    assert src == "metrics"
    assert caps == {"has_evidence": True, "has_stale": False, "has_abstention": True}

    # 旧版：顶层全 False 占位，但数据集实际有拒答题 → 回退扫数据集，来源 dataset
    ds = _write_dataset(
        tmp_path / "d.jsonl",
        [
            {
                "dialogue_id": "d1",
                "sessions": [_session()],
                "questions": [
                    {
                        "qid": "q1",
                        "question": "私人电话？",
                        "gold_answer": "<ABSTAIN>",
                        "category": "abstention",
                        "evidence": [],
                    }
                ],
            }
        ],
    )
    caps, src = _resolve_capabilities(
        {"has_evidence": False, "has_stale": False, "has_abstention": False},
        dataset_dir=str(tmp_path),
        spec_path=str(ds),
    )
    assert src == "dataset"
    assert caps["has_abstention"] is True

    # 顶层缺失、数据集与 results.jsonl 均不可得 → missing 全 False
    caps, src = _resolve_capabilities(
        {}, dataset_dir=str(tmp_path), spec_path=str(tmp_path / "nope.jsonl")
    )
    assert src == "missing"
    assert caps == {"has_evidence": False, "has_stale": False, "has_abstention": False}

    # 数据集不可得时回退按 results.jsonl 反推（evidence_hit / KU / 拒答哨兵）；
    # 逐题明细固定读 <数据集目录>/report/results.jsonl
    dataset_dir = tmp_path / "run1"
    report_dir = dataset_dir / layout.DIR_REPORT
    report_dir.mkdir(parents=True)
    (report_dir / layout.FILE_RESULTS).write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"category": "knowledge_update", "evidence_hit": False, "gold_answer": "x"},
                {"category": "privacy_refuse", "evidence_hit": True, "gold_answer": "<ABSTAIN>"},
            ]
        ),
        encoding="utf-8",
    )
    caps, src = _resolve_capabilities(
        {},
        dataset_dir=str(dataset_dir),
        spec_path=str(tmp_path / "nope2.jsonl"),
    )
    assert src == "results_jsonl"
    assert caps == {"has_evidence": True, "has_stale": True, "has_abstention": True}


def test_cli_evaluate_compare_dataset_flag(tmp_path, monkeypatch) -> None:
    """CLI evaluate-compare --dataset：两侧透传同一显式清单。"""
    import membench.evaluate_compare as ec_mod

    ds = tmp_path / "custom.jsonl"
    ds.write_text(
        json.dumps(
            {
                "dialogue_id": "d1",
                "sessions": [[{"role": "user", "content": "聊聊", "timestamp": None}]],
                "questions": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def _stub(baseline_adapter, candidate_adapter, judge, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return {
            "verdict": "PASS",
            "gates": [],
            "fused_diffs": {},
            "fused_baseline": {},
            "fused_candidate": {},
            "per_dataset": [],
            "compare_dir": str(tmp_path / "compare_x"),
        }

    monkeypatch.setattr(ec_mod, "evaluate_compare", _stub)
    rc = cli.main(
        [
            "evaluate-compare",
            "--baseline-adapter", "mock-strong",
            "--candidate-adapter", "mock-weak",
            "--dataset", str(ds),
        ]
    )
    assert rc == 0
    specs = captured["specs"]
    assert specs is not None and specs[0].path == str(ds.resolve())
    # 未传 --report-out 时交给缺省 output_root()；--out 已删除，不再有 out_dir 参数
    assert captured["report_out"] is None
    assert "out_dir" not in captured

    with pytest.raises(SystemExit, match="互斥"):
        cli.main(
            [
                "evaluate-compare",
                "--baseline-adapter", "mock-strong",
                "--candidate-adapter", "mock-weak",
                "--dataset", str(ds),
                "--quick",
            ]
        )


# ---------------------------------------------------------------------------
# 就地模式（task_dir）与快照开关
# ---------------------------------------------------------------------------


def test_evaluate_compare_task_dir_in_place(
    mini_datasets, scripts, tmp_path, monkeypatch
) -> None:
    """就地模式：两臂写 ``DIR/A`` ``DIR/B``，对比报告落 ``DIR/reports/``，不建 ``compare_*``。"""
    monkeypatch.setenv("MEMBENCH_OUTPUT_DIR", str(tmp_path / "output"))
    base = tmp_path / "out"
    task = Path(layout.init_task_dir(str(base), "ab", platform="t"))

    result = evaluate_compare(
        ScriptedAdapter(scripts["baseline"], name="base"),
        ScriptedAdapter(scripts["candidate"], name="cand"),
        "heuristic",
        task_dir=str(task),
        tag="base_vs_cand",
        specs=_specs(mini_datasets),
        adapter_version_baseline="v1",
        adapter_version_candidate="v2",
    )

    assert result["compare_dir"] == str(task)
    for arm in (layout.DIR_ARM_A, layout.DIR_ARM_B):
        arm_dir = task / arm
        for key in ("a", "b", "c"):
            dataset_report = arm_dir / layout.DIR_RUNS / key / layout.DIR_REPORT
            assert (dataset_report / layout.FILE_METRICS).is_file()
            assert (dataset_report / layout.FILE_RESULTS).is_file()
        for name in (
            layout.FILE_REPORT_MD,
            layout.FILE_METRICS,
            layout.FILE_SUMMARY,
            layout.FIG_DATASET_OVERVIEW,
        ):
            assert (arm_dir / layout.DIR_REPORTS / name).is_file(), f"{arm} 缺少 reports/{name}"
        # 该臂自身就是一个完整单臂实验形态：summary 里的 experiment_dir 就是它
        arm_summary = json.loads(
            (arm_dir / layout.DIR_REPORTS / layout.FILE_SUMMARY).read_text(encoding="utf-8")
        )
        assert Path(arm_summary["experiment_dir"]) == arm_dir

    reports = task / layout.DIR_REPORTS
    for name in (layout.FILE_COMPARISON_MD, FIG_PER_DATASET_DIFF, FIG_FUSED_DIFF):
        assert (reports / name).is_file(), f"缺少 reports/{name}"
    md = (reports / layout.FILE_COMPARISON_MD).read_text(encoding="utf-8")
    assert "**门禁结论：" in md and "基准集融合 diff" in md
    # 就地：没有另外新建 compare_* 实验目录
    assert [p.name for p in base.iterdir()] == [task.name]

    with pytest.raises(SystemExit, match="任务目录不存在"):
        evaluate_compare(task_dir=str(base / "nope"), specs=_specs(mini_datasets))
    with pytest.raises(SystemExit, match="互斥"):
        evaluate_compare(
            task_dir=str(task), report_out=str(base), specs=_specs(mini_datasets)
        )


def test_evaluate_compare_forwards_snapshot_to_both_sides(
    mini_datasets, scripts, tmp_path, monkeypatch
) -> None:
    """``snapshot`` 逐侧透传：默认不覆盖 Runner 默认值（导出），False 时两侧都收到 False。"""
    monkeypatch.setenv("MEMBENCH_OUTPUT_DIR", str(tmp_path / "output"))
    import membench.evaluate as ev_mod
    from membench.runner import Runner as RealRunner

    sent: list[object] = []

    class _RecordingRunner:
        """包一层真 Runner：只记 ``snapshot`` 实参，其余照旧（真跑，不是桩）。"""

        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            sent.append(kwargs.get("snapshot", "<未传>"))
            self._inner = RealRunner(*args, **kwargs)

        def run(self):
            return self._inner.run()

    monkeypatch.setattr(ev_mod, "Runner", _RecordingRunner)
    specs = _specs(mini_datasets)[:1]  # 只跑 A：每侧一个数据集 = 每侧一次 Runner

    evaluate_compare(
        ScriptedAdapter(scripts["baseline"], name="base"),
        ScriptedAdapter(scripts["candidate"], name="cand"),
        "heuristic",
        report_out=str(tmp_path / "out_default"),
        tag="base_vs_cand",
        specs=specs,
    )
    assert sent == ["<未传>", "<未传>"], "默认不表态：Runner 按自己的默认（导出）处理"

    sent.clear()
    evaluate_compare(
        ScriptedAdapter(scripts["baseline"], name="base"),
        ScriptedAdapter(scripts["candidate"], name="cand"),
        "heuristic",
        report_out=str(tmp_path / "out_quiet"),
        tag="base_vs_cand",
        specs=specs,
        snapshot=False,
    )
    assert sent == [False, False], "snapshot=False 要传到两侧的每个数据集 Runner"


def _arm(name: str, a_cats: dict[str, tuple[int, float]], b_cats: dict[str, tuple[int, float]]):
    """构造一个 ArmDatasetResult（只填 by_category 相关字段）。"""
    from membench.evaluate import DatasetSpec
    from membench.evaluate_compare import ArmDatasetResult

    def by_cat(cats: dict[str, tuple[int, float]]) -> dict:
        return {
            cat: {"n": n, "accuracy": acc, "token_f1": None} for cat, (n, acc) in cats.items()
        }

    return ArmDatasetResult(
        spec=DatasetSpec(key=name, name=name, path=f"/data/{name}.jsonl", source="fetch"),
        baseline={"by_category": by_cat(a_cats)},
        candidate={"by_category": by_cat(b_cats)},
        baseline_dataset_dir=f"/a/{name}",
        candidate_dataset_dir=f"/b/{name}",
    )


def test_top_improved_categories_orders_and_filters() -> None:
    """TOP 榜：按 Δ 降序；小样本（<10 题）与负向改进不进榜。"""
    from membench.evaluate_compare import top_improved_categories

    arms = [
        _arm(
            "X",
            {"single_hop": (100, 0.50), "multi_hop": (5, 0.0)},
            {"single_hop": (100, 0.70), "multi_hop": (5, 1.0)},
        ),
        _arm(
            "Y",
            {"temporal": (20, 0.60), "abstention": (30, 0.90)},
            {"temporal": (20, 0.75), "abstention": (30, 0.50)},
        ),
    ]
    top = top_improved_categories(arms)
    assert [(item["dataset"], item["category"]) for item in top] == [
        ("X", "single_hop"),
        ("Y", "temporal"),
    ]
    assert top[0]["delta"] == pytest.approx(0.2)
    assert top[0]["n"] == 100
    # 小样本 +100pt 不在榜上：会盖住大样本的真实改进
    assert all(item["category"] != "multi_hop" for item in top)
    # 负向改进不进榜
    assert all(item["category"] != "abstention" for item in top)


def test_top_improved_categories_empty_when_all_small() -> None:
    from membench.evaluate_compare import top_improved_categories

    arms = [_arm("X", {"single_hop": (3, 0.0)}, {"single_hop": (3, 1.0)})]
    assert top_improved_categories(arms) == []


def test_top_improved_categories_respects_limit() -> None:
    from membench.evaluate_compare import top_improved_categories

    cats_a = {f"c{i}": (20, 0.5) for i in range(8)}
    cats_b = {f"c{i}": (20, 0.5 + i * 0.01) for i in range(8)}
    top = top_improved_categories([_arm("X", cats_a, cats_b)], limit=3)
    assert len(top) == 3
    assert top[0]["category"] == "c7"  # Δ 最大
