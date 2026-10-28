"""加权融合口径测试：按参与题数加权、单集封顶、办公域槽位不参与。

覆盖产出方案的四条硬约束：
- 权重 = min(该指标的参与题数, 500)，办公域槽位不参与基准评级；
- 分母按指标取（检索类非拒答题数 / 拒答类拒答题数 / 陈旧率可机检题数），
  旧 run 缺 ``evidence_n`` / ``abstention_n`` 时回退 ``n_questions``；
- 基准集为空（只跑办公域槽位）时评级 n/a，业务结论与门禁照常；
- ``fusion`` 进可比性**次要维度**：只影响多数据集融合值，单数据集指标与它无关，
  因此不判不可比（旧产物归一为 ``legacy``，与两种新口径只在报告里标 WARN）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from membench import cli
from membench.adapters.base import BaseMemoryAdapter, IngestStats, QueryResult
from membench.compare import check_comparability
from membench.evaluate import (
    DatasetResult,
    DatasetSpec,
    evaluate,
    fuse_metrics,
    metric_denominator,
)


def _result(
    key: str,
    *,
    office: bool = False,
    values: dict[str, float] | None = None,
    n_questions: int = 100,
    evidence_n: int | None = None,
    abstention_n: int | None = None,
    stale_n: int | None = None,
    has_evidence: bool = False,
    has_abstention: bool = False,
    has_stale: bool = False,
) -> DatasetResult:
    """构造一个数据集结果（只带融合需要的字段）。"""
    overall: dict[str, object] = {"n_questions": n_questions}
    if evidence_n is not None:
        overall["evidence_n"] = evidence_n
    if abstention_n is not None:
        overall["abstention_n"] = abstention_n
    if stale_n is not None:
        overall["stale_n"] = stale_n
    overall.update(values or {})
    return DatasetResult(
        spec=DatasetSpec(key=key, name=key, path="x", source="builtin", is_office_slot=office),
        run_dir="x",
        metrics={"overall": overall},
        capabilities={
            "has_evidence": has_evidence,
            "has_stale": has_stale,
            "has_abstention": has_abstention,
        },
    )


def test_weighted_skips_office_and_caps_weight() -> None:
    """5 个集（86/1986/500/407/700 题）：office 不参与、locomo 封顶 500、equal 等权。"""
    results = [
        _result("office", office=True, values={"accuracy": 1.0}, n_questions=86),
        _result("locomo", values={"accuracy": 0.2}, n_questions=1986),
        _result("d500", values={"accuracy": 0.5}, n_questions=500),
        _result("d407", values={"accuracy": 0.6}, n_questions=407),
        _result("d700", values={"accuracy": 0.7}, n_questions=700),
    ]
    fused = fuse_metrics(results)
    acc = fused["accuracy"]

    assert acc["participants"] == ["locomo", "d500", "d407", "d700"]
    weights = {w["key"]: (w["denominator"], w["weight"]) for w in acc["weights"]}
    assert weights == {
        "locomo": (1986, 500),  # 封顶
        "d500": (500, 500),
        "d407": (407, 407),
        "d700": (700, 500),  # 封顶
    }
    assert acc["total_weight"] == 1907
    expected = (500 * 0.2 + 500 * 0.5 + 407 * 0.6 + 500 * 0.7) / 1907
    assert acc["value"] == pytest.approx(expected)

    equal = fuse_metrics(results, fusion="equal")["accuracy"]
    assert equal["participants"] == ["locomo", "d500", "d407", "d700"]
    assert [w["weight"] for w in equal["weights"]] == [1, 1, 1, 1]
    assert equal["value"] == pytest.approx((0.2 + 0.5 + 0.6 + 0.7) / 4)


def test_abstention_weight_uses_abstention_n_not_question_count() -> None:
    """拒答指标按拒答题数加权：6 : 446，不是 86 : 500（也不是 86 : 1986）。

    办公域槽位不参与融合，故用题量分布相同的基准集验证分母口径；办公集自身的
    分母同样按 ``abstention_n`` 记录（业务结论里读同一字段）。
    """
    office = _result(
        "office",
        office=True,
        values={"abstention_accuracy": 1.0},
        n_questions=86,
        abstention_n=6,
        has_abstention=True,
    )
    bench = _result(
        "locomo",
        values={"abstention_accuracy": 0.5},
        n_questions=1986,
        abstention_n=446,
        has_abstention=True,
    )
    # office 的分母本身也按 abstention_n（6）而不是 86——只是它不参与融合
    assert metric_denominator(office.metrics["overall"], "abstention_accuracy") == 6

    # 同规模基准集：6 道拒答题 / 86 题；与 locomo 的 446 道按 6:446 加权
    small = _result(
        "small_bench",
        values={"abstention_accuracy": 1.0},
        n_questions=86,
        abstention_n=6,
        has_abstention=True,
    )
    fused = fuse_metrics([office, small, bench])["abstention_accuracy"]
    assert fused["participants"] == ["small_bench", "locomo"]
    assert [(w["denominator"], w["weight"]) for w in fused["weights"]] == [
        (6, 6),
        (446, 446),
    ]
    assert fused["total_weight"] == 452
    assert fused["value"] == pytest.approx((6 * 1.0 + 446 * 0.5) / 452)


def test_legacy_run_without_counts_falls_back_to_n_questions() -> None:
    """旧 run 缺 evidence_n / abstention_n：分母与权重回退 n_questions。"""
    legacy = _result(
        "legacy",
        values={"evidence_recall": 0.8, "abstention_accuracy": 0.9},
        n_questions=10,
        has_evidence=True,
        has_abstention=True,
    )
    fresh = _result(
        "fresh",
        values={"evidence_recall": 0.4, "abstention_accuracy": 0.1},
        n_questions=1000,
        evidence_n=30,
        abstention_n=40,
        has_evidence=True,
        has_abstention=True,
    )
    assert metric_denominator(legacy.metrics["overall"], "evidence_recall") == 10
    assert metric_denominator(legacy.metrics["overall"], "abstention_accuracy") == 10

    evidence = fuse_metrics([legacy, fresh])["evidence_recall"]
    assert [(w["denominator"], w["weight"]) for w in evidence["weights"]] == [
        (10, 10),  # 回退 n_questions
        (30, 30),
    ]
    assert evidence["value"] == pytest.approx((10 * 0.8 + 30 * 0.4) / 40)

    abstention = fuse_metrics([legacy, fresh])["abstention_accuracy"]
    assert [(w["denominator"], w["weight"]) for w in abstention["weights"]] == [
        (10, 10),
        (40, 40),
    ]


def test_zero_or_missing_denominator_skips_dataset() -> None:
    """分母为 0 / 取不到的集在该指标上不参与（没有参与题就没有权重）。"""
    no_count = _result("no_count", values={"accuracy": 0.9}, n_questions=0)
    normal = _result("normal", values={"accuracy": 0.4}, n_questions=10)
    fused = fuse_metrics([no_count, normal])["accuracy"]
    assert fused["participants"] == ["normal"]
    assert fused["value"] == pytest.approx(0.4)

    # 等权口径不看分母：参与集就是一票
    equal = fuse_metrics([no_count, normal], fusion="equal")["accuracy"]
    assert equal["participants"] == ["no_count", "normal"]
    assert equal["value"] == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# 基准集为空（只跑办公域槽位）
# ---------------------------------------------------------------------------


class _StaticAdapter(BaseMemoryAdapter):
    """固定答案适配器：只为跑通一次最小 evaluate。"""

    name = "static"

    def reset(self) -> None:
        """无状态。"""

    def ingest(self, messages: list[dict]) -> IngestStats:
        """不存储。"""
        return IngestStats()

    def query(self, question: str) -> QueryResult:
        """按问题返回固定答案，q2 答旧值以制造 stale。"""
        if "旧值" in question:
            return QueryResult(answer="周三", retrieved=["截止日已改为周五"])
        return QueryResult(answer="周五", retrieved=["项目截止日是周五"])


def _write_office(path: Path) -> Path:
    """写一个带 stale 题的迷你办公集。"""
    dialogues = [
        {
            "dialogue_id": "d1",
            "sessions": [[{"role": "user", "content": "随便聊聊。", "timestamp": None}]],
            "questions": [
                {
                    "qid": "q1",
                    "question": "项目截止日是哪天？",
                    "gold_answer": "周五",
                    "category": "single_hop",
                    "evidence": ["项目截止日是周五"],
                },
                {
                    "qid": "q2",
                    "question": "现在项目截止日的旧值是哪天？",
                    "gold_answer": "周五",
                    "category": "knowledge_update",
                    "evidence": ["截止日已改为周五"],
                    "gold_stale": "周三",
                },
            ],
        }
    ]
    with path.open("w", encoding="utf-8") as f:
        for d in dialogues:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return path


def test_office_only_evaluate_has_no_baseline_rating(tmp_path: Path) -> None:
    """只评测办公域槽位：基准评级 n/a，业务结论与门禁照常输出。"""
    office = _write_office(tmp_path / "office.jsonl")
    report_out = tmp_path / "out"
    rc = evaluate(
        _StaticAdapter(),
        "heuristic",
        report_out=str(report_out),
        specs=[
            DatasetSpec(
                key="office",
                name="office（内置办公黄金集）",
                path=str(office),
                source="builtin-office",
                is_office_slot=True,
            )
        ],
    )
    eval_dir = sorted(report_out.glob("eval_*"))[-1]
    summary = json.loads((eval_dir / "reports" / "evaluate_summary.json").read_text("utf-8"))
    report = (eval_dir / "reports" / "report.md").read_text("utf-8")

    assert summary["fusion"] == "weighted"
    assert summary["fused"]["accuracy"]["value"] is None
    assert "**基准评级（不含办公域槽位）：n/a**" in report
    assert "## 业务结论（办公域槽位）" in report
    assert "陈旧率（stale_rate）" in report
    assert "**门禁结论：" in report
    assert summary["gate"]["status"] in ("PASS", "FAIL", "SKIP")
    # 门禁结论决定退出码，与基准评级无关
    assert rc == (1 if summary["gate"]["status"] == "FAIL" else 0)


# ---------------------------------------------------------------------------
# 可比性：fusion 是关键维度，旧产物归一为 legacy
# ---------------------------------------------------------------------------


def test_fusion_is_warn_comparability_dim() -> None:
    """weighted vs equal / legacy 记次要维度不一致（不判不可比）；旧产物归一 legacy。"""
    same = _base_metrics()
    weighted = check_comparability(dict(same, fusion="weighted"), dict(same, fusion="weighted"))
    assert weighted["comparable"] is True

    # 融合口径只影响多数据集融合值，单数据集指标与它无关 → 降级 WARN，
    # 不能因此把"单集 run 产物 vs evaluate 产物"判成不可比。
    mixed = check_comparability(dict(same, fusion="weighted"), dict(same, fusion="equal"))
    assert mixed["comparable"] is True
    assert [c["dimension"] for c in mixed["mismatches"]] == ["fusion"]
    assert all(c["severity"] == "warn" for c in mixed["mismatches"])
    assert "fusion" in " ".join(mixed["reasons"])

    legacy_vs_new = check_comparability(same, dict(same, fusion="weighted"))
    assert legacy_vs_new["comparable"] is True
    assert legacy_vs_new["mismatches"][0]["a"] == "legacy"

    # 两侧都是旧产物：同归一为 legacy，不算不一致
    legacy = check_comparability(same, dict(same))
    assert legacy["comparable"] is True


def _base_metrics() -> dict:
    """最小可核验的 metrics（关键维度都填好，只差 fusion）。"""
    return {
        "adapter": "mock",
        "adapter_version": "v1",
        "dataset_fingerprint": {"sha256": "aaaa"},
        "judge": "heuristic",
        "judge_model": None,
        "driver_llm": "none",
        "platform": "default",
        "run_mode": "full",
        "overall": {"n_questions": 10},
    }


def test_cli_passes_fusion_through(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """CLI --fusion 透传到 evaluate / evaluate-compare（缺省 weighted）。"""
    import membench.evaluate as ev_mod
    import membench.evaluate_compare as cmp_mod

    captured: dict[str, object] = {}

    def _stub_evaluate(adapter, judge, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return 0

    def _stub_compare(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return {"verdict": "PASS"}

    monkeypatch.setattr(ev_mod, "evaluate", _stub_evaluate)
    monkeypatch.setattr(cmp_mod, "evaluate_compare", _stub_compare)

    rc = cli.main(["evaluate", "--adapter", "mock-weak", "--fusion", "equal"])
    assert rc == 0 and captured["fusion"] == "equal"

    rc = cli.main(["evaluate-compare", "--baseline-adapter", "mock-weak", "--candidate-adapter", "mock-strong"])
    assert rc == 0 and captured["fusion"] == "weighted"
