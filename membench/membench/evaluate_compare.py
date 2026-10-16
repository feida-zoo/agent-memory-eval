"""evaluate_compare：两组（baseline vs candidate）× 多数据集融合 A/B 对比。

`compare_runs` 只对比**单数据集**两个 run；`evaluate` 只跑
**单组**多数据集融合；`matrix` 只对同数据集跨 label。两组 × 多数据集的融合
A/B 没有现成入口——本模块补齐，产出与 ``ab_<tag>_<ts>/`` 自包含结构一致的
逐数据集 diff + 跨数据集融合 diff + 门禁 + 对比图。

流程：
    plan_datasets（复用 evaluate.plan_datasets，与 evaluate 一致的数据集清单）
    -> 对每个数据集分别用 baseline / candidate 两个 adapter 各跑一次 Runner
    -> 逐数据集 diff（精度复用 :mod:`membench.compare` 的 _diff_overall / 门禁）
    -> 两组各自 fuse_metrics 得跨数据集宏平均融合值
    -> 融合 diff（b - a）+ 门禁（accuracy / stale_rate / latency_p95）
    -> 产物落 ``<root>/ab_<tag>_<ts>/``，按
       :func:`membench.layout.materialize_ab_experiment` 落 A/B 两侧 run 副本
       与 compare_report.md，外加两张图：
       ``fig_per_dataset_diff.png``（各数据集关键指标 diff）、``fig_fused_diff.png``
       （融合 diff）。

门禁与单数据集 compare 口径一致：
    accuracy 下降 > acc_threshold -> FAIL
    stale_rate 上升 > stale_threshold -> FAIL
    latency_p95 相对上升 > latency_warn -> WARN
    其余指标变动仅报告。

两组均跑同一 judge；adapter 元数据（adapter_version / commit / driver_llm）
独立传入，落到两侧 run 的 metrics.json。可比性前置校验由两侧各数据集的
dataset_fingerprint 自动覆盖（同一数据集两侧指纹必一致），故不重复。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Union

from membench import layout, paths
from membench.compare import _diff_overall, _eval_gates
from membench.data.schema import is_abstention
from membench.evaluate import (
    DatasetResult,
    DatasetSpec,
    fuse_metrics,
    plan_datasets,
    run_datasets,
    scan_capabilities,
)
from membench.report import (
    PALETTE,
    EFFICIENCY_DIFF_METRICS,
    _md_table,
    _setup_fonts,
    category_display,
    dim_label,
    metric_label,
    metric_labels,
)
from membench.ratings import format_metric, format_metric_delta, format_metric_value, overall_rating


@dataclass
class ArmDatasetResult:
    """A/B 两组下某数据集的指标：包含 baseline 与 candidate 两侧 metrics。

    Attributes:
        spec: 数据集槽位。
        baseline: 基线 run 的 metrics.json。
        candidate: 候选 run 的 metrics.json。
        baseline_run_dir: 基线 run 目录。
        candidate_run_dir: 候选 run 目录。
    """

    spec: DatasetSpec
    baseline: dict[str, Any]
    candidate: dict[str, Any]
    baseline_run_dir: str
    candidate_run_dir: str
    capabilities: dict[str, bool] = field(default_factory=dict)


def _rescan_capabilities(run_dir: str, spec_path: str) -> tuple[dict[str, bool], str]:
    """重算 capabilities：数据集扫描（准确）→ results.jsonl 反推（近似）→ 缺失。

    Returns:
        ``(capabilities, source)``，source ∈ ``dataset | results_jsonl | missing``。
    """
    try:
        return scan_capabilities(spec_path), "dataset"
    except (OSError, ValueError):
        pass

    jsonl_path = os.path.join(run_dir, "results.jsonl")
    if os.path.isfile(jsonl_path):
        try:
            has_evidence = False
            has_stale = False
            has_abstention = False
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    # results.jsonl 不保留题目原始 evidence / gold_stale 标注，
                    # 只能近似：evidence_hit 命中过任何一条 → 有 evidence 题；
                    # knowledge_update 题存在 → 视为有 stale 标注；gold 哨兵或
                    # category=abstention → 有拒答题。
                    if rec.get("evidence_hit"):
                        has_evidence = True
                    if str(rec.get("category", "")) == "knowledge_update":
                        has_stale = True
                    if is_abstention(
                        str(rec.get("category", "")), str(rec.get("gold_answer", ""))
                    ):
                        has_abstention = True
            return (
                {
                    "has_evidence": has_evidence,
                    "has_stale": has_stale,
                    "has_abstention": has_abstention,
                },
                "results_jsonl",
            )
        except (OSError, json.JSONDecodeError):
            pass

    return (
        {"has_evidence": False, "has_stale": False, "has_abstention": False},
        "missing",
    )


def _resolve_capabilities(
    metrics: dict[str, Any],
    run_dir: str,
    spec_path: str,
) -> tuple[dict[str, bool], str]:
    """解析某侧数据集的能力标注，优先读 metrics.json 顶层落账字段。

    旧版 run 的 metrics.json 顶层没有 has_evidence / has_stale / has_abstention
    （或全 False 占位），直接读会把 evidence_recall 等三项判成"未定义 / 参与集
    （无）"，与真实融合值不一致；此函数按"落账优先、缺失/全 False 时重算核对"
    的顺序回退。重算结果与落账一致时仍按 ``"metrics"`` 标注，避免报告噪音。

    Returns:
        ``(capabilities, source)``，source ∈
        ``metrics | dataset | results_jsonl | missing``（用于报告标注来源）。
    """
    keys = ("has_evidence", "has_stale", "has_abstention")
    stored = {key: metrics.get(key) for key in keys}
    complete = all(isinstance(v, bool) for v in stored.values())
    if complete and any(stored.values()):
        return dict(stored), "metrics"  # type: ignore[arg-type]

    caps, source = _rescan_capabilities(run_dir, spec_path)
    if complete and caps == stored:
        return dict(stored), "metrics"  # type: ignore[arg-type]
    return caps, source


def _arm_fuse(
    results: list[ArmDatasetResult], side: str
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """对一侧（baseline / candidate）的各数据集结果做宏平均融合。

    Args:
        results: 各数据集的 ArmDatasetResult。
        side: ``"baseline"`` 或 ``"candidate"``。

    Returns:
        ``(fused, sources)``：fused 为 ``{metric: {"value", "participants"}}``
        （结构与 :func:`membench.evaluate.fuse_metrics` 一致）；sources 为
        ``{数据集 key: capabilities 来源}``（metrics / dataset / results_jsonl /
        missing），供报告标注"未落账已自动推断"。
    """
    # 直接复用 fuse_metrics：构造一组 DatasetResult-like 即可。
    fake_results: list[DatasetResult] = []
    sources: dict[str, str] = {}
    for r in results:
        metrics = r.baseline if side == "baseline" else r.candidate
        run_dir = r.baseline_run_dir if side == "baseline" else r.candidate_run_dir
        caps, src = _resolve_capabilities(metrics, run_dir, r.spec.path)
        sources[r.spec.key] = src
        fake_results.append(
            DatasetResult(
                spec=r.spec,
                run_dir=run_dir,
                metrics=metrics,
                capabilities=caps,
            )
        )
    return fuse_metrics(fake_results), sources


def _per_dataset_diff(results: list[ArmDatasetResult]) -> list[dict[str, Any]]:
    """逐数据集 diff：candidate.overall - baseline.overall + 门禁。"""
    rows: list[dict[str, Any]] = []
    for r in results:
        a = r.baseline.get("overall", {})
        b = r.candidate.get("overall", {})
        diffs = _diff_overall(a, b)
        rows.append({"spec": r.spec, "diffs": diffs})
    return rows


def _fused_diff(
    fused_a: dict[str, dict[str, Any]], fused_b: dict[str, dict[str, Any]]
) -> dict[str, dict[str, float]]:
    """两组融合值的逐指标 diff（b - a）；任一侧为 None（未计算）时跳过该指标。"""
    out: dict[str, dict[str, float]] = {}
    for metric in fused_a:
        va = fused_a[metric].get("value")
        vb = fused_b[metric].get("value")
        if va is None or vb is None:
            continue
        out[metric] = {"a": float(va), "b": float(vb), "diff": float(vb) - float(va)}
    return out


def _fused_gates(
    fused_a: dict[str, dict[str, Any]],
    fused_b: dict[str, dict[str, Any]],
    acc_threshold: float,
    stale_threshold: float,
    latency_warn: float,
) -> list[dict[str, Any]]:
    """两组融合值的门禁评估（与 compare 口径一致）。"""
    a_overall = {k: v["value"] for k, v in fused_a.items() if v.get("value") is not None}
    b_overall = {k: v["value"] for k, v in fused_b.items() if v.get("value") is not None}
    if not a_overall or not b_overall:
        return []
    return _eval_gates(a_overall, b_overall, acc_threshold, stale_threshold, latency_warn)


def _per_dataset_gates(
    arms: list[ArmDatasetResult],
    acc_threshold: float,
    stale_threshold: float,
    latency_warn: float,
) -> list[dict[str, Any]]:
    """逐数据集门禁：单数据集 spike 不被融合平均抹平。

    返回 ``[{dataset, verdict, gates}, ...]``，verdict 由该数据集内 gates
    决定（任一 FAIL → FAIL；任一 WARN → WARN；否则 PASS）。
    """
    rows: list[dict[str, Any]] = []
    for arm in arms:
        a_overall = arm.baseline.get("overall", {})
        b_overall = arm.candidate.get("overall", {})
        gates = _eval_gates(a_overall, b_overall, acc_threshold, stale_threshold, latency_warn)
        verdict = "PASS"
        if any(g["status"] == "FAIL" for g in gates):
            verdict = "FAIL"
        elif any(g["status"] == "WARN" for g in gates):
            verdict = "WARN"
        elif not any(g["status"] == "PASS" for g in gates):
            verdict = "SKIP"
        rows.append({"spec": arm.spec, "verdict": verdict, "gates": gates})
    return rows


# ---------------------------------------------------------------------------
# 图表：每数据集 diff / 融合 diff
# ---------------------------------------------------------------------------


def _fig_per_dataset_diff(
    per_dataset: list[dict[str, Any]], out_path: str, metric: str = "accuracy"
) -> str:
    """逐数据集关键指标 diff（B - A）柱状图。"""
    _setup_fonts()
    names: list[str] = []
    diffs: list[float] = []
    for row in per_dataset:
        d = row["diffs"].get(metric, {})
        diff = float(d.get("diff", 0.0)) if "diff" in d else 0.0
        names.append(row["spec"].name)
        diffs.append(diff)

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(names)), 4.5))
    colors = [PALETTE[3] if abs(d) < 0.02 else (PALETTE[2] if d > 0 else PALETTE[4]) for d in diffs]
    ax.bar(names, diffs, color=colors)
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_ylabel(f"{metric_label(metric, with_key=False)} diff (B - A)")
    ax.set_title(f"各数据集 {metric_label(metric, with_key=False)} 差值 (B - A)")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    for i, d in enumerate(diffs):
        ax.text(i, d + (0.005 if d >= 0 else -0.01), f"{d:+.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _fig_fused_diff(
    fused_diffs: dict[str, dict[str, float]], out_path: str
) -> str:
    """融合 diff（B - A）柱状图。"""
    _setup_fonts()
    keys = list(fused_diffs.keys())
    vals = [float(fused_diffs[k]["diff"]) for k in keys]

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(keys)), 4.5))
    colors = [PALETTE[3] if abs(v) < 0.02 else (PALETTE[2] if v > 0 else PALETTE[4]) for v in vals]
    ax.bar([metric_label(k, with_key=False) for k in keys], vals, color=colors)
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_ylabel("diff (B - A)")
    ax.set_title("融合指标差值（B - A）")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    for i, v in enumerate(vals):
        ax.text(i, v + (0.005 if v >= 0 else -0.01), f"{v:+.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # matplotlib 为可选依赖
    plt = None  # type: ignore[assignment]


def _check_plt() -> None:
    if plt is None:
        raise RuntimeError("报告图表需要 matplotlib，请运行: pip install matplotlib")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def evaluate_compare(
    baseline_adapter: Any,
    candidate_adapter: Any,
    judge: Any,
    out_dir: Union[str, None] = None,
    platform: Union[str, None] = None,
    commit: Union[str, None] = None,
    quick: bool = False,
    beam_10m: bool = False,
    include_lme_s: bool = False,
    report_out: Union[str, None] = None,
    tag: Union[str, None] = None,
    specs: Union[list[DatasetSpec], None] = None,
    jobs: int = 1,
    driver_llm_baseline: Union[str, None] = None,
    driver_llm_candidate: Union[str, None] = None,
    adapter_version_baseline: Union[str, None] = None,
    adapter_version_candidate: Union[str, None] = None,
    acc_threshold: float = 0.02,
    stale_threshold: float = 0.02,
    latency_warn: float = 0.30,
    evidence_matcher: Any = None,
) -> dict[str, Any]:
    """两组（baseline vs candidate）× 多数据集融合 A/B 对比。

    Args:
        baseline_adapter: 基线适配器实例。
        candidate_adapter: 候选适配器实例。
        judge: judge 名字符串或 Judge 实例（与 Runner 约定一致）。
        out_dir: run 输出父目录；None 时用 :func:`membench.paths.runs_dir`。
        platform / commit: 平台与 commit 标签（写到两侧 run 元数据）。
        quick / beam_10m / include_lme_s: 同 :func:`membench.evaluate.evaluate`。
        report_out: 实验目录父目录；缺省 :func:`membench.paths.output_root`。
        tag: A/B 实验目录标签；缺省由两个 adapter 名拼成 ``<baseline>_vs_<candidate>``。
        specs: 显式数据集清单（默认 None 时按 quick/beam_10m/include_lme_s 调 plan_datasets）。
        jobs: 透传给 Runner 的题目级并发参数。
        driver_llm_baseline / driver_llm_candidate: 两侧各自的 driver LLM 标识。
        adapter_version_baseline / adapter_version_candidate: 两侧各自的 adapter 版本。
        acc_threshold: accuracy 允许最大下降（融合 diff 维度）。
        stale_threshold: stale_rate 允许最大上升（融合 diff 维度）。
        latency_warn: latency_p95 相对上升告警比例（融合 diff 维度）。

    Returns:
        ``{"compare_dir", "verdict", "gates", "fused_diffs", "per_dataset",
        "fused_baseline", "fused_candidate"}``。
    """
    if out_dir is None:
        out_dir = paths.runs_dir()
    if specs is None:
        specs = plan_datasets(quick=quick, beam_10m=beam_10m, include_lme_s=include_lme_s)

    judge_name = judge if isinstance(judge, str) else str(getattr(judge, "name", judge.__class__.__name__))
    baseline_name = str(getattr(baseline_adapter, "name", baseline_adapter.__class__.__name__))
    candidate_name = str(getattr(candidate_adapter, "name", candidate_adapter.__class__.__name__))

    if judge_name == "heuristic" and any(s.key.startswith("beam") for s in specs):
        print(
            "[membench] 警告: judge=heuristic 且评测清单含 BEAM；"
            "heuristic 判分可能失真，建议使用 --judge llm 或 --judge client。"
        )

    # 两侧各跑一次 Runner：用 evaluate.run_datasets 跑 baseline，临时复制 adapter
    # 接口（baseline 与 candidate 类型不同但 run_datasets 不感知）。
    print(f"[membench] ===== 基线 adapter: {baseline_name} =====")
    baseline_results = run_datasets(
        baseline_adapter,
        judge,
        specs,
        out_dir,
        platform=platform,
        commit=commit,
        jobs=jobs,
        driver_llm=driver_llm_baseline,
        adapter_version=adapter_version_baseline,
        evidence_matcher=evidence_matcher,
    )
    print(f"[membench] ===== 候选 adapter: {candidate_name} =====")
    candidate_results = run_datasets(
        candidate_adapter,
        judge,
        specs,
        out_dir,
        platform=platform,
        commit=commit,
        jobs=jobs,
        driver_llm=driver_llm_candidate,
        adapter_version=adapter_version_candidate,
        evidence_matcher=evidence_matcher,
    )

    # 对齐 baseline / candidate 各数据集（按 spec.key 匹配）
    baseline_by_key = {r.spec.key: r for r in baseline_results}
    candidate_by_key = {r.spec.key: r for r in candidate_results}
    keys = [s.key for s in specs if s.key in baseline_by_key and s.key in candidate_by_key]
    arms: list[ArmDatasetResult] = []
    for k in keys:
        b = baseline_by_key[k]
        c = candidate_by_key[k]
        arms.append(
            ArmDatasetResult(
                spec=b.spec,
                baseline=b.metrics,
                candidate=c.metrics,
                baseline_run_dir=b.run_dir,
                candidate_run_dir=c.run_dir,
                capabilities=b.capabilities,
            )
        )

    # 逐数据集 diff
    per_dataset = _per_dataset_diff(arms)

    # 两组融合（sources 记录 capabilities 来源，供报告标注）
    fused_baseline, cap_sources_a = _arm_fuse(arms, "baseline")
    fused_candidate, cap_sources_b = _arm_fuse(arms, "candidate")
    fused_diffs = _fused_diff(fused_baseline, fused_candidate)
    gates = _fused_gates(
        fused_baseline, fused_candidate, acc_threshold, stale_threshold, latency_warn
    )

    # per-dataset 门禁：单数据集 spike 不被融合平均抹平
    per_dataset_gates = _per_dataset_gates(
        arms, acc_threshold, stale_threshold, latency_warn
    )

    if any(g["status"] == "FAIL" for g in gates):
        verdict = "FAIL"
    elif any(g["status"] == "WARN" for g in gates):
        verdict = "WARN"
    else:
        verdict = "PASS"

    # per-dataset worst verdict 用于报告并列呈现（不改变 fused verdict）。
    per_dataset_verdict = "PASS"
    if any(row["verdict"] == "FAIL" for row in per_dataset_gates):
        per_dataset_verdict = "FAIL"
    elif any(row["verdict"] == "WARN" for row in per_dataset_gates):
        per_dataset_verdict = "WARN"
    elif per_dataset_gates and all(
        row["verdict"] == "SKIP" for row in per_dataset_gates
    ):
        per_dataset_verdict = "SKIP"

    # 产物落 <root>/ab_<tag>_<ts>/
    now = datetime.now()
    base = report_out or paths.output_root()
    ab_dir = layout.experiment_path(
        base, layout.ab_subdir(tag or f"{baseline_name}_vs_{candidate_name}", layout.ts_now(now))
    )

    # 两侧 run 副本就近归属
    for arm_label, results_iter in (
        ("A", baseline_results),
        ("B", candidate_results),
    ):
        arm_dir = os.path.join(ab_dir, arm_label)
        os.makedirs(arm_dir, exist_ok=True)
        run_dirs = [r.run_dir for r in results_iter]
        labels = [f"{arm_label}_{r.spec.key}" for r in results_iter]
        layout.copy_runs_into(os.path.join(arm_dir, "runs"), run_dirs, labels=labels)

    # 报告 + 图表
    _write_evaluate_compare_report(
        ab_dir,
        baseline_name,
        candidate_name,
        judge_name,
        platform,
        commit,
        now,
        arms,
        per_dataset,
        fused_baseline,
        fused_candidate,
        fused_diffs,
        gates,
        verdict,
        per_dataset_gates=per_dataset_gates,
        per_dataset_verdict=per_dataset_verdict,
        cap_sources={"baseline": cap_sources_a, "candidate": cap_sources_b},
    )

    _check_plt()
    _fig_per_dataset_diff(per_dataset, os.path.join(ab_dir, "fig_per_dataset_diff.png"))
    _fig_fused_diff(fused_diffs, os.path.join(ab_dir, "fig_fused_diff.png"))

    print(f"[membench] evaluate-compare verdict: {verdict}")
    for gate in gates:
        print(f"[membench]   gate {gate['name']}: {gate['status']} ({gate['detail']})")
    print(f"[membench] report: {os.path.join(ab_dir, 'compare_report.md')}")

    return {
        "compare_dir": ab_dir,
        "verdict": verdict,
        "gates": gates,
        "fused_diffs": fused_diffs,
        "fused_baseline": fused_baseline,
        "fused_candidate": fused_candidate,
        "per_dataset": per_dataset,
    }


def _write_evaluate_compare_report(
    ab_dir: str,
    baseline_name: str,
    candidate_name: str,
    judge_name: str,
    platform: Union[str, None],
    commit: Union[str, None],
    timestamp: datetime,
    arms: list[ArmDatasetResult],
    per_dataset: list[dict[str, Any]],
    fused_baseline: dict[str, dict[str, Any]],
    fused_candidate: dict[str, dict[str, Any]],
    fused_diffs: dict[str, dict[str, float]],
    gates: list[dict[str, Any]],
    verdict: str,
    per_dataset_gates: list[dict[str, Any]] | None = None,
    per_dataset_verdict: str | None = None,
    cap_sources: dict[str, dict[str, str]] | None = None,
) -> str:
    """生成 evaluate-compare 的 compare_report.md（中文）。"""
    lines: list[str] = []
    lines.append(f"# membench 两组多数据集对比报告 — {baseline_name} vs {candidate_name}")
    lines.append("")
    verdict_icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(verdict, "")
    lines.append(f"**门禁结论： {verdict} {verdict_icon}**")
    if per_dataset_verdict and per_dataset_verdict != verdict:
        pd_icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "SKIP": "⏭️"}.get(per_dataset_verdict, "")
        lines.append(
            f"**per-dataset 门禁最坏结论： {per_dataset_verdict} {pd_icon}**"
            "（单数据集 spike 不被融合平均抹平）"
        )
    lines.append("")
    lines.append(f"- 基线 A: `{baseline_name}`")
    lines.append(f"- 候选 B: `{candidate_name}`")
    lines.append(f"- judge: `{judge_name}`")
    lines.append(f"- platform: `{platform or 'default'}`")
    lines.append(f"- commit: `{commit or 'n/a'}`")
    lines.append(f"- 数据集数: {len(arms)}（{', '.join(r.spec.name for r in arms)}）")
    match_modes = sorted(
        {
            str(metrics_dict.get("retrieval_match") or "substring")
            for arm in arms
            for metrics_dict in (arm.baseline, arm.candidate)
        }
    )
    lines.append(f"- 检索匹配口径: `{', '.join(match_modes)}`")
    if len(match_modes) > 1:
        lines.append(
            "  ⚠️ 两侧检索匹配口径不一致，检索类指标（证据召回率 / 检索召回率@5）"
            "不可直接比较。"
        )
    lines.append(f"- 时间戳: {timestamp.isoformat(timespec='seconds')}")
    lines.append("")

    # capabilities 来源标注：任一侧有推断值时提示，避免"未定义/未落账"混淆。
    if cap_sources:
        parts: list[str] = []
        for side_key, side_label in (("baseline", "A"), ("candidate", "B")):
            inferred = [
                f"{k}:{v}"
                for k, v in (cap_sources.get(side_key) or {}).items()
                if v != "metrics"
            ]
            if inferred:
                parts.append(f"{side_label} 侧 " + "、".join(inferred))
        if parts:
            lines.append(
                "<sub>capabilities 未落账，已自动推断（"
                + "；".join(parts)
                + "；dataset=重扫数据集，results_jsonl=按逐题记录反推，missing=不可得按全 False）。</sub>"
            )
            lines.append("")

    # 1. 逐数据集 diff（关键指标）
    lines.append("## 逐数据集关键指标 diff（B - A）")
    lines.append("")
    diff_keys = [
        "accuracy",
        "token_f1",
        "bleu1",
        "evidence_recall",
        "abstention_accuracy",
        "stale_rate",
        "recall_at_5",
        "over_refusal_rate",
    ]
    header = ["数据集", *metric_labels(diff_keys)]
    rows: list[list[str]] = []
    for row in per_dataset:
        spec = row["spec"]
        diffs = row["diffs"]
        cells: list[str] = [spec.name]
        for k in diff_keys:
            d = diffs.get(k)
            cells.append(format_metric_delta(k, d["diff"]) if d else "—")
        rows.append(cells)
    lines.append(_md_table(header, rows or [["(无)", *[""] * len(diff_keys)]]))
    lines.append("")
    # 检索指标全 0 警示：疑似 evidence 与被检索文本格式不一致（摘要式存储）
    zero_sets: list[str] = []
    for arm in arms:
        for metrics_dict, run_dir in (
            (arm.baseline, arm.baseline_run_dir),
            (arm.candidate, arm.candidate_run_dir),
        ):
            ov = metrics_dict.get("overall", {}) or {}
            if ov.get("evidence_recall") != 0.0 or ov.get("recall_at_5") != 0.0:
                continue
            has_evidence = metrics_dict.get("has_evidence")
            if not isinstance(has_evidence, bool):
                has_evidence = _resolve_capabilities(
                    metrics_dict, run_dir, arm.spec.path
                )[0].get("has_evidence", False)
            if has_evidence and arm.spec.name not in zero_sets:
                zero_sets.append(arm.spec.name)
    if zero_sets:
        lines.append(
            "<sub>⚠️ "
            + "、".join(zero_sets)
            + " 标注了 evidence，但两项检索指标全为 0：疑似 evidence 与被检索文本"
            "格式不一致（如摘要式存储），并非真实零召回；可尝试 "
            "`--evidence-match overlap|llm` 调整口径，或让适配器返回原文片段后重跑。</sub>"
        )
        lines.append("")
    lines.append("![逐数据集准确率差值](fig_per_dataset_diff.png)")
    lines.append("")

    # 1.5 逐数据集 by_category diff：每个数据集按其实际出现的 raw category
    # 集合分别画 A vs B 准确率与 Δ；不跨数据集合并。
    lines.append("## 逐数据集题型 diff（B - A）")
    lines.append("")
    counts = "、".join(
        f"{arm.spec.name} {len(set(arm.baseline.get('by_category', {})) | set(arm.candidate.get('by_category', {})))} 类"
        for arm in arms
        if arm.baseline.get("by_category") or arm.candidate.get("by_category")
    )
    lines.append(
        "<sub>每个数据集按各自的 raw category 列出 A/B 准确率与差值"
        + (f"（本报告实际出现：{counts}）" if counts else "")
        + "；题型集合按数据集设计意图保留，不跨数据集强行合并。</sub>"
    )
    lines.append("")
    for arm in arms:
        a_by_cat = arm.baseline.get("by_category", {})
        b_by_cat = arm.candidate.get("by_category", {})
        if not a_by_cat and not b_by_cat:
            continue
        all_cats = sorted(set(a_by_cat) | set(b_by_cat))
        header = ["题型（category）", "n", "A 准确率", "B 准确率", "Δ"]
        rows_bc: list[list[str]] = []
        for cat in all_cats:
            a_entry = a_by_cat.get(cat, {})
            b_entry = b_by_cat.get(cat, {})
            n_a = int(a_entry.get("n", 0))
            n_b = int(b_entry.get("n", 0))
            n = n_a or n_b
            acc_a = float(a_entry.get("accuracy", 0.0)) if n_a else None
            acc_b = float(b_entry.get("accuracy", 0.0)) if n_b else None
            if acc_a is None and acc_b is None:
                continue
            if acc_a is None:
                diff_str = "—"
                a_str = "—"
                b_str = f"{acc_b * 100:.1f}%" if acc_b is not None else "—"
            elif acc_b is None:
                a_str = f"{acc_a * 100:.1f}%"
                b_str = "—"
                diff_str = "—"
            else:
                diff_str = f"{(acc_b - acc_a) * 100:+.1f}pt"
                a_str = f"{acc_a * 100:.1f}%"
                b_str = f"{acc_b * 100:.1f}%"
            rows_bc.append([category_display(cat), str(n), a_str, b_str, diff_str])
        if not rows_bc:
            continue
        lines.append(f"### {arm.spec.name}")
        lines.append("")
        lines.append(_md_table(header, rows_bc))
        lines.append("")

    # 1.6 效率成本 diff：逐数据集 A/B/Δ（Δ 为相对变化，与门禁阈值口径一致）
    lines.append("## 效率成本 diff（A → B）")
    lines.append("")
    lines.append(
        "<sub>逐数据集对照；Δ 为相对变化（B−A）/A，与 latency_p95 门禁的阈值口径"
        "一致（基线为 0 或缺失时不给 Δ；延迟单位 ms、Token 为每题均值）。</sub>"
    )
    lines.append("")
    for arm in arms:
        a_ov = arm.baseline.get("overall", {})
        b_ov = arm.candidate.get("overall", {})
        rows_eff: list[list[str]] = []
        for metric in EFFICIENCY_DIFF_METRICS:
            a_val = a_ov.get(metric)
            b_val = b_ov.get(metric)
            a_str = f"{float(a_val):.3f}" if isinstance(a_val, (int, float)) else "—"
            b_str = f"{float(b_val):.3f}" if isinstance(b_val, (int, float)) else "—"
            if (
                isinstance(a_val, (int, float))
                and isinstance(b_val, (int, float))
                and a_val
            ):
                delta = f"{(b_val - a_val) / a_val:+.1%}"
            else:
                delta = "—"
            rows_eff.append([metric_label(metric), a_str, b_str, delta])
        if not rows_eff:
            continue
        lines.append(f"### {arm.spec.name}")
        lines.append("")
        lines.append(_md_table(["指标", "A", "B", "Δ"], rows_eff))
        lines.append("")

    # 2. 融合 diff
    lines.append("## 跨数据集融合 diff（B - A）")
    lines.append("")
    fused_rows: list[list[str]] = []
    for k, d in fused_diffs.items():
        fused_rows.append(
            [
                metric_label(k),
                format_metric_value(k, d["a"]),
                format_metric_value(k, d["b"]),
                format_metric_delta(k, d["diff"]),
            ]
        )
    if fused_rows:
        lines.append(_md_table(["指标", "A 融合值", "B 融合值", "diff"], fused_rows))
    else:
        lines.append("（两组均无可参与融合的质量指标，跳过）")
    lines.append("")
    lines.append("![融合指标差值](fig_fused_diff.png)")
    lines.append("")

    # 3. 门禁明细
    lines.append("## 门禁明细")
    lines.append("")
    if gates:
        gate_rows: list[list[str]] = []
        for g in gates:
            icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "SKIP": "⏭️"}.get(str(g.get("status", "")), "")
            gate_rows.append([dim_label(str(g.get("name", ""))), str(g.get("status", "")), icon, str(g.get("detail", ""))])
        lines.append(_md_table(["门禁", "状态", "", "说明"], gate_rows))
    else:
        lines.append("（无可评估的门禁项——任一侧融合值缺失）")
    lines.append("")

    # 4. 两组融合指标完整表（含参与集与差值）
    lines.append("## 两组融合指标对照")
    lines.append("")
    fused_full_rows: list[list[str]] = []
    for metric in [
        "accuracy",
        "token_f1",
        "bleu1",
        "evidence_recall",
        "abstention_accuracy",
        "stale_rate",
        "recall_at_5",
        "over_refusal_rate",
    ]:
        a_entry = fused_baseline.get(metric, {})
        b_entry = fused_candidate.get(metric, {})
        a_val = a_entry.get("value")
        b_val = b_entry.get("value")
        a_str = format_metric(metric, a_val) if a_val is not None else "未定义"
        b_str = format_metric(metric, b_val) if b_val is not None else "未定义"
        if a_val is None or b_val is None:
            delta_str = "—"
        else:
            delta_str = format_metric_delta(metric, b_val - a_val)
        fused_full_rows.append(
            [
                metric_label(metric),
                a_str,
                b_str,
                delta_str,
                "、".join(a_entry.get("participants", [])) or "（无）",
                "、".join(b_entry.get("participants", [])) or "（无）",
            ]
        )
    lines.append(
        _md_table(
            ["指标", "A 融合", "B 融合", "Δ（B - A）", "A 参与集", "B 参与集"],
            fused_full_rows,
        )
    )
    lines.append("")
    lines.append(
        "<sub>检索召回率@5 与过度拒答率为诊断指标（展示与归因用），"
        "不参与综合评级与门禁。</sub>"
    )
    lines.append("")

    # 4.5 per-dataset 门禁明细：单数据集 spike 不被融合抹平
    if per_dataset_gates:
        lines.append("## per-dataset 门禁明细")
        lines.append("")
        lines.append(
            "<sub>每个数据集各自的 accuracy_regression / stale_rate_regression /"
            " latency_p95_regression 判定。融合 verdict 通过时，单数据集仍可能"
            " 出现 WARN/FAIL（融合平均会抹平 spike）。</sub>"
        )
        lines.append("")
        rows_pd: list[list[str]] = []
        for row in per_dataset_gates:
            spec = row["spec"]
            verdict_pd = row["verdict"]
            icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "SKIP": "⏭️"}.get(verdict_pd, "")
            fails = [g for g in row["gates"] if g["status"] != "PASS"]
            detail = "; ".join(f"{g['name']}: {g['detail']}" for g in fails) or "全部通过"
            rows_pd.append([spec.name, verdict_pd, icon, detail])
        lines.append(_md_table(["数据集", "verdict", "", "触发明细"], rows_pd))
        lines.append("")

    # 5. 各数据集完整指标（含参与能力标注与诊断指标）
    lines.append("## 各数据集完整指标（参考）")
    lines.append("")
    q_cols = [
        "accuracy",
        "token_f1",
        "bleu1",
        "evidence_recall",
        "abstention_accuracy",
        "stale_rate",
        "recall_at_5",
        "over_refusal_rate",
    ]
    header_q = ["数据集", "组", *metric_labels(q_cols)]
    detail_rows: list[list[str]] = []
    for arm in arms:
        for side_label, metrics_dict in (("A", arm.baseline), ("B", arm.candidate)):
            ov = metrics_dict.get("overall", {})
            row = [arm.spec.name, side_label]
            for m in q_cols:
                row.append(format_metric_value(m, ov.get(m)))
            detail_rows.append(row)
    lines.append(_md_table(header_q, detail_rows))
    lines.append("")

    # 6. 评级
    lines.append("## 两组综合评级")
    lines.append("")
    # 评级基于两组各自的融合 overall（与 evaluate 报告口径一致）
    fused_a_overall = {k: v["value"] for k, v in fused_baseline.items() if v.get("value") is not None}
    fused_b_overall = {k: v["value"] for k, v in fused_candidate.items() if v.get("value") is not None}
    rated_a = overall_rating(fused_a_overall) if fused_a_overall else None
    rated_b = overall_rating(fused_b_overall) if fused_b_overall else None
    if rated_a:
        lines.append(f"- A 综合评级（融合）：{rated_a[0]}（{rated_a[1]:.2f}/4.0）")
    if rated_b:
        lines.append(f"- B 综合评级（融合）：{rated_b[0]}（{rated_b[1]:.2f}/4.0）")
    if rated_a is None and rated_b is None:
        lines.append("（两组融合 overall 中无可评级指标）")
    lines.append("")
    lines.append(
        "<sub>评级规则：取六项质量指标（accuracy / token_f1 / bleu1 / "
        "evidence_recall / abstention_accuracy / stale_rate）的融合值连续点数"
        "均值（1.0~4.0，阈值 3.5 / 2.5 / 1.5）；效率成本与两项诊断指标"
        "（检索召回率@5 / 过度拒答率）不参与评级。</sub>"
    )
    lines.append("")

    out_path = os.path.join(ab_dir, "compare_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path
