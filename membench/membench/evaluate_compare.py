"""evaluate_compare：双臂（baseline vs candidate）× 多数据集融合 A/B 对比。

`compare_runs` 只对比**单数据集**两个 run；`evaluate` 只跑
**单臂**多数据集融合；`matrix` 只对同数据集跨 label。两臂 × 多数据集的融合
A/B 没有现成入口——本模块补齐，产出与 ``ab_<tag>_<ts>/`` 自包含结构一致的
逐数据集 diff + 跨数据集融合 diff + 门禁 + 对比图。

流程：
    plan_datasets（复用 evaluate.plan_datasets，与 evaluate 一致的数据集清单）
    -> 对每个数据集分别用 baseline / candidate 两个 adapter 各跑一次 Runner
    -> 逐数据集 diff（精度复用 :mod:`membench.compare` 的 _diff_overall / 门禁）
    -> 双臂各自 fuse_metrics 得跨数据集宏平均融合值
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

双臂均跑同一 judge；adapter 元数据（adapter_version / commit / driver_llm）
独立传入，落到两侧 run 的 metrics.json。可比性前置校验由两侧各数据集的
dataset_fingerprint 自动覆盖（同一数据集两侧指纹必一致），故不重复。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Union

from membench import layout, paths
from membench.compare import _diff_overall, _eval_gates
from membench.evaluate import (
    DatasetResult,
    DatasetSpec,
    fuse_metrics,
    plan_datasets,
    run_datasets,
)
from membench.report import (
    PALETTE,
    _md_table,
    _setup_fonts,
    category_display,
    metric_label,
    metric_labels,
)
from membench.ratings import format_metric, overall_rating


@dataclass
class ArmDatasetResult:
    """双臂 A/B 下某数据集的指标：包含 baseline 与 candidate 两侧 metrics。

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


def _arm_fuse(results: list[ArmDatasetResult], side: str) -> dict[str, dict[str, Any]]:
    """对一侧（baseline / candidate）的各数据集结果做宏平均融合。

    Args:
        results: 各数据集的 ArmDatasetResult。
        side: ``"baseline"`` 或 ``"candidate"``。

    Returns:
        ``{metric: {"value": 融合值, "participants": [...]}}``，结构与
        :func:`membench.evaluate.fuse_metrics` 一致。
    """
    # 直接复用 fuse_metrics：构造一组 DatasetResult-like 即可。
    fake_results: list[DatasetResult] = []
    for r in results:
        metrics = r.baseline if side == "baseline" else r.candidate
        fake_results.append(
            DatasetResult(
                spec=r.spec,
                run_dir=r.baseline_run_dir if side == "baseline" else r.candidate_run_dir,
                metrics=metrics,
                capabilities=r.capabilities,
            )
        )
    return fuse_metrics(fake_results)


def _per_dataset_diff(results: list[ArmDatasetResult]) -> list[dict[str, Any]]:
    """逐数据集 diff：candidate.overall - baseline.overall + 门禁。"""
    rows: list[dict[str, Any]] = []
    for r in results:
        a = r.baseline.get("overall", {})
        b = r.candidate.get("overall", {})
        diffs = _diff_overall(a, b)
        rows.append({"spec": r.spec, "diffs": diffs, "a": a, "b": b})
    return rows


def _fused_diff(
    fused_a: dict[str, dict[str, Any]], fused_b: dict[str, dict[str, Any]]
) -> dict[str, dict[str, float]]:
    """双臂融合值的逐指标 diff（b - a）；任一侧为 None（未计算）时跳过该指标。"""
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
    """双臂融合值的门禁评估（与 compare 口径一致）。"""
    a_overall = {k: v["value"] for k, v in fused_a.items() if v.get("value") is not None}
    b_overall = {k: v["value"] for k, v in fused_b.items() if v.get("value") is not None}
    if not a_overall or not b_overall:
        return []
    return _eval_gates(a_overall, b_overall, acc_threshold, stale_threshold, latency_warn)


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
) -> dict[str, Any]:
    """双臂（baseline vs candidate）× 多数据集融合 A/B 对比。

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

    # 双臂融合
    fused_baseline = _arm_fuse(arms, "baseline")
    fused_candidate = _arm_fuse(arms, "candidate")
    fused_diffs = _fused_diff(fused_baseline, fused_candidate)
    gates = _fused_gates(
        fused_baseline, fused_candidate, acc_threshold, stale_threshold, latency_warn
    )

    if any(g["status"] == "FAIL" for g in gates):
        verdict = "FAIL"
    elif any(g["status"] == "WARN" for g in gates):
        verdict = "WARN"
    else:
        verdict = "PASS"

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
) -> str:
    """生成 evaluate-compare 的 compare_report.md（中文）。"""
    lines: list[str] = []
    lines.append(f"# membench 双臂多数据集对比报告 — {baseline_name} vs {candidate_name}")
    lines.append("")
    verdict_icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(verdict, "")
    lines.append(f"**门禁结论： {verdict} {verdict_icon}**")
    lines.append("")
    lines.append(f"- 基线 A: `{baseline_name}`")
    lines.append(f"- 候选 B: `{candidate_name}`")
    lines.append(f"- judge: `{judge_name}`")
    lines.append(f"- platform: `{platform or 'default'}`")
    lines.append(f"- commit: `{commit or 'n/a'}`")
    lines.append(f"- 数据集数: {len(arms)}（{', '.join(r.spec.name for r in arms)}）")
    lines.append(f"- 时间戳: {timestamp.isoformat(timespec='seconds')}")
    lines.append("")

    # 1. 逐数据集 diff（关键指标）
    lines.append("## 逐数据集关键指标 diff（B - A）")
    lines.append("")
    diff_keys = ["accuracy", "token_f1", "bleu1", "evidence_recall", "abstention_accuracy", "stale_rate"]
    header = ["数据集", *metric_labels(diff_keys)]
    rows: list[list[str]] = []
    for row in per_dataset:
        spec = row["spec"]
        diffs = row["diffs"]
        cells: list[str] = [spec.name]
        for k in diff_keys:
            d = diffs.get(k)
            cells.append(f"{float(d['diff']):+.3f}" if d else "—")
        rows.append(cells)
    lines.append(_md_table(header, rows or [["(无)", *[""] * len(diff_keys)]]))
    lines.append("")
    lines.append("![逐数据集准确率差值](fig_per_dataset_diff.png)")
    lines.append("")

    # 1.5 逐数据集 by_category diff：每个数据集按其 raw category 集合分别画
    # A vs B 准确率与 Δ；不同数据集的 category 互不合并
    # （BEAM 10 类、gatemem 4 类、locomo 5 类、longmemeval 6 类各自保留）。
    lines.append("## 逐数据集题型 diff（B - A）")
    lines.append("")
    lines.append(
        "<sub>每个数据集按各自的 raw category 列出 A/B 准确率与差值；"
        "category 集合按数据集设计意图保留（BEAM 10 类 / gatemem 4 类 / "
        "locomo 5 类 / longmemeval 6 类），不跨数据集强行合并。</sub>"
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
                b_str = f"{acc_b:.3f}" if acc_b is not None else "—"
            elif acc_b is None:
                a_str = f"{acc_a:.3f}"
                b_str = "—"
                diff_str = "—"
            else:
                diff_str = f"{acc_b - acc_a:+.3f}"
                a_str = f"{acc_a:.3f}"
                b_str = f"{acc_b:.3f}"
            rows_bc.append([category_display(cat), str(n), a_str, b_str, diff_str])
        if not rows_bc:
            continue
        lines.append(f"### {arm.spec.name}")
        lines.append("")
        lines.append(_md_table(header, rows_bc))
        lines.append("")

    # 2. 融合 diff
    lines.append("## 跨数据集融合 diff（B - A）")
    lines.append("")
    fused_rows: list[list[str]] = []
    for k, d in fused_diffs.items():
        fused_rows.append(
            [
                metric_label(k),
                f"{d['a']:.3f}",
                f"{d['b']:.3f}",
                f"{d['diff']:+.3f}",
            ]
        )
    if fused_rows:
        lines.append(_md_table(["指标", "A 融合值", "B 融合值", "diff"], fused_rows))
    else:
        lines.append("（双臂均无可参与融合的质量指标，跳过）")
    lines.append("")
    lines.append("![融合指标差值](fig_fused_diff.png)")
    lines.append("")

    # 3. 门禁明细
    lines.append("## 门禁明细")
    lines.append("")
    if gates:
        gate_rows: list[list[str]] = []
        for g in gates:
            icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(str(g.get("status", "")), "")
            gate_rows.append([str(g.get("name", "")), str(g.get("status", "")), icon, str(g.get("detail", ""))])
        lines.append(_md_table(["门禁", "状态", "", "说明"], gate_rows))
    else:
        lines.append("（无可评估的门禁项——任一侧融合值缺失）")
    lines.append("")

    # 4. 双臂融合指标完整表（含参与集）
    lines.append("## 双臂融合指标对照")
    lines.append("")
    fused_full_rows: list[list[str]] = []
    for metric in ["accuracy", "token_f1", "bleu1", "evidence_recall", "abstention_accuracy", "stale_rate"]:
        a_entry = fused_baseline.get(metric, {})
        b_entry = fused_candidate.get(metric, {})
        a_str = format_metric(metric, a_entry.get("value")) if a_entry.get("value") is not None else "未定义"
        b_str = format_metric(metric, b_entry.get("value")) if b_entry.get("value") is not None else "未定义"
        fused_full_rows.append(
            [
                metric_label(metric),
                a_str,
                b_str,
                "、".join(a_entry.get("participants", [])) or "（无）",
                "、".join(b_entry.get("participants", [])) or "（无）",
            ]
        )
    lines.append(
        _md_table(
            ["指标", "A 融合", "B 融合", "A 参与集", "B 参与集"],
            fused_full_rows,
        )
    )
    lines.append("")

    # 5. 各数据集完整指标（含参与能力标注）
    lines.append("## 各数据集完整指标（参考）")
    lines.append("")
    q_cols = ["accuracy", "token_f1", "bleu1", "stale_rate", "abstention_accuracy", "evidence_recall"]
    header_q = ["数据集", "臂", *metric_labels(q_cols)]
    detail_rows: list[list[str]] = []
    for arm in arms:
        for side_label, metrics_dict in (("A", arm.baseline), ("B", arm.candidate)):
            ov = metrics_dict.get("overall", {})
            row = [arm.spec.name, side_label]
            for m in q_cols:
                v = ov.get(m)
                row.append(f"{float(v):.3f}" if isinstance(v, (int, float)) and v is not None else "未定义")
            detail_rows.append(row)
    lines.append(_md_table(header_q, detail_rows))
    lines.append("")

    # 6. 评级
    lines.append("## 双臂综合评级")
    lines.append("")
    # 评级基于双臂各自的融合 overall（与 evaluate 报告口径一致）
    fused_a_overall = {k: v["value"] for k, v in fused_baseline.items() if v.get("value") is not None}
    fused_b_overall = {k: v["value"] for k, v in fused_candidate.items() if v.get("value") is not None}
    rated_a = overall_rating(fused_a_overall) if fused_a_overall else None
    rated_b = overall_rating(fused_b_overall) if fused_b_overall else None
    if rated_a:
        lines.append(f"- A 综合评级（融合）：{rated_a[0]}（{rated_a[1]:.2f}/4.0）")
    if rated_b:
        lines.append(f"- B 综合评级（融合）：{rated_b[0]}（{rated_b[1]:.2f}/4.0）")
    if rated_a is None and rated_b is None:
        lines.append("（双臂融合 overall 中无可评级指标）")
    lines.append("")

    out_path = os.path.join(ab_dir, "compare_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path
