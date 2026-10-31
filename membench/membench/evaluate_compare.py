"""evaluate_compare：两组（baseline vs candidate）× 多数据集融合 A/B 对比。

`compare_runs` 只对比**单数据集**两个 run；`evaluate` 只跑
**单组**多数据集融合；`matrix` 只对同数据集跨 label。两组 × 多数据集的融合
A/B 没有现成入口——本模块补齐：逐数据集 diff + 跨数据集融合 diff + 门禁 + 对比图。

流程：
    plan_datasets（复用 evaluate.plan_datasets，与 evaluate 一致的数据集清单）
    -> 建对比实验目录 ``<report_out>/compare_<name>_<platform>_<ts>/``，其下
       ``A/``（基线）、``B/``（候选）各是一个完整 eval 实验形态（runs/ + reports/）
       （``task_dir`` 给定时容器即该已存在的任务目录，不新建 ``compare_*``）
    -> 每个数据集分别用 baseline / candidate 两个 adapter 各跑一次 Runner
       （write_task_reports=False，直接落进该侧的 ``runs/<dataset>/``）
    -> 逐数据集 diff（精度复用 :mod:`membench.compare` 的 _diff_overall / 门禁）
    -> 两组各自 fuse_metrics 得**基准集**融合值（默认按题数加权、办公域槽位不参与）
       + 各写该侧任务级 reports/
    -> 基准集融合 diff（b - a）+ 办公域槽位的业务 red line 门禁 + 门禁
       （accuracy / stale_rate / latency_p95）
    -> 对比产物落 ``reports/``：comparison.md + fig_per_dataset_diff.png
       （各数据集关键指标 diff）+ fig_fused_diff.png（基准集融合 diff）

两侧复用的"产物目录"（``--baseline-run`` / ``--candidate-run`` 的 ``KEY=DIR``）
统一经 :func:`membench.evaluate.resolve_dataset_dir` 解析（实验目录或数据集目录）；
若某侧给的是**一个已存在的实验目录**，整树复制进该侧目录（可比性检查之后）——
该侧的 runs/ 与 reports/ 一并带过来；其余情形按目标清单补跑 / 并入数据集产物。

门禁与单数据集 compare 口径一致：
    accuracy 下降 > acc_threshold -> FAIL
    stale_rate 上升 > stale_threshold -> FAIL
    latency_p95 相对上升 > latency_warn -> WARN
    其余指标变动仅报告。

两组均跑同一 judge；adapter 元数据（adapter_version / commit / driver_llm）
独立传入，落到两侧 run 的 metrics.json。跨侧可比性按单数据集 compare 的维度表
逐数据集核对：两侧都已复用的数据集对在**开跑前**打印问题清单，跑完（含本轮补跑
的数据集）在控制台与报告「两侧可比性（逐数据集）」小节呈现——只测量差异，不改变
accuracy / stale / latency 门禁结论。
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Union

from membench import layout, runtime
from membench.compare import _diff_overall, _eval_gates, check_comparability
from membench.data.schema import is_abstention
from membench.evaluate import (
    FUSION_CHOICES,
    FUSION_WEIGHTED,
    DatasetResult,
    DatasetSpec,
    adopt_dataset_dir,
    assert_gap_matches_provided,
    assert_single_identity,
    evaluate_gate,
    expected_fresh_metadata,
    fuse_metrics,
    fusion_label,
    load_provided_result,
    metadata_inconsistencies,
    participants_with_weight,
    plan_datasets,
    run_datasets,
    scan_capabilities,
    stamp_fusion,
    write_task_reports,
)
from membench.layout import (
    DIR_ARM_A,
    DIR_ARM_B,
    FIG_FUSED_DIFF,
    FIG_PER_DATASET_DIFF,
)
from membench.report import (
    CATEGORY_MIN_N,
    PALETTE,
    EFFICIENCY_DIFF_METRICS,
    _fmt_opt,
    _md_table,
    _setup_fonts,
    category_display,
    dim_label,
    metric_label,
    metric_label_with_role,
    metric_labels,
)
from membench.ratings import format_metric, format_metric_delta, format_metric_value, overall_rating
from membench.version import version_display


@dataclass
class ArmDatasetResult:
    """A/B 两组下某数据集的指标：包含 baseline 与 candidate 两侧 metrics。

    Attributes:
        spec: 数据集槽位。
        baseline: 基线侧的 metrics.json。
        candidate: 候选侧的 metrics.json。
        baseline_dataset_dir: 基线侧数据集产物目录（``<A>/runs/<dataset>``）。
        candidate_dataset_dir: 候选侧数据集产物目录（``<B>/runs/<dataset>``）。
    """

    spec: DatasetSpec
    baseline: dict[str, Any]
    candidate: dict[str, Any]
    baseline_dataset_dir: str
    candidate_dataset_dir: str
    capabilities: dict[str, bool] = field(default_factory=dict)


#: 「提升 TOP N 题型」列多少条。
TOP_IMPROVED_CATEGORIES = 5


def _category_diff_rows(arm: ArmDatasetResult) -> list[dict[str, Any]]:
    """把某数据集的 ``by_category`` 拉平成逐题型行（A/B 准确率与题数）。

    两侧 category 取并集、按名称排序；两侧都没有准确率时跳过。某侧缺该题型时
    记 ``None``（不按 0 处理——"没跑"不是"全错"）。
    """
    a_by_cat = arm.baseline.get("by_category", {}) or {}
    b_by_cat = arm.candidate.get("by_category", {}) or {}
    rows: list[dict[str, Any]] = []
    for cat in sorted(set(a_by_cat) | set(b_by_cat)):
        a_entry = a_by_cat.get(cat, {}) or {}
        b_entry = b_by_cat.get(cat, {}) or {}
        n_a = int(a_entry.get("n", 0) or 0)
        n_b = int(b_entry.get("n", 0) or 0)
        acc_a = (
            float(a_entry["accuracy"])
            if n_a and a_entry.get("accuracy") is not None
            else None
        )
        acc_b = (
            float(b_entry["accuracy"])
            if n_b and b_entry.get("accuracy") is not None
            else None
        )
        if acc_a is None and acc_b is None:
            continue
        rows.append(
            {"category": cat, "n_a": n_a, "n_b": n_b, "acc_a": acc_a, "acc_b": acc_b}
        )
    return rows


def top_improved_categories(
    arms: "Sequence[ArmDatasetResult]", limit: Union[int, None] = None
) -> list[dict[str, Any]]:
    """按 Δ 准确率排出「提升最多」的 (数据集, 题型)。

    只收**正向**改进、且两侧题量都达到 :data:`membench.report.CATEGORY_MIN_N` 的组合——
    小样本题型的一两题翻转会制造 +50pt 的假象，那类仍留在逐数据集表里。

    Args:
        arms: 各数据集两臂结果。
        limit: 取前几条（缺省 :data:`TOP_IMPROVED_CATEGORIES`）。

    Returns:
        ``[{"dataset","category","n","acc_a","acc_b","delta"}, ...]``，按 Δ 降序；
        同 Δ 按题量降序（题量大的更可信）。
    """
    found: list[dict[str, Any]] = []
    for arm in arms:
        for row in _category_diff_rows(arm):
            if row["acc_a"] is None or row["acc_b"] is None:
                continue
            if min(row["n_a"], row["n_b"]) < CATEGORY_MIN_N:
                continue
            delta = float(row["acc_b"]) - float(row["acc_a"])
            if delta <= 0:
                continue
            found.append(
                {
                    "dataset": arm.spec.name,
                    "category": row["category"],
                    "n": min(row["n_a"], row["n_b"]),
                    "acc_a": float(row["acc_a"]),
                    "acc_b": float(row["acc_b"]),
                    "delta": delta,
                }
            )
    found.sort(key=lambda item: (-item["delta"], -item["n"]))
    return found[: (TOP_IMPROVED_CATEGORIES if limit is None else limit)]


def _rescan_capabilities(dataset_dir: str, spec_path: str) -> tuple[dict[str, bool], str]:
    """重算 capabilities：数据集扫描（准确）→ results.jsonl 反推（近似）→ 缺失。

    Returns:
        ``(capabilities, source)``，source ∈ ``dataset | results_jsonl | missing``。
    """
    try:
        return scan_capabilities(spec_path), "dataset"
    except (OSError, ValueError):
        pass

    jsonl_path = os.path.join(dataset_dir, layout.DIR_REPORT, layout.FILE_RESULTS)
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
    dataset_dir: str,
    spec_path: str,
) -> tuple[dict[str, bool], str]:
    """解析某侧数据集的能力标注，优先读 metrics.json 顶层落账字段。

    早期产物的 metrics.json 顶层没有 has_evidence / has_stale / has_abstention
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

    caps, source = _rescan_capabilities(dataset_dir, spec_path)
    if complete and caps == stored:
        return dict(stored), "metrics"  # type: ignore[arg-type]
    return caps, source


def _side_fuse(
    results: list[DatasetResult],
    fusion: str = FUSION_WEIGHTED,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """对一侧（基线 / 候选）的各数据集结果做基准集融合。

    Args:
        results: 该侧各数据集的评测结果。
        fusion: 融合口径（``weighted`` / ``equal``，两侧必须一致）。

    Returns:
        ``(fused, sources)``：fused 为
        ``{metric: {"value", "participants", "weights", "total_weight"}}``
        （结构与 :func:`membench.evaluate.fuse_metrics` 一致）；sources 为
        ``{数据集 key: capabilities 来源}``（metrics / dataset / results_jsonl /
        missing），供报告标注"未落账已自动推断"。
    """
    # 直接复用 fuse_metrics：capabilities 落账优先、缺失或全 False 时重算核对。
    resolved: list[DatasetResult] = []
    sources: dict[str, str] = {}
    for r in results:
        caps, src = _resolve_capabilities(r.metrics, r.run_dir, r.spec.path)
        sources[r.spec.key] = src
        resolved.append(
            DatasetResult(
                spec=r.spec,
                run_dir=r.run_dir,
                metrics=r.metrics,
                capabilities=caps,
            )
        )
    return fuse_metrics(resolved, fusion=fusion), sources


def _single_source_experiment(raw_runs: Union[dict[str, str], None]) -> Union[str, None]:
    """该侧复用产物是否整体来自同一个已存在的实验目录（是则返回它）。

    只有全部入参都指向同一个实验目录（含 ``runs/``）时才整树复制；入参是数据集
    目录、或来自多个目录时返回 None——改为逐个并入数据集产物，避免误判复制范围。
    """
    if not raw_runs:
        return None
    paths = {os.path.abspath(raw) for raw in raw_runs.values()}
    if len(paths) != 1:
        return None
    only = paths.pop()
    if os.path.isdir(layout.runs_root(only)) and not os.path.isfile(
        os.path.join(only, layout.DIR_REPORT, layout.FILE_METRICS)
    ):
        return only
    return None


def _stage_side_dir(
    container_dir: str,
    arm_label: str,
    provided: dict[str, DatasetResult],
    raw_runs: Union[dict[str, str], None],
) -> dict[str, DatasetResult]:
    """把该侧复用的产物摆进 ``<container_dir>/<arm_label>/``，返回重定位后的结果。

    入参同为一个已存在的实验目录时整树复制（该侧的 ``runs/`` 与 ``reports/``
    一并带过来）；其余情形逐个把数据集产物并入该侧目录的 ``runs/`` 下。产物本就
    在该侧目录内时不做复制。
    """
    side_dir = layout.arm_dir(container_dir, arm_label)
    source_exp = _single_source_experiment(raw_runs)
    staged: dict[str, DatasetResult] = {}
    if source_exp:
        shutil.copytree(source_exp, side_dir, dirs_exist_ok=True)
        print(f"[membench] {arm_label} 侧整树复制已有实验目录: {source_exp} -> {side_dir}")
        for key, res in provided.items():
            rel = os.path.relpath(res.run_dir, source_exp)
            staged[key] = DatasetResult(
                spec=res.spec,
                run_dir=os.path.abspath(os.path.join(side_dir, rel)),
                metrics=res.metrics,
                capabilities=res.capabilities,
            )
        return staged
    for key, res in provided.items():
        staged[key] = DatasetResult(
            spec=res.spec,
            run_dir=adopt_dataset_dir(side_dir, res.run_dir),
            metrics=res.metrics,
            capabilities=res.capabilities,
        )
    return staged


def _precheck_side(
    label: str,
    provided: dict[str, DatasetResult],
    missing: list[DatasetSpec],
    adapter: Any,
    judge: Any,
    driver_llm_value: Union[str, None],
    adapter_version_value: Union[str, None],
    evidence_matcher: Any,
) -> None:
    """某一侧的跑前预检：身份/口径唯一、补跑参数与已有产物一致、缺项可补跑。

    调用方必须在**建实验目录与复制产物之前**调用：预检失败时 ``report_out`` 下
    不留 ``compare_*/{A,B}/`` 半成品树（与单臂 ``evaluate`` 的预检位置对称）。

    Args:
        label: 侧标签（如 ``"基线侧"``），用于报错定位。
        provided: 该侧复用的已有产物。
        missing: 该侧需要补跑的数据集槽位。
        adapter: 该侧适配器（补跑所需）；纯复用可为 None。
        judge: judge 字符串或实例（补跑所需）；纯复用可为 None。
        driver_llm_value / adapter_version_value: 该侧的驱动 LLM / 适配器版本标注。
        evidence_matcher: 检索命中口径匹配器。

    Raises:
        SystemExit: 该侧混用多个被测对象身份或评测口径；需要补跑但缺 adapter /
            judge；或补跑参数与已有产物不一致。
    """
    if provided:
        assert_single_identity(list(provided.values()), label)
    else:
        missing_meta = [
            option
            for option, value in (
                ("--driver-llm-a/-b", driver_llm_value),
                ("--adapter-version-a/-b", adapter_version_value),
            )
            if value is None
        ]
        if missing_meta:
            print(
                f"[membench] 提示: {label}为全新评测且未标注 {' / '.join(missing_meta)}；"
                "严格可比性（compare --require-comparable）会判这些关键维度「无法核验」。"
                "建议补齐参数，省略的标注无法事后补记。"
            )
    if not missing:
        return
    if adapter is None or judge is None:
        raise SystemExit(
            f"{label} 以下数据集没有可复用的产物，补跑需要适配器与 judge: "
            + "、".join(s.key for s in missing)
        )
    assert_gap_matches_provided(
        provided,
        expected_fresh_metadata(
            adapter, judge, driver_llm_value, adapter_version_value, evidence_matcher
        ),
        label,
    )


def _per_dataset_diff(results: list[ArmDatasetResult]) -> list[dict[str, Any]]:
    """逐数据集 diff：candidate.overall - baseline.overall + 门禁。"""
    rows: list[dict[str, Any]] = []
    for r in results:
        a = r.baseline.get("overall", {})
        b = r.candidate.get("overall", {})
        diffs = _diff_overall(a, b)
        rows.append({"spec": r.spec, "diffs": diffs})
    return rows


#: 跨侧可比性清单中不呈现的维度：evaluate-compare 的契约就是两侧各固定一个被测对象
#: （A 基线 / B 候选），``adapter / adapter_version`` 跨侧不同是本实验的设计变量；
#: ``commit`` 等同属 info 级（不出清单）。仪器类维度（数据集指纹 / judge / 判分模型 /
#: driver LLM / platform / 检索口径）才决定报告能否归因。
_CROSS_SIDE_EXPECTED_DIFFS: tuple[str, ...] = ("adapter", "adapter_version")


def _cross_side_findings(
    pairs: list[tuple[DatasetSpec, dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    """逐数据集核对两侧 run 是否可比（复用单数据集 compare 的维度表与分级）。

    只测量差异、不判定两侧是否只差一个变量；被测变量（adapter / adapter_version）
    与 info 级维度（commit / 数据集路径等）不进清单——它们跨侧不同是预期内的。
    ``kind`` 区分「两侧不一致」（``mismatch``）与「关键维度缺失、无法核验」
    （``missing``，如未标注 driver LLM）；两者含义不同，报告分开计数。

    Args:
        pairs: ``[(spec, baseline_metrics, candidate_metrics)]``。

    Returns:
        每项 ``{"key", "name", "dimension", "severity", "kind", "a", "b", "detail"}``。
    """
    findings: list[dict[str, Any]] = []
    for spec, baseline_metrics, candidate_metrics in pairs:
        comparability = check_comparability(baseline_metrics, candidate_metrics)
        missing_dims = {c["dimension"] for c in comparability["unverifiable"]}
        for check in comparability["checks"]:
            if check["severity"] == "info" or check["same"]:
                continue
            if check["dimension"] in _CROSS_SIDE_EXPECTED_DIFFS:
                continue
            findings.append(
                {
                    "key": spec.key,
                    "name": spec.name,
                    "dimension": check["dimension"],
                    "severity": check["severity"],
                    "kind": "missing" if check["dimension"] in missing_dims else "mismatch",
                    "a": check.get("a"),
                    "b": check.get("b"),
                    "detail": check.get("detail", ""),
                }
            )
    return findings


def _finding_level(finding: dict[str, Any]) -> str:
    """finding 的级别显示名（``关键缺失`` 区别于 ``关键`` 不一致）。"""
    if finding.get("kind") == "missing":
        return "关键缺失"
    return "关键" if finding["severity"] == "critical" else "次要"


def _finding_counts(findings: list[dict[str, Any]]) -> tuple[int, int, int]:
    """按「关键不一致 / 关键缺失 / 次要」三类计数。"""
    critical = sum(
        1 for f in findings if f["kind"] == "mismatch" and f["severity"] == "critical"
    )
    missing = sum(1 for f in findings if f["kind"] == "missing")
    return critical, missing, len(findings) - critical - missing


def _print_cross_side_findings(findings: list[dict[str, Any]], title: str) -> None:
    """把两侧可比性问题打到控制台（跑前与跑后共用）。"""
    if not findings:
        return
    critical, missing, secondary = _finding_counts(findings)
    print(
        f"[membench] ⚠️ {title}"
        f"（关键不一致 {critical} 项 / 关键缺失 {missing} 项 / 次要 {secondary} 项）"
    )
    for f in findings:
        print(f"[membench]   - {f['name']} [{_finding_level(f)}] {f['dimension']}: {f['detail']}")


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
    baseline_adapter: Any = None,
    candidate_adapter: Any = None,
    judge: Any = None,
    platform: Union[str, None] = None,
    commit: Union[str, None] = None,
    quick: bool = False,
    beam_10m: bool = False,
    no_beam: bool = False,
    no_lme_s: bool = False,
    common: bool = False,
    report_out: Union[str, None] = None,
    tag: Union[str, None] = None,
    specs: Union[list[DatasetSpec], None] = None,
    baseline_runs: Union[dict[str, str], None] = None,
    candidate_runs: Union[dict[str, str], None] = None,
    jobs: int = 1,
    driver_llm_baseline: Union[str, None] = None,
    driver_llm_candidate: Union[str, None] = None,
    adapter_version_baseline: Union[str, None] = None,
    adapter_version_candidate: Union[str, None] = None,
    acc_threshold: float = 0.02,
    stale_threshold: float = 0.02,
    latency_warn: float = 0.30,
    evidence_matcher: Any = None,
    task_dir: Union[str, None] = None,
    snapshot: Union[bool, None] = None,
    trace_state: Union[bool, None] = None,
    abstention_recheck: Union[bool, None] = None,
    fusion: str = FUSION_WEIGHTED,
) -> dict[str, Any]:
    """两组（baseline vs candidate）× 多数据集融合 A/B 对比。

    Args:
        baseline_adapter / candidate_adapter: 两侧适配器实例；对应侧存在需要
            补跑的数据集时必需，纯复用可为 None（此时该侧名称取已有产物的元数据）。
        judge: judge 名字符串或 Judge 实例（与 Runner 约定一致）；纯复用可为 None。
        baseline_runs / candidate_runs: 汇总模式：``{数据集 key: 产物目录}``，目录
            可以是实验目录或数据集目录。目标清单中有对应产物的复用、缺失的**补跑**
            （需要该侧 adapter 与 judge 可用）；未给 ``specs`` 时目标清单即两侧
            key 的并集。某侧给的是同一个已存在的实验目录时，该侧整树复制进
            ``A/``/``B/``（可比性检查之后）。**每侧**的被测对象与评测口径
            （:data:`membench.evaluate._IDENTITY_KEYS`）在**开跑之前**（建对比
            实验目录与复制产物之前）校验：已有产物之间混用、或补跑参数与已有产物
            不一致，直接拒绝（A/B 的差异变量只应发生在两侧之间）；某侧全新评测缺
            driver_llm / adapter_version 时启动即提示（严格可比性会判无法核验）。
        platform / commit: 平台与 commit 标签（写到两侧 run 元数据）。platform
            缺省取两侧产物的 platform；两侧不同时实验目录名用 ``p1-p2``（跨平台对比）。
        quick / beam_10m / no_beam / no_lme_s / common: 同 :func:`membench.evaluate.evaluate`。
        report_out: 对比实验目录父目录；缺省 :func:`membench.paths.output_root`。
        tag: 实验目录标签；缺省由两个 adapter 名拼成 ``<baseline>_vs_<candidate>``。
        specs: 显式数据集清单（默认 None 时按 quick/beam_10m/no_beam/no_lme_s/common 调 plan_datasets）。
        jobs: 透传给 Runner 的题目级并发参数。
        driver_llm_baseline / driver_llm_candidate: 两侧各自的 driver LLM 标识。
        adapter_version_baseline / adapter_version_candidate: 两侧各自的 adapter 版本。
        acc_threshold: accuracy 允许最大下降（融合 diff 维度）。
        stale_threshold: stale_rate 允许最大上升（融合 diff 维度）。
        latency_warn: latency_p95 相对上升告警比例（融合 diff 维度）。
        evidence_matcher: 检索命中口径匹配器（默认 substring）。
        task_dir: **就地模式**：输出容器就是这个已存在的任务目录——两臂分别写
            ``<task_dir>/A`` ``<task_dir>/B``，对比报告与两张对比图写
            ``<task_dir>/reports/``，不新建 ``compare_*`` 目录。与 ``report_out`` 互斥。
        snapshot: 是否导出对话状态快照。None（缺省）= 用 Runner 默认（导出）；
            False = 关闭（``--no-snapshot``），逐侧透传到每个数据集的 Runner。
        trace_state: 失败归因（``--trace-state``），逐侧透传给 Runner：只对
            interleaved 对话生效，sequence 对话的提问时点就是末尾 ``memory_snapshot/``。
            None（缺省）= Runner 的**自动**判定（有 interleaved 对话含 safety 题时
            开启），``--no-trace-state`` 传 False 两侧同时关闭。
        fusion: 两侧共用的基准集融合口径（``weighted`` 默认 / ``equal``）。
            两侧必须同口径，否则 diff 不可归因——本函数只有一个参数，天然一致；
            口径会写进两侧各数据集与任务级 metrics（可比性维度）。

    Returns:
        ``{"compare_dir", "verdict", "gates", "fused_diffs", "per_dataset",
        "fused_baseline", "fused_candidate", "fusion"}``；``compare_dir`` 指向本次
        输出容器（新建的 ``compare_<name>_<platform>_<ts>`` 实验目录，或就地模式下的
        ``task_dir``）。
    """
    if fusion not in FUSION_CHOICES:
        raise SystemExit(
            f"未知融合口径: {fusion!r}（可选 {' / '.join(FUSION_CHOICES)}）"
        )
    if task_dir is not None:
        if report_out is not None:
            raise SystemExit(
                "evaluate-compare 的 task_dir 与 report_out 互斥：就地写进任务目录时不再给父目录"
            )
        if not os.path.isdir(task_dir):
            raise SystemExit(
                f"任务目录不存在: {task_dir}；"
                "请先运行 python -m membench.cli task init --name <NAME> --arms 2"
            )
    if (
        (baseline_runs or candidate_runs)
        and specs is None
        and (quick or beam_10m or no_beam or no_lme_s or common)
    ):
        raise SystemExit("--baseline-run/--candidate-run 与预设开关互斥")

    baseline_provided: dict[str, DatasetResult] = {}
    for key, product_dir in (baseline_runs or {}).items():
        baseline_provided[key] = load_provided_result(key, product_dir)
    candidate_provided: dict[str, DatasetResult] = {}
    for key, product_dir in (candidate_runs or {}).items():
        candidate_provided[key] = load_provided_result(key, product_dir)

    if specs is None:
        if baseline_provided or candidate_provided:
            # 纯汇总：目标清单即两侧 key 的并集（基线顺序在前）
            keys = list(baseline_provided) + [
                k for k in candidate_provided if k not in baseline_provided
            ]
            specs = [
                (baseline_provided.get(k) or candidate_provided[k]).spec for k in keys
            ]
        else:
            specs = plan_datasets(
                quick=quick, beam_10m=beam_10m, no_beam=no_beam, no_lme_s=no_lme_s, common=common
            )
    else:
        target_keys = {s.key for s in specs}
        unknown = [k for k in {**baseline_provided, **candidate_provided} if k not in target_keys]
        if unknown:
            raise SystemExit(
                f"以下复用 run 不在目标数据集清单中: {', '.join(unknown)}"
                "（请加入 --dataset 或删除对应的 --baseline-run / --candidate-run）"
            )

    # 头信息：汇总模式优先以已有 run 的元数据为准
    first_b = next(iter(baseline_provided.values())).metrics if baseline_provided else None
    first_c = next(iter(candidate_provided.values())).metrics if candidate_provided else None
    if judge is None:
        src = first_b or first_c
        if src is None:
            raise SystemExit("未提供 judge：无已有 run 可复用时必须指定 --judge")
        judge_name = str(src.get("judge") or "heuristic")
    else:
        judge_name = judge if isinstance(judge, str) else str(getattr(judge, "name", judge.__class__.__name__))
    if baseline_adapter is not None:
        baseline_name = str(getattr(baseline_adapter, "name", baseline_adapter.__class__.__name__))
    elif first_b is not None:
        baseline_name = str(first_b.get("adapter") or "baseline")
    else:
        raise SystemExit("未提供基线适配器：该侧没有可复用的 run 时必须指定 --baseline-adapter")
    if candidate_adapter is not None:
        candidate_name = str(getattr(candidate_adapter, "name", candidate_adapter.__class__.__name__))
    elif first_c is not None:
        candidate_name = str(first_c.get("adapter") or "candidate")
    else:
        raise SystemExit("未提供候选适配器：该侧没有可复用的 run 时必须指定 --candidate-adapter")
    if platform is None:
        platform = (first_b or first_c or {}).get("platform")
    if commit is None:
        commit = (first_b or first_c or {}).get("commit")
    # 两侧 platform 各自解析：缺省取该侧产物元数据；两侧不同则目录名用 p1-p2
    platform_a = str((first_b or {}).get("platform") or platform or "default")
    platform_b = str((first_c or {}).get("platform") or platform or "default")
    dir_platform = platform_a if platform_a == platform_b else f"{platform_a}-{platform_b}"

    if judge_name == "heuristic" and any(s.key.startswith("beam") for s in specs):
        print(
            "[membench] 警告: judge=heuristic 且评测清单含 BEAM；"
            "heuristic 判分可能失真，建议使用 --judge llm 或 --judge client。"
        )

    # 跑前预检：两侧都已复用的数据集对先把可比性问题摆出来（避免跑完才发现报告不可用）
    _print_cross_side_findings(
        _cross_side_findings(
            [
                (s, baseline_provided[s.key].metrics, candidate_provided[s.key].metrics)
                for s in specs
                if s.key in baseline_provided and s.key in candidate_provided
            ]
        ),
        "两侧可比性问题（两侧均复用产物，跑前检查）",
    )

    # 跑前预检：两侧的身份 / 口径与补跑参数问题必须在**建实验目录与复制产物之前**
    # 暴露——否则 report_out 下会残留 compare_*/{A,B}/ 半成品树（与单臂 evaluate 对称）。
    _precheck_side(
        "基线侧",
        baseline_provided,
        [s for s in specs if s.key not in baseline_provided],
        baseline_adapter,
        judge,
        driver_llm_baseline,
        adapter_version_baseline,
        evidence_matcher,
    )
    _precheck_side(
        "候选侧",
        candidate_provided,
        [s for s in specs if s.key not in candidate_provided],
        candidate_adapter,
        judge,
        driver_llm_candidate,
        adapter_version_candidate,
        evidence_matcher,
    )

    # 产物落 <report_out>/compare_<name>_<platform>_<ts>/；就地模式则落给定的任务目录
    now = datetime.now()
    if task_dir is not None:
        compare_dir = os.path.abspath(task_dir)
        print(f"[membench] 任务目录（就地）: {compare_dir}")
    else:
        compare_dir = layout.experiment_path(
            report_out,
            layout.compare_subdir(
                tag or f"{baseline_name}_vs_{candidate_name}", dir_platform, layout.ts_now(now)
            ),
        )
        print(f"[membench] 对比实验目录: {compare_dir}")
    # A/B 两侧落位之前登记：两侧补跑的 Runner 会把同一条命令的日志继续写在这里
    runtime.set_dir(compare_dir)

    # A/B 两侧各摆成完整 eval 实验形态（产物齐备与身份预检已在上方做完，再复制 / 补跑）
    baseline_provided = _stage_side_dir(compare_dir, DIR_ARM_A, baseline_provided, baseline_runs)
    candidate_provided = _stage_side_dir(compare_dir, DIR_ARM_B, candidate_provided, candidate_runs)

    # 两侧组装：已有产物直接复用，缺失数据集用该侧 adapter 补跑
    def _assemble_side(
        label: str,
        arm_label: str,
        adapter: Any,
        provided: dict[str, DatasetResult],
        platform_value: Union[str, None],
        driver_llm_value: Union[str, None],
        adapter_version_value: Union[str, None],
    ) -> list[DatasetResult]:
        side_dir = layout.arm_dir(compare_dir, arm_label)
        by_key = dict(provided)
        for spec in specs:
            if spec.key in provided:
                print(f"[membench] 复用已有产物（{label}）: {spec.key} <- {provided[spec.key].run_dir}")
        for r in run_datasets(
            adapter,
            judge,
            [s for s in specs if s.key not in by_key],
            side_dir,
            platform=platform_value,
            commit=commit,
            jobs=jobs,
            driver_llm=driver_llm_value,
            adapter_version=adapter_version_value,
            evidence_matcher=evidence_matcher,
            snapshot=snapshot,
            trace_state=trace_state,
            abstention_recheck=abstention_recheck,
        ):
            by_key[r.spec.key] = r
        out: list[DatasetResult] = []
        for spec in specs:
            res = by_key[spec.key]
            if spec.key in provided:
                res = DatasetResult(
                    spec=spec,
                    run_dir=res.run_dir,
                    metrics=res.metrics,
                    capabilities=res.capabilities,
                )
            out.append(res)
        return out

    print(f"[membench] ===== 基线 adapter: {baseline_name} =====")
    baseline_results = _assemble_side(
        "基线侧",
        DIR_ARM_A,
        baseline_adapter,
        baseline_provided,
        platform_a,
        driver_llm_baseline,
        adapter_version_baseline,
    )
    print(f"[membench] ===== 候选 adapter: {candidate_name} =====")
    candidate_results = _assemble_side(
        "候选侧",
        DIR_ARM_B,
        candidate_adapter,
        candidate_provided,
        platform_b,
        driver_llm_candidate,
        adapter_version_candidate,
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
                baseline_dataset_dir=b.run_dir,
                candidate_dataset_dir=c.run_dir,
                capabilities=b.capabilities,
            )
        )

    # 融合口径随产物走：两侧各数据集的 metrics.json 都写入当次口径，可比性
    # 维度（fusion）才读得到；旧产物缺该字段时归一为 legacy。
    stamp_fusion(baseline_results, fusion)
    stamp_fusion(candidate_results, fusion)

    # 两侧可比性（逐数据集）：与单数据集 compare 同一套维度表；只测量差异、不进门禁结论
    comparability_findings = _cross_side_findings(
        [(arm.spec, arm.baseline, arm.candidate) for arm in arms]
    )

    # 汇总模式的元数据一致性提示（混产物风险）；notes 直接作为报告顶部提示行
    notes: list[str] = []
    for label, side in (("基线侧", baseline_results), ("候选侧", candidate_results)):
        for note in metadata_inconsistencies(side):
            line = f"{label}内部：{note}"
            notes.append(f"> ⚠️ **汇总模式**：{line}")
            print(f"[membench] ⚠️ 汇总模式：{line}")
    if platform_a != platform_b:
        notes.append(
            f"> ⚠️ 两侧 platform 不同（A=`{platform_a}` / B=`{platform_b}`）：跨平台对比，"
            "平台差异同时计入「两侧可比性」维度。"
        )

    # 两侧可比性问题（含本轮补跑的数据集）在控制台同步提示；详情见报告同名小节
    _print_cross_side_findings(comparability_findings, "两侧可比性问题（详见报告「两侧可比性」）")

    # 逐数据集 diff
    per_dataset = _per_dataset_diff(arms)

    # 两组融合（sources 记录 capabilities 来源，供报告标注）
    fused_baseline, cap_sources_a = _side_fuse(baseline_results, fusion)
    fused_candidate, cap_sources_b = _side_fuse(candidate_results, fusion)
    fused_diffs = _fused_diff(fused_baseline, fused_candidate)
    gates = _fused_gates(
        fused_baseline, fused_candidate, acc_threshold, stale_threshold, latency_warn
    )
    # 办公域槽位不参与基准融合，但业务红线仍要进顶层门禁：把 office 移出融合
    # 不等于把红线移出结论（否则陈旧率回归会因融合值为 None 而 SKIP）。
    office_arm = next((arm for arm in arms if arm.spec.is_office_slot), None)
    if office_arm is not None:
        for gate in _eval_gates(
            office_arm.baseline.get("overall", {}),
            office_arm.candidate.get("overall", {}),
            acc_threshold,
            stale_threshold,
            latency_warn,
        ):
            gates.append(
                {
                    **gate,
                    "name": f"office_{gate['name']}",
                    "detail": f"办公域槽位（{office_arm.spec.name}）：{gate['detail']}",
                }
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

    # 产物落位：A/B 两侧各写一份任务级 reports/（该侧自身就是一个完整 eval 实验形态），
    # 对比报告与两张对比图写输出容器的 reports/（就地模式下即任务目录的）。
    _check_plt()
    for arm_label, arm_name, side_results, side_platform, side_fused in (
        (DIR_ARM_A, baseline_name, baseline_results, platform_a, fused_baseline),
        (DIR_ARM_B, candidate_name, candidate_results, platform_b, fused_candidate),
    ):
        write_task_reports(
            layout.arm_dir(compare_dir, arm_label),
            adapter_name=arm_name,
            judge_name=judge_name,
            platform=side_platform,
            commit=commit,
            timestamp=now,
            results=side_results,
            fused=side_fused,
            gate=evaluate_gate(side_results),
            fusion=fusion,
        )

    reports_dir = layout.task_report_dir(compare_dir)
    os.makedirs(reports_dir, exist_ok=True)
    _fig_per_dataset_diff(per_dataset, os.path.join(reports_dir, FIG_PER_DATASET_DIFF))
    _fig_fused_diff(fused_diffs, os.path.join(reports_dir, FIG_FUSED_DIFF))
    _write_evaluate_compare_report(
        compare_dir,
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
        comparability_findings=comparability_findings,
        cap_sources={"baseline": cap_sources_a, "candidate": cap_sources_b},
        fusion=fusion,
        extra_notes=notes,
    )

    print(f"[membench] evaluate-compare verdict: {verdict}")
    for gate in gates:
        print(f"[membench]   gate {gate['name']}: {gate['status']} ({gate['detail']})")
    print(f"[membench] comparison 报告: {os.path.join(reports_dir, layout.FILE_COMPARISON_MD)}")

    return {
        "compare_dir": compare_dir,
        "verdict": verdict,
        "gates": gates,
        "fused_diffs": fused_diffs,
        "fused_baseline": fused_baseline,
        "fused_candidate": fused_candidate,
        "per_dataset": per_dataset,
        "fusion": fusion,
    }


def _write_evaluate_compare_report(
    container_dir: str,
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
    fusion: str = FUSION_WEIGHTED,
    extra_notes: list[str] | None = None,
    comparability_findings: list[dict[str, Any]] | None = None,
) -> str:
    """生成输出容器 ``reports/`` 下的 ``comparison.md``（中文）。

    ``comparability_findings`` 非空时：抬头给一行汇总，并在「门禁明细」后附
    「两侧可比性（逐数据集）」小节（逐数据集列出关键不一致 / 关键缺失 / 次要差异，
    口径与单数据集 compare 一致）。
    """
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
    # 评测代码自身版本：两侧同一次 compare 由同一版本产出，这里只记一行
    # 供回查（判分口径随代码变，历史分数差异要靠它对账）。
    lines.append(
        "- membench 版本: "
        + version_display((arms[0].baseline if arms else {}).get("membench_version"))
    )
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
    lines.append(
        f"- 融合口径: `{fusion}`（{fusion_label(fusion)}；办公域槽位不参与基准评级）"
    )
    lines.append(f"- 时间戳: {timestamp.isoformat(timespec='seconds')}")
    findings = comparability_findings or []
    if findings:
        critical_n, missing_n, secondary_n = _finding_counts(findings)
        lines.append(
            f"- ⚠️ 两侧可比性：关键不一致 {critical_n} 项、关键缺失 {missing_n} 项、"
            f"次要 {secondary_n} 项——详见「两侧可比性（逐数据集）」小节"
        )
    else:
        lines.append("- 两侧可比性：各数据集关键维度一致且可核验")
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

    for note in extra_notes or []:
        lines.append(note)
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
        for metrics_dict, dataset_dir in (
            (arm.baseline, arm.baseline_dataset_dir),
            (arm.candidate, arm.candidate_dataset_dir),
        ):
            ov = metrics_dict.get("overall", {}) or {}
            if ov.get("evidence_recall") != 0.0 or ov.get("recall_at_5") != 0.0:
                continue
            has_evidence = metrics_dict.get("has_evidence")
            if not isinstance(has_evidence, bool):
                has_evidence = _resolve_capabilities(
                    metrics_dict, dataset_dir, arm.spec.path
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
    lines.append(f"![逐数据集准确率差值]({FIG_PER_DATASET_DIFF})")
    lines.append("")

    # 1.5 逐数据集 by_category diff：每个数据集按其实际出现的 raw category
    # 集合分别画 A vs B 准确率与 Δ；不跨数据集合并。
    lines.append("## 逐数据集题型 diff（B - A）")
    lines.append("")
    # 题型集合与类数都按本报告实际出现的 category 统计（不写死任何数据集的设计类数）
    counts = "、".join(
        f"{arm.spec.name} {len(set(arm.baseline.get('by_category', {})) | set(arm.candidate.get('by_category', {})))} 类"
        for arm in arms
        if arm.baseline.get("by_category") or arm.candidate.get("by_category")
    )
    lines.append(
        "<sub>每个数据集按各自的 raw category 列出 A/B 准确率与差值"
        + (f"（本报告实际出现：{counts}）" if counts else "（两侧均无 by_category 数据）")
        + "；题型集合按各数据集实际出现的 category 保留，不跨数据集强行合并。</sub>"
    )
    lines.append("")
    # 提升 TOP N：先给头条结论，再下钻逐数据集表
    top = top_improved_categories(arms)
    if top:
        lines.append(f"### 提升最多的题型（TOP {len(top)}）")
        lines.append("")
        lines.append(
            f"<sub>按 Δ 准确率降序；只收**两侧题量都 ≥ {CATEGORY_MIN_N}** 且正向改进的"
            "组合——小样本题型一两题翻转就能 +50pt，会盖住大样本的真实改进（它们仍在下方"
            "各表里）。同 Δ 按题量降序。</sub>"
        )
        lines.append("")
        top_rows = [
            [
                item["dataset"],
                category_display(item["category"]),
                str(item["n"]),
                f"{item['acc_a'] * 100:.1f}%",
                f"{item['acc_b'] * 100:.1f}%",
                f"{item['delta'] * 100:+.1f}pt",
            ]
            for item in top
        ]
        lines.append(
            _md_table(["数据集", "题型", "n", "A 准确率", "B 准确率", "Δ"], top_rows)
        )
        lines.append("")
    else:
        lines.append(
            f"<sub>本次没有满足条件的提升项（要求两侧题量 ≥ {CATEGORY_MIN_N} 且 Δ > 0）。</sub>"
        )
        lines.append("")
    for arm in arms:
        rows_bc: list[list[str]] = []
        for row in _category_diff_rows(arm):
            n = row["n_a"] or row["n_b"]
            acc_a, acc_b = row["acc_a"], row["acc_b"]
            if acc_a is None:
                a_str, b_str = "—", f"{acc_b * 100:.1f}%"
                diff_str = "—"
            elif acc_b is None:
                a_str, b_str, diff_str = f"{acc_a * 100:.1f}%", "—", "—"
            else:
                a_str = f"{acc_a * 100:.1f}%"
                b_str = f"{acc_b * 100:.1f}%"
                diff_str = f"{(acc_b - acc_a) * 100:+.1f}pt"
            rows_bc.append(
                [category_display(row["category"]), str(n), a_str, b_str, diff_str]
            )
        if not rows_bc:
            continue
        lines.append(f"### {arm.spec.name}")
        lines.append("")
        lines.append(_md_table(["题型（category）", "n", "A 准确率", "B 准确率", "Δ"], rows_bc))
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

    # 2. 基准集融合 diff（主结论）：办公域槽位不参与
    lines.append(f"## 基准集融合 diff（B - A，{fusion_label(fusion)}）")
    lines.append("")
    lines.append(
        "<sub>办公域槽位不参与本节：它与基准集不同源，单列在下一节「业务结论 diff」。</sub>"
    )
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
        lines.append("（两组均无可参与融合的基准集质量指标，跳过）")
    lines.append("")
    lines.append("![融合指标差值](fig_fused_diff.png)")
    lines.append("")

    # 2.5 业务结论 diff（办公域槽位单独一段）：业务红线不混进基准评级
    lines.append("## 业务结论 diff（办公域槽位，B - A）")
    lines.append("")
    office_arm = next((arm for arm in arms if arm.spec.is_office_slot), None)
    if office_arm is None:
        lines.append("（本次数据集清单无办公域槽位，业务结论与 stale 门禁跳过。）")
        lines.append("")
    else:
        a_ov = office_arm.baseline.get("overall", {}) or {}
        b_ov = office_arm.candidate.get("overall", {}) or {}
        biz_rows: list[list[str]] = []
        for key in ("accuracy", "stale_rate", "abstention_accuracy"):
            a_val = a_ov.get(key)
            b_val = b_ov.get(key)
            a_str = format_metric(key, a_val) if a_val is not None else "未定义"
            b_str = format_metric(key, b_val) if b_val is not None else "未定义"
            if a_val is None or b_val is None:
                delta_str = "—"
            else:
                delta_str = format_metric_delta(key, float(b_val) - float(a_val))
            biz_rows.append([metric_label(key), a_str, b_str, delta_str])
        lines.append(f"### {office_arm.spec.name}")
        lines.append("")
        lines.append(_md_table(["指标", "A", "B", "Δ（B - A）"], biz_rows))
        lines.append("")
        lines.append(
            "<sub>陈旧率与拒答准确率是业务红线（陈旧率只看可机检的 stale 题，"
            "覆盖率见各侧报告），不参与基准评级；门禁结论见上方「门禁明细」"
            "（准确的逐数据集判定见「per-dataset 门禁明细」）。</sub>"
        )
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

    # 3.5 两侧可比性（逐数据集）
    lines.append("## 两侧可比性（逐数据集）")
    lines.append("")
    lines.append(
        "<sub>与单数据集 `compare` 同一套维度表：关键维度（dataset_sha256 / judge / "
        "judge_model / driver_llm / platform）不一致（级别「关键」）或两侧都未标注、"
        "无法核验（级别「关键缺失」）时，该数据集的对比结果不可归因"
        "（`compare --require-comparable` 两种情况都判不可比）；次要维度（检索命中口径等）"
        "不一致时检索类指标不可直接比较。两侧 `adapter / adapter_version` 与 `commit` "
        "是被测变量，差异属预期，不计入本表。</sub>"
    )
    lines.append("")
    if findings:
        comp_rows = [
            [
                f["name"],
                dim_label(str(f["dimension"])),
                _finding_level(f),
                _fmt_opt(f.get("a")),
                _fmt_opt(f.get("b")),
                str(f.get("detail", "")),
            ]
            for f in findings
        ]
        lines.append(_md_table(["数据集", "维度", "级别", "A (基线)", "B (候选)", "说明"], comp_rows))
    else:
        lines.append("（无：各数据集两侧关键与次要维度均一致且可核验）")
    lines.append("")

    # 4. 两组融合指标完整表（含参与集、权重与差值）
    lines.append("## 两组融合指标对照（基准集）")
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
                metric_label_with_role(metric),
                a_str,
                b_str,
                delta_str,
                participants_with_weight(a_entry, fusion),
                participants_with_weight(b_entry, fusion),
            ]
        )
    lines.append(
        _md_table(
            ["指标", "A 融合", "B 融合", "Δ（B - A）", "A 参与集（题数 / 权重）", "B 参与集（题数 / 权重）"],
            fused_full_rows,
        )
    )
    lines.append("")
    lines.append(
        "<sub>token_f1 / bleu1（字面重合度，非语义正确性）与检索召回率@5 / "
        "过度拒答率为诊断指标，仅展示与归因，不参与基准评级与门禁。"
        f"融合口径：`{fusion}`（{fusion_label(fusion)}），办公域槽位不参与。</sub>"
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

    # 6. 基准评级（不含办公域槽位）
    lines.append("## 两组基准评级（不含办公域槽位）")
    lines.append("")
    # 评级基于两组各自的基准集融合 overall（与 evaluate 报告口径一致）
    fused_a_overall = {k: v["value"] for k, v in fused_baseline.items() if v.get("value") is not None}
    fused_b_overall = {k: v["value"] for k, v in fused_candidate.items() if v.get("value") is not None}
    rated_a = overall_rating(fused_a_overall) if fused_a_overall else None
    rated_b = overall_rating(fused_b_overall) if fused_b_overall else None
    if rated_a:
        lines.append(f"- A 基准评级（融合）：{rated_a[0]}（{rated_a[1]:.2f}/4.0）")
    if rated_b:
        lines.append(f"- B 基准评级（融合）：{rated_b[0]}（{rated_b[1]:.2f}/4.0）")
    if rated_a is None and rated_b is None:
        lines.append("（两组基准集融合 overall 中无可评级指标：基准集为空或指标不可评级）")
    lines.append("")
    lines.append(
        "<sub>评级规则：基准集融合值"
        f"（{fusion_label(fusion)}；办公域槽位不参与）取四项质量指标"
        "（accuracy / evidence_recall / abstention_accuracy / stale_rate）的连续点数"
        "均值（1.0~4.0，阈值 3.5 / 2.5 / 1.5）；token_f1 / bleu1"
        "（字面重合度，非语义正确性）与 recall_at_5 / over_refusal_rate"
        "为诊断指标，仅展示不参与评级；效率成本指标同样不参与评级。"
        "办公域槽位的业务口径见「业务结论 diff」。</sub>"
    )
    lines.append("")

    out_path = os.path.join(
        layout.task_report_dir(container_dir), layout.FILE_COMPARISON_MD
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path
