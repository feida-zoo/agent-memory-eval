"""compare：对比两个 run（基线 vs 候选），产出逐指标 diff 与回归门禁判定。

门禁规则（SPEC 3.6，阈值可由 CLI 覆盖）：
- accuracy 下降 > acc_threshold（默认 0.02） -> FAIL
- stale_rate 上升 > stale_threshold（默认 0.02） -> FAIL
- latency_p95 上升 > latency_warn（默认 30%） -> WARN
- 其余指标变动仅报告，不门禁。

产物布局：``<root>/ab_<tag>_<ts>/``（自包含，A/B 各自带 run 副本），由
:func:`membench.layout.materialize_ab_experiment` 落地；``<root>`` 取 ``out_dir``
或 :func:`membench.paths.output_root`。

可比性前置校验（接入侧反馈 1）：比较前先核对驱动/判分模型、适配器实现与数据集
指纹等维度；关键维度缺失或不一致时给出 ``comparability`` 门禁，避免结果不可归因
却照给 PASS。
"""

from __future__ import annotations

import json
import os
from typing import Any

from membench import layout
from membench.paths import output_root
from membench.report import render_compare_report

# 参与逐指标 diff 的 overall 指标（排除计数项）
_DIFF_KEYS: list[str] = [
    "accuracy",
    "token_f1",
    "bleu1",
    "evidence_recall",
    "abstention_accuracy",
    "stale_rate",
    "latency_p50_ms",
    "latency_p95_ms",
    "avg_input_tokens",
    "avg_llm_calls",
]

#: 可比性维度（接入侧反馈 1）：任一 critical 维度不一致即视为不可比；
#: 缺值时不记 mismatch，而记入 ``unverifiable``（strict 下同样判不可比）。
CRITICAL_COMPARABILITY_DIMS: tuple[str, ...] = (
    "adapter",
    "adapter_version",
    "dataset_sha256",
    "judge",
    "judge_model",
    "driver_llm",
    "platform",
)

#: 次要维度：不一致仅降级为 WARN 门禁，不判 FAIL。
WARN_COMPARABILITY_DIMS: tuple[str, ...] = (
    "limit",
    "shards",
    "shard_index",
    "n_questions",
    "adapter_meta",
)

#: 提示维度：仅记录（``same=False``），不影响结论与门禁。
#: ``commit`` 不同是 A/B 的被测变量，**不得**判为不可比。
INFO_COMPARABILITY_DIMS: tuple[str, ...] = ("commit", "dataset_path", "dataset")

#: 维度 -> 严重级别。
_SEVERITY_BY_DIM: dict[str, str] = {
    **{dim: "critical" for dim in CRITICAL_COMPARABILITY_DIMS},
    **{dim: "warn" for dim in WARN_COMPARABILITY_DIMS},
    **{dim: "info" for dim in INFO_COMPARABILITY_DIMS},
}

#: 全量比较维度（critical -> warn -> info，决定 checks 顺序）。
_COMPARABILITY_DIMS: tuple[str, ...] = (
    CRITICAL_COMPARABILITY_DIMS + WARN_COMPARABILITY_DIMS + INFO_COMPARABILITY_DIMS
)


def _load_metrics(run_dir: str) -> dict[str, Any]:
    """读取 run 目录下的 metrics.json。"""
    path = os.path.join(run_dir, "metrics.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"metrics.json 不存在: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _diff_overall(overall_a: dict[str, Any], overall_b: dict[str, Any]) -> dict[str, dict[str, float]]:
    """逐指标 diff（b - a）。"""
    diffs: dict[str, dict[str, float]] = {}
    for key in _DIFF_KEYS:
        if key in overall_a and key in overall_b:
            va = float(overall_a[key])
            vb = float(overall_b[key])
            diffs[key] = {"a": va, "b": vb, "diff": vb - va}
    return diffs


def _eval_gates(
    overall_a: dict[str, Any],
    overall_b: dict[str, Any],
    acc_threshold: float,
    stale_threshold: float,
    latency_warn: float,
) -> list[dict[str, Any]]:
    """按 SPEC 3.6 逐条评估门禁。"""
    gates: list[dict[str, Any]] = []

    acc_a = float(overall_a.get("accuracy", 0.0))
    acc_b = float(overall_b.get("accuracy", 0.0))
    acc_diff = acc_b - acc_a
    gates.append(
        {
            "name": "accuracy_regression",
            "status": "FAIL" if acc_diff < -acc_threshold else "PASS",
            "detail": f"accuracy diff={acc_diff:+.3f}，阈值 -{acc_threshold}",
        }
    )

    stale_a = float(overall_a.get("stale_rate", 0.0))
    stale_b = float(overall_b.get("stale_rate", 0.0))
    stale_diff = stale_b - stale_a
    gates.append(
        {
            "name": "stale_rate_regression",
            "status": "FAIL" if stale_diff > stale_threshold else "PASS",
            "detail": f"stale_rate diff={stale_diff:+.3f}，阈值 +{stale_threshold}",
        }
    )

    p95_a = float(overall_a.get("latency_p95_ms", 0.0))
    p95_b = float(overall_b.get("latency_p95_ms", 0.0))
    if p95_a > 0:
        p95_rel = (p95_b - p95_a) / p95_a
        gates.append(
            {
                "name": "latency_p95_regression",
                "status": "WARN" if p95_rel > latency_warn else "PASS",
                "detail": f"latency_p95 相对变化={p95_rel:+.1%}，告警阈值 +{latency_warn:.0%}",
            }
        )
    else:
        gates.append(
            {
                "name": "latency_p95_regression",
                "status": "PASS",
                "detail": "基线 latency_p95 为 0，跳过相对变化判定",
            }
        )
    return gates


def _comparability_value(metrics: dict[str, Any], dim: str) -> Any:
    """取某维度在 metrics 中的值（``dataset_sha256`` 走指纹子字段）。"""
    if dim == "dataset_sha256":
        fingerprint = metrics.get("dataset_fingerprint") or {}
        return fingerprint.get("sha256") if isinstance(fingerprint, dict) else None
    return metrics.get(dim)


def _is_missing(value: Any) -> bool:
    """critical 维度的「缺失」判定（None / 空串）；warn/info 维度的 None 是合法取值。"""
    return value is None or value == ""


#: 规则判分器：不存在"判分模型"这一维度。两侧同为规则判分时 judge_model 记
#: 「不适用」（same=True），而非记成无法核验——否则用 heuristic 判分的 A/B
#: 在 strict 模式下会因"判分模型缺失"无条件 FAIL。
_RULE_BASED_JUDGES: tuple[str, ...] = ("heuristic",)


def _is_not_applicable(dim: str, metrics_a: dict[str, Any], metrics_b: dict[str, Any]) -> bool:
    """判断某维度是否「不适用」（区别于元数据缺失，如规则判分无模型标识）。"""
    if dim != "judge_model":
        return False
    judge_a = metrics_a.get("judge")
    return judge_a == metrics_b.get("judge") and judge_a in _RULE_BASED_JUDGES


def _check_reason(check: dict[str, Any]) -> str:
    """把一条 check 转成可读的「不可比原因」文本。"""
    label = {"critical": "关键", "warn": "次要", "info": "提示"}.get(
        check["severity"], check["severity"]
    )
    return f"[{label}] {check['dimension']}: {check['detail']}"


def check_comparability(metrics_a: dict[str, Any], metrics_b: dict[str, Any]) -> dict[str, Any]:
    """核对两个 run 的元数据是否可比（接入侧反馈 1）。

    取值规则：``dataset_sha256`` 取 ``metrics["dataset_fingerprint"]["sha256"]``，
    其余维度取 metrics 顶层字段。

    - critical 维度任一值缺失 -> 记入 ``unverifiable``（不记 mismatch）；
      两值都在且不同 -> 记入 ``mismatches``；相同 -> ``same=True``。
      例外：两侧判分器同为规则判分（``heuristic``）时 ``judge_model`` 记
      「不适用」（``same=True``）——规则判分本就没有模型标识。
    - warn / info 维度同样比较：warn 不一致只降级 WARN 门禁，info 不一致仅记录
      （``same=False``）但不影响结论与门禁。
    - ``comparable = 无 critical mismatch``（缺值不算 mismatch）。

    Returns:
        ``{"comparable": bool, "checks": [...], "mismatches": [...],
        "unverifiable": [...], "reasons": [str]}``。
    """
    checks: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    unverifiable: list[dict[str, Any]] = []

    for dim in _COMPARABILITY_DIMS:
        severity = _SEVERITY_BY_DIM[dim]
        a = _comparability_value(metrics_a, dim)
        b = _comparability_value(metrics_b, dim)
        if severity == "critical" and _is_not_applicable(dim, metrics_a, metrics_b):
            check = {
                "dimension": dim,
                "a": a,
                "b": b,
                "same": True,
                "severity": severity,
                "detail": "两侧均未记录（规则判分无模型标识，不适用）",
            }
        elif severity == "critical" and (_is_missing(a) or _is_missing(b)):
            check = {
                "dimension": dim,
                "a": a,
                "b": b,
                "same": False,
                "severity": severity,
                "detail": f"关键维度缺失，无法核验（A={a!r}, B={b!r}）",
            }
            unverifiable.append(check)
        elif a != b:
            check = {
                "dimension": dim,
                "a": a,
                "b": b,
                "same": False,
                "severity": severity,
                "detail": f"A={a} != B={b}",
            }
            mismatches.append(check)
        else:
            check = {
                "dimension": dim,
                "a": a,
                "b": b,
                "same": True,
                "severity": severity,
                "detail": "一致",
            }
        checks.append(check)

    comparable = not any(c["severity"] == "critical" for c in mismatches)
    # 「不可比原因」只列 mismatch（非 info 级）与无法核验项；info 级差异
    # （commit / dataset_path / dataset）只在逐维度表里呈现。
    reasons = [_check_reason(c) for c in mismatches if c["severity"] != "info"] + [
        _check_reason(c) for c in unverifiable
    ]
    return {
        "comparable": comparable,
        "checks": checks,
        "mismatches": mismatches,
        "unverifiable": unverifiable,
        "reasons": reasons,
    }


def _comparability_gate(
    comparability: dict[str, Any], require_comparable: bool
) -> dict[str, Any]:
    """把可比性结论转成首位门禁条目。

    strict（``require_comparable``）下 critical mismatch 或 critical unverifiable
    -> FAIL；非 strict 下 critical mismatch -> WARN；任一 warn 级 mismatch -> WARN
    （strict 下同样 WARN，不升 FAIL）；否则 PASS。
    """
    critical_mismatch = [c for c in comparability["mismatches"] if c["severity"] == "critical"]
    warn_mismatch = [c for c in comparability["mismatches"] if c["severity"] == "warn"]
    unverifiable = comparability["unverifiable"]

    if require_comparable and (critical_mismatch or unverifiable):
        status = "FAIL"
    elif critical_mismatch or warn_mismatch:
        status = "WARN"
    else:
        status = "PASS"

    strict_note = "（strict 模式）" if require_comparable else ""
    detail = (
        f"关键维度不一致 {len(critical_mismatch)} 项，"
        f"次要维度不一致 {len(warn_mismatch)} 项，"
        f"无法核验 {len(unverifiable)} 项{strict_note}"
    )
    return {"name": "comparability", "status": status, "detail": detail}


def compare_runs(
    dir_a: str,
    dir_b: str,
    acc_threshold: float = 0.02,
    stale_threshold: float = 0.02,
    latency_warn: float = 0.30,
    require_comparable: bool = False,
    out_dir: str | None = None,
    tag: str | None = None,
) -> dict[str, Any]:
    """对比两个 run 目录。

    Args:
        dir_a: 基线 run 目录（含 metrics.json）。
        dir_b: 候选 run 目录（含 metrics.json）。
        acc_threshold: accuracy 允许的最大下降幅度。
        stale_threshold: stale_rate 允许的最大上升幅度。
        latency_warn: latency_p95 相对上升的告警比例（0.30 = 30%）。
        require_comparable: strict 模式；关键维度不一致或缺失 -> 门禁 FAIL。
        out_dir: A/B 实验目录的父目录；缺省 :func:`membench.paths.output_root`。
        tag: A/B 实验目录标签；缺省 ``{run_a}_vs_{run_b}``。

    Returns:
        ``{"verdict", "diffs", "gates", "run_a", "run_b", "compare_dir",
        "comparability"}``；``gates`` 首位为 ``comparability`` 门禁，
        ``compare_dir`` 指向落地的 ``ab_*`` 实验目录。
    """
    metrics_a = _load_metrics(dir_a)
    metrics_b = _load_metrics(dir_b)
    overall_a = metrics_a.get("overall", {})
    overall_b = metrics_b.get("overall", {})

    diffs = _diff_overall(overall_a, overall_b)
    comparability = check_comparability(metrics_a, metrics_b)
    gates = [
        _comparability_gate(comparability, require_comparable),
        *_eval_gates(overall_a, overall_b, acc_threshold, stale_threshold, latency_warn),
    ]

    if any(g["status"] == "FAIL" for g in gates):
        verdict = "FAIL"
    elif any(g["status"] == "WARN" for g in gates):
        verdict = "WARN"
    else:
        verdict = "PASS"

    id_a = str(metrics_a.get("run_id") or os.path.basename(os.path.normpath(dir_a)))
    id_b = str(metrics_b.get("run_id") or os.path.basename(os.path.normpath(dir_b)))
    base = out_dir or output_root()
    ab_dir = layout.experiment_path(base, layout.ab_subdir(tag or f"{id_a}_vs_{id_b}"))
    render_compare_report(
        ab_dir, metrics_a, metrics_b, diffs, gates, verdict, comparability=comparability
    )
    layout.materialize_ab_experiment(ab_dir, dir_a, dir_b, ab_dir)

    # 逐条打印不可比原因（含"关键维度无法核验"），默认模式也不静默
    for reason in comparability["reasons"]:
        print(f"[membench] 不可比原因: {reason}")

    print(f"[membench] compare {id_a} -> {id_b}: {verdict}")
    for gate in gates:
        print(f"[membench]   gate {gate['name']}: {gate['status']} ({gate['detail']})")
    print(f"[membench] report: {os.path.join(ab_dir, 'compare_report.md')}")

    return {
        "verdict": verdict,
        "diffs": diffs,
        "gates": gates,
        "run_a": id_a,
        "run_b": id_b,
        "compare_dir": ab_dir,
        "comparability": comparability,
    }
