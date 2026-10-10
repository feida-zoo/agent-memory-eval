"""compare：对比两个 run（基线 vs 候选），产出逐指标 diff 与回归门禁判定。

门禁规则（SPEC 3.6，阈值可由 CLI 覆盖）：
- accuracy 下降 > acc_threshold（默认 0.02） -> FAIL
- stale_rate 上升 > stale_threshold（默认 0.02） -> FAIL
- latency_p95 上升 > latency_warn（默认 30%） -> WARN
- 其余指标变动仅报告，不门禁。

输出目录：``{runs}/compare_{a}_vs_{b}/compare_report.md`` + ``fig_metric_diff.png``，
其中 ``{runs}`` 取 baseline run 目录的父目录。
"""

from __future__ import annotations

import json
import os
from typing import Any

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


def compare_runs(
    dir_a: str,
    dir_b: str,
    acc_threshold: float = 0.02,
    stale_threshold: float = 0.02,
    latency_warn: float = 0.30,
) -> dict[str, Any]:
    """对比两个 run 目录。

    Args:
        dir_a: 基线 run 目录（含 metrics.json）。
        dir_b: 候选 run 目录（含 metrics.json）。
        acc_threshold: accuracy 允许的最大下降幅度。
        stale_threshold: stale_rate 允许的最大上升幅度。
        latency_warn: latency_p95 相对上升的告警比例（0.30 = 30%）。

    Returns:
        ``{"verdict": "PASS"|"WARN"|"FAIL", "diffs": {...}, "gates": [...], ...}``
    """
    metrics_a = _load_metrics(dir_a)
    metrics_b = _load_metrics(dir_b)
    overall_a = metrics_a.get("overall", {})
    overall_b = metrics_b.get("overall", {})

    diffs = _diff_overall(overall_a, overall_b)
    gates = _eval_gates(overall_a, overall_b, acc_threshold, stale_threshold, latency_warn)

    if any(g["status"] == "FAIL" for g in gates):
        verdict = "FAIL"
    elif any(g["status"] == "WARN" for g in gates):
        verdict = "WARN"
    else:
        verdict = "PASS"

    id_a = str(metrics_a.get("run_id") or os.path.basename(os.path.normpath(dir_a)))
    id_b = str(metrics_b.get("run_id") or os.path.basename(os.path.normpath(dir_b)))
    base_dir = os.path.dirname(os.path.abspath(os.path.normpath(dir_a)))
    out_dir = os.path.join(base_dir, f"compare_{id_a}_vs_{id_b}")
    render_compare_report(out_dir, metrics_a, metrics_b, diffs, gates, verdict)

    print(f"[membench] compare {id_a} -> {id_b}: {verdict}")
    for gate in gates:
        print(f"[membench]   gate {gate['name']}: {gate['status']} ({gate['detail']})")
    print(f"[membench] report: {os.path.join(out_dir, 'compare_report.md')}")

    return {
        "verdict": verdict,
        "diffs": diffs,
        "gates": gates,
        "run_a": id_a,
        "run_b": id_b,
        "compare_dir": out_dir,
    }
