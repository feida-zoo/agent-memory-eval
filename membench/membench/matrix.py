"""matrix：多 run 矩阵对比（platform × commit 维度）。

:func:`compare_matrix` 读取若干带标签 run 的 metrics.json，产出：

1. 若标签均可解析为 ``platform@commit`` 且恰好覆盖 2 平台 × 2 commit，
   对 accuracy / stale_rate / evidence_recall / abstention_accuracy /
   latency_p95_ms 各生成一张 platform × commit 透视表；
2. 所有 label 的逐指标汇总表（行=指标，列=label）；
3. 所有两两组合的门禁判定（复用 :mod:`membench.compare` 的规则）；
4. 分组条形图 ``fig_matrix_accuracy.png``（x=commit，hue=platform，
   或反之取合理者；标签不可解析时退化为按 label 单系列）。

产物写到 ``out_dir``：``matrix_report.md``、``matrix_result.json``、
``fig_matrix_accuracy.png``。
"""

from __future__ import annotations

import itertools
import json
import os
from typing import Any

from membench import layout
from membench.compare import _eval_gates, _load_metrics
from membench.report import PALETTE, _md_table, _setup_fonts

try:
    import matplotlib

    matplotlib.use("Agg")  # 无显示环境可用
    import matplotlib.pyplot as plt
except ImportError:  # matplotlib 为可选依赖（pip install membench[report]）
    plt = None  # type: ignore[assignment]

#: 参与透视表 / 汇总表的指标。
MATRIX_METRICS: list[str] = [
    "accuracy",
    "stale_rate",
    "evidence_recall",
    "abstention_accuracy",
    "latency_p95_ms",
]


def _parse_label(label: str) -> tuple[str, str] | None:
    """把 "platform@commit" 解析为 (platform, commit)；解析失败返回 None。"""
    if "@" not in label:
        return None
    platform, _, commit = label.partition("@")
    if not platform or not commit:
        return None
    return (platform, commit)


def _build_pivots(
    parsed: dict[str, tuple[str, str]],
    overalls: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, float]]] | None:
    """构造各指标的 platform × commit 透视表；不满足 2×2 全覆盖时返回 None。"""
    platforms = sorted({p for p, _ in parsed.values()})
    commits = sorted({c for _, c in parsed.values()})
    if len(platforms) != 2 or len(commits) != 2:
        return None
    cells = {(p, c) for p, c in parsed.values()}
    if set(itertools.product(platforms, commits)) != cells:
        return None
    # 每个 (platform, commit) 组合必须唯一对应一个 label
    combo_to_label: dict[tuple[str, str], str] = {}
    for label, (p, c) in parsed.items():
        if (p, c) in combo_to_label:
            return None
        combo_to_label[(p, c)] = label

    pivots: dict[str, dict[str, dict[str, float]]] = {}
    for metric in MATRIX_METRICS:
        table: dict[str, dict[str, float]] = {}
        for p in platforms:
            table[p] = {}
            for c in commits:
                label = combo_to_label[(p, c)]
                table[p][c] = float(overalls[label].get(metric, 0.0))
        pivots[metric] = table
    return pivots


def fig_matrix_accuracy(
    summary: dict[str, dict[str, float]],
    parsed: dict[str, tuple[str, str]] | None,
    out_path: str,
) -> str:
    """矩阵 accuracy 分组条形图。

    标签可解析为 platform@commit 时：x 轴取基数较小的维度，hue 取另一维度；
    否则退化为按 label 的单系列条形图。
    """
    _setup_fonts()
    labels = list(summary.keys())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if parsed:
        platforms = sorted({p for p, _ in parsed.values()})
        commits = sorted({c for _, c in parsed.values()})
        # x 轴取数量较少的一维，hue 取另一维（相等时 x=commit, hue=platform）
        if len(commits) <= len(platforms):
            x_vals, hue_vals = commits, platforms
            key_of = lambda lbl: (parsed[lbl][1], parsed[lbl][0])  # (x, hue)
            ax.set_xlabel("commit")
        else:
            x_vals, hue_vals = platforms, commits
            key_of = lambda lbl: (parsed[lbl][0], parsed[lbl][1])
            ax.set_xlabel("platform")
        width = 0.8 / max(len(hue_vals), 1)
        x = list(range(len(x_vals)))
        for hi, hue in enumerate(hue_vals):
            xs, ys = [], []
            for xi, xv in enumerate(x_vals):
                hit = [
                    lbl
                    for lbl in labels
                    if key_of(lbl) == (xv, hue)
                ]
                if not hit:
                    continue
                xs.append(xi - 0.4 + width * (hi + 0.5))
                ys.append(float(summary[hit[0]].get("accuracy", 0.0)))
            ax.bar(
                xs,
                ys,
                width,
                label=hue,
                color=PALETTE[hi % len(PALETTE)],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(x_vals)
        ax.legend(title="hue")
    else:
        vals = [float(summary[lbl].get("accuracy", 0.0)) for lbl in labels]
        x = list(range(len(labels)))
        ax.bar(
            x,
            vals,
            color=[PALETTE[i % len(PALETTE)] for i in range(len(labels))],
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("accuracy")
    ax.set_title("矩阵对比 accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _verdict_of(gates: list[dict[str, Any]]) -> str:
    if any(g["status"] == "FAIL" for g in gates):
        return "FAIL"
    if any(g["status"] == "WARN" for g in gates):
        return "WARN"
    return "PASS"


def compare_matrix(labeled_runs: list[tuple[str, str]], out_dir: str) -> dict[str, Any]:
    """多 run 矩阵对比。

    Args:
        labeled_runs: ``[(label, run_dir), ...]``，label 推荐格式
            ``"platform@commit"``（也接受任意字符串）。
        out_dir: 产物输出目录（写 matrix_report.md / matrix_result.json /
            fig_matrix_accuracy.png）。

    Returns:
        结果 dict：``{"labels", "summary", "pivot_tables", "pairwise",
        "out_dir", "report_path", "result_path", "fig_path"}``。
    """
    if len(labeled_runs) < 2:
        raise ValueError("matrix 至少需要 2 个 labeled run")

    labels: list[str] = []
    overalls: dict[str, dict[str, Any]] = {}
    metrics_by_label: dict[str, dict[str, Any]] = {}
    for label, run_dir in labeled_runs:
        if label in overalls:
            raise ValueError(f"重复的 label: {label}")
        metrics = _load_metrics(run_dir)
        metrics_by_label[label] = metrics
        overalls[label] = metrics.get("overall", {})
        labels.append(label)

    # 1. 逐指标汇总表（行=指标，列=label）
    summary: dict[str, dict[str, float]] = {
        label: {
            m: float(overalls[label].get(m, 0.0))
            for m in MATRIX_METRICS
            if m in overalls[label]
        }
        for label in labels
    }

    # 2. platform × commit 透视表（满足 2×2 全覆盖才有）
    parsed: dict[str, tuple[str, str]] = {}
    for label in labels:
        pc = _parse_label(label)
        if pc is None:
            parsed = {}
            break
        parsed[label] = pc
    pivots = _build_pivots(parsed, overalls) if parsed else None

    # 3. 两两组合门禁判定（复用 compare 的规则与默认阈值）
    pairwise: list[dict[str, Any]] = []
    for label_a, label_b in itertools.combinations(labels, 2):
        gates = _eval_gates(overalls[label_a], overalls[label_b], 0.02, 0.02, 0.30)
        pairwise.append(
            {
                "a": label_a,
                "b": label_b,
                "verdict": _verdict_of(gates),
                "gates": gates,
            }
        )

    # 4. 分组条形图
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, "fig_matrix_accuracy.png")
    fig_matrix_accuracy(summary, parsed or None, fig_path)

    result: dict[str, Any] = {
        "labels": labels,
        "summary": summary,
        "pivot_tables": pivots,
        "pairwise": pairwise,
        "out_dir": out_dir,
        "fig_path": fig_path,
    }

    report_path = _write_matrix_report_md(
        out_dir, labels, metrics_by_label, summary, pivots, pairwise
    )
    result["report_path"] = report_path
    result_path = os.path.join(out_dir, "matrix_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    result["result_path"] = result_path

    # 各臂 run 副本落 out_dir/runs/<label>/，使矩阵实验目录自包含
    layout.materialize_matrix_experiment(out_dir, labeled_runs)

    print(f"[membench] matrix: {len(labels)} runs -> {out_dir}")
    return result


def _write_matrix_report_md(
    out_dir: str,
    labels: list[str],
    metrics_by_label: dict[str, dict[str, Any]],
    summary: dict[str, dict[str, float]],
    pivots: dict[str, dict[str, dict[str, float]]] | None,
    pairwise: list[dict[str, Any]],
) -> str:
    """生成 matrix_report.md（中文）。"""
    lines: list[str] = []
    lines.append("# membench 矩阵对比报告")
    lines.append("")
    lines.append(f"- run 数: {len(labels)}")
    for label in labels:
        m = metrics_by_label[label]
        lines.append(
            f"- `{label}`: adapter={m.get('adapter', 'n/a')}, "
            f"dataset={m.get('dataset', 'n/a')}, run_id={m.get('run_id', 'n/a')}"
        )
    lines.append("")

    # 1. platform × commit 透视表
    lines.append("## platform × commit 透视表")
    lines.append("")
    if pivots:
        for metric in MATRIX_METRICS:
            table = pivots[metric]
            platforms = sorted(table.keys())
            commits = sorted(next(iter(table.values())).keys())
            lines.append(f"### {metric}")
            lines.append("")
            rows = [
                [p] + [f"{table[p][c]:.3f}" for c in commits]
                for p in platforms
            ]
            lines.append(_md_table(["platform \\ commit", *commits], rows))
            lines.append("")
    else:
        lines.append("（标签不构成 2 平台 × 2 commit 全覆盖，跳过透视表）")
        lines.append("")

    # 2. 逐指标汇总表（行=指标，列=label）
    lines.append("## 逐指标汇总")
    lines.append("")
    rows = [
        [metric] + [f"{summary[lbl].get(metric, 0.0):.3f}" for lbl in labels]
        for metric in MATRIX_METRICS
    ]
    lines.append(_md_table(["指标", *labels], rows))
    lines.append("")
    lines.append("![矩阵 accuracy](fig_matrix_accuracy.png)")
    lines.append("")

    # 3. 两两门禁判定
    lines.append("## 两两门禁判定")
    lines.append("")
    pair_rows = []
    for p in pairwise:
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(p["verdict"], "")
        fails = [g for g in p["gates"] if g["status"] != "PASS"]
        detail = "; ".join(f"{g['name']}: {g['detail']}" for g in fails) or "全部通过"
        pair_rows.append([p["a"], p["b"], p["verdict"], icon, detail])
    lines.append(_md_table(["A", "B", "结论", "", "说明"], pair_rows))
    lines.append("")

    out_path = os.path.join(out_dir, "matrix_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path
