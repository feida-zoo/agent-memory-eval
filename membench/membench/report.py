"""报告生成：report.md / compare_report.md 与配套图表（matplotlib, Agg 后端）。

图表统一低饱和配色；尽量自动探测中文字体以避免乱码。
公共入口：
- :func:`render_run_report`     单个 run 的 report.md + fig_category_accuracy.png + fig_latency_hist.png
- :func:`render_compare_report` compare_report.md + fig_metric_diff.png
- :func:`copy_report`           --report-out：把报告连带引用图片复制到指定目录
"""

from __future__ import annotations

import os
import re
import shutil
from typing import Any, Iterable

try:
    import matplotlib

    matplotlib.use("Agg")  # 必须在 pyplot 导入前设置，保证无显示环境可用
    import matplotlib.pyplot as plt
except ImportError:  # matplotlib 为可选依赖（pip install membench[report]）
    plt = None  # type: ignore[assignment]

_MPL_HINT = "报告图表需要 matplotlib，请运行: pip install matplotlib"


def _require_plt() -> None:
    """画图前检查 matplotlib 是否可用，缺失时给出清晰提示而非 ImportError 堆栈。"""
    if plt is None:
        raise RuntimeError(_MPL_HINT)

from .ratings import format_metric, overall_rating, rate_metric

# 统一低饱和配色（SPEC 第 7 节）
PALETTE: list[str] = ["#3b6ea5", "#c9a86a", "#5e8c61", "#8f8f8f", "#b03a2e"]

# 分题型 category 显示名映射（仅显示层；数据层 category 值保持不变）。
# 查不到的未知 category 原样显示，向后兼容自定义题型。
CATEGORY_ZH: dict[str, str] = {
    "single_hop": "单跳（single_hop）",
    "multi_hop": "多跳（multi_hop）",
    "temporal": "时间推理（temporal）",
    "knowledge_update": "知识更新（knowledge_update）",
    "abstention": "拒答（abstention）",
    "preference": "偏好（preference）",
    "open_ended": "开放问答（open_ended）",
}

# 效率统计表的指标显示名（中文，单位保留）。
METRIC_ZH: dict[str, str] = {
    "query latency mean (ms)": "查询延迟均值（ms）",
    "query latency p50 (ms)": "查询延迟 p50（ms）",
    "query latency p95 (ms)": "查询延迟 p95（ms）",
    "avg input tokens / query": "平均输入 token / 题",
    "avg llm calls / query": "平均 LLM 调用次数 / 题",
}


def category_display(cat: str) -> str:
    """category 的显示名：已知题型返回「中文（英文）」，未知题型原样返回。"""
    return CATEGORY_ZH.get(cat, cat)


# 验收基线（report.md 对照表）：(指标 key, 展示名「中文（英文 key）」, 目标描述, 判定函数)
BASELINES: list[tuple[str, str, str, Any]] = [
    ("accuracy", "正确性（accuracy）", ">= 0.85", lambda v: v >= 0.85),
    ("evidence_recall", "检索质量（evidence_recall）", ">= 0.90", lambda v: v >= 0.90),
    ("stale_rate", "记忆动态（stale_rate）", "<= 0.03", lambda v: v <= 0.03),
    ("abstention_accuracy", "拒答（abstention_accuracy）", ">= 0.80", lambda v: v >= 0.80),
]

# compare 分组条形图使用的关键指标（均为 [0,1] 区间，可同图比较）
DIFF_METRICS: list[str] = [
    "accuracy",
    "token_f1",
    "bleu1",
    "evidence_recall",
    "abstention_accuracy",
    "stale_rate",
]

_FONTS_READY = False


def _setup_fonts() -> None:
    """探测可用中文字体并配置 rcParams（找不到则回退默认字体）。"""
    global _FONTS_READY
    _require_plt()
    if _FONTS_READY:
        return
    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK",
        "WenQuanYi Zen Hei",
        "WenQuanYi Micro Hei",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "Arial Unicode MS",
    ]
    try:
        from matplotlib import font_manager

        installed = {f.name for f in font_manager.fontManager.ttflist}
        for name in candidates:
            if name in installed:
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                break
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False
    _FONTS_READY = True


# ---------------------------------------------------------------------------
# 图表
# ---------------------------------------------------------------------------

def fig_category_accuracy(by_category: dict[str, dict[str, Any]], out_path: str) -> str:
    """按 category 的准确率横向条形图。"""
    _setup_fonts()
    cats = sorted(by_category.keys())
    accs = [float(by_category[c].get("accuracy", 0.0)) for c in cats]
    ns = [int(by_category[c].get("n", 0)) for c in cats]

    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.5 * len(cats) + 1.5)))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(cats))]
    ax.barh(cats, accs, color=colors)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("accuracy")
    ax.set_title("分题型准确率 (by_category)")
    ax.invert_yaxis()
    for i, (acc, n) in enumerate(zip(accs, ns)):
        ax.text(min(acc + 0.01, 0.95), i, f"{acc:.2f} (n={n})", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def fig_latency_hist(
    latencies: Iterable[float],
    p50: float,
    p95: float,
    out_path: str,
) -> str:
    """query 延迟直方图，叠加 p50 / p95 竖线。"""
    _setup_fonts()
    vals = [float(v) for v in latencies]

    fig, ax = plt.subplots(figsize=(8, 4))
    if vals:
        bins = min(30, max(5, len(vals) // 2))
        ax.hist(vals, bins=bins, color=PALETTE[0], edgecolor="white")
    ax.axvline(p50, color=PALETTE[2], linestyle="--", linewidth=1.5, label=f"p50 = {p50:.0f} ms")
    ax.axvline(p95, color=PALETTE[4], linestyle="--", linewidth=1.5, label=f"p95 = {p95:.0f} ms")
    ax.set_xlabel("query latency (ms)")
    ax.set_ylabel("count")
    ax.set_title("查询延迟分布")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def fig_metric_diff(
    overall_a: dict[str, Any],
    overall_b: dict[str, Any],
    label_a: str,
    label_b: str,
    out_path: str,
) -> str:
    """compare 用：关键指标 A vs B 分组条形图。"""
    _setup_fonts()
    keys = [k for k in DIFF_METRICS if k in overall_a and k in overall_b]
    vals_a = [float(overall_a.get(k, 0.0)) for k in keys]
    vals_b = [float(overall_b.get(k, 0.0)) for k in keys]

    x = list(range(len(keys)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(keys)), 4.5))
    ax.bar([i - width / 2 for i in x], vals_a, width, label=label_a, color=PALETTE[0])
    ax.bar([i + width / 2 for i in x], vals_b, width, label=label_b, color=PALETTE[1])
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("value")
    ax.set_title("关键指标对比 (A vs B)")
    ax.legend()
    for i, (va, vb) in enumerate(zip(vals_a, vals_b)):
        ax.text(i - width / 2, va + 0.01, f"{va:.2f}", ha="center", fontsize=8)
        ax.text(i + width / 2, vb + 0.01, f"{vb:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Markdown 报告
# ---------------------------------------------------------------------------

def _fmt(value: Any) -> str:
    """数值格式化：float 保留 3 位，其余原样。"""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _md_cell(text: Any) -> str:
    """转义 markdown 表格单元格内容：换行折叠为 <br>，竖线转义。

    失败样例等表格直接嵌入模型回答 / 数据集原文，含换行与竖线时会把
    整张表撑裂，统一在这里兜底。
    """
    s = str(text).replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")
    return s.replace("|", "\\|")


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(_md_cell(c) for c in rows_i) + " |" for rows_i in rows]
    return "\n".join(lines)


def _baseline_rows(overall: dict[str, Any]) -> list[list[str]]:
    rows = []
    for key, label, target, ok in BASELINES:
        if key not in overall:
            rows.append([label, target, "n/a", "—", "❓"])
            continue
        value = float(overall[key])
        rating = rate_metric(key, value)
        rows.append([label, target, f"{value:.3f}", rating if rating else "—", "✅" if ok(value) else "❌"])
    return rows


def write_run_report_md(run_dir: str, records: list[dict], metrics: dict[str, Any]) -> str:
    """生成 report.md（章节按 SPEC 第 7 节，中文）。"""
    overall = metrics.get("overall", {})
    by_category = metrics.get("by_category", {})

    lines: list[str] = []
    lines.append(f"# membench 评测报告 — {metrics.get('run_id', '')}")
    lines.append("")

    # 0. 评测配置（硬需求：必须位于标题之后、正文第一段之前）
    lines.append("## 评测配置")
    lines.append("")
    config_rows = [
        ["测试集路径", f"`{metrics.get('dataset_path', metrics.get('dataset', 'n/a'))}`"],
        ["测试集来源", str(metrics.get("dataset_source", "n/a"))],
        ["平台", str(metrics.get("platform", "default"))],
        ["commit", str(metrics.get("commit") or "n/a")],
        ["判分器", str(metrics.get("judge", "n/a"))],
    ]
    lines.append(_md_table(["项目", "值"], config_rows))
    lines.append("")

    lines.append(f"- adapter: `{metrics.get('adapter', 'n/a')}`")
    lines.append(f"- dataset: `{metrics.get('dataset', 'n/a')}`")
    lines.append(f"- judge: `{metrics.get('judge', 'n/a')}`")
    lines.append(f"- timestamp: {metrics.get('timestamp', 'n/a')}")
    lines.append(f"- n_questions: {overall.get('n_questions', len(records))}")
    n_ignored = int(metrics.get("n_ignored", 0) or 0)
    if n_ignored:
        lines.append(f"- 忽略题数: {n_ignored}（ignore=true，未参与评分）")
    lines.append("")

    # 1. 摘要表（六维核心指标）
    lines.append("## 摘要（六维核心指标）")
    lines.append("")
    summary_rows = [
        ["正确性", "accuracy", format_metric("accuracy", overall.get("accuracy", "n/a"))],
        ["正确性", "token_f1", format_metric("token_f1", overall.get("token_f1", "n/a"))],
        ["正确性", "bleu1", format_metric("bleu1", overall.get("bleu1", "n/a"))],
        ["检索质量", "evidence_recall", format_metric("evidence_recall", overall.get("evidence_recall", "n/a"))],
        ["效率成本", "latency_p50_ms", _fmt(overall.get("latency_p50_ms", "n/a"))],
        ["效率成本", "latency_p95_ms", _fmt(overall.get("latency_p95_ms", "n/a"))],
        ["效率成本", "avg_input_tokens", _fmt(overall.get("avg_input_tokens", "n/a"))],
        ["效率成本", "avg_llm_calls", _fmt(overall.get("avg_llm_calls", "n/a"))],
        ["记忆动态", "stale_rate", format_metric("stale_rate", overall.get("stale_rate", "n/a"))],
        ["拒答", "abstention_accuracy", format_metric("abstention_accuracy", overall.get("abstention_accuracy", "n/a"))],
    ]
    lines.append(_md_table(["维度", "指标", "值"], summary_rows))
    lines.append("")

    # 综合评级（质量指标评级点数均值；无可评级指标则不输出）
    overall_rated = overall_rating(overall)
    if overall_rated is not None:
        label, mean_points = overall_rated
        lines.append(f"**综合评级：{label}（{mean_points:.1f}/4.0）**")
        lines.append("")
        lines.append(
            "<sub>评级规则：取六项质量指标（accuracy / token_f1 / bleu1 / "
            "evidence_recall / abstention_accuracy / stale_rate）的评级点数"
            "（优秀=4、良好=3、中等=2、较差=1）均值，阈值 3.5 / 2.5 / 1.5；"
            "效率成本指标不参与评级。</sub>"
        )
        lines.append("")

    # 2. by_category 表 + 图
    lines.append("## 分题型准确率")
    lines.append("")
    cat_rows = [
        [
            category_display(cat),
            format_metric("accuracy", by_category[cat].get("accuracy", "n/a")),
            str(by_category[cat].get("n", 0)),
        ]
        for cat in sorted(by_category.keys())
    ]
    lines.append(_md_table(["category", "accuracy", "n"], cat_rows or [["(无)", "", ""]]))
    lines.append("")
    lines.append("![分题型准确率](fig_category_accuracy.png)")
    lines.append("")

    # 3. 效率统计
    lines.append("## 效率统计")
    lines.append("")
    lat = [float(r.get("latency_ms", 0.0)) for r in records]
    eff_rows = [
        [METRIC_ZH["query latency mean (ms)"], f"{(sum(lat) / len(lat)):.1f}" if lat else "n/a"],
        [METRIC_ZH["query latency p50 (ms)"], _fmt(overall.get("latency_p50_ms", "n/a"))],
        [METRIC_ZH["query latency p95 (ms)"], _fmt(overall.get("latency_p95_ms", "n/a"))],
        [METRIC_ZH["avg input tokens / query"], _fmt(overall.get("avg_input_tokens", "n/a"))],
        [METRIC_ZH["avg llm calls / query"], _fmt(overall.get("avg_llm_calls", "n/a"))],
    ]
    lines.append(_md_table(["指标", "值"], eff_rows))
    lines.append("")
    lines.append("![查询延迟分布](fig_latency_hist.png)")
    lines.append("")

    # 4. 失败样例 Top10
    lines.append("## 失败样例 Top10")
    lines.append("")
    fails = [r for r in records if not r.get("judge", {}).get("correct", False)][:10]
    if fails:
        fail_rows = []
        for r in fails:
            pred = str(r.get("prediction", ""))
            if len(pred) > 80:
                pred = pred[:77] + "..."
            question = str(r.get("question", ""))
            if len(question) > 100:
                question = question[:97] + "..."
            gold = str(r.get("gold_answer", ""))
            if len(gold) > 160:
                gold = gold[:157] + "..."
            fail_rows.append([
                str(r.get("qid", "")),
                str(r.get("category", "")),
                question,
                gold,
                pred,
            ])
        lines.append(_md_table(["qid", "category", "问题", "期望", "实际"], fail_rows))
    else:
        lines.append("（无失败样例 🎉）")
    lines.append("")

    # 5. 验收基线对照
    lines.append("## 验收基线对照")
    lines.append("")
    lines.append(_md_table(["指标", "目标", "实际", "评价", "结果"], _baseline_rows(overall)))
    lines.append("")

    out_path = os.path.join(run_dir, "report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def render_run_report(run_dir: str, records: list[dict], metrics: dict[str, Any]) -> None:
    """生成 run 目录下的 report.md 与两张图表。"""
    os.makedirs(run_dir, exist_ok=True)
    fig_category_accuracy(
        metrics.get("by_category", {}), os.path.join(run_dir, "fig_category_accuracy.png")
    )
    fig_latency_hist(
        [r.get("latency_ms", 0.0) for r in records],
        float(metrics.get("overall", {}).get("latency_p50_ms", 0.0)),
        float(metrics.get("overall", {}).get("latency_p95_ms", 0.0)),
        os.path.join(run_dir, "fig_latency_hist.png"),
    )
    write_run_report_md(run_dir, records, metrics)


def write_compare_report_md(
    out_dir: str,
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
    diffs: dict[str, Any],
    gates: list[dict[str, Any]],
    verdict: str,
) -> str:
    """生成 compare_report.md：逐指标 diff 表 + 门禁结论（中文）。"""
    id_a = str(metrics_a.get("run_id", "A"))
    id_b = str(metrics_b.get("run_id", "B"))

    lines: list[str] = []
    lines.append(f"# membench 对比报告 — {id_a} vs {id_b}")
    lines.append("")
    verdict_icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(verdict, "")
    lines.append(f"**门禁结论： {verdict} {verdict_icon}**")
    lines.append("")
    lines.append(f"- 基线 A: `{id_a}`")
    rated_a = overall_rating(metrics_a.get("overall", {}))
    if rated_a is not None:
        lines.append(f"- A 综合评级：{rated_a[0]}（{rated_a[1]:.1f}/4.0）")
    lines.append(f"- 候选 B: `{id_b}`")
    rated_b = overall_rating(metrics_b.get("overall", {}))
    if rated_b is not None:
        lines.append(f"- B 综合评级：{rated_b[0]}（{rated_b[1]:.1f}/4.0）")
    lines.append("")

    lines.append("## 逐指标 diff（B - A）")
    lines.append("")
    rows = []
    for key in sorted(diffs.keys()):
        d = diffs[key]
        rows.append([key, _fmt(d.get("a", "n/a")), _fmt(d.get("b", "n/a")), _fmt(d.get("diff", "n/a"))])
    lines.append(_md_table(["指标", "A (基线)", "B (候选)", "diff"], rows or [["(无)", "", "", ""]]))
    lines.append("")
    lines.append("![关键指标对比](fig_metric_diff.png)")
    lines.append("")

    lines.append("## 门禁明细")
    lines.append("")
    gate_rows = []
    for g in gates:
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(str(g.get("status", "")), "")
        gate_rows.append([str(g.get("name", "")), str(g.get("status", "")), icon, str(g.get("detail", ""))])
    lines.append(_md_table(["门禁", "状态", "", "说明"], gate_rows or [["(无)", "", "", ""]]))
    lines.append("")

    out_path = os.path.join(out_dir, "compare_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def render_compare_report(
    out_dir: str,
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
    diffs: dict[str, Any],
    gates: list[dict[str, Any]],
    verdict: str,
) -> None:
    """生成 compare 目录下的 compare_report.md 与 fig_metric_diff.png。"""
    os.makedirs(out_dir, exist_ok=True)
    fig_metric_diff(
        metrics_a.get("overall", {}),
        metrics_b.get("overall", {}),
        str(metrics_a.get("run_id", "A")),
        str(metrics_b.get("run_id", "B")),
        os.path.join(out_dir, "fig_metric_diff.png"),
    )
    write_compare_report_md(out_dir, metrics_a, metrics_b, diffs, gates, verdict)


# ---------------------------------------------------------------------------
# 报告导出（--report-out）
# ---------------------------------------------------------------------------

_IMG_REF_RE = re.compile(r"(!\[[^\]]*\]\()([^)]+)(\))")


def copy_report(
    md_path: str,
    dest_dir: str,
    dest_filename: str,
    ts_prefix: str,
    extra_files: "Iterable[str] | None" = None,
) -> str:
    """把 markdown 报告复制到指定目录，并连带复制其引用的本地图片。

    引用的图片复制到同一目标目录；``ts_prefix`` 非空时图片文件名加前缀
    避免冲突，为空时保持原文件名（每次评测一个独立子目录的场景）。
    复制后的 md 中图片引用会被修正为新文件名。非本地引用（http/https）
    保持不变。

    Args:
        md_path: 源报告 markdown 路径。
        dest_dir: 目标目录（不存在则创建）。
        dest_filename: 目标 md 文件名（如 ``report.md``）。
        ts_prefix: 图片重命名前缀（空串表示不加前缀、保留原名）。
        extra_files: 额外一并复制的文件路径（如 run 目录下的
            ``results.jsonl`` / ``metrics.json``），按原文件名复制；
            不存在的文件跳过。

    Returns:
        复制后的 md 完整路径。
    """
    os.makedirs(dest_dir, exist_ok=True)
    with open(md_path, encoding="utf-8") as f:
        content = f.read()
    src_dir = os.path.dirname(os.path.abspath(md_path))

    def _replace(match: "re.Match[str]") -> str:
        prefix, ref, suffix = match.group(1), match.group(2), match.group(3)
        if "://" in ref:
            return match.group(0)
        src_img = os.path.join(src_dir, ref)
        new_name = f"{ts_prefix}_{os.path.basename(ref)}" if ts_prefix else os.path.basename(ref)
        dest_img = os.path.join(dest_dir, new_name)
        if os.path.isfile(src_img):
            shutil.copyfile(src_img, dest_img)
        return f"{prefix}{new_name}{suffix}"

    new_content = _IMG_REF_RE.sub(_replace, content)
    dest_md = os.path.join(dest_dir, dest_filename)
    with open(dest_md, "w", encoding="utf-8") as f:
        f.write(new_content)

    for extra in extra_files or ():
        if os.path.isfile(extra):
            shutil.copyfile(extra, os.path.join(dest_dir, os.path.basename(extra)))
    return dest_md
