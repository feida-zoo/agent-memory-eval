"""报告生成：report.md / comparison.md 与配套图表（matplotlib, Agg 后端）。

图表统一低饱和配色；尽量自动探测中文字体以避免乱码。
公共入口：
- :func:`render_dialogue_report`  对话级 report.md（纯文字，不出图）
- :func:`render_dataset_report`   数据集级 report.md + fig_category_accuracy.png + fig_latency_hist.png
- :func:`render_compare_report`   comparison.md + fig_metric_diff.png（质量+效率）+ fig_category_diff.png（分题型）
- :func:`fig_dataset_overview`    evaluate 报告用：各数据集质量 + 效率分组条形图
- :func:`fig_series_accuracy`     多线对比报告用：accuracy 随线序的走势（多数据集多条折线 + 融合粗线）
- :func:`copy_report`             把带图报告连同引用图片与附带文件复制到目标目录
"""

from __future__ import annotations

import json
import os
import re
import shutil
from typing import Any, Iterable

from .layout import (
    DIR_SNAPSHOT,
    DIR_TRACE_STATE,
    FIG_CATEGORY,
    FIG_CATEGORY_DIFF,
    FIG_LATENCY,
    FIG_METRIC_DIFF,
    FILE_COMPARISON_MD,
    FILE_REPORT_MD,
)
from .snapshot import MANIFEST_NAME

try:
    import matplotlib

    matplotlib.use("Agg")  # 必须在 pyplot 导入前设置，保证无显示环境可用
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except ImportError:  # matplotlib 为可选依赖（pip install membench[report]）
    plt = None  # type: ignore[assignment]
    Patch = None  # type: ignore[assignment]

_MPL_HINT = "报告图表需要 matplotlib，请运行: pip install matplotlib"


def _require_plt() -> None:
    """画图前检查 matplotlib 是否可用，缺失时给出清晰提示而非 ImportError 堆栈。"""
    if plt is None:
        raise RuntimeError(_MPL_HINT)

from .ratings import (
    DIAGNOSTIC_METRICS,
    format_metric,
    format_metric_delta,
    format_metric_value,
    overall_rating,
    rate_metric,
)
from .data.schema import is_abstention
from .metrics.summary import ATTRIBUTION_BUCKETS
from .version import version_display

# 统一低饱和配色（SPEC 第 7 节）
PALETTE: list[str] = ["#3b6ea5", "#c9a86a", "#5e8c61", "#8f8f8f", "#b03a2e"]

# 分题型 category 显示名映射（仅显示层；数据层 category 值保持不变）。
# 不同数据集保留自己的 raw category，不再做多对一合并，故 CATEGORY_ZH
# 扩到 7 通用 + 7 BEAM 增量 + 4 gatemem 增量；查不到的未知 category
# 原样显示，向后兼容自定义题型。
CATEGORY_ZH: dict[str, str] = {
    # membench 通用 7 类（locomo / longmemeval / office 也用这些 key）
    "single_hop": "单跳（single_hop）",
    "multi_hop": "多跳（multi_hop）",
    "temporal": "时间推理（temporal）",
    "knowledge_update": "知识更新（knowledge_update）",
    "abstention": "拒答（abstention）",
    "preference": "偏好（preference）",
    "open_ended": "开放问答（open_ended）",
    # BEAM 原始 10 类（1:1 直传后保留）
    "information_extraction": "信息抽取（information_extraction）",
    "multi_session_reasoning": "多 session 推理（multi_session_reasoning）",
    "event_ordering": "事件排序（event_ordering）",
    "temporal_reasoning": "时序推理（temporal_reasoning）",
    "contradiction_resolution": "矛盾消解（contradiction_resolution）",
    "instruction_following": "指令遵循（instruction_following）",
    "preference_following": "偏好遵循（preference_following）",
    "summarization": "摘要（summarization）",
    # gatemem 4 类（按 query_type + expected_action 拆分）
    "utility_answer": "实用-应答（utility_answer）",
    "privacy_refuse": "隐私-拒答（privacy_refuse）",
    "privacy_answer_redacted": "隐私-脱敏应答（privacy_answer_redacted）",
    "safety_no_memory": "安全-不再记（safety_no_memory）",
}

#: 效率统计表的指标显示名（中文，单位保留）。
METRIC_ZH: dict[str, str] = {
    "query latency mean (ms)": "查询延迟均值（ms）",
    "query latency p50 (ms)": "查询延迟 p50（ms）",
    "query latency p95 (ms)": "查询延迟 p95（ms）",
    "avg input tokens / query": "平均输入 token / 题",
    "avg output tokens / query": "平均输出 token / 题",
    "avg ingest latency (ms)": "按题摊分的对话 ingest 耗时（ms）",
    "avg llm calls / query": "平均 LLM 调用次数 / 题",
}

#: 全量指标 key -> 中文显示名。
#: 缺失键回退原 key；report 默认渲染用「中文（key）」双标形式以兼顾可读性
#: 与可检索性。指标 key 在数据层与导出接口保持不变，仅渲染层做本地化。
#: 单位（ms / Token）随 with_key=False 调用方或值单元格附带，避免与双标
#: 「中文（key）」结构重复（如「延迟 P50（ms）（latency_p50_ms）」）。
METRIC_LABEL_ZH: dict[str, str] = {
    "accuracy": "正确率",
    "token_f1": "Token F1",
    "bleu1": "BLEU-1",
    "evidence_recall": "证据召回率",
    "abstention_accuracy": "拒答准确率",
    "stale_rate": "陈旧率",
    "latency_p50_ms": "延迟 P50",
    "latency_p95_ms": "延迟 P95",
    "avg_input_tokens": "平均输入 Token",
    "avg_output_tokens": "平均输出 Token",
    "avg_llm_calls": "平均 LLM 调用次数",
    "avg_ingest_latency_ms": "按题摊分的对话 ingest 耗时",
    "n_questions": "题目数",
    "n_ignored": "忽略题数",
    "mean_points": "综合点数",
    "recall_at_5": "检索召回率@5",
    "over_refusal_rate": "过度拒答率",
    "rubric_item_recall": "要点命中率",
}


def category_display(cat: str) -> str:
    """category 的显示名：已知题型返回「中文（英文）」，未知题型原样返回。"""
    return CATEGORY_ZH.get(cat, cat)


#: 题型进榜的最小题量（「最需要优化的题型」与「提升最多的题型」共用）：
#: 只有 2~4 题的题型，一两题翻转就是 25~50pt，会挤掉真正该修的大题型。
CATEGORY_MIN_N = 10

#: 「最需要优化的题型」列多少条。
WORST_CATEGORY_TOP = 5


def worst_categories(
    records: list[dict[str, Any]],
    *,
    limit: int = WORST_CATEGORY_TOP,
    min_n: int = CATEGORY_MIN_N,
) -> list[dict[str, Any]]:
    """按准确率升序排出最需要优化的题型，并标出该题型的失败形态。

    为什么设题量门槛：题型只有 2~4 题时，一两题答错就是 25~50pt 的落差，会把
    真正该修的大题型挤下去。小样本题型仍在「分题型准确率」表里，只是不进榜。

    失败形态只做**可机检**的粗分（字段都已在逐题记录里，不额外调用判分器），
    且**每个判错题只归一类**（分桶互斥、加总等于判错题数，不会被读者误当成漏算）：

    拒答类（``is_abstention``）按优先级：

    1. ``未拒答``——``abstained_pred`` 为假（该拒答却没拒）；
    2. ``泄露红线``——``judge.leak`` 为真（拒答了但说出了敏感内容）；
    3. ``状态未遗忘``——``evidence_in_state`` 为真（回答没问题，记忆里还留着）；
    4. ``其他``——以上都不成立（例如 rubric 的 include 要点没满足）。

    其余题型：

    1. ``状态未遗忘``——``evidence_in_state`` 为真；
    2. ``证据未命中``——``evidence_hit`` 为假（证据没进过检索上下文）；
    3. ``检索到仍答错``——``evidence_hit`` 为真；
    4. ``无判定依据``——``evidence_hit`` 为 ``None``（该数据集没有证据标注）。

    拒答类的判定按 category + 该题型**第一条判错题**的 ``gold_answer``：数据集里
    category 与 gold 类型是绑定的（``abstention`` / ``safety_no_memory`` /
    ``privacy_refuse``），不会出现同题型混杂。

    Args:
        records: 逐题记录（``results.jsonl`` 风格）。
        limit: 取前几名。
        min_n: 进榜的最小题目数。

    Returns:
        ``[{"category","n","accuracy","n_wrong","issue"}, ...]``，按准确率升序；
        同准确率按题量降序（大题优先）。
    """
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        buckets.setdefault(str(record.get("category", "unknown")), []).append(record)

    def _judge(record: dict[str, Any]) -> dict[str, Any]:
        return record.get("judge") or {}

    rows: list[dict[str, Any]] = []
    for cat, items in buckets.items():
        n = len(items)
        if n < min_n:
            continue
        wrong = [r for r in items if not bool(_judge(r).get("correct"))]
        accuracy = (n - len(wrong)) / n if n else 0.0
        issue = ""
        if wrong:
            abstain_category = is_abstention(
                cat, str((wrong[0].get("gold_answer") or ""))
            )
            counts: dict[str, int] = {}
            for record in wrong:
                if abstain_category:
                    if not bool(_judge(record).get("abstained_pred")):
                        label = "未拒答"
                    elif bool(_judge(record).get("leak")):
                        label = "泄露红线"
                    elif record.get("evidence_in_state") is True:
                        label = "状态未遗忘"
                    else:
                        label = "其他"
                else:
                    if record.get("evidence_in_state") is True:
                        label = "状态未遗忘"
                    elif record.get("evidence_hit") is False:
                        label = "证据未命中"
                    elif record.get("evidence_hit") is True:
                        label = "检索到仍答错"
                    else:
                        label = "无判定依据"
                counts[label] = counts.get(label, 0) + 1
            # 分桶互斥：按固定顺序输出，加总恒等于判错题数
            order = (
                ("未拒答", "泄露红线", "状态未遗忘", "其他")
                if abstain_category
                else ("证据未命中", "检索到仍答错", "状态未遗忘", "无判定依据")
            )
            issue = "；".join(f"{label} {counts[label]}" for label in order if counts.get(label))
        rows.append(
            {
                "category": cat,
                "n": n,
                "accuracy": accuracy,
                "n_wrong": len(wrong),
                "issue": issue,
            }
        )
    rows.sort(key=lambda item: (item["accuracy"], -item["n"]))
    return rows[:limit]


def metric_label(key: str, with_key: bool = True) -> str:
    """指标的显示名。

    Args:
        key: 指标 key（数据层不变）。
        with_key: True 时返回「中文（key）」双标形式（默认，便于检索/调试）；
            False 时仅返回中文名（图表轴标签等紧凑场景用）。

    Returns:
        显示名字符串；未知 key 原样返回。
    """
    zh = METRIC_LABEL_ZH.get(key)
    if zh is None:
        return str(key)
    return f"{zh}（{key}）" if with_key else zh


def metric_labels(keys: Iterable[str], with_key: bool = True) -> list[str]:
    """批量取指标的显示名（缺值过滤掉，但保留顺序）。"""
    return [metric_label(k, with_key=with_key) for k in keys if k]


def metric_label_with_role(key: str) -> str:
    """指标表显示名；诊断指标（不参与综合评级）追加「（诊断）」后缀。"""
    label = metric_label(key)
    return f"{label}（诊断）" if key in DIAGNOSTIC_METRICS else label


#: ``metrics.json`` 的 ``token_usage`` 块字段 -> 中文显示名（报告与控制台共用）。
TOKEN_USAGE_FIELD_ZH: dict[str, str] = {
    "ingest_input_tokens": "ingest 写入 token（估算）",
    "query_input_tokens": "query 输入 token 合计",
    "query_output_tokens": "query 输出 token 合计",
    "ingest_llm_calls": "ingest LLM 调用",
    "query_llm_calls": "query LLM 调用",
    "n_dialogues": "对话数",
    "n_questions": "题目数",
}


def token_usage_totals(usage: "dict[str, Any] | None") -> dict[str, int]:
    """从 ``token_usage`` 块取可求和字段（缺字段按 0 计）。

    只保留 :data:`TOKEN_USAGE_FIELD_ZH` 里的计数器字段——``n_dialogues`` /
    ``n_questions`` 是上下文，不参与求和。
    """
    if not usage:
        return {}
    return {
        key: int(usage.get(key, 0) or 0)
        for key in TOKEN_USAGE_FIELD_ZH
        if key not in ("n_dialogues", "n_questions")
    }


def format_token_usage(usage: "dict[str, Any] | None") -> str:
    """token 消耗单行摘要（run 控制台 / 报告摘要共用）。

    ``usage`` 为空（旧 run 未落账）时返回 ``n/a``；适配器未上报 token 时各项为 0，
    摘要显式注明"未上报"，避免读成"真的没消耗"。
    """
    if not usage:
        return "n/a（该 run 未落账 token_usage）"
    totals = token_usage_totals(usage)
    ingest = totals.get("ingest_input_tokens", 0)
    q_in = totals.get("query_input_tokens", 0)
    q_out = totals.get("query_output_tokens", 0)
    calls = totals.get("ingest_llm_calls", 0) + totals.get("query_llm_calls", 0)
    text = (
        f"ingest {ingest} + query 输入 {q_in} / 输出 {q_out}"
        f"（合计约 {ingest + q_in + q_out}），LLM 调用 {calls} 次"
    )
    if usage.get("ingest_skipped"):
        text += "；快照回放：ingest 已跳过（0 = 没跑，非零成本）"
    elif ingest == 0 and q_in == 0 and q_out == 0:
        text += "；全为 0 = 适配器未上报该字段，非真实零消耗"
    return text


#: 门禁 / 可比性维度 key -> 中文显示名（数据层 key 不变，仅渲染层本地化）。
DIM_LABEL_ZH: dict[str, str] = {
    # compare 门禁名
    "comparability": "可比性",
    "accuracy_regression": "准确率回归",
    "stale_rate_regression": "陈旧率回归",
    "latency_p95_regression": "延迟 P95 回归",
    # 办公域槽位的业务红线（不参与基准融合，但单独进门禁）
    "office_accuracy_regression": "准确率回归（办公域）",
    "office_stale_rate_regression": "陈旧率回归（办公域）",
    "office_latency_p95_regression": "延迟 P95 回归（办公域）",
    # 可比性维度
    "adapter": "适配器",
    "adapter_version": "适配器版本",
    "dataset_sha256": "数据集指纹",
    "judge": "判分器",
    "judge_model": "判分模型",
    "driver_llm": "驱动 LLM",
    "platform": "平台",
    "commit": "commit",
    "retrieval_match": "检索匹配口径",
    "run_mode": "运行模式",
    "ingest_strategy": "摄入策略",
    "fusion": "融合口径",
    "limit": "题目上限",
    "categories": "题型过滤",
    "shards": "分片数",
    "shard_index": "分片序号",
    "n_questions": "题目数",
    "adapter_meta": "adapter 元数据",
    "dataset_path": "数据集路径",
    "dataset": "数据集",
}


def dim_label(key: str) -> str:
    """门禁 / 可比性维度 key 的中文显示名。未知 key 原样返回。"""
    return DIM_LABEL_ZH.get(key, str(key))


#: metadata 字段标签（run / evaluate / compare 报告顶部「评测配置」段）。
METADATA_FIELD_ZH: dict[str, str] = {
    "adapter": "adapter",
    "dataset": "数据集",
    "dialogue_id": "对话 id",
    "dataset_path": "测试集路径",
    "dataset_source": "测试集来源",
    "platform": "平台",
    "commit": "commit",
    "task_name": "任务名",
    "categories": "题型过滤",
    "judge": "判分器",
    "driver_llm": "驱动 LLM",
    "judge_model": "判分模型",
    "adapter_version": "adapter 版本",
    "adapter_meta": "adapter 元数据",
    "run_mode": "运行模式",
    "n_questions": "题目数",
    "n_ignored": "忽略题数",
    "timestamp": "时间戳",
    "limit": "题目上限",
    "categories": "题型过滤",
    "shards": "分片数",
    "shard_index": "分片序号",
    "shard_by": "分片维度",
    "dataset_fingerprint": "数据集指纹",
    "fused": "融合产物",
    "fused_from": "来源分片",
    "schema_version": "schema 版本",
    "membench_version": "membench 版本",
    "retrieval_match": "检索匹配口径",
    "retrieval_match_threshold": "检索匹配阈值",
    "retrieval_match_fallbacks": "检索匹配回退数",
    "ingest_strategy": "摄入策略",
    "as_of_questions": "带 as_of 的题数",
    "ingest_blocks": "摄入块",
}


def metadata_label(key: str) -> str:
    """metadata 字段的中文标签。未知 key 原样返回。"""
    return METADATA_FIELD_ZH.get(key, str(key))


#: ``state_snapshot.role`` -> 显示文案。``unsupported`` 是"请求了导出但适配器没声明快照
#: 能力"，不是快照坏掉：渲染成 ``n/a`` / ``vNone`` 会被读成产物质损，这里如实写成一句话。
SNAPSHOT_ROLE_ZH: dict[str, str] = {
    "replayed": "回放（读取侧口径）",
    "exported": "导出（本次为全量 run）",
    "unsupported": "不支持（适配器未声明快照能力）",
}


def snapshot_role_display(role: Any) -> str:
    """快照角色的显示文案；未知角色原样返回，缺失显示 ``n/a``。"""
    if role is None or role == "":
        return "n/a"
    return SNAPSHOT_ROLE_ZH.get(str(role), str(role))


#: ``trace_state.role`` -> 显示文案（--trace-state 的逐 as_of 状态）。
TRACE_ROLE_ZH: dict[str, str] = {
    "exported": "已导出",
    "unsupported": "不支持（适配器未声明快照能力）",
    "not_applicable": "不适用（本 run 没有 interleaved 对话）",
    "not_exported": "未导出",
}

#: 失败归因四桶 -> 中文显示名。
ATTRIBUTION_BUCKET_ZH: dict[str, str] = {
    "not_written": "未写入",
    "not_retrieved": "未检索到",
    "wrong_answer": "检索到但答错",
    "unknown": "不可判定",
}

#: 失败归因样例里问题 / 回答的截断长度（完整原文在 results.jsonl）。
ATTRIBUTION_SAMPLE_CHARS = 80


def trace_role_display(role: Any) -> str:
    """追踪状态角色的显示文案；未知角色原样返回，缺失显示 ``n/a``。"""
    if role is None or role == "":
        return "n/a"
    return TRACE_ROLE_ZH.get(str(role), str(role))


def ingest_blocks_display(blocks: dict[str, Any]) -> str:
    """「摄入块」一行的值：``31 块（4~60 条/块，中位 5）``。

    中位数可能是 .5（偶数个块），用 ``:g`` 去掉无意义的小数点。
    """

    def _num(value: Any) -> str:
        return f"{float(value):g}"

    return (
        f"{int(blocks.get('n') or 0)} 块"
        f"（{_num(blocks.get('min') or 0)}~{_num(blocks.get('max') or 0)} 条/块，"
        f"中位 {_num(blocks.get('median') or 0)}）"
    )


def shard_scope_text(metrics: dict[str, Any]) -> "str | None":
    """本片范围短标（``分片 1/2``）；不分片与 fuse 融合产物返回 None。

    ``shard_index`` 为 None 表示这份产物不是某一分片而是**整场融合结果**（fuse 只记
    分片总数）——它覆盖全部题目，标"仅覆盖部分题目"就成了假声明。
    """
    shards = metrics.get("shards")
    index = metrics.get("shard_index")
    if not isinstance(shards, int) or not isinstance(index, int) or shards <= 1:
        return None
    return f"分片 {index}/{shards}"


def shard_note(metrics: dict[str, Any]) -> "str | None":
    """分片产物的顶部标注文案；非分片产物返回 None（口径见 :func:`shard_scope_text`）。"""
    scope = shard_scope_text(metrics)
    if scope is None:
        return None
    by = str(metrics.get("shard_by") or "question")
    return f"{scope}（按 {by}）——仅覆盖部分题目"


# 验收基线（report.md 对照表）：(指标 key, 展示名「中文（英文 key）」, 目标描述, 判定函数)
BASELINES: list[tuple[str, str, str, Any]] = [
    ("accuracy", metric_label("accuracy"), ">= 0.85", lambda v: v >= 0.85),
    ("evidence_recall", metric_label("evidence_recall"), ">= 0.90", lambda v: v >= 0.90),
    ("stale_rate", metric_label("stale_rate"), "<= 0.03", lambda v: v <= 0.03),
    ("abstention_accuracy", metric_label("abstention_accuracy"), ">= 0.80", lambda v: v >= 0.80),
]

#: 对比报告的指标展示顺序：四项评级质量指标 + 四项诊断指标。
DIFF_METRICS: list[str] = [
    "accuracy",
    "token_f1",
    "bleu1",
    "evidence_recall",
    "abstention_accuracy",
    "stale_rate",
    "recall_at_5",
    "over_refusal_rate",
]

#: 效率成本指标（compare 图第二子图；单位各异，仅看相对高低，值标签直接标注）。
EFFICIENCY_DIFF_METRICS: list[str] = [
    "latency_p50_ms",
    "latency_p95_ms",
    "avg_input_tokens",
    "avg_output_tokens",
    "avg_ingest_latency_ms",
    "avg_llm_calls",
]

#: B 条高亮色：明显优化 / 明显劣化。图例里单列为语义项，不参与身份区分——
#: 否则 B 的身份色会被高亮色顶掉，读者看不出哪根是 B。
HIGHLIGHT_UP = "#2e8b57"
HIGHLIGHT_DOWN = "#cc3333"

#: A/B 变化高亮阈值：质量指标 diff 绝对值超过该值即视为"明显"优化/劣化，
#: 与 compare 门禁默认阈值（--acc-threshold / --stale-threshold 0.02）同口径。
HIGHLIGHT_THRESHOLD: float = 0.02

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
    """compare 用：A vs B 分组条形图，左质量指标、右效率成本。

    图例给「A（基线）· 短名 / B（候选）· 短名」两个身份项（短名由调用方从
    adapter / commit 拼出，不用 run_id——那会压住柱体且分不出 A/B）；B 条的
    「明显优化 / 明显劣化」高亮另列语义图例项，不冒充身份色。质量子图对
    "明显变化"高亮：diff（B-A，stale_rate 反向）绝对值 >
    :data:`HIGHLIGHT_THRESHOLD` 时 B 条染绿（优化）/ 红（劣化），其余保持
    身份色。效率子图单位各异，仅看相对高低并直接标注数值。
    """
    _setup_fonts()
    # 双侧都有非 None 值的指标才画（任一侧"未计算"时不出柱，避免把 None 画成 0）。
    keys = [
        k for k in DIFF_METRICS
        if overall_a.get(k) is not None and overall_b.get(k) is not None
    ]
    vals_a = [float(overall_a[k]) for k in keys]
    vals_b = [float(overall_b[k]) for k in keys]

    eff_keys = [
        k for k in EFFICIENCY_DIFF_METRICS
        if overall_a.get(k) is not None and overall_b.get(k) is not None
    ]
    eff_a = [float(overall_a[k]) for k in eff_keys]
    eff_b = [float(overall_b[k]) for k in eff_keys]

    id_handles = [
        Patch(facecolor=PALETTE[0], label=f"基线 · {label_a}"),
        Patch(facecolor=PALETTE[1], label=f"候选 · {label_b}"),
    ]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(max(12, 1.4 * len(keys)), 4.8))
    width = 0.38
    x = list(range(len(keys)))
    ax.bar([i - width / 2 for i in x], vals_a, width, color=PALETTE[0])
    colors_b = []
    for k, va, vb in zip(keys, vals_a, vals_b):
        diff = vb - va
        if k == "stale_rate":
            diff = -diff  # 越低越好，反向计"改进量"
        if diff > HIGHLIGHT_THRESHOLD:
            colors_b.append(HIGHLIGHT_UP)  # 明显优化
        elif diff < -HIGHLIGHT_THRESHOLD:
            colors_b.append(HIGHLIGHT_DOWN)  # 明显劣化
        else:
            colors_b.append(PALETTE[1])
    ax.bar([i + width / 2 for i in x], vals_b, width, color=colors_b)
    ax.set_xticks(x)
    ax.set_xticklabels([metric_label(k, with_key=False) for k in keys], rotation=20, ha="right")
    # 顶部留出图例带：柱体最高 1.0、值标签到 ~1.03，图例放 1.1~1.2 不可能压柱
    ax.set_ylim(0, 1.24)
    ax.set_ylabel("value")
    ax.set_title("质量指标 (A vs B)")
    q_handles = list(id_handles)
    if HIGHLIGHT_UP in colors_b:
        q_handles.append(
            Patch(facecolor=HIGHLIGHT_UP, label=f"B 明显优化（> +{HIGHLIGHT_THRESHOLD}）")
        )
    if HIGHLIGHT_DOWN in colors_b:
        q_handles.append(
            Patch(facecolor=HIGHLIGHT_DOWN, label=f"B 明显劣化（< -{HIGHLIGHT_THRESHOLD}）")
        )
    ax.legend(handles=q_handles, loc="upper center", fontsize=8, ncol=2, framealpha=0.92)
    for i, (va, vb) in enumerate(zip(vals_a, vals_b)):
        ax.text(i - width / 2, va + 0.01, f"{va:.2f}", ha="center", fontsize=8)
        ax.text(i + width / 2, vb + 0.01, f"{vb:.2f}", ha="center", fontsize=8)

    x2 = list(range(len(eff_keys)))
    ax2.bar([i - width / 2 for i in x2], eff_a, width, color=PALETTE[0])
    ax2.bar([i + width / 2 for i in x2], eff_b, width, color=PALETTE[1])
    ax2.set_xticks(x2)
    ax2.set_xticklabels([metric_label(k, with_key=False) for k in eff_keys], rotation=20, ha="right")
    # 效率值无上界：按最大值留出图例带（值标签到 ~1.02×max）
    ax2.set_ylim(0, max(1.0, max(eff_a + eff_b, default=1.0) * 1.45))
    ax2.set_ylabel("cost (units differ)")
    ax2.set_title("效率成本 (A vs B)")
    ax2.legend(handles=id_handles, loc="upper center", fontsize=8, ncol=2, framealpha=0.92)
    for i, (va, vb) in enumerate(zip(eff_a, eff_b)):
        ax2.text(i - width / 2, va, f"{va:.0f}", ha="center", va="bottom", fontsize=7)
        ax2.text(i + width / 2, vb, f"{vb:.0f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def fig_category_diff(
    by_category_a: dict[str, dict[str, Any]],
    by_category_b: dict[str, dict[str, Any]],
    label_a: str,
    label_b: str,
    out_path: str,
) -> str:
    """compare 用：分题型准确率 A vs B 横向分组条形图。

    图例同 :func:`fig_metric_diff`：明确 A（基线）/ B（候选）与短名；右侧留出
    图例带（``xlim`` 扩到 1.5），图例不压柱体。
    """
    _setup_fonts()
    cats = sorted(set(by_category_a) | set(by_category_b))
    accs_a = [float(by_category_a.get(c, {}).get("accuracy", 0.0)) for c in cats]
    accs_b = [float(by_category_b.get(c, {}).get("accuracy", 0.0)) for c in cats]

    y = list(range(len(cats)))
    height = 0.38
    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.6 * len(cats) + 1.5)))
    ax.barh([i + height / 2 for i in y], accs_a, height, color=PALETTE[0])
    ax.barh([i - height / 2 for i in y], accs_b, height, color=PALETTE[1])
    ax.set_yticks(y)
    ax.set_yticklabels([category_display(c) for c in cats])
    ax.invert_yaxis()
    # 柱体最长到 1.0（值标签到 0.95），1.05~1.5 留给图例
    ax.set_xlim(0, 1.5)
    ax.set_xlabel("accuracy")
    ax.set_title("分题型准确率对比 (A vs B)")
    ax.legend(
        handles=[
            Patch(facecolor=PALETTE[0], label=f"基线 · {label_a}"),
            Patch(facecolor=PALETTE[1], label=f"候选 · {label_b}"),
        ],
        loc="lower right",
        fontsize=8,
        framealpha=0.92,
    )
    for i, (va, vb) in enumerate(zip(accs_a, accs_b)):
        ax.text(min(va + 0.01, 0.95), i + height / 2, f"{va:.2f}", va="center", fontsize=8)
        ax.text(min(vb + 0.01, 0.95), i - height / 2, f"{vb:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def fig_dataset_overview(
    datasets: list[tuple[str, dict[str, Any]]],
    out_path: str,
) -> str:
    """evaluate 用：各数据集质量指标与效率成本分组条形图。

    缺失的指标（如数据集未标注 evidence 时 ``evidence_recall`` 无意义）留空
    不画柱——避免业务方误读为"召回率为 0"；仅绘制至少一个数据集实际产出
    该指标的列。

    Args:
        datasets: ``[(数据集显示名, 该集 overall 指标字典), ...]``。
        out_path: 输出 png 路径。
    """
    _setup_fonts()
    q_keys = [k for k in DIFF_METRICS if any(k in ov and ov.get(k) is not None for _, ov in datasets)]
    eff_keys = [k for k in EFFICIENCY_DIFF_METRICS if any(k in ov and ov.get(k) is not None for _, ov in datasets)]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(max(12, 1.4 * max(len(q_keys), len(eff_keys), 1)), 4.8)
    )
    width = min(0.8 / max(len(datasets), 1), 0.38)
    x = list(range(len(q_keys)))
    for di, (name, ov) in enumerate(datasets):
        vals = [float(ov.get(k)) if ov.get(k) is not None else float("nan") for k in q_keys]
        offset = (di - (len(datasets) - 1) / 2) * width
        ax.bar([i + offset for i in x], vals, width, label=name, color=PALETTE[di % len(PALETTE)])
    ax.set_xticks(x)
    ax.set_xticklabels([metric_label(k, with_key=False) for k in q_keys], rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("value")
    ax.set_title("各数据集质量指标")
    ax.legend(fontsize=8)

    x2 = list(range(len(eff_keys)))
    for di, (name, ov) in enumerate(datasets):
        vals = [float(ov.get(k)) if ov.get(k) is not None else float("nan") for k in eff_keys]
        offset = (di - (len(datasets) - 1) / 2) * width
        ax2.bar([i + offset for i in x2], vals, width, label=name, color=PALETTE[di % len(PALETTE)])
    ax2.set_xticks(x2)
    ax2.set_xticklabels([metric_label(k, with_key=False) for k in eff_keys], rotation=20, ha="right")
    ax2.set_ylabel("cost (units differ)")
    ax2.set_title("各数据集效率成本")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _nan_or(value: Any) -> float:
    """``None``（未计算）画成缺口而不是 0——"没算"不等于"分数为 0"。"""
    return float("nan") if value is None else float(value)


def _annotate_bars(ax: Any, container: Any, *, decimals: int = 0) -> None:
    """在柱顶标数值（比率按百分比）。

    柱高为 0 时柱体本身看不见，没标注就会被读成"没数据"——这和"分数为 0"
    是两回事，必须写出来；``NaN``（该线未计算该指标）跳过不标。
    """
    for bar in container:
        height = bar.get_height()
        if height != height:  # NaN
            continue
        text = f"{height * 100:.{decimals}f}%"
        positive = height >= 0
        ax.annotate(
            text,
            (bar.get_x() + bar.get_width() / 2, height),
            textcoords="offset points",
            xytext=(0, 2 if positive else -2),
            ha="center",
            va="bottom" if positive else "top",
            fontsize=5.5,
            rotation=90,
            color="#333333",
        )


def fig_series_fused(
    names: list[str],
    values_by_metric: "dict[str, list[Any]]",
    out_path: str,
    *,
    title: str = "融合层各指标（按线）",
) -> str:
    """多线对比用：融合层各指标 × 各线的分组柱。

    行不写表也能看出"哪条线在哪一项上抬起来了"：x 轴是指标（全是 [0,1] 比率），
    每个指标下一组柱对应各线。缺失值（该线未计算该指标）留空不画，避免被读成 0。

    Args:
        names: 各线显示名（顺序即柱的先后）。
        values_by_metric: ``{指标 key: [各线取值]}``。
        out_path: 输出 png 路径。
        title: 图标题。
    """
    _setup_fonts()
    metrics = list(values_by_metric)
    fig, ax = plt.subplots(figsize=(max(9.0, 1.5 * len(metrics)), 4.8))
    width = min(0.8 / max(len(names), 1), 0.3)
    x = list(range(len(metrics)))
    for li, name in enumerate(names):
        vals = [_nan_or(values_by_metric[m][li]) for m in metrics]
        offset = (li - (len(names) - 1) / 2) * width
        _annotate_bars(
            ax,
            ax.bar(
                [i + offset for i in x],
                vals,
                width,
                label=name,
                color=PALETTE[li % len(PALETTE)],
            ),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [metric_label(m, with_key=False) for m in metrics], rotation=20, ha="right"
    )
    # 上边界留白：柱顶标注（100%）不要顶到标题上
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def fig_series_per_dataset(
    names: list[str],
    per_dataset: "dict[str, dict[str, list[Any]]]",
    out_path: str,
    *,
    metrics: "list[tuple[str, str]] | None" = None,
) -> str:
    """多线对比用：逐数据集关键指标的分组柱（每个指标一个面板）。

    为什么按指标分面而不是按数据集分面：同一指标的面板里各数据集共用 y 轴
    （都是 [0,1] 比率），跨数据集高低一眼可比；若按数据集分面，读者要在
    不同面板间对着不同量纲比。

    缺失指标（该数据集未标注对应能力）留空不画柱；整个面板全无值时不建该
    面板——"没算"不能画成 0。

    Args:
        names: 各线显示名。
        per_dataset: ``{数据集: {指标: [各线取值]}}``。
        out_path: 输出 png 路径。
        metrics: ``[(指标 key, 面板标题), ...]``；缺省取四项质量指标
            （accuracy / evidence_recall / abstention_accuracy / stale_rate）。
    """
    _setup_fonts()
    if metrics is None:
        metrics = [
            ("accuracy", "正确率"),
            ("evidence_recall", "证据召回率"),
            ("abstention_accuracy", "拒答准确率"),
            ("stale_rate", "陈旧率（越低越好）"),
        ]
    panels = [
        (key, label)
        for key, label in metrics
        if any(
            any(value is not None for value in per_dataset[ds].get(key, []))
            for ds in per_dataset
        )
    ]
    if not panels:
        raise ValueError("没有可画的逐数据集指标（全部为 None）")
    cols = 2
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(max(10.0, 1.6 * len(per_dataset)), 3.6 * rows), squeeze=False
    )
    datasets = list(per_dataset)
    width = min(0.8 / max(len(names), 1), 0.3)
    x = list(range(len(datasets)))
    for index, (key, label) in enumerate(panels):
        ax = axes[index // cols][index % cols]
        for li, name in enumerate(names):
            vals = [_nan_or(per_dataset[ds].get(key, [None] * len(names))[li]) for ds in datasets]
            offset = (li - (len(names) - 1) / 2) * width
            _annotate_bars(
                ax,
                ax.bar(
                    [i + offset for i in x],
                    vals,
                    width,
                    label=name,
                    color=PALETTE[li % len(PALETTE)],
                ),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=15, ha="right")
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("value")
        ax.set_title(label)
        ax.legend(fontsize=7, loc="best", framealpha=0.9)
    for index in range(len(panels), rows * cols):
        axes[index // cols][index % cols].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def fig_series_delta(
    names: list[str],
    vs_base: list[Any],
    vs_prev: list[Any],
    out_path: str,
) -> str:
    """多线对比用：每条线的 accuracy 差值（vs 基线 / vs 上一步）分组柱。

    这张图直接回答"每一步优化提升多少"：两组柱并排，``vs 基线`` 看累计收益、
    ``vs 上一步`` 看单步收益；差值为负说明回退。基线自身没有差值，柱为空。

    Args:
        names: 各线显示名。
        vs_base: 各线 accuracy 相对基线的差值（小数，0.018 = +1.8pp）。
        vs_prev: 各线 accuracy 相对上一步的差值。
        out_path: 输出 png 路径。
    """
    _setup_fonts()
    fig, ax = plt.subplots(figsize=(max(7.5, 1.4 * len(names)), 4.2))
    x = list(range(len(names)))
    width = 0.36
    series = [("vs 基线（累计）", vs_base, PALETTE[0]), ("vs 上一步（单步）", vs_prev, PALETTE[1])]
    for si, (label, values, color) in enumerate(series):
        vals = [_nan_or(v) for v in values]
        offset = (si - 0.5) * width
        # 差值多是零点几个百分点，取 1 位小数（四舍五入到整数会把 18.5% → 19%）
        _annotate_bars(
            ax,
            ax.bar([i + offset for i in x], vals, width, label=label, color=color),
            decimals=1,
        )
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.margins(y=0.22)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Δaccuracy")
    ax.set_title("每条线的 accuracy 变化（正=提升）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def fig_series_accuracy(
    names: list[str],
    headline: list[Any],
    out_path: str,
    *,
    dataset_series: "dict[str, list[Any]] | None" = None,
    headline_label: str = "融合值",
) -> str:
    """多线对比用：accuracy 随线序的走势。

    x 轴是各线的报告名（中文按顺序即时间顺序），y 轴是 accuracy。多数据集时每条
    数据集各画一条细折线（同一套低饱和配色），融合值/单数据集汇总值画一条粗线；
    ``None``（该线未计算该值）留缺口，不按 0 画——"没算"不是"分数为 0"。

    Args:
        names: 各线的报告名（顺序即 x 轴）。
        headline: 各线的汇总 accuracy（多数据集为融合值，单数据集为该集的值）。
        out_path: 输出 png 路径。
        dataset_series: 多数据集时 ``{数据集名: [各线 accuracy]}``；单数据集传 None。
        headline_label: 粗线的图例名（多数据集「融合值」/ 单数据集「accuracy」）。
    """
    _setup_fonts()
    fig, ax = plt.subplots(figsize=(max(7.0, 1.5 * len(names)), 4.5))
    x = list(range(len(names)))

    def _ys(values: list[Any]) -> list[float]:
        return [float(v) if v is not None else float("nan") for v in values]

    if dataset_series:
        for i, (label, values) in enumerate(dataset_series.items()):
            ax.plot(
                x,
                _ys(values),
                marker="o",
                linewidth=1.2,
                color=PALETTE[(i + 1) % len(PALETTE)],
                label=label,
            )
    ax.plot(
        x,
        _ys(headline),
        marker="o",
        linewidth=2.8,
        color=PALETTE[0],
        label=headline_label,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("accuracy")
    ax.set_title("多线 accuracy 走势")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Markdown 报告
# ---------------------------------------------------------------------------

def _fmt(value: Any) -> str:
    """数值格式化：float 保留 3 位；``None``（未计算）显示 ``n/a``；其余原样。"""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _fmt_opt(value: Any) -> str:
    """可选值格式化：None / 空串显示为 ``n/a``，其余转字符串。"""
    if value is None or value == "":
        return "n/a"
    return str(value)


def stale_rate_display(overall: dict[str, Any]) -> str:
    """陈旧率单元格：区分"没有可机检题"与"真实 0%"，有覆盖缺口时标注覆盖率。

    - ``stale_n == 0``：本次没有任何可机检的 stale 题，显示
      ``n/a（无可机检的 stale 注解）``——不能渲染成「优秀（0.0%）」；
    - ``0 < stale_n < stale_total``：值后追加 ``（可机检 n/total）``，
      且不给评价标签（分母不是全部注解题，标签会被读成全量结论）；
    - 旧产物没有 ``stale_n`` 字段：沿用 `format_metric` 的旧渲染。
    """
    value = overall.get("stale_rate", "n/a")
    stale_n = overall.get("stale_n")
    if not isinstance(stale_n, int):
        return format_metric("stale_rate", value)
    if stale_n == 0:
        return "n/a（无可机检的 stale 注解）"
    stale_total = overall.get("stale_total")
    if isinstance(stale_total, int) and stale_n < stale_total:
        # 覆盖不全时不给评价标签：分母不是全部注解题，标签会被读成全量结论。
        return (
            f"{format_metric_value('stale_rate', value)}"
            f"（可机检 {stale_n}/{stale_total}）"
        )
    return format_metric("stale_rate", value)


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


#: 适配器自描述值的最大展示长度：超出即截断（完整值仍在 metrics.json）。
ADAPTER_DESCRIBE_VALUE_MAX = 200


def adapter_describe_rows(describe: Any) -> list[list[str]]:
    """``metrics["adapter_describe"]`` -> 报告表格行（非 dict / 空 dict 返回空列表）。

    自描述由适配器自由给出，可能是嵌套结构或超长配置：字符串原样展示，其余
    值转紧凑 JSON，并统一折叠换行、截断到 :data:`ADAPTER_DESCRIBE_VALUE_MAX`，
    避免把表格撑裂。该信息不参与评分，渲染只求可读。
    """
    if not isinstance(describe, dict) or not describe:
        return []
    rows: list[list[str]] = []
    for key, value in describe.items():
        if isinstance(value, str):
            text = value
        else:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")
        if len(text) > ADAPTER_DESCRIBE_VALUE_MAX:
            text = text[: ADAPTER_DESCRIBE_VALUE_MAX - 1] + "…"
        rows.append([str(key), text])
    return rows


def _human_bytes(value: Any) -> "str | None":
    """字节数的可读写法（``1.2 MB（1258291 字节）``）；非数值返回 None。"""
    try:
        size = int(value)
    except (TypeError, ValueError):
        return None
    for unit, step in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if size >= step:
            return f"{size / step:.1f} {unit}（{size} 字节）"
    return f"{size} 字节"


def _snapshot_entry(snapshot_dir: str) -> "dict[str, Any] | None":
    """读快照清单的 ``entry`` 段；目录 / 清单缺失或损坏按"读不到"处理（返回 None）。

    报告是只读消费者：清单不在只说明本次没有条目可写（``--no-snapshot``、适配器不
    支持快照、回放读的是源 run 的快照），不能因此把整份报告写成错误。
    """
    try:
        with open(os.path.join(snapshot_dir, MANIFEST_NAME), encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return None
    entry = manifest.get("entry") if isinstance(manifest, dict) else None
    return entry if isinstance(entry, dict) else None


def _entry_counts(entry: dict[str, Any]) -> str:
    """清单 ``entry`` 的规模简述：``3 session / 42 消息 / 14 题``（缺字段不写该段）。"""
    parts = [
        f"{entry[key]} {unit}"
        for key, unit in (
            ("n_sessions", "session"),
            ("n_messages", "消息"),
            ("n_questions", "题"),
        )
        if isinstance(entry.get(key), int)
    ]
    return " / ".join(parts)


def _path_cell(target: str, out_dir: str) -> str:
    """报告里的路径单元：目录存在就给相对链接，否则给不可点击的绝对路径。

    不可点击也要写绝对路径：报告可能被移动，绝对路径至少能说明"去哪找"。
    """
    if not os.path.isdir(target):
        return f"`{target}`"
    rel = os.path.relpath(target, out_dir)
    suffix = "" if rel.endswith(os.sep) else "/"
    return f"[`{rel}{suffix}`]({rel}{suffix})"


def dialogue_snapshot_rows(
    out_dir: str, state: "dict[str, Any] | None" = None
) -> list[list[str]]:
    """对话级报告里的快照行：可点击的相对链接 + 清单条目规模与体积。

    「跑完 ingest 就能翻记忆文件」是快照的第一价值，因此只在磁盘上真实存在目标
    目录时才给 markdown 链接——报告里不留打不开的相对路径。两种落点：

    - 普通 run：``<对话目录>/memory_snapshot/``（报告目录 ``report/`` 的
      ``../memory_snapshot/``）；
    - 回放 run：任务级内容寻址池里的该对话桶 ``<snapshots_dir>/<bucket>/``（相对
      报告目录可能跨几层，按 ``os.path.relpath`` 现算），并写明这个桶是复用还是新写。

    Args:
        out_dir: 对话级报告目录（``<实验目录>/runs/<dataset>/<对话>/report``）。
        state: 该对话 metrics 的 ``state_snapshot`` 块（回放时带 ``bucket`` /
            ``reused`` / ``snapshots_dir``）。

    Returns:
        ``[[项目, 值], ...]``；该对话没有快照时返回空列表。
    """
    state = state or {}
    if state.get("role") == "replayed" and state.get("snapshots_dir") and state.get("bucket"):
        bucket_dir = os.path.join(str(state["snapshots_dir"]), str(state["bucket"]))
        rows = [["本对话快照桶", _path_cell(bucket_dir, out_dir)]]
        rows.append(
            [
                "桶来源",
                "复用（池内已有同样内容的状态，未新增）"
                if state.get("reused")
                else "新写（该状态在池内首次出现）",
            ]
        )
        entry = _snapshot_entry(bucket_dir)
        if entry is None:
            return rows
        counts = _entry_counts(entry)
        if counts:
            rows.append(["快照条目", counts])
        size = _human_bytes(entry.get("bytes"))
        if size:
            rows.append(["快照体积", size])
        return rows

    snapshot_dir = os.path.normpath(os.path.join(out_dir, os.pardir, DIR_SNAPSHOT))
    if not os.path.isdir(snapshot_dir):
        return []
    rel = f"../{DIR_SNAPSHOT}/"
    rows = [["本对话快照目录", f"[`{rel}`]({rel})"]]
    entry = _snapshot_entry(snapshot_dir)
    if entry is None:
        return rows
    counts = _entry_counts(entry)
    if counts:
        rows.append(["快照条目", counts])
    size = _human_bytes(entry.get("bytes"))
    if size:
        rows.append(["快照体积", size])
    return rows


def dialogue_trace_rows(
    out_dir: str, state: "dict[str, Any] | None" = None
) -> list[list[str]]:
    """对话级报告里的逐 as_of 追踪状态链接（``--trace-state``）。

    与 :func:`dialogue_snapshot_rows` 同一原则：磁盘上真存在才给可点击链接。
    """
    state = state or {}
    if state.get("role") != "exported":
        return []
    trace_dir = os.path.normpath(os.path.join(out_dir, os.pardir, DIR_TRACE_STATE))
    if not os.path.isdir(trace_dir):
        return []
    rel = f"../{DIR_TRACE_STATE}/"
    return [["逐 as_of 追踪状态目录", f"[`{rel}`]({rel})"]]


def _baseline_rows(overall: dict[str, Any]) -> list[list[str]]:
    rows = []
    for key, label, target, ok in BASELINES:
        if key not in overall:
            rows.append([label, target, "n/a", "—", "❓"])
            continue
        raw = overall[key]
        # 指标为 None 表示"未计算"（数据集无该题型），不参与基线判定。
        if raw is None:
            rows.append([label, target, "未定义", "—", "—"])
            continue
        value = float(raw)
        rating = rate_metric(key, value)
        rows.append(
            [
                label,
                target,
                format_metric_value(key, value),
                rating if rating else "—",
                "✅" if ok(value) else "❌",
            ]
        )
    return rows


def write_run_report_md(
    out_dir: str,
    records: list[dict],
    metrics: dict[str, Any],
    *,
    with_figures: bool = True,
) -> str:
    """生成 report.md（章节按 SPEC 第 7 节，中文）。

    Args:
        out_dir: 报告目录（对话级 / 数据集级 / 任务级均可）。
        records: 参与本次报告的评测记录。
        metrics: ``aggregate`` 结果 + 元信息。
        with_figures: 是否在正文引用两张分题型 / 延迟图；对话级报告不出图，
            置 False 以免留下打不开的相对引用。

    Returns:
        写出的 report.md 路径。
    """
    overall = metrics.get("overall", {})
    by_category = metrics.get("by_category", {})

    lines: list[str] = []
    lines.append(f"# membench 评测报告 — {metrics.get('run_id', '')}")
    lines.append("")
    # 分片产物：先说清"这份报告不是全量结论"，它的题目数只是本片的
    shard_scope = shard_scope_text(metrics)
    if shard_scope:
        lines.append(
            f"> 📌 **{shard_note(metrics)}**：本报告只覆盖本分片负责的对话与题目，"
            "数据集级完整结论见 `fuse` 融合产物。"
        )
        lines.append("")

    # 0. 评测配置（硬需求：必须位于标题之后、正文第一段之前）
    lines.append("## 评测配置")
    lines.append("")
    fingerprint = metrics.get("dataset_fingerprint") or {}
    fp_sha = fingerprint.get("sha256") if isinstance(fingerprint, dict) else None
    config_rows = [
        [metadata_label("dataset_path"), f"`{metrics.get('dataset_path', metrics.get('dataset', 'n/a'))}`"],
        [metadata_label("dataset_source"), str(metrics.get("dataset_source", "n/a"))],
        [metadata_label("platform"), str(metrics.get("platform", "default"))],
        [metadata_label("commit"), str(metrics.get("commit") or "n/a")],
        *(
            [[metadata_label("task_name"), str(metrics["task_name_display"])]]
            if metrics.get("task_name_display")
            else []
        ),
        *(
            [[metadata_label("categories"), "、".join(metrics["categories"])]]
            if metrics.get("categories")
            else []
        ),
        [metadata_label("membench_version"), version_display(metrics.get("membench_version"))],
        [metadata_label("judge"), str(metrics.get("judge", "n/a"))],
        [metadata_label("driver_llm"), _fmt_opt(metrics.get("driver_llm"))],
        [metadata_label("judge_model"), _fmt_opt(metrics.get("judge_model"))],
        [metadata_label("adapter_version"), _fmt_opt(metrics.get("adapter_version"))],
        [metadata_label("dataset_fingerprint"), _fmt_opt(fp_sha)],
        [metadata_label("retrieval_match"), str(metrics.get("retrieval_match") or "substring")],
        [metadata_label("ingest_strategy"), str(metrics.get("ingest_strategy") or "sequence")],
    ]
    ingest_blocks = metrics.get("ingest_blocks")
    if isinstance(ingest_blocks, dict) and ingest_blocks:
        config_rows.append(
            [metadata_label("ingest_blocks"), ingest_blocks_display(ingest_blocks)]
        )
    if metrics.get("dialogue_id"):
        # 对话级报告：先声明这份报告只覆盖哪一段对话，避免被读成数据集级结论
        config_rows.insert(2, [metadata_label("dialogue_id"), f"`{metrics['dialogue_id']}`"])
    as_of_questions = int(metrics.get("as_of_questions") or 0)
    if as_of_questions:
        config_rows.append(
            [
                metadata_label("as_of_questions"),
                f"{as_of_questions}（"
                + ("已强制" if metrics.get("as_of_enforced", True) else "未强制，时点错位")
                + "）",
            ]
        )
    lines.append(_md_table(["项目", "值"], config_rows))
    lines.append("")
    # 非默认检索命中口径提示：与 substring 的历史跑分不可直接比较
    match_mode = str(metrics.get("retrieval_match") or "substring")
    if match_mode != "substring":
        match_thr = metrics.get("retrieval_match_threshold")
        lines.append(
            "> ⚠️ 检索命中口径：**"
            + match_mode
            + "**"
            + (f"（阈值 {match_thr}）" if match_thr else "")
            + "；与默认 substring 口径的跑分不可直接比较。"
        )
        lines.append("")
    # 检索诊断：条数规模 / 空转题数 / LLM 命中回退数。recall@5 的窗口只有 5 条，
    # 适配器返回 20~40 条时 top-5 天然偏低；不给条数，读者会把"窗口窄"读成
    # "检索差"。空转（retrieved 为空）与"检索到了但没进前 5"必须分开。
    retrieval_notes: list[str] = []
    retrieved_median = overall.get("retrieved_median")
    n_empty = overall.get("n_empty_retrieved")
    if isinstance(retrieved_median, (int, float)) and retrieved_median:
        window_note = (
            "——窗口远窄于返回列表，两个指标的差距主要来自窗口，不全是排序问题"
            if retrieved_median >= 10
            else "——窗口与返回列表规模接近，差距更能反映排序质量"
        )
        retrieval_notes.append(
            f"检索条数中位 **{retrieved_median:.0f}** 条（`recall@5` 只看前 5 条"
            f"{window_note}）。"
        )
    if isinstance(n_empty, int) and n_empty:
        retrieval_notes.append(
            f"**{n_empty} 道题检索结果为空**（`retrieved=[]`）：这类 0 命中不是"
            "「检索差」，而是「没东西可检索」——常见于 ingest 还没落盘就被查"
            "（接入侧异步写入），先查这一类再谈排序。"
        )
    fallbacks = metrics.get("retrieval_match_fallbacks")
    if isinstance(fallbacks, int) and fallbacks:
        retrieval_notes.append(
            f"LLM 命中口径回退 **{fallbacks}** 次（模型输出不可解析 / 调用失败时"
            "退回子串判定）：这些题的 evidence_hit 口径与其余题不同，复核时注意。"
        )
    if retrieval_notes:
        lines.append("> 检索诊断：")
        for note in retrieval_notes:
            lines.append(f"> - {note}")
        lines.append("")
    # 快照体检：疑似日志/轨迹文件会被状态扫描当记忆读，污染 safety 状态红线与归因。
    # 只提示不排除——若这类文件同时被检索覆盖，排掉会把真实泄露藏起来。
    state_scan = metrics.get("state_scan") or {}
    suspect_files = state_scan.get("suspect_files") or []
    if suspect_files:
        shown = "、".join(f"`{name}`" for name in suspect_files[:5])
        more = f" 等 {len(suspect_files)} 个" if len(suspect_files) > 5 else ""
        lines.append(
            f"> ⚠️ 快照里出现疑似日志/查询轨迹文件：{shown}{more}。状态扫描会把它们当"
            "记忆内容读取——safety 的「状态仍含被删除的串」与失败归因的"
            "「证据是否已写入」都可能把「日志里出现过」判成「记忆仍记得」。"
            "快照只应放**可检索记忆**；日志 / transcript / 逐轮归档请写到对话目录的 "
            "`runtime/`（框架不扫描），然后重跑。"
        )
        lines.append("")
    # 检索指标全 0 警示：疑似 evidence 与被检索文本格式不一致（摘要式存储）
    if (
        metrics.get("has_evidence")
        and overall.get("evidence_recall") == 0.0
        and overall.get("recall_at_5") == 0.0
    ):
        lines.append(
            "> ⚠️ 本数据集标注了 evidence，但「证据召回率」与「检索召回率@5」全为 0："
            "疑似 evidence 与被检索文本格式不一致（如记忆以摘要/改写形式存储），"
            "并非真实零召回。可尝试 `--evidence-match overlap|llm` 调整命中口径，"
            "或让适配器返回可核对的原文片段后重跑。"
        )
        lines.append("")
    # as_of 未强制：题目带时点标注却按整段 ingest 跑，问的是"结尾时才看得到"的状态
    if as_of_questions and not metrics.get("as_of_enforced", True):
        lines.append(
            f"> ⚠️ 本 run 按 **{metrics.get('ingest_strategy') or 'sequence'}** 摄入："
            f"{as_of_questions} 道题带 `as_of` 时点标注但**未强制**，题目看到的是整段"
            "对话的全部消息（时点错位）——更新型事实可能被按新值判分。需要「当时能看到"
            "什么」的口径，请用 `ingest_strategy=interleaved` 的数据集重跑。"
        )
        lines.append("")

    lines.append(f"- {metadata_label('adapter')}: `{metrics.get('adapter', 'n/a')}`")
    lines.append(f"- {metadata_label('dataset')}: `{metrics.get('dataset', 'n/a')}`")
    lines.append(f"- {metadata_label('judge')}: `{metrics.get('judge', 'n/a')}`")
    lines.append(f"- {metadata_label('timestamp')}: {metrics.get('timestamp', 'n/a')}")
    n_questions = str(overall.get("n_questions", len(records)))
    if shard_scope:
        # 分片产物：这个数字是**本片**的题数（不是整场），写清分母与完整结果的去向
        n_questions += f"（{shard_scope}，总量见融合报告）"
    lines.append(f"- {metadata_label('n_questions')}: {n_questions}")
    n_ignored = int(metrics.get("n_ignored", 0) or 0)
    if n_ignored:
        lines.append(f"- {metadata_label('n_ignored')}: {n_ignored}（ignore=true，未参与评分）")
    judge_fallback = metrics.get("judge_fallback")
    if judge_fallback:
        fb_count = int(judge_fallback.get("count", 0) or 0)
        fb_ratio = float(judge_fallback.get("ratio", 0.0) or 0.0)
        lines.append(
            f"- LLM judge 回退: {fb_count} 题（{fb_ratio:.1%}，judge=llm-fallback）"
        )
    # capabilities 落账在 metrics.json 顶层（has_evidence / has_stale /
    # has_abstention），直接读三个顶层字段。
    cap_items = [
        ("has_evidence", "evidence 标注"),
        ("has_stale", "可机检 stale 注解"),
        ("has_abstention", "abstention 题"),
    ]
    if any(key in metrics for key, _ in cap_items):
        cap_strs = [
            f"{label}={'✓' if metrics.get(key) else '✗'}"
            for key, label in cap_items
        ]
        lines.append(f"- 能力标注: {'、'.join(cap_strs)}")
    lines.append("")

    # 0.1 适配器自描述（可选；只有适配器给出非空 describe() 时才出现）
    describe_rows = adapter_describe_rows(metrics.get("adapter_describe"))
    if describe_rows:
        lines.append("### 适配器自描述")
        lines.append("")
        lines.append(
            "<sub>由适配器 `describe()` 提供，供了解被测系统的配置 / 版本；"
            "不参与评分，也不是 compare 的可比性维度。</sub>"
        )
        lines.append("")
        lines.append(_md_table(["配置项", "值"], describe_rows))
        lines.append("")

    # 1. 摘要表（六维核心指标）
    lines.append("## 摘要（六维核心指标）")
    lines.append("")
    summary_rows = [
        ["正确性", metric_label("accuracy"), format_metric("accuracy", overall.get("accuracy", "n/a"))],
        ["正确性", metric_label_with_role("token_f1"), format_metric("token_f1", overall.get("token_f1", "n/a"))],
        ["正确性", metric_label_with_role("bleu1"), format_metric("bleu1", overall.get("bleu1", "n/a"))],
        ["检索质量", metric_label("evidence_recall"), format_metric("evidence_recall", overall.get("evidence_recall", "n/a"))],
        ["检索质量", metric_label_with_role("recall_at_5"), format_metric("recall_at_5", overall.get("recall_at_5", "n/a"))],
        ["正确性", metric_label_with_role("rubric_item_recall"), format_metric("rubric_item_recall", overall.get("rubric_item_recall", "n/a"))],
        ["效率成本", metric_label("latency_p50_ms"), _fmt(overall.get("latency_p50_ms", "n/a"))],
        ["效率成本", metric_label("latency_p95_ms"), _fmt(overall.get("latency_p95_ms", "n/a"))],
        ["效率成本", metric_label("avg_input_tokens"), _fmt(overall.get("avg_input_tokens", "n/a"))],
        ["效率成本", metric_label("avg_llm_calls"), _fmt(overall.get("avg_llm_calls", "n/a"))],
        ["记忆动态", metric_label("stale_rate"), stale_rate_display(overall)],
        ["拒答", metric_label("abstention_accuracy"), format_metric("abstention_accuracy", overall.get("abstention_accuracy", "n/a"))],
        ["拒答", metric_label_with_role("over_refusal_rate"), format_metric("over_refusal_rate", overall.get("over_refusal_rate", "n/a"))],
    ]
    lines.append(_md_table(["维度", "指标", "值"], summary_rows))
    lines.append("")
    lines.append(
        "<sub>Token F1 / BLEU-1 为诊断指标：只测与参考答案的字面重合度，"
        "不代表语义正确性——正确但表述更完整的回答也会因字面重合低而失分。"
        "它们与检索召回率@5 / 过度拒答率 / 要点命中率一样照常展示具体值，"
        "但不参与综合评级。"
        "「拒答准确率」在 judge=llm 时是**混合口径**：词表已识别的拒答直接算，"
        "词表漏掉的改写式拒答由一次 LLM 复核补判（可用 `--no-abstention-recheck` 关闭，"
        "关掉后该指标退回词表口径、会系统性偏低）。</sub>"
    )
    lines.append("")
    # 有 stale 注解但没有任何可机检题：解释 n/a 的来源，避免被读成"零陈旧"。
    stale_n = overall.get("stale_n")
    stale_total = overall.get("stale_total")
    if (
        isinstance(stale_n, int)
        and isinstance(stale_total, int)
        and stale_n == 0
        and stale_total > 0
    ):
        lines.append(
            f"<sub>本数据集有 {stale_total} 道 stale 注解题，但当前判分器拿不到可机检的"
            "裸旧值（`gold_stale` 为空，仅有 `gold_stale_note` 描述）；陈旧率不参与评级。"
            "改用 LLM judge 可基于描述判定。</sub>"
        )
        lines.append("")

    # LLM judge 回退比例 >5% 时在指标表下方给出警示（判分质量可能退化）。
    if judge_fallback:
        fb_ratio = float(judge_fallback.get("ratio", 0.0) or 0.0)
        if fb_ratio > 0.05:
            lines.append(
                f"> ⚠️ LLM judge 回退比例 {fb_ratio:.1%} 超过 5%：大量题目由启发式兜底判分，"
                "判分质量可能退化，建议复核判分配置（base_url / API key / 模型可用性）后重跑。"
            )
            lines.append("")

    # 综合评级（质量指标连续点数均值；无可评级指标则不输出）
    overall_rated = overall_rating(overall)
    if overall_rated is not None:
        label, mean_points = overall_rated
        lines.append(f"**综合评级：{label}（{mean_points:.2f}/4.0）**")
        lines.append("")
        lines.append(
            "<sub>评级规则：取四项质量指标（accuracy / evidence_recall / "
            "abstention_accuracy / stale_rate）的连续点数"
            "（1.0~4.0，档内随数值线性变化）均值，阈值 3.5 / 2.5 / 1.5；"
            "token_f1 / bleu1（字面重合度，非语义正确性）与 "
            "recall_at_5 / over_refusal_rate / rubric_item_recall 为诊断指标，"
            "仅展示不参与评级（`rubric_item_recall` 是 judge_spec include 的逐条"
            "命中比例——accuracy 对这类题全命中才算对，两者差得越多说明判分口径"
            "把近失记成了完全答错）；"
            "stale_rate 的分母只含可机检的 stale 题（有裸 gold_stale；"
            "LLM judge 下含 gold_stale_note）；"
            "效率成本指标同样不参与评级。</sub>"
        )
        lines.append("")

    # 2. by_category 表 + 图
    # 1.1 GateMem 官方口径（U / A / F / MGS）：与六维指标并列，不替换。
    # 论文（arXiv:2606.18829）用 LLM judge + 结构化动作算 MGS = U·(1−A)·(1−F)；
    # 我们这里动作是**从文本推断**的，泄露用字面命中近似，state_leak 不参与。
    official = metrics.get("official") or {}
    if official:
        lines.append("## GateMem 官方口径（U / A / F / MGS）")
        lines.append("")
        lines.append(
            "<sub>对齐 GateMem 论文（arXiv:2606.18829）的三支柱与乘法主指标："
            "`U` 合法请求的有效性、`A` 越权泄露率、`F` 遗忘失败率，"
            "`MGS = U × (1−A) × (1−F)`。**差异**：官方由 LLM judge 依据隐藏标注"
            "（expected action / judge spec / leak targets）判决并做人工一致性验证；"
            "本实现的动作标签是**从回答文本推断**的（`action_source=inferred`），"
            "泄露用字面命中 `leak_targets` 近似，`state_leak`（记忆状态红线）"
            "不参与 MGS。因此数值不能与论文表格直接对比，只用于同一实现下的纵向比较。"
            "**本节是参考视图：不参与多数据集融合、综合评级与门禁**——融合与评级仍走"
            "六维指标（`accuracy` / `evidence_recall` / `abstention_accuracy` / `stale_rate`）。</sub>"
        )
        lines.append("")

        def _pct(key: str, hit: str, total: str) -> str:
            value = official.get(key)
            count = official.get(hit)
            n = official.get(total)
            if value is None:
                return "未定义"
            return f"{float(value) * 100:.1f}%" + (
                f"（{int(count)}/{int(n)}）" if count is not None and n else ""
            )

        official_rows = [
            ["U（utility 有效性）", _pct("U", "n_utility_correct", "n_utility"), "答对且动作是 answer"],
            ["OR（过度拒答率）", _pct("OR", "n_over_refusal", "n_utility"), "utility 题上动作不是 answer"],
            ["A（越权泄露率）", _pct("A", "n_access_violation", "n_access"), "泄露目标，或动作不在 {refuse, answer_redacted}"],
            ["F（遗忘失败率）", _pct("F", "n_forget_failure", "n_forget"), "复述被删信息，或动作不是 no_memory"],
            [
                "MGS（= U × (1−A) × (1−F)）",
                f"{float(official['MGS']) * 100:.1f}%" if official.get("MGS") is not None else "未定义",
                "三支柱缺一即为未定义（不猜）",
            ],
        ]
        lines.append(_md_table(["指标", "值", "口径"], official_rows))
        lines.append("")
        counts = official.get("action_counts") or {}
        if counts:
            lines.append(
                "<sub>推断动作分布：" + "、".join(
                    f"{name} {int(counts.get(name, 0))}"
                    for name in ("answer", "answer_redacted", "refuse", "no_memory")
                ) + "（safety 题只认 `no_memory`；`refuse` 在官方口径里是 evasive mismatch）。</sub>"
            )
            lines.append("")

    lines.append("## 分题型准确率")
    lines.append("")
    cat_rows = [
        [
            category_display(cat),
            format_metric("accuracy", by_category[cat].get("accuracy", "n/a")),
            format_metric("token_f1", by_category[cat].get("token_f1", "n/a")),
            str(by_category[cat].get("n", 0)),
        ]
        for cat in sorted(by_category.keys())
    ]
    lines.append(
        _md_table(
            ["题型", metric_label("accuracy"), metric_label_with_role("token_f1"), "题数"],
            cat_rows or [["(无)", "", "", ""]],
        )
    )
    lines.append("")
    if with_figures:
        lines.append(f"![分题型准确率]({FIG_CATEGORY})")
        lines.append("")
    # 2.1 最需要优化的题型：头条结论 + 可机检的失败形态
    worst = worst_categories(records)
    lines.append(f"## 最需要优化的题型（TOP {WORST_CATEGORY_TOP}）")
    lines.append("")
    if worst:
        lines.append(
            f"<sub>按准确率升序；只收题量 ≥ {CATEGORY_MIN_N} 的题型（小样本一两题"
            "翻转就是 25~50pt，会挤掉真正该修的大题型——它们仍在上一张表里）。"
            "「主要问题」只做可机检的粗分，且每个判错题**只归一类**（各分桶加总等于判错题数）："
            "拒答类看「未拒答 / 泄露红线 / 状态未遗忘」，其余看「证据未命中 / 检索到仍答错 / "
            "状态未遗忘 / 无判定依据（无证据标注）」。</sub>"
        )
        lines.append("")
        worst_rows = [
            [
                category_display(item["category"]),
                str(item["n"]),
                format_metric("accuracy", item["accuracy"]),
                str(item["n_wrong"]),
                item["issue"] or "—",
            ]
            for item in worst
        ]
        lines.append(
            _md_table(
                ["题型", "题数", metric_label("accuracy"), "判错题数", "主要问题"],
                worst_rows,
            )
        )
    else:
        lines.append(
            f"<sub>没有题量 ≥ {CATEGORY_MIN_N} 的题型（或该数据集无逐题记录），"
            "不列榜。</sub>"
        )
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
        [METRIC_ZH["avg output tokens / query"], _fmt(overall.get("avg_output_tokens", "n/a"))],
        [METRIC_ZH["avg ingest latency (ms)"], _fmt(overall.get("avg_ingest_latency_ms", "n/a"))],
        [METRIC_ZH["avg llm calls / query"], _fmt(overall.get("avg_llm_calls", "n/a"))],
    ]
    ingest_block = metrics.get("ingest_latency") or {}
    if ingest_block:
        # 同一对话的每题都记同一个对话级耗时：按题摊分会随"每题对话题数"变化，
        # 必须把对话级口径一起给出来，否则 locomo / beam 会显示夸张的单题成本
        eff_rows.extend(
            [
                ["对话级 ingest 总计（ms）", _fmt(ingest_block.get("total_ms", "n/a"))],
                ["对话级 ingest 中位（ms）", _fmt(ingest_block.get("median_ms", "n/a"))],
                ["对话数", str(ingest_block.get("n_dialogues", "n/a"))],
            ]
        )
    lines.append(_md_table(["指标", "值"], eff_rows))
    lines.append("")
    if ingest_block:
        lines.append(
            "<sub>**ingest 耗时的两个口径**：对话级（总计 / 中位）= 每个对话 ingest"
            " 一次的真实耗时；「按题摊分」= 把该对话耗时记到它每一道题上再按题平均，"
            "随「每题对话有多少题」变化，**不能读成单题写入成本**，也不要跨数据集横比。</sub>"
        )
        lines.append("")
    if with_figures:
        lines.append(f"![查询延迟分布]({FIG_LATENCY})")
        lines.append("")

    # 3.5 全流程 token 消耗（被测系统上报口径，不参与评级/门禁）
    lines.append("## Token 消耗（全流程）")
    lines.append("")
    usage = metrics.get("token_usage")
    lines.append(
        "<sub>被测系统（适配器）上报口径：ingest 按每次对话写入累计、query 逐题累计；"
        "不含判分器 / 驱动侧消耗。适配器未填报字段显示 0（= 未上报，非真实零消耗）。</sub>"
    )
    lines.append("")
    if usage:
        totals = token_usage_totals(usage)
        grand = (
            totals.get("ingest_input_tokens", 0)
            + totals.get("query_input_tokens", 0)
            + totals.get("query_output_tokens", 0)
        )
        token_rows = [
            [TOKEN_USAGE_FIELD_ZH["ingest_input_tokens"], str(totals.get("ingest_input_tokens", 0))],
            [TOKEN_USAGE_FIELD_ZH["query_input_tokens"], str(totals.get("query_input_tokens", 0))],
            [TOKEN_USAGE_FIELD_ZH["query_output_tokens"], str(totals.get("query_output_tokens", 0))],
            ["合计（ingest + query 输入 + 输出）", str(grand)],
            [TOKEN_USAGE_FIELD_ZH["ingest_llm_calls"], str(totals.get("ingest_llm_calls", 0))],
            [TOKEN_USAGE_FIELD_ZH["query_llm_calls"], str(totals.get("query_llm_calls", 0))],
            [TOKEN_USAGE_FIELD_ZH["n_dialogues"], str(usage.get("n_dialogues", "n/a"))],
            [TOKEN_USAGE_FIELD_ZH["n_questions"], str(usage.get("n_questions", "n/a"))],
        ]
        lines.append(_md_table(["项目", "值"], token_rows))
    else:
        lines.append("（本次 run 未落账 `token_usage`：适配器未上报或产物来自旧版本）")
    if isinstance(usage, dict) and usage.get("ingest_skipped"):
        lines.append("")
        lines.append(
            "<sub>⚠️ 本 run 为快照回放：**ingest 被跳过**，上表 ingest 行为 0 是"
            "「没跑」而非「零成本」。</sub>"
        )
    lines.append("")

    # 3.6 快照与回放（写入侧固定时必须在报告里说清，避免被当完整评测读）
    state = metrics.get("state_snapshot") or {}
    verify = metrics.get("replay_verify") or {}
    trace = metrics.get("trace_state") or {}
    # 对话级报告额外给"去哪翻记忆文件"：磁盘上真实存在才给链接（见 dialogue_snapshot_rows）
    snap_rows = (
        dialogue_snapshot_rows(out_dir, state) if metrics.get("dialogue_id") else []
    )
    if state or verify or snap_rows or trace:
        lines.append("## 快照与回放")
        lines.append("")
        role = str(state.get("role") or "")
        if role == "replayed":
            lines.append(
                "> ⚠️ **本 run 为快照回放（写入侧固定）**：跳过 reset + ingest，"
                "分数只覆盖**读取 / 检索 / 回答层**，不得与全量 run 直接对比"
                "（compare 会把 `run_mode` 判为关键维度不一致）。涉及写入侧改动"
                "的结论必须来自重新 ingest。"
            )
            lines.append("")
        snapshot_rows: list[list[str]] = []
        if state:
            snapshot_rows.append(["快照角色", snapshot_role_display(state.get("role"))])
            if role == "unsupported":
                # 适配器没声明快照能力：没有可回放的根目录、也没有格式版本与清单指纹，
                # 空值会被读成"快照坏了"，故只给对话数与原因
                snapshot_rows.append(["对话数", str(state.get("n_dialogues", "n/a"))])
                snapshot_rows.append(
                    [
                        "说明",
                        "适配器未声明快照能力，本 run 未导出快照、也无法用 `--replay` 免重跑 "
                        "ingest；适配器实现 `snapshot()` 并让 `snapshot_capabilities()` 返回 "
                        "supported 后即可导出（默认开启，不需要时用 `--no-snapshot` 关闭）。",
                    ]
                )
            else:
                replayable = str(metrics.get("ingest_strategy") or "sequence") == "sequence"
                snapshot_rows.append(
                    [
                        "快照根目录" + ("（`--replay` 入参）" if replayable else ""),
                        f"`{state.get('root', 'n/a')}`",
                    ]
                )
                if state.get("format_version") is not None:
                    snapshot_rows.append(["格式版本", f"v{state['format_version']}"])
                if state.get("manifest_sha256"):
                    snapshot_rows.append(["清单 sha256（逐对话清单合并）", f"`{state['manifest_sha256']}`"])
                snapshot_rows.append(["对话数", str(state.get("n_dialogues", "n/a"))])
                if role == "replayed":
                    snapshot_rows.append(["源 run", f"`{state.get('source_run', 'n/a')}`"])
                    snapshot_rows.append(["回放模式", str(state.get("replay_mode", "n/a"))])
        if role == "replayed" and state.get("snapshots_dir"):
            # 回放的状态已按内容去重写入任务级池：报告要指向桶，而不是对话目录
            snapshot_rows.append(
                ["快照池（任务级，内容寻址）", _path_cell(str(state["snapshots_dir"]), out_dir)]
            )
            n_buckets = state.get("n_buckets")
            if isinstance(n_buckets, int):
                reused = int(state.get("n_reused") or 0)
                new = int(state.get("n_new") or (n_buckets - reused))
                snapshot_rows.append(
                    ["桶数（本 run 复用 / 新写）", f"{n_buckets}（复用 {reused} / 新写 {new}）"]
                )
        if trace.get("enabled"):
            snapshot_rows.append(
                [
                    "逐 as_of 追踪状态（`--trace-state`）",
                    f"{trace_role_display(trace.get('role'))}；"
                    f"{int(trace.get('n_states') or 0)} 份",
                ]
            )
        if metrics.get("dialogue_id"):
            snapshot_rows.extend(dialogue_trace_rows(out_dir, trace))
        snapshot_rows.extend(snap_rows)
        lines.append(_md_table(["项目", "值"], snapshot_rows))
        lines.append("")
        if trace.get("enabled"):
            lines.append(
                "**失败归因**用的是「该题提问时点」的状态：interleaved 对话取 "
                "`trace_state/t<as_of>/`，sequence 对话取对话末尾的 "
                "`memory_snapshot/`。`trace_state.role=unsupported` 时没有逐时点状态，"
                "归因记 `unknown`（不猜）。"
            )
            lines.append("")
        if role == "exported":
            if str(metrics.get("ingest_strategy") or "sequence") == "sequence":
                lines.append(
                    "改**读取侧**代码后可用 `--replay <快照根目录>`（数据集目录，或含它的"
                    "实验目录）免重跑 ingest；改写入侧（抽取 / 更新 / 去重）则必须重新 ingest。"
                )
            else:
                lines.append(
                    "本 run 的摄入策略是 **interleaved / mixed**：`--replay` 只有对话末尾"
                    "一份快照，给不出每个 `as_of` 时点的状态，会被拒绝；改读取侧代码需要"
                    "重新 ingest（`--trace-state` 的逐时点状态只用于失败归因，不是回放入口）。"
                )
            lines.append("")
        if role == "replayed" and state.get("snapshots_dir"):
            lines.append(
                "本次回放的状态已按内容去重写入任务级快照池（桶名 = 状态内容 sha256 前 "
                "12 位）：同一源回放到另一组时内容一致的对话共用同一个桶，差异即"
                "「回放不保真」的证据；回放任务本身也可继续作为 `--replay` 源（链式回放）。"
            )
            lines.append("")
        if verify:
            if "error" in verify:
                lines.append(f"> ⚠️ 回放保真校验未完成：{verify['error']}")
            else:
                lines.append("### 回放保真校验（与源 run 逐题比对）")
                lines.append("")
                lines.append(
                    "<sub>这是**回放误差地板**：读取侧 A/B 的差值必须大于它才有意义。"
                    "未改动系统时答案应逐题一致。</sub>"
                )
                lines.append("")
                rate = verify.get("answer_match_rate")
                verify_rows = [
                    ["共同题数", f"{verify.get('n_common')}（源 {verify.get('n_source')} / 回放 {verify.get('n_replay')}）"],
                    ["答案不一致", str(verify.get("n_answer_diff"))],
                    ["检索不一致", str(verify.get("n_retrieved_diff"))],
                    ["判分不一致", str(verify.get("n_correct_diff"))],
                    ["答案一致率", f"{rate:.1%}" if isinstance(rate, (int, float)) else "n/a"],
                ]
                lines.append(_md_table(["项目", "值"], verify_rows))
                mismatches = verify.get("mismatch_qids") or []
                if mismatches:
                    lines.append("")
                    lines.append(f"- 不一致示例：{'、'.join('`' + str(q) + '`' for q in mismatches)}")
            lines.append("")

    # 3.9 safety 状态泄露（拒答 + 真遗忘才算对；与"回答是否泄露"是两回事）
    leak = metrics.get("state_leak") or {}
    n_safety = int(leak.get("n_safety") or 0)
    n_checked = int(leak.get("n_checked") or 0)
    n_leak = int(leak.get("n_leak") or 0)
    if n_safety:
        lines.append("## safety 状态泄露")
        lines.append("")
        if n_checked:
            rate = float(leak.get("rate") or 0.0)
            lines.append(
                f"- 状态泄露：{n_leak}/{n_checked}（{rate:.1%}）——"
                "拒答但记忆仍含被删除的敏感串"
            )
            n_echo = int(leak.get("n_echo_skipped") or 0)
            if n_echo:
                lines.append(
                    f"- 题面回显：{n_echo} 道题的状态扫描剔除了题面自带的 needle"
                    "（探针把值注入了状态，不能当「未遗忘」）——这些题只在未出现在"
                    "题面的 needle 上判定，剔除后无可判定 needle 的题记 `unknown`"
                )
            uncovered = n_safety - n_checked
            lines.append(
                f"- 覆盖率：{n_checked}/{n_safety} 道 safety 题的状态可读"
                + (
                    f"（其余 {uncovered} 道状态不可读，只按回答判分："
                    "`--trace-state` 关闭 / 适配器不支持快照 / 状态里没有可读文本）"
                    if uncovered
                    else ""
                )
            )
        else:
            # 区分"为什么没查"：显式关闭 / 适配器不支持 / 没开（含旧版本没自动开）
            # / 状态不可读。四者的处置完全不同，笼统一句"未检查"读者只能猜。
            trace_meta = metrics.get("trace_state") or {}
            role = str(trace_meta.get("role") or "")
            if role == "disabled":
                why = "本次显式 `--no-trace-state` 关闭"
            elif role == "unsupported":
                why = "适配器不支持快照，`snapshot_capabilities.supported=false`"
            elif trace_meta.get("enabled"):
                why = "有些时点的状态目录不可读：二进制 / 超大 / 没有可读文本"
            else:
                why = "本次未开启 `--trace-state`——interleaved + safety 本应自动开启，未开说明跑的是旧版本或手工关闭"
            lines.append(
                f"- 状态泄露：未检查（覆盖率 0/{n_safety}）——本 run {n_safety} 道 "
                f"safety 题的提问时点状态都不可读（{why}），safety 只按回答判分、"
                "状态未判定，记 `unknown`（不猜）"
            )
        lines.append("")
        lines.append(
            "<sub>这是**状态**里仍含敏感串，不等同于回答泄露：safety 题只有"
            "「拒答 + 真遗忘」才算对——回答是否泄露由 judge 按 `leak_targets` 判定，"
            "记忆里是否还留着被删除的串由该题提问时点的状态扫描判定，任一成立即"
            "判错。分母只含**状态可读**的 safety 题；状态不可读时**不改判**"
            "（记 `unknown`，不猜）。</sub>"
        )
        lines.append("")

    # 4. 失败归因（只有存在判错题时出现）
    attribution = metrics.get("attribution") or {}
    if int(attribution.get("n_wrong") or 0) > 0:
        lines.append("## 失败归因")
        lines.append("")
        lines.append(
            "<sub>回答「为什么答错」：只有**判错**且带 evidence 标注的题进入四桶——"
            "证据不在该题提问时点的状态里（`not_written`）→ 在状态里但没被检索进"
            "回答上下文（`not_retrieved`）→ 两者都有仍答错（`wrong_answer`，推理 / 表述"
            "问题）；状态读不到 / 没有快照记 `unknown`（不猜）。既然证据已经被检索进"
            "回答上下文（`evidence_hit=true`），失败就只可能在回答——这类题**不需要**"
            "状态快照也直接落 `wrong_answer`，所以没开 `--trace-state` 的运行里这一桶"
            "仍会有值。没有 evidence 标注的"
            "判错题无法判断「证据是否已写入」，单独计数、不塞进任何桶。</sub>"
        )
        lines.append("")
        lines.append(
            f"- 判错 {int(attribution.get('n_wrong') or 0)} 题："
            f"带 evidence {int(attribution.get('n_wrong_with_evidence') or 0)}、"
            f"无 evidence（无法归因）{int(attribution.get('n_wrong_without_evidence') or 0)}；"
            f"提问时点状态可读 {int(attribution.get('n_state_readable') or 0)} 题"
        )
        lines.append("")
        by_category = attribution.get("by_category") or {}
        samples = attribution.get("samples") or []
        cat_rows = []
        for cat in sorted(by_category):
            info = by_category[cat]
            sample_qids = [
                f"`{s.get('qid')}`（{ATTRIBUTION_BUCKET_ZH.get(str(s.get('bucket')), str(s.get('bucket')))}）"
                for s in samples
                if str(s.get("category")) == str(cat)
            ]
            cat_rows.append(
                [
                    category_display(cat),
                    str(info.get("n_wrong", 0)),
                    str(info.get("n_wrong_with_evidence", 0)),
                    *(str(info.get(bucket, 0)) for bucket in ATTRIBUTION_BUCKETS),
                    "、".join(sample_qids) or "—",
                ]
            )
        lines.append(
            _md_table(
                [
                    "题型",
                    "判错题",
                    "带 evidence",
                    *[ATTRIBUTION_BUCKET_ZH[b] for b in ATTRIBUTION_BUCKETS],
                    "样例 qid（桶）",
                ],
                cat_rows or [["(无)", "0", "0", "0", "0", "0", "0", "—"]],
            )
        )
        lines.append("")
        if not int(attribution.get("n_wrong_with_evidence") or 0):
            lines.append(
                "（本次判错题都没有 evidence 标注：四桶全为 0 是「无法归因」，"
                "不是「没有失败」。）"
            )
            lines.append("")
        record_by_key = {
            (str(r.get("dialogue_id", "")), str(r.get("qid", ""))): r for r in records
        }

        def _short(text: Any) -> str:
            value = str(text or "")
            if len(value) > ATTRIBUTION_SAMPLE_CHARS:
                value = value[: ATTRIBUTION_SAMPLE_CHARS - 1] + "…"
            return value

        sample_rows = []
        for sample in samples:
            record = record_by_key.get(
                (str(sample.get("dialogue_id", "")), str(sample.get("qid", "")))
            )
            if record is None:
                continue
            in_state = record.get("evidence_in_state")
            sample_rows.append(
                [
                    ATTRIBUTION_BUCKET_ZH.get(
                        str(sample.get("bucket")), str(sample.get("bucket"))
                    ),
                    f"`{sample.get('qid')}`",
                    _short(record.get("question")),
                    _short(record.get("prediction")),
                    "null" if in_state is None else ("true" if in_state else "false"),
                    (
                        "null"
                        if record.get("evidence_hit") is None
                        else ("true" if record.get("evidence_hit") else "false")
                    ),
                ]
            )
        if sample_rows:
            lines.append("### 归因样例")
            lines.append("")
            lines.append(
                _md_table(
                    [
                        "归因",
                        "qid",
                        "问题",
                        "回答摘要",
                        "evidence_in_state",
                        "evidence_hit",
                    ],
                    sample_rows,
                )
            )
            lines.append("")

    # 4.1 失败样例 Top10
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

    out_path = os.path.join(out_dir, FILE_REPORT_MD)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def render_dialogue_report(
    out_dir: str, records: list[dict], metrics: dict[str, Any]
) -> str:
    """写对话级 ``report.md``（纯文字，不出图）。

    对话的题目往往只有个位数，出图既没信息量又会把产物撑成两倍；整场评测的
    图表留给数据集级报告。因此这里也不引用图片，避免留下打不开的相对路径。
    正文除六维指标外另给两件对话级专属信息：该对话的**记忆快照**（
    ``../memory_snapshot/`` 的链接与清单条目规模，跑完 ingest 就能翻记忆文件），
    以及分片产物（``shards > 1``）的"仅覆盖部分题目"标注。
    """
    os.makedirs(out_dir, exist_ok=True)
    return write_run_report_md(out_dir, records, metrics, with_figures=False)


def render_dataset_report(
    out_dir: str, records: list[dict], metrics: dict[str, Any]
) -> None:
    """写数据集级报告：``report.md`` + 分题型准确率图 + 查询延迟分布图。"""
    os.makedirs(out_dir, exist_ok=True)
    fig_category_accuracy(
        metrics.get("by_category", {}), os.path.join(out_dir, FIG_CATEGORY)
    )
    fig_latency_hist(
        [r.get("latency_ms", 0.0) for r in records],
        float(metrics.get("overall", {}).get("latency_p50_ms", 0.0)),
        float(metrics.get("overall", {}).get("latency_p95_ms", 0.0)),
        os.path.join(out_dir, FIG_LATENCY),
    )
    write_run_report_md(out_dir, records, metrics, with_figures=True)


def write_compare_report_md(
    out_dir: str,
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
    diffs: dict[str, Any],
    gates: list[dict[str, Any]],
    verdict: str,
    comparability: "dict[str, Any] | None" = None,
) -> str:
    """生成 comparison.md：逐指标 diff 表 + 门禁结论（中文）。

    ``comparability`` 非 None 时附加「可比性前置校验」小节：逐维度对照表 +
    不可比原因列表，说明 A/B 是否具备结果归因的前提。
    """
    id_a = str(metrics_a.get("run_id", "A"))
    id_b = str(metrics_b.get("run_id", "B"))

    lines: list[str] = []
    lines.append(f"# membench 对比报告 — {id_a} vs {id_b}")
    lines.append("")
    verdict_icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(verdict, "")
    lines.append(f"**门禁结论： {verdict} {verdict_icon}**")
    lines.append("")
    lines.append(f"- 基线: `{id_a}`")
    rated_a = overall_rating(metrics_a.get("overall", {}))
    if rated_a is not None:
        lines.append(f"- A 综合评级：{rated_a[0]}（{rated_a[1]:.2f}/4.0）")
    lines.append(f"- 候选: `{id_b}`")
    rated_b = overall_rating(metrics_b.get("overall", {}))
    if rated_b is not None:
        lines.append(f"- B 综合评级：{rated_b[0]}（{rated_b[1]:.2f}/4.0）")
    lines.append("")
    lines.append(
        "<sub>评级取四项质量指标（accuracy / evidence_recall / "
        "abstention_accuracy / stale_rate）连续点数均值；token_f1 / bleu1"
        "（字面重合度，非语义正确性）与 recall_at_5 / over_refusal_rate"
        "为诊断指标，仅展示不参与评级；stale_rate 的分母只含可机检的 stale 题"
        "（有裸 gold_stale；LLM judge 下含 gold_stale_note）；"
        "效率成本指标同样不参与评级。</sub>"
    )
    lines.append("")

    lines.append("## 逐指标 diff（B - A）")
    lines.append("")
    rows = []
    for key in sorted(diffs.keys()):
        d = diffs[key]
        rows.append(
            [
                metric_label(key),
                format_metric_value(key, d["a"]) if "a" in d else "n/a",
                format_metric_value(key, d["b"]) if "b" in d else "n/a",
                format_metric_delta(key, d["diff"]) if "diff" in d else "n/a",
            ]
        )
    lines.append(_md_table(["指标", "基线", "候选", "diff"], rows or [["(无)", "", "", ""]]))
    lines.append("")
    lines.append(f"![关键指标对比]({FIG_METRIC_DIFF})")
    lines.append("")
    lines.append(f"![分题型准确率对比]({FIG_CATEGORY_DIFF})")
    lines.append("")
    short_a, short_b = ab_series_labels(metrics_a, metrics_b)
    lines.append(
        f"<sub>图中 **A（基线）** = `{short_a}`、**B（候选）** = `{short_b}`；"
        "B 条染绿/红表示该项明显优化/劣化（|diff| > 0.02，stale_rate 反向）。"
        "完整 run_id 见上方。</sub>"
    )
    lines.append("")

    lines.append("## 门禁明细")
    lines.append("")
    gate_rows = []
    for g in gates:
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "SKIP": "⏭️"}.get(str(g.get("status", "")), "")
        gate_rows.append([dim_label(str(g.get("name", ""))), str(g.get("status", "")), icon, str(g.get("detail", ""))])
    lines.append(_md_table(["门禁", "状态", "", "说明"], gate_rows or [["(无)", "", "", ""]]))
    lines.append("")

    if comparability is not None:
        lines.append("## 可比性前置校验")
        lines.append("")
        checks = comparability.get("checks", []) or []
        comp_rows = []
        for c in checks:
            same = c.get("same")
            same_txt = "是" if same else ("否" if same is False else "—")
            comp_rows.append(
                [
                    dim_label(str(c.get("dimension", ""))),
                    _fmt_opt(c.get("a")),
                    _fmt_opt(c.get("b")),
                    same_txt,
                    str(c.get("detail", "")),
                ]
            )
        lines.append(
            _md_table(["维度", "A", "B", "是否一致", "说明"], comp_rows or [["(无)", "", "", "", ""]])
        )
        lines.append("")
        lines.append("### 不可比原因")
        lines.append("")
        reasons = comparability.get("reasons", []) or []
        if reasons:
            lines.extend(f"- {r}" for r in reasons)
        else:
            lines.append("（无可比性问题）")
        lines.append("")

    out_path = os.path.join(out_dir, FILE_COMPARISON_MD)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def ab_series_labels(metrics_a: dict[str, Any], metrics_b: dict[str, Any]) -> tuple[str, str]:
    """A/B 图的系列短名（adapter@commit 优先，退回 run_id）。

    图例不用 run_id：那既长（压住柱体）又分不出哪根是基线——短名 + 「A（基线）/
    B（候选）」前缀才是可读的身份标识。两侧短名相同时退回 run_id 保证可区分。
    """

    def _short(metrics: dict[str, Any]) -> str:
        adapter = str(metrics.get("adapter") or "").strip()
        commit = metrics.get("commit")
        if adapter and commit:
            return f"{adapter}@{str(commit)[:8]}"
        return adapter or str(metrics.get("run_id") or "?")

    label_a, label_b = _short(metrics_a), _short(metrics_b)
    if label_a == label_b:
        label_a = str(metrics_a.get("run_id") or label_a)
        label_b = str(metrics_b.get("run_id") or label_b)
    return label_a, label_b


def render_compare_report(
    out_dir: str,
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
    diffs: dict[str, Any],
    gates: list[dict[str, Any]],
    verdict: str,
    comparability: "dict[str, Any] | None" = None,
) -> None:
    """生成对比报告目录下的 comparison.md 与两张对比图
    （fig_metric_diff.png 质量+效率、fig_category_diff.png 分题型）。

    图例用短名（``adapter@commit``）而非 run_id，完整 run_id 见报告正文头部。
    """
    os.makedirs(out_dir, exist_ok=True)
    label_a, label_b = ab_series_labels(metrics_a, metrics_b)
    fig_metric_diff(
        metrics_a.get("overall", {}),
        metrics_b.get("overall", {}),
        label_a,
        label_b,
        os.path.join(out_dir, FIG_METRIC_DIFF),
    )
    fig_category_diff(
        metrics_a.get("by_category", {}),
        metrics_b.get("by_category", {}),
        label_a,
        label_b,
        os.path.join(out_dir, FIG_CATEGORY_DIFF),
    )
    write_compare_report_md(out_dir, metrics_a, metrics_b, diffs, gates, verdict, comparability)


# ---------------------------------------------------------------------------
# 报告导出（把报告连同图片与数据复制到目标目录）
# ---------------------------------------------------------------------------

_IMG_REF_RE = re.compile(r"(!\[[^\]]*\]\()([^)]+)(\))")


def copy_report(
    md_path: str,
    dest_dir: str,
    dest_filename: str,
    extra_files: "Iterable[str] | None" = None,
) -> str:
    """把 markdown 报告复制到指定目录，并连带复制其引用的本地图片。

    图片按原文件名复制到同一目标目录（报告与图片始终同目录同名引用）；
    非本地引用（http/https）保持不变。

    Args:
        md_path: 源报告 markdown 路径。
        dest_dir: 目标目录（不存在则创建）。
        dest_filename: 目标 md 文件名（如 ``report.md`` / ``comparison.md``）。
        extra_files: 额外一并复制的文件路径（如数据集目录下的
            ``metrics.json`` / ``results.jsonl``），按原文件名复制；
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
        new_name = os.path.basename(ref)
        dest_img = os.path.join(dest_dir, new_name)
        # 目标目录就是源目录时（报告与图片已在位）跳过，否则 copyfile 抛 SameFileError
        if os.path.isfile(src_img) and os.path.abspath(src_img) != os.path.abspath(dest_img):
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
