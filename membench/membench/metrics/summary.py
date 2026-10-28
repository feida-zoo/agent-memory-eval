"""六维指标汇总（SPEC 3.5）。

``aggregate(records)`` 消费 runner 写入 results.jsonl 的单条评测记录
（SPEC 3.4），输出 ``{"overall": {...}, "by_category": {...}}``；
run_id / adapter / dataset / judge / timestamp 等元信息由 runner 补充。

分位数为手写线性插值实现，不依赖 numpy / pandas。
"""

from __future__ import annotations

from typing import Any, Iterable

from membench.data.schema import is_abstention

__all__ = ["percentile", "aggregate"]


def percentile(values: Iterable[float], p: float) -> float:
    """线性插值分位数（与 numpy 默认 'linear' 方法一致）。

    Args:
        values: 数值序列。
        p: 分位点，取值 [0, 100]。

    Returns:
        分位数值；空序列返回 0.0。
    """
    xs = sorted(float(v) for v in values)
    n = len(xs)
    if n == 0:
        return 0.0
    if n == 1:
        return xs[0]
    rank = (p / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


def _mean(values: Iterable[float]) -> float | None:
    """算术均值；空序列返回 ``None``（显式区分"未计算"与"真实 0"）。

    返回 ``None`` 让 ``format_metric`` / ``fig_*`` / ``fuse_metrics`` 一律走
    "未定义"路径，避免旧版把"数据集无 evidence"误算成"召回率为 0"。
    """
    xs = [float(v) for v in values]
    return sum(xs) / len(xs) if xs else None


def aggregate(
    records: list[dict[str, Any]],
    capabilities: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """汇总单条评测记录为六维指标字典（SPEC 3.5）。

    Args:
        records: results.jsonl 风格的记录列表，每条含
            ``category / gold_answer / evidence_hit / token_f1 / bleu1 /
            latency_ms / input_tokens / llm_calls / ingest_llm_calls`` 以及
            ``judge``（含 correct / abstained_pred / stale）字段。
        capabilities: 可选，由 runner 从题目集扫描得到的能力标注
            （``has_evidence / has_stale / has_abstention``）；传 None 时
            ``aggregate`` 从 records 自身推断（仅以"是否有非 abstention
            题 / abstention 题 / 带 stale 注解的题"为依据——这是
            fallback，主要路径是 runner 直接传 capabilities）。
            ``has_stale`` 最终以 records 的 ``judge.stale_checkable`` 为准
            （见 :func:`membench.judges.base.JudgeResult`）：传进来的静态
            标注只说明数据集有注解，不保证本次判分真的能机检。

    Returns:
        ``{"overall": {...}, "by_category": {category: {"accuracy", "n"}},
        "has_evidence": bool, "has_stale": bool, "has_abstention": bool}``；
        ``overall`` 另含 ``stale_n``（stale_rate 分母：可机检题数）、
        ``stale_total``（有 stale 注解的题数）与 ``evidence_n`` / ``abstention_n``
        （非拒答题数 / 拒答题数，分别是检索类与拒答类指标的分母）：报告据此
        标注覆盖率，跨数据集加权融合据此取权重（见
        :func:`membench.evaluate.fuse_metrics`）；无对应题型的指标返回 ``None``
        （区别于"真实 0"）；使用了 LLM judge 时附加
        ``"judge_fallback": {"count", "ratio"}``。

    指标口径：
        - accuracy: ``judge.correct`` 均值。
        - evidence_recall: ``evidence_hit`` 均值（仅非拒答题）。
        - recall_at_5: 非拒答题上 ``recall@5`` 均值（严格检索覆盖：证据条目
          进前 5 条检索结果的比例；仅 has_evidence 数据集参与——诊断指标，
          不参与综合评级）。
        - abstention_accuracy: 拒答题（category 为 abstention 或 gold 为拒答
          哨兵值，见 ``schema.is_abstention``）上 ``judge.abstained_pred`` 均值。
        - over_refusal_rate: 非拒答题上 ``judge.abstained_pred`` 均值
          （"不该拒答却拒答"的比例；诊断指标，不参与综合评级）。
        - stale_rate: ``judge.stale_checkable`` 为真的题上 ``judge.stale``
          均值；无这类题时为 ``None``（不把"没有判定依据"记成 0）。
        - latency_p50_ms / latency_p95_ms: ``latency_ms`` 分位数（线性插值）。
        - avg_input_tokens / avg_output_tokens: 每题输入 / 输出 token 均值。
        - avg_ingest_latency_ms: 每题 ingest 侧耗时均值（"记记忆"本身的开销）。
        - avg_llm_calls: 每题 ``llm_calls + ingest_llm_calls`` 均值
          （ingest 成本按 runner 的记录归属口径计入）。
        - by_category 除 accuracy 外附带 token_f1 均值（分题型质量）。
    """
    n = len(records)

    def _judge(rec: dict[str, Any]) -> dict[str, Any]:
        return rec.get("judge", {}) or {}

    def _is_abstention_rec(rec: dict[str, Any]) -> bool:
        return is_abstention(
            str(rec.get("category", "")), str(rec.get("gold_answer", ""))
        )

    non_abstention = [r for r in records if not _is_abstention_rec(r)]
    abstention = [r for r in records if _is_abstention_rec(r)]
    # stale 分母：本次判分确实拿到可用旧值的题（不是"有 gold_stale 注解"）。
    stale_checkable_recs = [r for r in records if _judge(r).get("stale_checkable")]
    stale_n = len(stale_checkable_recs)
    # 覆盖率分母：有 stale 注解的题（裸旧值或原始描述），不依赖判分器。
    stale_total = sum(1 for r in records if _judge(r).get("stale_annotated"))

    # capabilities：优先用 runner 传入的（基于题目集扫描，含被忽略题统计），
    # fallback 到从 records 推断（仅看 records 中出现过的 category / evidence_hit）。
    if capabilities is not None:
        caps = dict(capabilities)
    else:
        caps = {
            "has_evidence": any(bool(r.get("evidence_hit")) for r in non_abstention),
            "has_stale": stale_total > 0,
            "has_abstention": bool(abstention),
        }

    overall = {
        "accuracy": _mean(bool(_judge(r).get("correct")) for r in records),
        "token_f1": _mean(r.get("token_f1", 0.0) for r in records),
        "bleu1": _mean(r.get("bleu1", 0.0) for r in records),
        # capabilities 为 False 时显式 None（"未计算"），而非把"无该题型"
        # 误算成 0（污染总分/评级）。
        "evidence_recall": (
            _mean(bool(r.get("evidence_hit")) for r in non_abstention)
            if caps.get("has_evidence", False)
            else None
        ),
        # 严格检索覆盖（诊断指标，不参与评级）：非拒答题上前 5 条检索的
        # 证据条目覆盖率（results.jsonl 的 recall@5）；无 evidence 标注的
        # 数据集为 None。
        "recall_at_5": (
            _mean(float(r.get("recall@5", 0.0)) for r in non_abstention)
            if caps.get("has_evidence", False)
            else None
        ),
        "abstention_accuracy": (
            _mean(bool(_judge(r).get("abstained_pred")) for r in abstention)
            if caps.get("has_abstention", False)
            else None
        ),
        # 过度拒答率（诊断指标，不参与评级）：非拒答题上出现拒答信号的
        # 比例——与 abstention_accuracy（漏拒）互为两个方向。
        "over_refusal_rate": _mean(
            bool(_judge(r).get("abstained_pred")) for r in non_abstention
        ),
        "stale_rate": (
            _mean(bool(_judge(r).get("stale")) for r in stale_checkable_recs)
            if stale_n
            else None
        ),
        # stale_n=分母（可机检）；stale_total=有 stale 注解的题数（含不可机检）。
        # 两者都落 overall，报告据此渲染 n/a 与覆盖率。
        "stale_n": stale_n,
        "stale_total": stale_total,
        # 检索类（evidence_recall / recall_at_5 / over_refusal_rate）与拒答类的
        # 分母：跨数据集加权融合按它取权重，不按总题数（见 fuse_metrics）。
        "evidence_n": len(non_abstention),
        "abstention_n": len(abstention),
        "latency_p50_ms": percentile((r.get("latency_ms", 0.0) for r in records), 50),
        "latency_p95_ms": percentile((r.get("latency_ms", 0.0) for r in records), 95),
        "avg_input_tokens": _mean(r.get("input_tokens", 0) for r in records),
        "avg_output_tokens": _mean(r.get("output_tokens", 0) for r in records),
        "avg_ingest_latency_ms": _mean(r.get("ingest_latency_ms", 0.0) for r in records),
        "avg_llm_calls": _mean(
            float(r.get("llm_calls", 0)) + float(r.get("ingest_llm_calls", 0))
            for r in records
        ),
        "n_questions": n,
    }

    by_category: dict[str, dict[str, Any]] = {}
    for rec in records:
        cat = str(rec.get("category", "unknown"))
        bucket = by_category.setdefault(cat, {"_correct": 0.0, "_f1": 0.0, "n": 0})
        bucket["_correct"] += 1.0 if _judge(rec).get("correct") else 0.0
        bucket["_f1"] += float(rec.get("token_f1", 0.0))
        bucket["n"] += 1
    for cat, bucket in by_category.items():
        bucket["accuracy"] = bucket["_correct"] / bucket["n"] if bucket["n"] else 0.0
        bucket["token_f1"] = bucket["_f1"] / bucket["n"] if bucket["n"] else 0.0
        del bucket["_correct"]
        del bucket["_f1"]

    result: dict[str, Any] = {
        "overall": overall,
        "by_category": by_category,
        # capabilities 落账到 metrics.json 顶层（has_evidence / has_stale /
        # has_abstention）：caller 直接读，不必再扫 results.jsonl 二次推断。
        **caps,
        # has_stale 的口径是"本次至少有 1 道可机检的 stale 题"，而不是
        # "数据集里出现过 gold_stale 字段"——没有判定依据的题不构成能力。
        "has_stale": stale_n > 0,
    }

    # LLM judge 回退统计：任一记录由 llm / llm-fallback 判分时，
    # 统计 judge=="llm-fallback" 的数量与占比（无 LLM judge 时省略该键）。
    uses_llm_judge = any(
        str(_judge(r).get("judge", "")).startswith("llm") for r in records
    )
    if uses_llm_judge:
        n_fallback = sum(
            1 for r in records if str(_judge(r).get("judge", "")) == "llm-fallback"
        )
        result["judge_fallback"] = {
            "count": n_fallback,
            "ratio": (n_fallback / n) if n else 0.0,
        }

    return result
