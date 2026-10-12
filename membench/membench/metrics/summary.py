"""六维指标汇总（SPEC 3.5）。

``aggregate(records)`` 消费 runner 写入 results.jsonl 的单条评测记录
（SPEC 3.4），输出 ``{"overall": {...}, "by_category": {...}}``；
run_id / adapter / dataset / judge / timestamp 等元信息由 runner 补充。

分位数为手写线性插值实现，不依赖 numpy / pandas。
"""

from __future__ import annotations

from typing import Any, Iterable

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


def _mean(values: Iterable[float]) -> float:
    """算术均值；空序列返回 0.0。"""
    xs = [float(v) for v in values]
    return sum(xs) / len(xs) if xs else 0.0


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总单条评测记录为六维指标字典（SPEC 3.5）。

    Args:
        records: results.jsonl 风格的记录列表，每条含
            ``category / evidence_hit / token_f1 / bleu1 / latency_ms /
            input_tokens / llm_calls / ingest_llm_calls`` 以及
            ``judge``（含 correct / abstained_pred / stale）字段。

    Returns:
        ``{"overall": {...}, "by_category": {category: {"accuracy", "n"}}}``；
        使用了 LLM judge 时附加 ``"judge_fallback": {"count", "ratio"}``
        （``judge=="llm-fallback"`` 的记录占比）。

    指标口径：
        - accuracy: ``judge.correct`` 均值。
        - evidence_recall: ``evidence_hit`` 均值（仅非 abstention 题）。
        - abstention_accuracy: abstention 题上 ``judge.abstained_pred`` 均值。
        - stale_rate: knowledge_update 题上 ``judge.stale`` 均值。
        - latency_p50_ms / latency_p95_ms: ``latency_ms`` 分位数（线性插值）。
        - avg_llm_calls: 每题 ``llm_calls + ingest_llm_calls`` 均值
          （ingest 成本按 runner 的记录归属口径计入）。
    """
    n = len(records)

    def _judge(rec: dict[str, Any]) -> dict[str, Any]:
        return rec.get("judge", {}) or {}

    non_abstention = [r for r in records if r.get("category") != "abstention"]
    abstention = [r for r in records if r.get("category") == "abstention"]
    knowledge_update = [r for r in records if r.get("category") == "knowledge_update"]

    overall = {
        "accuracy": _mean(bool(_judge(r).get("correct")) for r in records),
        "token_f1": _mean(r.get("token_f1", 0.0) for r in records),
        "bleu1": _mean(r.get("bleu1", 0.0) for r in records),
        "evidence_recall": _mean(
            bool(r.get("evidence_hit")) for r in non_abstention
        ),
        "abstention_accuracy": _mean(
            bool(_judge(r).get("abstained_pred")) for r in abstention
        ),
        "stale_rate": _mean(bool(_judge(r).get("stale")) for r in knowledge_update),
        "latency_p50_ms": percentile((r.get("latency_ms", 0.0) for r in records), 50),
        "latency_p95_ms": percentile((r.get("latency_ms", 0.0) for r in records), 95),
        "avg_input_tokens": _mean(r.get("input_tokens", 0) for r in records),
        "avg_llm_calls": _mean(
            float(r.get("llm_calls", 0)) + float(r.get("ingest_llm_calls", 0))
            for r in records
        ),
        "n_questions": n,
    }

    by_category: dict[str, dict[str, Any]] = {}
    for rec in records:
        cat = str(rec.get("category", "unknown"))
        bucket = by_category.setdefault(cat, {"_correct": 0.0, "n": 0})
        bucket["_correct"] += 1.0 if _judge(rec).get("correct") else 0.0
        bucket["n"] += 1
    for cat, bucket in by_category.items():
        bucket["accuracy"] = bucket["_correct"] / bucket["n"] if bucket["n"] else 0.0
        del bucket["_correct"]

    result: dict[str, Any] = {"overall": overall, "by_category": by_category}

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
