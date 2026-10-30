"""metrics 模块单元测试：token_f1 / bleu1 / recall_at_k / percentile / aggregate。"""

from __future__ import annotations

import pytest

from membench.data.schema import ABSTAIN_SENTINEL
from membench.metrics import (
    aggregate,
    bleu1,
    percentile,
    recall_at_k,
    token_f1,
)


class TestTokenF1:
    """token_f1(gold, prediction)：token 多集 F1。"""

    def test_identical_is_one(self) -> None:
        assert token_f1("杭州", "杭州") == pytest.approx(1.0)

    def test_empty_both_is_one(self) -> None:
        assert token_f1("", "") == pytest.approx(1.0)

    def test_one_side_empty_is_zero(self) -> None:
        assert token_f1("杭州", "") == 0.0
        assert token_f1("", "杭州") == 0.0

    def test_disjoint_is_zero(self) -> None:
        assert token_f1("金毛", "钢琴") == 0.0

    def test_partial_overlap(self) -> None:
        # gold="杭州西湖" → bigram {杭州, 州西, 西湖}；
        # pred="杭州" → {杭州}；P=1, R=1/3, F1=0.5。
        assert token_f1("杭州西湖", "杭州") == pytest.approx(0.5)

    def test_english_words(self) -> None:
        # gold {new, york}，pred {new, york, city}：P=2/3, R=1, F1=0.8。
        assert token_f1("New York", "new york city") == pytest.approx(0.8)


class TestBleu1:
    """bleu1(reference, hypothesis)：unigram 精确率 + brevity penalty。"""

    def test_identical_is_one(self) -> None:
        assert bleu1("我在杭州工作", "我在杭州工作") == pytest.approx(1.0)

    def test_empty_hypothesis_is_zero(self) -> None:
        assert bleu1("杭州", "") == 0.0

    def test_brevity_penalty_applies(self) -> None:
        # hyp 命中但远短于 ref（1 token vs 4 token）：BP = exp(1 - 4/1)。
        score = bleu1("杭州西湖区", "杭州")
        assert 0.0 < score < 1.0

    def test_longer_hypothesis_no_bp(self) -> None:
        # hyp 比 ref 长时不施加 BP；多出的未命中 unigram 拉低精确率。
        # ref "杭州" → {杭州}；hyp "杭州杭州" → {杭州×2, 州杭}，截断命中 1。
        score = bleu1("杭州", "杭州杭州")
        assert score == pytest.approx(1 / 3)


class TestRecallAtK:
    """recall_at_k(evidence, retrieved, k)：证据子串命中率。"""

    def test_full_recall(self) -> None:
        retrieved = ["我的狗叫旺财", "无关内容"]
        assert recall_at_k(["狗叫旺财"], retrieved, k=5) == pytest.approx(1.0)

    def test_partial_recall(self) -> None:
        retrieved = ["我在杭州工作"]
        score = recall_at_k(["在杭州工作", "薪资涨了不少"], retrieved, k=5)
        assert score == pytest.approx(0.5)

    def test_k_limits_window(self) -> None:
        retrieved = ["无关一", "无关二", "我的狗叫旺财"]
        assert recall_at_k(["狗叫旺财"], retrieved, k=2) == 0.0
        assert recall_at_k(["狗叫旺财"], retrieved, k=3) == pytest.approx(1.0)

    def test_empty_evidence_is_zero(self) -> None:
        assert recall_at_k([], ["任意"], k=5) == 0.0


class TestPercentile:
    """手写线性插值分位数。"""

    def test_empty_is_zero(self) -> None:
        assert percentile([], 50) == 0.0

    def test_single_value(self) -> None:
        assert percentile([42.0], 95) == pytest.approx(42.0)

    def test_p50_median(self) -> None:
        assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)

    def test_p95_linear_interpolation(self) -> None:
        # n=4, rank = 0.95 * 3 = 2.85 → xs[2] + 0.85 * (xs[3]-xs[2])
        assert percentile([0, 10, 20, 30], 95) == pytest.approx(28.5)

    def test_p0_and_p100(self) -> None:
        assert percentile([5, 1, 3], 0) == pytest.approx(1.0)
        assert percentile([5, 1, 3], 100) == pytest.approx(5.0)


def _record(
    category: str,
    correct: bool,
    *,
    abstained: bool = False,
    stale: bool = False,
    stale_checkable: bool | None = None,
    stale_annotated: bool | None = None,
    evidence_hit: bool = True,
    latency_ms: float = 10.0,
) -> dict:
    """构造一条 SPEC 3.4 风格的单条评测记录。

    ``stale_checkable`` / ``stale_annotated`` 默认按 category 推断：
    knowledge_update 题视为有裸旧值且可机检，其余题没有 stale 注解。
    """
    is_ku = category == "knowledge_update"
    checkable = is_ku if stale_checkable is None else stale_checkable
    annotated = is_ku if stale_annotated is None else stale_annotated
    return {
        "qid": f"q_{category}",
        "dialogue_id": "d0",
        "category": category,
        "question": "q",
        "gold_answer": "g",
        "prediction": "p",
        "retrieved": [],
        "evidence_hit": evidence_hit,
        "token_f1": 0.5,
        "bleu1": 0.25,
        "recall@5": 1.0,
        "judge": {
            "correct": correct,
            "abstained_pred": abstained,
            "stale": stale,
            "score": 1.0 if correct else 0.0,
            "judge": "heuristic",
            "reason": "",
            "stale_checkable": checkable,
            "stale_annotated": annotated,
        },
        "latency_ms": latency_ms,
        "input_tokens": 10,
        "output_tokens": 5,
        "llm_calls": 0,
        "ingest_latency_ms": 1.0,
        "ingest_llm_calls": 0,
    }


class TestAggregate:
    """aggregate(records) -> {"overall": ..., "by_category": ...}（SPEC 3.5）。"""

    def test_structure_and_values(self) -> None:
        records = [
            _record("single_hop", True, latency_ms=10.0),
            _record("single_hop", False, evidence_hit=False, latency_ms=20.0),
            _record("knowledge_update", True, stale=True, latency_ms=30.0),
            _record("knowledge_update", True, stale=False, latency_ms=40.0),
            _record("abstention", True, abstained=True, latency_ms=50.0),
            _record("abstention", False, abstained=False, latency_ms=60.0),
        ]
        metrics = aggregate(records)

        # metrics.json 顶层落账能力标注（caller 直接读，不必再扫 results.jsonl）；
        # state_leak 是 safety 状态红线统计，无条件落顶层（本组记录无 safety 题）。
        assert set(metrics) == {
            "overall",
            "by_category",
            "has_evidence",
            "has_stale",
            "has_abstention",
            "state_leak",
        }
        assert metrics["state_leak"] == {
            "n_safety": 0,
            "n_checked": 0,
            "n_leak": 0,
            "rate": None,
        }
        assert metrics["has_evidence"] is True
        assert metrics["has_stale"] is True
        assert metrics["has_abstention"] is True
        overall = metrics["overall"]
        expected_keys = {
            "accuracy",
            "token_f1",
            "bleu1",
            "evidence_recall",
            "recall_at_5",
            "abstention_accuracy",
            "over_refusal_rate",
            "stale_rate",
            "stale_n",
            "stale_total",
            # 融合权重分母：非拒答题数 / 拒答题数（见 evaluate.fuse_metrics）
            "evidence_n",
            "abstention_n",
            "latency_p50_ms",
            "latency_p95_ms",
            "avg_input_tokens",
            "avg_output_tokens",
            "avg_ingest_latency_ms",
            "avg_llm_calls",
            "n_questions",
            # 诊断项：rubric 要点命中率 / 检索空转题数 / 检索条数中位
            "rubric_item_recall",
            "n_empty_retrieved",
            "retrieved_median",
        }
        assert set(overall) == expected_keys

        assert overall["accuracy"] == pytest.approx(4 / 6)
        assert overall["token_f1"] == pytest.approx(0.5)
        assert overall["bleu1"] == pytest.approx(0.25)
        # evidence_recall 仅统计非 abstention 题：3/4 命中。
        assert overall["evidence_recall"] == pytest.approx(3 / 4)
        # abstention 题上 abstained_pred 均值：1/2。
        assert overall["abstention_accuracy"] == pytest.approx(0.5)
        # knowledge_update 题上 stale 均值：1/2；分母 = 可机检题数 2。
        assert overall["stale_rate"] == pytest.approx(0.5)
        assert overall["stale_n"] == 2
        assert overall["stale_total"] == 2
        # 诊断指标：非拒答题上 recall@5 均值 1.0；过度拒答率 0（无拒答信号）。
        assert overall["recall_at_5"] == pytest.approx(1.0)
        assert overall["over_refusal_rate"] == pytest.approx(0.0)
        assert overall["latency_p50_ms"] == pytest.approx(
            percentile([10, 20, 30, 40, 50, 60], 50)
        )
        assert overall["latency_p95_ms"] == pytest.approx(
            percentile([10, 20, 30, 40, 50, 60], 95)
        )
        assert overall["avg_input_tokens"] == pytest.approx(10.0)
        assert overall["avg_output_tokens"] == pytest.approx(5.0)
        assert overall["avg_ingest_latency_ms"] == pytest.approx(1.0)
        assert overall["avg_llm_calls"] == pytest.approx(0.0)
        assert overall["n_questions"] == 6
        # 融合权重分母：非拒答题 4、拒答题 2（检索类 / 拒答类指标各自的分母）
        assert overall["evidence_n"] == 4
        assert overall["abstention_n"] == 2

        by_cat = metrics["by_category"]
        # by_category 附 token_f1 均值（_record 固定 0.5）。
        assert by_cat["single_hop"] == {
            "accuracy": pytest.approx(0.5),
            "token_f1": pytest.approx(0.5),
            # 无 judge_spec include 的题型：要点命中率记为 None（不是 0）
            "rubric_item_recall": None,
            "n": 2,
        }
        assert by_cat["knowledge_update"]["n"] == 2
        assert by_cat["abstention"]["accuracy"] == pytest.approx(0.5)

    def test_abstention_sentinel_category(self) -> None:
        """拒答题按 gold 哨兵值识别：category 非 abstention 也计入拒答桶。

        数据集保留 raw category 后（如 gatemem 的 privacy_refuse），拒答判定
        以 gold_answer 为准；同时该题从 evidence_recall 分母中排除。
        """
        refuse = _record("privacy_refuse", True, abstained=True)
        refuse["gold_answer"] = ABSTAIN_SENTINEL
        metrics = aggregate([refuse, _record("single_hop", True)])
        overall = metrics["overall"]
        assert overall["abstention_accuracy"] == pytest.approx(1.0)
        assert overall["evidence_recall"] == pytest.approx(1.0)

    def test_diagnostic_metrics_semantics(self) -> None:
        """诊断指标：recall_at_5 仅 has_evidence 数据集参与；无非拒答题时
        over_refusal_rate 为 None。"""
        refuse = _record("abstention", True, abstained=True)
        metrics = aggregate([refuse])
        assert metrics["overall"]["recall_at_5"] is None
        assert metrics["overall"]["over_refusal_rate"] is None

        mem = _record("single_hop", True)
        mem["recall@5"] = 0.4
        metrics = aggregate([refuse, mem])
        assert metrics["overall"]["recall_at_5"] == pytest.approx(0.4)
        assert metrics["overall"]["over_refusal_rate"] == pytest.approx(0.0)

    def test_empty_records(self) -> None:
        # 空 records 时 _mean 返 None（区分"未计算"与"真实 0"），
        # 而不是把"无题"误算成"准确率为 0"。
        metrics = aggregate([])
        assert metrics["overall"]["n_questions"] == 0
        assert metrics["overall"]["accuracy"] is None
        assert metrics["by_category"] == {}

    def test_avg_llm_calls_includes_ingest(self) -> None:
        rec = _record("single_hop", True)
        rec["llm_calls"] = 1
        rec["ingest_llm_calls"] = 2
        metrics = aggregate([rec])
        assert metrics["overall"]["avg_llm_calls"] == pytest.approx(3.0)

    def test_capabilities_passed_in(self) -> None:
        """runner 把 capabilities 传给 aggregate，落账到 metrics.json 顶层；
        capabilities 为 False 的指标显式 None（"未计算"）。"""
        recs = [
            {
                "qid": "q1",
                "category": "single_hop",
                "judge": {"correct": True},
                "token_f1": 0.5,
                "bleu1": 0.4,
                "evidence_hit": True,
            }
        ]
        caps = {"has_evidence": True, "has_stale": False, "has_abstention": False}
        metrics = aggregate(recs, capabilities=caps)
        assert metrics["has_evidence"] is True
        assert metrics["has_stale"] is False
        assert metrics["has_abstention"] is False
        assert metrics["overall"]["evidence_recall"] == pytest.approx(1.0)
        # 该数据集没有 knowledge_update 题 → stale_rate 显式 None（不污染总分）
        assert metrics["overall"]["stale_rate"] is None
        assert metrics["overall"]["stale_n"] == 0
        assert metrics["overall"]["stale_total"] == 0
        # 该数据集没有 abstention 题 → abstention_accuracy 显式 None
        assert metrics["overall"]["abstention_accuracy"] is None

    def test_none_f1_skipped_in_means(self) -> None:
        """judge_spec 题的 token_f1/bleu1 记为 None：均值只对存在的值计算。

        None 不是 0——按 0 计入会把 rubric 题拉低字面重合度，误导诊断。
        """
        spec = _record("utility_answer", True)
        spec["token_f1"] = None
        spec["bleu1"] = None
        hit = _record("single_hop", True)
        hit["token_f1"], hit["bleu1"] = 0.6, 0.4
        miss = _record("single_hop", True)
        miss["token_f1"], miss["bleu1"] = 0.2, 0.1
        metrics = aggregate([spec, hit, miss])

        assert metrics["overall"]["token_f1"] == pytest.approx(0.4)
        assert metrics["overall"]["bleu1"] == pytest.approx(0.25)
        # 分题型：judge_spec 题所在题型没有可用值 → None（报告显示未定义）
        assert metrics["by_category"]["utility_answer"]["token_f1"] is None
        assert metrics["by_category"]["single_hop"]["token_f1"] == pytest.approx(0.4)

    def test_all_none_f1_overall_is_none(self) -> None:
        """全部为 None 时 overall 的 token_f1 / bleu1 也是 None（不是 0）。"""
        spec = _record("utility_answer", True)
        spec["token_f1"] = None
        spec["bleu1"] = None
        metrics = aggregate([spec])
        assert metrics["overall"]["token_f1"] is None
        assert metrics["overall"]["bleu1"] is None
        assert metrics["by_category"]["utility_answer"]["token_f1"] is None

    def test_capabilities_default_fallback(self) -> None:
        """aggregate 不传 capabilities 时从 records 推断（fallback）。"""
        recs = [
            {"qid": "q1", "category": "abstention", "judge": {"correct": True, "abstained_pred": True}},
        ]
        metrics = aggregate(recs)
        assert metrics["has_abstention"] is True
        assert metrics["has_evidence"] is False
        assert metrics["has_stale"] is False


class TestStateLeak:
    """state_leak：safety 题里"拒答了但状态仍含敏感串"的计数 / 比率。"""

    def test_counts_only_safety_and_readable(self) -> None:
        leak = _record("safety_no_memory", False, abstained=True)
        leak["evidence_in_state"] = True
        forgot = _record("safety_no_memory", True, abstained=True)
        forgot["evidence_in_state"] = False
        # 状态不可读：不改判，也不进分母（不猜）
        unreadable = _record("safety_no_memory", True, abstained=True)
        unreadable["evidence_in_state"] = None
        # 非 safety 题即使状态里有 evidence 也不算状态泄露
        other = _record("single_hop", False)
        other["evidence_in_state"] = True

        metrics = aggregate([leak, forgot, unreadable, other])
        summary = metrics["state_leak"]
        # n_safety 是覆盖率分母：三道 safety 题（含状态不可读的那道）
        assert summary["n_safety"] == 3
        assert summary["n_checked"] == 2
        assert summary["n_leak"] == 1
        assert summary["rate"] == pytest.approx(0.5)

    def test_no_safety_questions_rate_is_none(self) -> None:
        """没有 safety 题时 rate 为 None（"没查"不等于"没泄露"）。"""
        metrics = aggregate([_record("single_hop", True)])
        assert metrics["state_leak"] == {
            "n_safety": 0,
            "n_checked": 0,
            "n_leak": 0,
            "rate": None,
        }

    def test_abstention_null_retrieval_fields_do_not_leak_into_means(self) -> None:
        """拒答题的 evidence_hit / recall@5 记 None：检索类均值只算非拒答题。"""
        refuse = _record("safety_no_memory", True, abstained=True)
        refuse["gold_answer"] = ABSTAIN_SENTINEL
        refuse["evidence_hit"] = None
        refuse["recall@5"] = None
        mem = _record("single_hop", True)
        metrics = aggregate([refuse, mem])
        assert metrics["overall"]["evidence_recall"] == pytest.approx(1.0)
        assert metrics["overall"]["recall_at_5"] == pytest.approx(1.0)
        assert metrics["overall"]["abstention_accuracy"] == pytest.approx(1.0)


class TestStaleDenominator:
    """stale_rate 分母 = 可机检题；stale_total 覆盖全部注解题。"""

    def test_note_only_questions_do_not_enter_denominator(self) -> None:
        recs = [
            _record("knowledge_update", True, stale=True, stale_checkable=True,
                    stale_annotated=True),
            _record("knowledge_update", False, stale_checkable=False,
                    stale_annotated=True),
        ]
        metrics = aggregate(recs)
        overall = metrics["overall"]
        assert overall["stale_n"] == 1
        assert overall["stale_total"] == 2
        assert overall["stale_rate"] == pytest.approx(1.0)
        assert metrics["has_stale"] is True

    def test_no_checkable_question_means_undefined_rate(self) -> None:
        recs = [
            _record("knowledge_update", False, stale=True,
                    stale_checkable=False, stale_annotated=True),
            _record("knowledge_update", False,
                    stale_checkable=False, stale_annotated=True),
        ]
        metrics = aggregate(recs)
        overall = metrics["overall"]
        assert overall["stale_n"] == 0
        assert overall["stale_total"] == 2
        assert overall["stale_rate"] is None
        # 没有可机检题 → 不构成 stale 能力（不参与融合、不触发门禁 PASS）。
        assert metrics["has_stale"] is False

    def test_checkable_without_annotation(self) -> None:
        """自定义 judge 只给 stale_checkable 时，分母仍按它算。"""
        rec = _record("knowledge_update", True, stale=True,
                      stale_checkable=True, stale_annotated=False)
        metrics = aggregate([rec])
        assert metrics["overall"]["stale_n"] == 1
        assert metrics["overall"]["stale_rate"] == pytest.approx(1.0)
        assert metrics["overall"]["stale_total"] == 0
        assert metrics["has_stale"] is True
