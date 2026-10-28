"""评级系统（membench.ratings）及报告集成测试。

覆盖：
- 各指标阈值边界（如 accuracy 0.90->优秀、0.899->良好；stale_rate 0.03->优秀、0.031->良好）
- 不可评级指标返回 None
- overall_rating 混合指标点数均值与 None 场景
- format_metric 的 "评价（百分比）" 输出格式
- write_run_report_md 报告集成（综合评级行与评价呈现）
"""

from __future__ import annotations

import os

import pytest

from membench.ratings import (
    DIAGNOSTIC_METRICS,
    METRIC_THRESHOLDS,
    RATE_METRICS,
    format_metric,
    format_metric_delta,
    format_metric_value,
    overall_rating,
    rate_metric,
    rating_points,
)
from membench.report import render_dialogue_report, write_run_report_md

# ---------------------------------------------------------------------------
# rate_metric：各指标阈值边界
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.90, "优秀"),
        (1.0, "优秀"),
        (0.899, "良好"),
        (0.75, "良好"),
        (0.749, "中等"),
        (0.60, "中等"),
        (0.599, "较差"),
        (0.0, "较差"),
    ],
)
def test_rate_accuracy_thresholds(value: float, expected: str) -> None:
    """accuracy 阈值边界：0.90/0.75/0.60。"""
    assert rate_metric("accuracy", value) == expected


@pytest.mark.parametrize(
    "key",
    ["token_f1", "bleu1", "recall_at_5", "over_refusal_rate"],
)
def test_rate_metric_diagnostic_keys(key: str) -> None:
    """诊断指标不评级（含字面重合的 token_f1 / bleu1），返回 None。"""
    assert rate_metric(key, 0.0) is None
    assert rate_metric(key, 1.0) is None


def test_metric_roles() -> None:
    """评级 / 诊断 / 比率指标集合的口径固定。"""
    assert frozenset(METRIC_THRESHOLDS) == frozenset(
        {"accuracy", "evidence_recall", "abstention_accuracy", "stale_rate"}
    )
    assert DIAGNOSTIC_METRICS == frozenset(
        {"token_f1", "bleu1", "recall_at_5", "over_refusal_rate"}
    )
    assert DIAGNOSTIC_METRICS.isdisjoint(METRIC_THRESHOLDS)
    assert RATE_METRICS == frozenset(METRIC_THRESHOLDS) | DIAGNOSTIC_METRICS


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.95, "优秀"),
        (0.949, "良好"),
        (0.85, "良好"),
        (0.849, "中等"),
        (0.70, "中等"),
        (0.699, "较差"),
    ],
)
def test_rate_evidence_recall_thresholds(value: float, expected: str) -> None:
    """evidence_recall 阈值边界：0.95/0.85/0.70。"""
    assert rate_metric("evidence_recall", value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.90, "优秀"),
        (0.899, "良好"),
        (0.80, "良好"),
        (0.799, "中等"),
        (0.60, "中等"),
        (0.599, "较差"),
    ],
)
def test_rate_abstention_accuracy_thresholds(value: float, expected: str) -> None:
    """abstention_accuracy 阈值边界：0.90/0.80/0.60。"""
    assert rate_metric("abstention_accuracy", value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.03, "优秀"),
        (0.0, "优秀"),
        (0.031, "良好"),
        (0.10, "良好"),
        (0.101, "中等"),
        (0.20, "中等"),
        (0.201, "较差"),
        (1.0, "较差"),
    ],
)
def test_rate_stale_rate_thresholds(value: float, expected: str) -> None:
    """stale_rate 越低越好，阈值边界：0.03/0.10/0.20。"""
    assert rate_metric("stale_rate", value) == expected


@pytest.mark.parametrize(
    "key",
    ["latency_p50_ms", "latency_p95_ms", "avg_input_tokens", "avg_llm_calls", "n_questions", "unknown_metric"],
)
def test_rate_metric_unrated_keys(key: str) -> None:
    """效率成本指标及未知指标不评级，返回 None。"""
    assert rate_metric(key, 123.4) is None


# ---------------------------------------------------------------------------
# rating_points
# ---------------------------------------------------------------------------


def test_rating_points_mapping() -> None:
    """点数映射：优秀=4、良好=3、中等=2、较差=1。"""
    assert rating_points("优秀") == 4
    assert rating_points("良好") == 3
    assert rating_points("中等") == 2
    assert rating_points("较差") == 1


def test_rating_points_invalid_label() -> None:
    """非法标签抛出 KeyError。"""
    with pytest.raises(KeyError):
        rating_points("完美")


# ---------------------------------------------------------------------------
# overall_rating
# ---------------------------------------------------------------------------


def test_overall_rating_mixed_metrics() -> None:
    """四项均值：accuracy 0.92(4.0) + evidence_recall 0.85(3.0) -> 3.5。

    诊断指标（token_f1 / bleu1）与效率成本指标都不参与均值。
    """
    overall = {
        "accuracy": 0.92,
        "evidence_recall": 0.85,
        "token_f1": 0.0,
        "bleu1": 0.0,
        "latency_p50_ms": 120.0,  # 不可评级，应被忽略
        "n_questions": 10,
    }
    result = overall_rating(overall)
    assert result is not None
    label, mean = result
    assert mean == pytest.approx(3.5)
    assert label == "优秀"


def test_overall_rating_four_quality_metrics() -> None:
    """四项质量指标全部优秀 -> 均值 4.0 -> 优秀（诊断指标不稀释）。"""
    overall = {
        "accuracy": 0.95,
        "evidence_recall": 0.97,
        "abstention_accuracy": 0.92,
        "stale_rate": 0.01,
        "token_f1": 0.0,
        "bleu1": 0.0,
    }
    result = overall_rating(overall)
    assert result is not None
    assert result == ("优秀", pytest.approx(4.0))


def test_overall_rating_ignores_diagnostic_metrics() -> None:
    """诊断指标拉高或拉低都不改变综合评级（与评级口径解耦）。"""
    base = {
        "accuracy": 0.80,
        "evidence_recall": 0.80,
        "abstention_accuracy": 0.85,
        "stale_rate": 0.05,
    }
    low = dict(base, token_f1=0.0, bleu1=0.0, recall_at_5=0.0, over_refusal_rate=1.0)
    high = dict(base, token_f1=1.0, bleu1=1.0, recall_at_5=1.0, over_refusal_rate=0.0)
    assert overall_rating(low) == overall_rating(high) == overall_rating(base)


def test_overall_rating_threshold_boundaries() -> None:
    """综合评级均值阈值边界：3.5 优秀、2.5 良好、1.5 中等、其余较差。"""
    # 均值 3.5：accuracy 0.90(4.0) + evidence_recall 0.85(3.0)
    assert overall_rating({"accuracy": 0.90, "evidence_recall": 0.85})[0] == "优秀"
    # 均值 2.5：accuracy 0.75(3.0) + evidence_recall 0.70(2.0)
    assert overall_rating({"accuracy": 0.75, "evidence_recall": 0.70})[0] == "良好"
    # 均值 1.5：accuracy 0.30（较差档中点）-> 1.5
    assert overall_rating({"accuracy": 0.30})[0] == "中等"
    # 均值 1.0：仅一项较差
    assert overall_rating({"accuracy": 0.10})[0] == "较差"


def test_overall_rating_none_when_no_ratable() -> None:
    """overall 中没有任何可评级指标时返回 None。"""
    assert overall_rating({}) is None
    assert overall_rating({"latency_p50_ms": 120.0, "n_questions": 10}) is None
    # 只有诊断指标（含 token_f1 / bleu1）不足以给出综合评级
    assert overall_rating({"token_f1": 1.0, "bleu1": 1.0, "recall_at_5": 1.0}) is None


def test_overall_rating_skips_non_numeric() -> None:
    """非数值的可评级键被跳过；全为非数值时返回 None。"""
    assert overall_rating({"accuracy": "n/a"}) is None
    result = overall_rating({"accuracy": "n/a", "evidence_recall": 0.95})
    assert result is not None and result[0] == "优秀"
    # 诊断指标不因数值合法而让 overall 有评级
    assert overall_rating({"accuracy": "n/a", "bleu1": 0.5}) is None


# ---------------------------------------------------------------------------
# continuous_points：档内连续点数
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        # accuracy 阈值边界：与离散点数精确一致
        ("accuracy", 0.90, 4.0),
        ("accuracy", 0.75, 3.0),
        ("accuracy", 0.60, 2.0),
        ("accuracy", 0.0, 1.0),
        # accuracy 档内线性
        ("accuracy", 0.825, 3.5),   # 良好档中点
        ("accuracy", 0.675, 2.5),   # 中等档中点
        ("accuracy", 0.30, 1.5),    # 较差档中点
        # stale_rate 阈值边界
        ("stale_rate", 0.03, 4.0),
        ("stale_rate", 0.10, 3.0),
        ("stale_rate", 0.20, 2.0),
        ("stale_rate", 1.0, 1.0),
        # stale_rate 档内线性
        ("stale_rate", 0.065, 3.5),  # 良好档中点
        ("stale_rate", 0.15, 2.5),   # 中等档中点
        ("stale_rate", 0.60, 1.5),   # 较差档中点
        # 不可评级指标（诊断指标与效率成本）
        ("latency_p50_ms", 100.0, None),
        ("token_f1", 0.25, None),
        ("bleu1", 0.5, None),
    ],
)
def test_continuous_points(key: str, value: float, expected: float | None) -> None:
    """连续点数：阈值边界与离散一致，档内线性插值。"""
    from membench.ratings import continuous_points

    result = continuous_points(key, value)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected, abs=1e-6)


def test_overall_rating_sensitive_to_within_bucket_improvement() -> None:
    """回归：档内改进（不跨档）应反映到综合评级点数上。

    案例：accuracy 0.647→0.715（+6.8pt）、abstention 0.476→0.567（+9.1pt），
    四项质量指标无一跨档，离散评级下综合评级看不出提升。
    连续点数口径下，综合评级分数应有可见上升。
    """
    before = {
        "accuracy": 0.647,
        "token_f1": 0.60,
        "bleu1": 0.35,
        "evidence_recall": 0.75,
        "abstention_accuracy": 0.476,
        "stale_rate": 0.10,
    }
    after = {
        "accuracy": 0.715,
        "token_f1": 0.60,
        "bleu1": 0.35,
        "evidence_recall": 0.75,
        "abstention_accuracy": 0.567,
        "stale_rate": 0.10,
    }
    r_before = overall_rating(before)
    r_after = overall_rating(after)
    assert r_before is not None and r_after is not None
    assert r_after[1] > r_before[1]
    # 四项均值下这次档内改进刚好跨过 2.5 阈值：中等 -> 良好
    assert (r_before[0], r_after[0]) == ("中等", "良好")


# ---------------------------------------------------------------------------
# format_metric
# ---------------------------------------------------------------------------


def test_format_metric_rated() -> None:
    """可评级比率指标输出 "评价（百分比）"（保留 1 位小数）。"""
    assert format_metric("accuracy", 0.9234) == "优秀（92.3%）"
    assert format_metric("accuracy", 0.8) == "良好（80.0%）"
    assert format_metric("stale_rate", 0.05) == "良好（5.0%）"
    # 诊断指标无评价标签，只给百分比（token_f1 / bleu1 也一样）
    assert format_metric("token_f1", 0.25) == "25.0%"
    assert format_metric("bleu1", 0.5) == "50.0%"
    assert format_metric("recall_at_5", 0.429) == "42.9%"
    assert format_metric("over_refusal_rate", 0.1) == "10.0%"


def test_format_metric_value_and_delta() -> None:
    """纯数值/差值呈现：比率型百分比与百分点，其余保留 3 位小数。"""
    assert format_metric_value("accuracy", 0.9234) == "92.3%"
    assert format_metric_value("latency_p50_ms", 123.4567) == "123.457"
    assert format_metric_value("accuracy", None) == "未定义"
    assert format_metric_delta("accuracy", 0.023) == "+2.3pt"
    assert format_metric_delta("latency_p50_ms", -12.0) == "-12.000"
    assert format_metric_delta("accuracy", None) == "n/a"


def test_format_metric_unrated() -> None:
    """不可评级指标返回原数值格式化字符串；None / 'n/a' 一律返回「未定义」。"""
    assert format_metric("latency_p50_ms", 123.4567) == "123.457"
    assert format_metric("n_questions", 100) == "100"
    # 缺失值统一渲染为「未定义」，业务方不会误读为 0。
    assert format_metric("accuracy", None) == "未定义"
    assert format_metric("accuracy", "n/a") == "未定义"
    assert format_metric("evidence_recall", "None") == "未定义"


# ---------------------------------------------------------------------------
# 报告集成：write_run_report_md
# ---------------------------------------------------------------------------


def _minimal_metrics() -> dict:
    """构造最小可用的 metrics（含可评级质量指标与效率指标）。"""
    return {
        "run_id": "test-run",
        "adapter": "mock",
        "dataset": "mock.jsonl",
        "judge": "heuristic",
        "timestamp": "2026-01-01T00:00:00",
        "overall": {
            "accuracy": 0.92,
            "token_f1": 0.80,
            "bleu1": 0.50,
            "evidence_recall": 0.90,
            "abstention_accuracy": 0.85,
            "stale_rate": 0.05,
            "latency_p50_ms": 100.0,
            "latency_p95_ms": 200.0,
            "avg_input_tokens": 512.0,
            "avg_llm_calls": 2.0,
            "n_questions": 1,
        },
        "by_category": {"single_hop": {"accuracy": 0.92, "token_f1": 0.5, "n": 1}},
    }


def test_write_run_report_md_contains_ratings(tmp_path) -> None:
    """report.md 含综合评级行与 "评价（评分）" 形式的单元格呈现。"""
    records = [{"qid": "q1", "category": "single_hop", "judge": {"correct": True}}]
    out_path = write_run_report_md(str(tmp_path), records, _minimal_metrics())
    assert os.path.isfile(out_path)
    with open(out_path, encoding="utf-8") as f:
        content = f.read()

    # 综合评级行 + 评级规则小字说明
    assert "**综合评级：" in content
    assert "/4.0）**" in content
    assert "评级规则" in content

    # 摘要表 "值" 列以 "评价（百分比）" 形式呈现
    assert "优秀（92.0%）" in content  # accuracy = 0.92
    assert "（5.0%）" in content  # stale_rate = 0.05

    # 效率成本指标保持原数值呈现，不带评价；指标列展示「中文（key）」双标
    assert "123" not in content
    assert "| 效率成本 | 延迟 P50（latency_p50_ms） | 100.000 |" in content
    assert "证据召回率（evidence_recall）" in content  # 摘要表的中文化列名
    assert "陈旧率（stale_rate）" in content
    assert "拒答准确率（abstention_accuracy）" in content

    # token_f1 / bleu1 标为诊断指标：照常展示百分比，但没有评价标签
    assert "| 正确性 | Token F1（token_f1）（诊断） | 80.0% |" in content
    assert "| 正确性 | BLEU-1（bleu1）（诊断） | 50.0% |" in content
    assert "token_f1 / bleu1（字面重合度，非语义正确性）" in content
    assert "Token F1 / BLEU-1 为诊断指标" in content

    # 分题型表：正确率 + Token F1 双列，Token F1 无评价标签
    assert "| 单跳（single_hop） | 优秀（92.0%） | 50.0% | 1 |" in content
    assert "较差（50.0%）" not in content

    # 验收基线对照表含 "评价" 列
    assert "| 指标 | 目标 | 实际 | 评价 | 结果 |" in content


def test_run_report_warns_all_zero_retrieval(tmp_path) -> None:
    """标注了 evidence 但两项检索指标全 0 -> 报告给出"格式不一致"警示；
    非默认检索口径在配置表与提示中可见。

    经对话级报告入口（``render_dialogue_report``）落盘：它负责建目录，
    两份口径各写自己的 ``report/``，后一次不覆盖前一次的产物。
    """
    records = [{"qid": "q1", "category": "single_hop", "judge": {"correct": True}}]

    metrics = _minimal_metrics()
    metrics.update({"has_evidence": True, "retrieval_match": "substring"})
    metrics["overall"]["evidence_recall"] = 0.0
    metrics["overall"]["recall_at_5"] = 0.0
    out_path = render_dialogue_report(str(tmp_path / "ds1" / "report"), records, metrics)
    content = open(out_path, encoding="utf-8").read()
    assert "疑似 evidence 与被检索文本格式不一致" in content
    assert "| 检索匹配口径 | substring |" in content

    metrics2 = _minimal_metrics()
    metrics2.update({"retrieval_match": "overlap", "retrieval_match_threshold": 0.6})
    out_path2 = render_dialogue_report(str(tmp_path / "ds2" / "report"), records, metrics2)
    content2 = open(out_path2, encoding="utf-8").read()
    assert "检索命中口径：**overlap**（阈值 0.6）" in content2
    assert "| 检索匹配口径 | overlap |" in content2
    # 第一份报告未被第二份覆盖
    assert os.path.isfile(out_path) and out_path != out_path2


def test_write_run_report_md_no_overall_rating(tmp_path) -> None:
    """overall 中无可评级指标时不输出综合评级行。"""
    metrics = _minimal_metrics()
    metrics["overall"] = {"latency_p50_ms": 100.0, "n_questions": 1}
    records = [{"qid": "q1", "category": "single_hop", "judge": {"correct": True}}]
    out_path = write_run_report_md(str(tmp_path), records, metrics)
    with open(out_path, encoding="utf-8") as f:
        content = f.read()
    assert "**综合评级：" not in content


def test_stale_row_n_a_when_no_checkable_question(tmp_path) -> None:
    """stale_n=0：显示 n/a（无可机检的 stale 注解），不得出现「优秀（0.0%）」。"""
    metrics = _minimal_metrics()
    metrics["overall"].update(
        {
            # 低分避免综合评级本身出现「优秀」，让断言能精确检查 stale 行。
            "accuracy": 0.40,
            "evidence_recall": 0.50,
            "abstention_accuracy": 0.40,
            "stale_rate": None,
            "stale_n": 0,
            "stale_total": 70,
        }
    )
    records = [{"qid": "q1", "category": "single_hop", "judge": {"correct": False}}]
    metrics["by_category"] = {"single_hop": {"accuracy": 0.40, "token_f1": 0.5, "n": 1}}
    out_path = write_run_report_md(str(tmp_path), records, metrics)
    content = open(out_path, encoding="utf-8").read()

    assert "n/a（无可机检的 stale 注解）" in content
    assert "优秀（0.0%）" not in content
    assert "优秀" not in content
    # 解释 n/a 的来源：有 70 道 stale 注解题但当前判分器不可机检。
    assert "70 道 stale 注解题" in content


def test_stale_row_marks_coverage_when_partial(tmp_path) -> None:
    """0 < stale_n < stale_total：值后标注覆盖率。"""
    metrics = _minimal_metrics()
    metrics["overall"].update({"stale_rate": 0.0, "stale_n": 56, "stale_total": 70})
    records = [{"qid": "q1", "category": "single_hop", "judge": {"correct": True}}]
    out_path = write_run_report_md(str(tmp_path), records, metrics)
    content = open(out_path, encoding="utf-8").read()

    assert "（可机检 56/70）" in content
    assert "0.0%（可机检 56/70）" in content
    # 评级规则小注说明 stale_rate 的分母口径。
    assert "stale_rate 的分母只含可机检的 stale 题" in content
