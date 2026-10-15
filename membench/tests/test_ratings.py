"""评级系统（membench.ratings）及报告集成测试。

覆盖：
- 各指标阈值边界（如 accuracy 0.90->优秀、0.899->良好；stale_rate 0.03->优秀、0.031->良好）
- 不可评级指标返回 None
- overall_rating 混合指标点数均值与 None 场景
- format_metric 的 "评价（评分）" 输出格式
- write_run_report_md 报告集成（综合评级行与评价呈现）
"""

from __future__ import annotations

import os

import pytest

from membench.ratings import (
    format_metric,
    overall_rating,
    rate_metric,
    rating_points,
)
from membench.report import write_run_report_md

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
    ("value", "expected"),
    [
        (0.90, "优秀"),
        (0.899, "良好"),
        (0.60, "中等"),
        (0.599, "较差"),
    ],
)
def test_rate_token_f1_thresholds(value: float, expected: str) -> None:
    """token_f1 与 accuracy 共用同一组阈值。"""
    assert rate_metric("token_f1", value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.60, "优秀"),
        (0.599, "良好"),
        (0.45, "良好"),
        (0.449, "中等"),
        (0.30, "中等"),
        (0.299, "较差"),
    ],
)
def test_rate_bleu1_thresholds(value: float, expected: str) -> None:
    """bleu1 阈值边界：0.60/0.45/0.30。"""
    assert rate_metric("bleu1", value) == expected


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
    """混合指标：accuracy 0.92(4.0) + bleu1 0.35(2.33) -> 均值 3.17 -> 良好。"""
    overall = {
        "accuracy": 0.92,
        "bleu1": 0.35,
        "latency_p50_ms": 120.0,  # 不可评级，应被忽略
        "n_questions": 10,
    }
    result = overall_rating(overall)
    assert result is not None
    label, mean = result
    # accuracy 0.92 >= 0.90 → 4.0；bleu1 0.35 在 [0.30,0.45) 档内 → 2.333
    assert mean == pytest.approx((4.0 + 2.0 + 0.05 / 0.15) / 2)
    assert label == "良好"


def test_overall_rating_full_six_metrics() -> None:
    """六项质量指标全部优秀 -> 均值 4.0 -> 优秀。"""
    overall = {
        "accuracy": 0.95,
        "token_f1": 0.93,
        "bleu1": 0.70,
        "evidence_recall": 0.97,
        "abstention_accuracy": 0.92,
        "stale_rate": 0.01,
    }
    result = overall_rating(overall)
    assert result is not None
    assert result == ("优秀", pytest.approx(4.0))


def test_overall_rating_threshold_boundaries() -> None:
    """综合评级均值阈值边界：3.5 优秀、2.5 良好、1.5 中等、其余较差。"""
    # 均值 3.5：两项优秀 + 两项良好 -> (4+4+3+3)/4 = 3.5
    assert overall_rating({"accuracy": 0.95, "token_f1": 0.95, "bleu1": 0.50, "evidence_recall": 0.90})[0] == "优秀"
    # 均值 2.5：一项良好 + 一项中等 -> (3+2)/2 = 2.5
    assert overall_rating({"accuracy": 0.80, "bleu1": 0.35})[0] == "良好"
    # 均值 1.5：一项中等 + 一项较差 -> (2+1)/2 = 1.5
    assert overall_rating({"accuracy": 0.65, "bleu1": 0.10})[0] == "中等"
    # 均值 1.0：仅一项较差
    assert overall_rating({"accuracy": 0.10})[0] == "较差"


def test_overall_rating_none_when_no_ratable() -> None:
    """overall 中没有任何可评级指标时返回 None。"""
    assert overall_rating({}) is None
    assert overall_rating({"latency_p50_ms": 120.0, "n_questions": 10}) is None


def test_overall_rating_skips_non_numeric() -> None:
    """非数值的可评级键被跳过；全为非数值时返回 None。"""
    assert overall_rating({"accuracy": "n/a"}) is None
    result = overall_rating({"accuracy": "n/a", "bleu1": 0.5})
    assert result is not None and result[0] == "良好"


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
        # 不可评级指标
        ("latency_p50_ms", 100.0, None),
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
    """接入侧反馈回归：档内改进（不跨档）应反映到综合评级点数上。

    反馈案例：accuracy 0.647→0.715（+6.8pt）、abstention 0.476→0.567（+9.1pt），
    六项指标无一跨档，旧口径下综合评级同为"中等（1.7/4.0）"，看不出提升。
    新口径（连续点数）下，综合评级分数应有可见上升。
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
    # 标签不变（都没跨档），但点数应上升
    assert r_before[0] == r_after[0]
    assert r_after[1] > r_before[1]


# ---------------------------------------------------------------------------
# format_metric
# ---------------------------------------------------------------------------


def test_format_metric_rated() -> None:
    """可评级指标输出 "评价（评分）"，评分保留 3 位小数。"""
    assert format_metric("accuracy", 0.9234) == "优秀（0.923）"
    assert format_metric("accuracy", 0.8) == "良好（0.800）"
    assert format_metric("stale_rate", 0.05) == "良好（0.050）"


def test_format_metric_unrated() -> None:
    """不可评级指标返回原数值格式化字符串。"""
    assert format_metric("latency_p50_ms", 123.4567) == "123.457"
    assert format_metric("n_questions", 100) == "100"
    assert format_metric("accuracy", "n/a") == "n/a"


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
        "by_category": {"single_hop": {"accuracy": 0.92, "n": 1}},
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

    # 摘要表 "值" 列以 "评价（0.xxx）" 形式呈现
    assert "优秀（0.920）" in content  # accuracy = 0.92
    assert "（0." in content

    # 效率成本指标保持原数值呈现，不带评价
    assert "123" not in content
    assert "| 效率成本 | latency_p50_ms | 100.000 |" in content

    # 分题型准确率表同样带评价
    assert "| 单跳（single_hop） | 优秀（0.920） | 1 |" in content

    # 验收基线对照表含 "评价" 列
    assert "| 指标 | 目标 | 实际 | 评价 | 结果 |" in content


def test_write_run_report_md_no_overall_rating(tmp_path) -> None:
    """overall 中无可评级指标时不输出综合评级行。"""
    metrics = _minimal_metrics()
    metrics["overall"] = {"latency_p50_ms": 100.0, "n_questions": 1}
    records = [{"qid": "q1", "category": "single_hop", "judge": {"correct": True}}]
    out_path = write_run_report_md(str(tmp_path), records, metrics)
    with open(out_path, encoding="utf-8") as f:
        content = f.read()
    assert "**综合评级：" not in content
