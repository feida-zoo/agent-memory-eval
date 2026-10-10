"""评级系统：在数值分数之外给出定性评价（优秀/良好/中等/较差）。

评价以"评价（评分）"形式呈现，如 ``优秀（0.923）``。

四级评价及点数映射：
    优秀 = 4、良好 = 3、中等 = 2、较差 = 1。

单指标阈值在模块内以数据表 :data:`METRIC_THRESHOLDS` 定义，便于后续调整。
效率成本类指标（latency / tokens / llm_calls / n_questions）依赖部署环境，
强制定性会误导，故不参与评级（:func:`rate_metric` 返回 ``None``）。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "LABELS",
    "LABEL_POINTS",
    "METRIC_THRESHOLDS",
    "OVERALL_LABEL_POINTS",
    "rate_metric",
    "rating_points",
    "overall_rating",
    "format_metric",
]

# 四级评价标签（由优到劣）
LABELS: tuple[str, str, str, str] = ("优秀", "良好", "中等", "较差")

# 评价标签 -> 点数映射
LABEL_POINTS: dict[str, int] = {
    "优秀": 4,
    "良好": 3,
    "中等": 2,
    "较差": 1,
}

# 单指标阈值表：
#   key -> (direction, [(阈值, 标签), ...])，阈值按从严到宽排列。
#   direction == "higher" 表示越高越好（value >= 阈值 即命中）；
#   direction == "lower"  表示越低越好（value <= 阈值 即命中）。
# 未出现在表中的指标视为不可评级。
METRIC_THRESHOLDS: dict[str, tuple[str, list[tuple[float, str]]]] = {
    "accuracy": ("higher", [(0.90, "优秀"), (0.75, "良好"), (0.60, "中等")]),
    "token_f1": ("higher", [(0.90, "优秀"), (0.75, "良好"), (0.60, "中等")]),
    "bleu1": ("higher", [(0.60, "优秀"), (0.45, "良好"), (0.30, "中等")]),
    "evidence_recall": ("higher", [(0.95, "优秀"), (0.85, "良好"), (0.70, "中等")]),
    "abstention_accuracy": ("higher", [(0.90, "优秀"), (0.80, "良好"), (0.60, "中等")]),
    "stale_rate": ("lower", [(0.03, "优秀"), (0.10, "良好"), (0.20, "中等")]),
}

# 综合评级阈值：对可评级质量指标的点数取均值后按此映射为标签。
OVERALL_THRESHOLDS: list[tuple[float, str]] = [
    (3.5, "优秀"),
    (2.5, "良好"),
    (1.5, "中等"),
]


def rate_metric(key: str, value: float) -> str | None:
    """对单个指标给出定性评价标签。

    Args:
        key: 指标名（如 ``accuracy``、``stale_rate``）。
        value: 指标数值。

    Returns:
        评价标签（优秀/良好/中等/较差）；不可评级的指标返回 ``None``。
    """
    spec = METRIC_THRESHOLDS.get(key)
    if spec is None:
        return None
    direction, thresholds = spec
    v = float(value)
    for threshold, label in thresholds:
        if direction == "higher":
            if v >= threshold:
                return label
        else:  # lower is better
            if v <= threshold:
                return label
    return LABELS[-1]  # 全部阈值均未命中 -> 较差


def rating_points(label: str) -> int:
    """评价标签对应的点数（优秀=4、良好=3、中等=2、较差=1）。

    Raises:
        KeyError: 标签不在四级评价之内。
    """
    return LABEL_POINTS[label]


def overall_rating(overall: dict[str, Any]) -> tuple[str, float] | None:
    """对 overall 汇总给出综合评级。

    取 overall 中"存在且可评级"的质量指标
    （accuracy / token_f1 / bleu1 / evidence_recall /
    abstention_accuracy / stale_rate）各自的评级点数均值，
    均值 >= 3.5 优秀、>= 2.5 良好、>= 1.5 中等，否则较差。

    Args:
        overall: 汇总指标字典（aggregate 输出的 ``metrics["overall"]``）。

    Returns:
        ``(标签, 点数均值)``；overall 中不存在任何可评级指标时返回 ``None``。
    """
    points: list[float] = []
    for key in METRIC_THRESHOLDS:
        if key not in overall:
            continue
        try:
            value = float(overall[key])
        except (TypeError, ValueError):
            continue
        label = rate_metric(key, value)
        if label is not None:
            points.append(float(rating_points(label)))
    if not points:
        return None
    mean = sum(points) / len(points)
    for threshold, label in OVERALL_THRESHOLDS:
        if mean >= threshold:
            return label, mean
    return LABELS[-1], mean


def format_metric(key: str, value: Any) -> str:
    """把指标值格式化为报告呈现字符串。

    可评级指标返回 ``"评价（评分）"``（评分保留 3 位小数），
    如 ``"优秀（0.923）"``；不可评级指标返回原数值的格式化字符串
    （float 保留 3 位，其余原样，与 report 的 ``_fmt`` 口径一致）。

    Args:
        key: 指标名。
        value: 指标数值（允许非数值，如 "n/a"）。

    Returns:
        报告单元格字符串。
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        label = rate_metric(key, float(value))
        if label is not None:
            return f"{label}（{float(value):.3f}）"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)
    return str(value)
