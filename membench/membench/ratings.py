"""评级系统：在数值分数之外给出定性评价（优秀/良好/中等/较差）。

评价以"评价（百分比）"形式呈现，如 ``优秀（92.3%）``——比率型指标统一
按百分比展示具体值，避免只给一个看不出量级的分数。

四级评价及点数映射：
    优秀 = 4、良好 = 3、中等 = 2、较差 = 1。

单指标阈值在模块内以数据表 :data:`METRIC_THRESHOLDS` 定义，便于后续调整。
效率成本类指标（latency / tokens / llm_calls / n_questions）依赖部署环境，
强制定性会误导，故不参与评级（:func:`rate_metric` 返回 ``None``）。

综合评级使用 :func:`continuous_points` 的连续点数（1.0~4.0），
在阈值边界处与离散点数一致，但档内随数值线性变化，
使"档内改进"也能反映到综合评级上。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "LABELS",
    "LABEL_POINTS",
    "METRIC_THRESHOLDS",
    "OVERALL_THRESHOLDS",
    "RATE_METRICS",
    "continuous_points",
    "format_metric",
    "format_metric_delta",
    "format_metric_value",
    "rate_metric",
    "rating_points",
    "overall_rating",
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

# 比率型指标（[0,1]，报告中以百分比展示具体值）：六项可评级质量指标 +
# 两项诊断指标（严格检索召回率@5 / 过度拒答率——展示但不参与评级）。
RATE_METRICS: frozenset[str] = frozenset(METRIC_THRESHOLDS) | {
    "recall_at_5",
    "over_refusal_rate",
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


def continuous_points(key: str, value: float) -> float | None:
    """对单个指标给出连续点数（1.0~4.0），保留档内位置信息。

    与 :func:`rate_metric` 的离散标签在阈值边界处取值一致
    （如 accuracy 0.60→2.0、0.75→3.0、0.90→4.0），但档内随数值线性变化，
    使综合评级对"档内改进"敏感（如 accuracy 0.647→0.715 会从 2.31 升到 2.77）。

    分段线性映射（以 accuracy 为例，阈值 0.90/0.75/0.60）：
        v >= 0.90        → 4.0
        0.75 <= v < 0.90 → 3.0 + (v-0.75)/0.15  ∈ [3.0, 4.0)
        0.60 <= v < 0.75 → 2.0 + (v-0.60)/0.15  ∈ [2.0, 3.0)
        v < 0.60         → 1.0 + v/0.60          ∈ [1.0, 2.0)
    lower 方向（stale_rate）对称处理。

    Args:
        key: 指标名。
        value: 指标数值。

    Returns:
        [1.0, 4.0] 的连续点数；不可评级的指标返回 ``None``。
    """
    spec = METRIC_THRESHOLDS.get(key)
    if spec is None:
        return None
    direction, thresholds = spec
    v = float(value)
    t_excellent, t_good, t_medium = (t for t, _ in thresholds)
    if direction == "higher":
        if v >= t_excellent:
            return 4.0
        if v >= t_good:
            return 3.0 + (v - t_good) / (t_excellent - t_good)
        if v >= t_medium:
            return 2.0 + (v - t_medium) / (t_good - t_medium)
        return 1.0 + v / t_medium
    else:  # lower is better
        if v <= t_excellent:
            return 4.0
        if v <= t_good:
            return 3.0 + (t_good - v) / (t_good - t_excellent)
        if v <= t_medium:
            return 2.0 + (t_medium - v) / (t_medium - t_good)
        return 1.0 + (1.0 - v) / (1.0 - t_medium)


def overall_rating(overall: dict[str, Any]) -> tuple[str, float] | None:
    """对 overall 汇总给出综合评级。

    取 overall 中"存在且可评级"的质量指标
    （accuracy / token_f1 / bleu1 / evidence_recall /
    abstention_accuracy / stale_rate）各自的**连续点数**均值
    （:func:`continuous_points`，档内随数值线性变化），
    均值 >= 3.5 优秀、>= 2.5 良好、>= 1.5 中等，否则较差。

    Args:
        overall: 汇总指标字典（aggregate 输出的 ``metrics["overall"]``）。

    Returns:
        ``(标签, 连续点数均值)``；overall 中不存在任何可评级指标时返回 ``None``。
    """
    points: list[float] = []
    for key in METRIC_THRESHOLDS:
        if key not in overall:
            continue
        try:
            value = float(overall[key])
        except (TypeError, ValueError):
            continue
        pts = continuous_points(key, value)
        if pts is not None:
            points.append(pts)
    if not points:
        return None
    mean = sum(points) / len(points)
    for threshold, label in OVERALL_THRESHOLDS:
        if mean >= threshold:
            return label, mean
    return LABELS[-1], mean


def format_metric_value(key: str, value: Any) -> str:
    """指标值的纯数值呈现（不带评价标签）。

    比率型指标（:data:`RATE_METRICS`）以百分比呈现，如 ``"92.3%"``；其余
    指标保留 3 位小数（与 report 的 ``_fmt`` 口径一致）；``None`` 与
    ``"n/a"`` 一律视为"未定义"（避免把"数据集没标 evidence"误读为"召回率
    为 0"）。

    Args:
        key: 指标名。
        value: 指标数值（允许非数值，如 "n/a"）。

    Returns:
        报告单元格字符串。
    """
    if value is None:
        return "未定义"
    if isinstance(value, str) and value.strip().lower() in ("n/a", "none", "null", ""):
        return "未定义"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if key in RATE_METRICS:
            return f"{float(value) * 100:.1f}%"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)
    return str(value)


def format_metric(key: str, value: Any) -> str:
    """把指标值格式化为报告呈现字符串（值 + 可选评价标签）。

    比率型指标以百分比呈现具体值：可评级指标返回 ``"评价（百分比）"``
    （如 ``"优秀（92.3%）"``），不可评级的诊断指标（如 recall_at_5）直接
    返回百分比；其余指标返回 ``format_metric_value`` 的格式化数值。

    Args:
        key: 指标名。
        value: 指标数值（允许非数值，如 "n/a"）。

    Returns:
        报告单元格字符串。
    """
    value_str = format_metric_value(key, value)
    if value_str == "未定义":
        return value_str
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        label = rate_metric(key, float(value))
        if label is not None:
            return f"{label}（{value_str}）"
    return value_str


def format_metric_delta(key: str, value: Any) -> str:
    """指标差值（B - A）呈现：比率型指标用百分点（``"+2.3pt"``），
    其余保留 3 位小数（``"+0.023"``）；``None`` 返回 ``"n/a"``。"""
    if value is None:
        return "n/a"
    v = float(value)
    if key in RATE_METRICS:
        return f"{v * 100:+.1f}pt"
    return f"{v:+.3f}"
