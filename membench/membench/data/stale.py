"""过期旧值（``gold_stale``）解析：把 stale 标注解析成可机检的裸旧值。

``gold_stale`` 的规范口径是**裸旧值**（如 ``"4 study sessions"``、
``"9月20日"``），judge 对它做包含与位置序判定。但旧版 BEAM 转换器曾把整句
``potential_confusion``（如 ``"LLM might incorrectly recall the original 4
study sessions instead of the updated 5 study sessions."``）直接写进该字段，
而已下载的数据集不会随转换器升级自动更新。因此判定侧需要一个兜底：

- :func:`resolve_stale` 是判定侧的唯一入口：裸旧值原样返回；整句描述尝试
  抽取裸旧值；抽不出时返回 ``(None, False)``——**不可机检**，不进入
  ``stale_rate`` 分母，报告显示 n/a，而不是把"没有判定依据"记成 0。
- :func:`extract_stale_value` 从描述句抽值，转换器与判定侧共用同一套正则。

模块纯标准库、无副作用。
"""

from __future__ import annotations

import re

#: potential_confusion 的三种句式（大小写不敏感）；每条规则的第 1 捕获组是旧值。
#: 第 2/3 种（previously … now / originally … now）共用同一条 ``now`` 规则。
#: 允许标记词后的逗号（``Originally, …``）、值前的 ``the``、以及
#: ``instead of`` 前的逗号；捕获到句号 / 分号 / 换行 / 结尾为止。
_STALE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"original(?:ly)?\b\s*[,;:，；：]?\s*(?:the\s+)?(.+?)"
        r"\s*[,;，；—–]?\s*(?:instead\s+of|rather\s+than)\s+"
        r"(?:the\s+)?(?:updated\s+|new\s+|current\s+)?(.+?)(?:[.;。；\n]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:previously|originally)\b\s*[,;:，；：]?\s*(?:the\s+)?(.+?)"
        r"\s*[,;，；—–]?\s*(?:but\s+)?now\b\s*[,;:，；：]?\s*"
        r"(?:the\s+)?(.+?)(?:[.;。；\n]|$)",
        re.IGNORECASE,
    ),
)

#: 裸旧值的最大长度：更长的多半是整句描述而非"值"，按解析失败处理。
_MAX_STALE_LEN = 80

#: 归一裸旧值时剥离的引号 / 反引号与首尾标点。
_STALE_QUOTES = "\"'`“”‘’「」『』"
_STALE_LEADING = " \t\r\n,;:，；：、—–"
_STALE_TRAILING = " \t\r\n.,;:!?，。；：！？、"

#: 描述句里的主语 / 系动词前缀（如 "the user had 4 study sessions"、
#: "the deadline was March 15"）。反复剥离到不再命中，留下值本身。
_STALE_CLAUSE_RE = re.compile(
    r"^.*?\b(?:said|says|say|had|has|have|completed|planned|chose|used|"
    r"was|were|is|are|stated|mentioned|reported|updated(?:\s+to)?|"
    r"changed(?:\s+to)?|switched(?:\s+to)?)\b\s+",
    re.IGNORECASE,
)

#: 描述句的对比连词（小写）：真实裸值里几乎不会出现，却是 BEAM 描述句的固定句式。
_PROSE_MARKERS: tuple[str, ...] = ("instead of", "rather than", "but now")

#: 描述句的常见开头（小写）。补上连词缺失的句式（``previously X now Y``），
#: 也覆盖"只提旧值、没有新旧对照"的标注句（``LLM might … original X``）。
_PROSE_PREFIXES: tuple[str, ...] = (
    "llm might",
    "the llm might",
    "the model might",
    "previously",
    "originally",
)


def _normalize_stale_value(value: str) -> str:
    """归一裸旧值：去引号 / 反引号、首尾空白与首尾标点，并剥离主语 / 系动词前缀。"""
    text = (value or "").strip()
    while text and (text[0] in _STALE_QUOTES or text[0] in _STALE_LEADING):
        text = text[1:]
    while text and (text[-1] in _STALE_QUOTES or text[-1] in _STALE_TRAILING):
        text = text[:-1]
    text = text.strip()
    # "the user had 4 study sessions" -> "4 study sessions"；
    # 循环处理 "said the deadline was March 15" -> "March 15"。
    while True:
        trimmed = _STALE_CLAUSE_RE.sub("", text, count=1).strip()
        if trimmed == text or not trimmed:
            break
        text = trimmed
    return text


def _same_value(a: str, b: str) -> bool:
    """两个值归一后是否相同（忽略大小写、空白与常见标点）。"""
    def _norm(text: str) -> str:
        return re.sub(r"[\s，。！？、；：,.!?;:'\"“”‘’（）()]+", "", text.lower())

    return bool(_norm(a)) and _norm(a) == _norm(b)


def extract_stale_value(
    confusion: "str | None", gold_answer: str
) -> "str | None":
    """从 potential_confusion 描述句中抽取裸旧值。

    抽不到、旧值过长（> 80 字符）或与 gold_answer 相同（归一后）时返回
    ``None``——调用方应保留原始描述到 ``gold_stale_note``，并把
    ``gold_stale`` 留空（不可机检），而不是把描述句塞进值字段。

    Args:
        confusion: 原始描述句（BEAM 的 ``potential_confusion``）。
        gold_answer: 该题的标准答案（新值），用于排除"旧值=新值"的误抽。

    Returns:
        归一后的裸旧值；解析失败返回 ``None``。
    """
    if not isinstance(confusion, str) or not confusion.strip():
        return None
    for pattern in _STALE_PATTERNS:
        match = pattern.search(confusion)
        if match is None:
            continue
        value = _normalize_stale_value(match.group(1))
        if not value or len(value) > _MAX_STALE_LEN:
            continue
        if _same_value(value, gold_answer):
            continue
        return value
    return None


def looks_like_prose(value: "str | None") -> bool:
    """判断 ``gold_stale`` 的值是否是一句自然语言描述而非裸旧值。

    信号：含对比连词（``instead of`` / ``rather than`` / ``but now``）、
    以标注者口吻或时间副词开头（``LLM might`` / ``previously`` /
    ``originally``），或长度超过 :data:`_MAX_STALE_LEN`。

    误判方向是"把裸值当描述"：抽不出旧值时按不可机检处理，分母缩小、
    报告显示 n/a——这比保留原来的"分母齐全但恒 0.0%"更诚实。

    Args:
        value: ``gold_stale`` 字段原值（可能为 None）。

    Returns:
        True 表示应按描述句尝试抽取裸旧值。
    """
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    if len(text) > _MAX_STALE_LEN:
        return True
    lowered = text.lower()
    if any(marker in lowered for marker in _PROSE_MARKERS):
        return True
    return lowered.startswith(_PROSE_PREFIXES)


def resolve_stale(
    gold_stale: "str | None", gold_answer: str
) -> "tuple[str | None, bool]":
    """把 ``gold_stale`` 解析成可机检的裸旧值。

    判定侧（heuristic / llm / client）的唯一入口。旧缓存里 ``gold_stale``
    可能仍是整句描述，直接拿它做逐字匹配会恒不命中：``stale_rate`` 分母齐全
    却恒为 0.0，报告反而显示"优秀"。统一兜底口径：

    - 空值 → ``(None, False)``；
    - 描述句（:func:`looks_like_prose`）→ 用 :func:`extract_stale_value`
      抽裸旧值：抽到 ``(旧值, True)``，抽不到 ``(None, False)``（不可机检，
      不进入 ``stale_rate`` 分母）；
    - 其余视为裸旧值 → ``(原值, True)``。

    Args:
        gold_stale: ``gold_stale`` 字段原值（可能为 None）。
        gold_answer: 标准答案（新值），抽取时用于排除"旧值=新值"。

    Returns:
        ``(裸旧值 | None, 是否可机检)``。
    """
    if not isinstance(gold_stale, str) or not gold_stale.strip():
        return None, False
    if looks_like_prose(gold_stale):
        value = extract_stale_value(gold_stale, gold_answer)
        return (value, True) if value is not None else (None, False)
    return gold_stale, True
