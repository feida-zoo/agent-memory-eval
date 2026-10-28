"""HeuristicJudge：零成本的启发式判分器（默认 judge）。

判分规则：
- 常规题：``token_f1(gold, prediction) >= 0.5`` 判 correct；
  另外当 gold 按标点切分后的每个片段都被 prediction 完整覆盖时
  （token 完全命中，可容忍模板化长回答），同样判 correct。
- stale：``gold_stale``（裸旧值）出现在 prediction 中，且没有先把 gold
  （新值）作为当前事实给出 → ``stale=True``。判定按位置序（见
  :func:`is_stale`）：新值逐字在前即豁免；新值只靠同义命中时用数字锚点
  定位；旧值命中却找不到新值锚点则判陈旧。
- ``stale_checkable``：仅当题目有裸 ``gold_stale`` 时为真——stale_rate 的
  分母只含这类题。
- ``stale_annotated``：题目是否有 stale 注解（``gold_stale`` 或
  ``gold_stale_note`` 非空），供报告计算覆盖率（可机检 / 全部注解题）。
- abstention 题（gold 为 ``"<ABSTAIN>"`` 或 category 为 abstention）：
  prediction 含拒答信号词 → ``abstained_pred=True``，且以此判定 correct。
"""

from __future__ import annotations

import re

from membench.adapters.mock import content_tokens
from membench.data.schema import ABSTAIN_SENTINEL
from membench.metrics.text import token_f1

from .base import Judge, JudgeResult

#: 拒答信号词（均为多字片段，避免 “没有”/“no”/“not” 等单字误伤正常回答，
#: 例如 "There is no charge for parking" 不应判为拒答）。
#: 中文信号不受大小写影响；英文信号以小写存储，匹配时对 prediction 做 lower()。
REFUSAL_SIGNALS: tuple[str, ...] = (
    # 中文信号
    "不知道",
    "无法确定",
    "不记得",
    "不在我的记忆",
    "没有相关",
    "没有提到",
    "没有任何记录",
    "不清楚",
    "无法回答",
    "无法确认",
    # 英文信号（小写，多字片段）
    "i don't know",
    "i do not know",
    "don't have",
    "do not have",
    "no information",
    "not mentioned",
    "no mention of",
    "no record",
    "cannot determine",
    "can't determine",
    "i'm not sure",
    "i am not sure",
    "no relevant",
    "unable to answer",
    "cannot answer",
    "don't recall",
    "do not recall",
    "no memory of",
)

#: gold 片段切分符（中英文标点与空白）。
_GOLD_SPLIT_RE = re.compile(r"[，,、；;。\s]+")

#: 归一化时剔除的标点与空白字符。
_STRIP_RE = re.compile(r"[\s，。！？、；：,.!?;:'\"“”‘’（）()]+")


def normalize(text: str) -> str:
    """归一化文本：转小写并剔除空白与常见中英文标点。"""
    return _STRIP_RE.sub("", (text or "").lower())


def contains_refusal(prediction: str) -> bool:
    """检测 prediction 是否包含拒答信号词（大小写不敏感）。

    对 prediction 先 ``lower()`` 再做子串匹配：英文信号以小写存储，因此
    "I DON'T KNOW" 也能命中 "i don't know"；中文信号不受 lower 影响。
    """
    pred_lower = (prediction or "").lower()
    return any(sig in pred_lower for sig in REFUSAL_SIGNALS)


#: 新值命中豁免的 token 召回阈值：gold 的内容 token 中至少该比例出现在
#: prediction 里，即视为"系统已知更新"。阈值取 0.8：既容忍同义改写
#: （如"下周一下午"→"周一下午"），又能挡住"只命中个别字词"的伪命中。
_GOLD_RECALL_THRESHOLD: float = 0.8

#: 数字型 token（如 ``5``、``52``）：新值字面不占先时，用它在预测中的位置
#: 判断"新值是否先于旧值给出"。
_DIGIT_RE = re.compile(r"\d+")


def _numeric_anchors(gold: str) -> set[str]:
    """gold 中的数字型 token（完整数字串，去重）。"""
    return set(_DIGIT_RE.findall(normalize(gold)))


def _find_value(text: str, value: str, start: int = 0) -> int:
    """在归一化文本中查找 value，跳过与相邻数字粘连的伪命中。

    归一化会去掉空白与标点，``"14 study sessions"`` 因此包含子串
    ``"4 study sessions"``。裸旧值以数字开头/结尾时要求两侧不是数字，
    避免把 ``14`` 当成旧值 ``4``（反之亦然）。
    """
    if not value:
        return -1
    pos = text.find(value, start)
    while pos >= 0:
        end = pos + len(value)
        left_ok = not (value[0].isdigit() and pos > 0 and text[pos - 1].isdigit())
        right_ok = not (
            value[-1].isdigit() and end < len(text) and text[end].isdigit()
        )
        if left_ok and right_ok:
            return pos
        pos = text.find(value, pos + 1)
    return -1


def gold_covered(gold: str, prediction: str) -> bool:
    """gold 按标点切分后，每个非空片段是否都是 prediction 的子串。"""
    pred_n = normalize(prediction)
    if not pred_n:
        return False
    parts = [normalize(p) for p in _GOLD_SPLIT_RE.split(gold)]
    parts = [p for p in parts if p]
    if not parts:
        return False
    return all(p in pred_n for p in parts)


def gold_recall_hit(gold: str, prediction: str, threshold: float = _GOLD_RECALL_THRESHOLD) -> bool:
    """gold 的内容 token 召回率是否达标（容忍同义改写的新值命中口径）。

    与 ``gold_covered`` 的逐字子串不同，这里按 token 多集算召回：
    回答把新值换了说法（如"下周一下午"→"周一下午"、"流量高峰打满"→"高峰打满"）
    也能命中。gold 过短（< 4 个内容 token）时阈值收紧为 1.0，避免
    短 gold 被个别字词误命中。
    """
    gold_toks = content_tokens(gold or "")
    if not gold_toks:
        return False
    pred_toks = content_tokens(prediction or "")
    if not pred_toks:
        return False
    hit = len(gold_toks & pred_toks) / len(gold_toks)
    effective = 1.0 if len(gold_toks) < 4 else threshold
    return hit >= effective


def is_stale(gold_stale: "str | None", gold: str, prediction: str) -> bool:
    """陈旧判定：预测里旧值出现，且**没有先给出更新后的值**才算 stale。

    只做「旧值出现即陈旧」会把"引用旧值作为变更上下文、同时给出新值"的正确
    回答误判为陈旧；反过来只做「新值出现即豁免」又会被"主答案给旧值、括号里
    顺带提一句新值"骗过（BEAM 的典型失败）。因此按**位置序**判定：

    1. ``gold_stale`` 为空 / 未出现在预测中 → 非 stale。
    2. 新值（``gold`` 归一后）逐字出现，且**首现早于**旧值 → 非 stale
       （如 ``"5 sessions, not 4"``：新值在前，旧值只是对照）；旧值整段落在
       新值内部（如旧值 ``4 sessions``、新值 ``4 sessions per week``）同样豁免。
    3. 新值字面没有占先，但 ``gold_covered`` / ``gold_recall_hit`` 命中
       （同义改写、被拆开）→ 再看 gold 的**数字型 token**（如 ``5``、``52``）
       在预测里出现的位置：任一数字锚点早于旧值 → 非 stale；数字锚点全部
       在旧值之后（如主答案 ``4 sessions``、括号里 ``5 sessions``）→ stale。
       gold 没有数字锚点时，沿用"新值语义命中即豁免"（无法定位先后，不
       凭空判陈旧）。
    4. 旧值出现且找不到任何新值锚点 → stale（宁可判陈旧，让 LLM 判分纠正）。

    已知边界：``"原来 4 个，现在 5 个"`` 这类"旧值在前、新值在后"的正确回答
    会被判 stale（数字锚点在旧值之后）——规则只认位置序，不解析语义；这是
    刻意的保守方向，由 stale_rate 的可机检口径与 LLM judge 兜底共同约束。
    """
    if not gold_stale:
        return False
    pred_n = normalize(prediction or "")
    old_n = normalize(gold_stale)
    if not old_n:
        return False
    old_pos = _find_value(pred_n, old_n)
    if old_pos < 0:
        return False
    if not gold or gold == ABSTAIN_SENTINEL:
        return True
    gold_n = normalize(gold)
    new_pos = _find_value(pred_n, gold_n) if gold_n else -1
    if 0 <= new_pos < old_pos:
        return False
    # 旧值是新值的子串（"4 sessions" ⊂ "4 sessions per week"）：
    # 覆盖旧值的那次新值命中说明给的是完整新值，不是旧值本身。
    if 0 <= new_pos <= old_pos and new_pos + len(gold_n) >= old_pos + len(old_n):
        return False
    if gold_covered(gold, prediction or "") or gold_recall_hit(gold, prediction or ""):
        anchors = _numeric_anchors(gold)
        if not anchors:
            return False
        # 数字锚点按"完整数字串"匹配，避免把 2025 里的 5 当成新值。
        for match in _DIGIT_RE.finditer(pred_n):
            if match.group(0) in anchors and match.start() < old_pos:
                return False
        return True
    return True


class HeuristicJudge(Judge):
    """启发式判分器：token 包含匹配 + 拒答检测 + 陈旧命中检测。"""

    name = "heuristic"

    #: 判对所需的 token_f1 阈值。
    f1_threshold: float = 0.5

    def grade(self, question: "Question", gold: str, prediction: str) -> JudgeResult:  # noqa: F821
        """按启发式规则给单条预测打分。

        ``question`` 既可以是 Question 数据类对象（读取其 category /
        gold_stale / gold_stale_note），也可以是普通字符串（视为无附加元信息）。
        """
        category = getattr(question, "category", None)
        gold_stale = getattr(question, "gold_stale", None)
        gold_stale_note = getattr(question, "gold_stale_note", None)

        abstained = contains_refusal(prediction or "")
        stale = is_stale(gold_stale, gold, prediction or "")
        # 有裸旧值才可机检：只有描述句（gold_stale_note）时启发式判不了 stale。
        stale_checkable = bool((gold_stale or "").strip())
        stale_annotated = bool(
            (gold_stale or "").strip() or (gold_stale_note or "").strip()
        )

        # abstention 题：gold 为哨兵值，期望拒答。
        if gold == ABSTAIN_SENTINEL:
            return JudgeResult(
                correct=abstained,
                abstained_pred=abstained,
                stale=stale,
                score=1.0 if abstained else 0.0,
                judge=self.name,
                reason="abstention 题：" + ("正确拒答" if abstained else "应拒答但未拒答"),
                stale_checkable=stale_checkable,
                stale_annotated=stale_annotated,
            )

        f1 = token_f1(gold, prediction or "")
        covered = gold_covered(gold, prediction or "")
        correct = f1 >= self.f1_threshold or covered
        score = 1.0 if correct else f1
        if f1 >= self.f1_threshold:
            reason = f"token_f1={f1:.3f} >= {self.f1_threshold}"
        elif covered:
            reason = f"gold 片段被完整覆盖（token_f1={f1:.3f}）"
        else:
            reason = f"token_f1={f1:.3f} < {self.f1_threshold} 且 gold 未覆盖"
        if stale:
            reason += f"；命中过期事实 '{gold_stale}' 且未先给出更新值"
        return JudgeResult(
            correct=correct,
            abstained_pred=abstained,
            stale=stale,
            score=score,
            judge=self.name,
            reason=reason,
            stale_checkable=stale_checkable,
            stale_annotated=stale_annotated,
        )
