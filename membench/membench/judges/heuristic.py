"""HeuristicJudge：零成本的启发式判分器（默认 judge）。

判分规则：
- 常规题：``token_f1(gold, prediction) >= 0.5`` 判 correct；
  另外当 gold 按标点切分后的每个片段都被 prediction 完整覆盖时
  （token 完全命中，可容忍模板化长回答），同样判 correct。
- stale：``gold_stale`` 出现在 prediction 中 → ``stale=True``。
- abstention 题（gold 为 ``"<ABSTAIN>"`` 或 category 为 abstention）：
  prediction 含拒答信号词 → ``abstained_pred=True``，且以此判定 correct。
"""

from __future__ import annotations

import re

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


def is_stale(gold_stale: "str | None", gold: str, prediction: str) -> bool:
    """陈旧判定：旧值命中且新值未命中才算 stale。

    纯子串匹配会把"引用旧值作为变更上下文"的正确更新回答误判为 stale，
    因此当 gold（新值）也出现在 prediction 中时说明系统已知更新，不判 stale。
    新值命中口径与判分逻辑一致（``gold_covered`` 片段覆盖，gold 可含元信息）。
    abstention 题（gold 为哨兵值）不做新值豁免，保持原有判定。
    """
    if not gold_stale:
        return False
    pred_n = normalize(prediction or "")
    if normalize(gold_stale) not in pred_n:
        return False
    if gold and gold != ABSTAIN_SENTINEL and gold_covered(gold, prediction or ""):
        return False
    return True


class HeuristicJudge(Judge):
    """启发式判分器：token 包含匹配 + 拒答检测 + 陈旧命中检测。"""

    name = "heuristic"

    #: 判对所需的 token_f1 阈值。
    f1_threshold: float = 0.5

    def grade(self, question: "Question", gold: str, prediction: str) -> JudgeResult:  # noqa: F821
        """按启发式规则给单条预测打分。

        ``question`` 既可以是 Question 数据类对象（读取其 category /
        gold_stale），也可以是普通字符串（视为无附加元信息）。
        """
        category = getattr(question, "category", None)
        gold_stale = getattr(question, "gold_stale", None)

        abstained = contains_refusal(prediction or "")
        stale = is_stale(gold_stale, gold, prediction or "")

        # abstention 题：gold 为哨兵值，期望拒答。
        if gold == ABSTAIN_SENTINEL:
            return JudgeResult(
                correct=abstained,
                abstained_pred=abstained,
                stale=stale,
                score=1.0 if abstained else 0.0,
                judge=self.name,
                reason="abstention 题：" + ("正确拒答" if abstained else "应拒答但未拒答"),
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
            reason += f"；命中过期事实 '{gold_stale}' 且未包含更新值"
        return JudgeResult(
            correct=correct,
            abstained_pred=abstained,
            stale=stale,
            score=score,
            judge=self.name,
            reason=reason,
        )
