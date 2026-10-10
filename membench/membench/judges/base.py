"""Judge 协议与 JudgeResult 数据结构（SPEC 3.3）。

判分双轨：启发式判分（默认，零成本）+ LLM-as-Judge（可选）。
所有 Judge 实现 ``grade(question, gold, prediction) -> JudgeResult``。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 仅类型标注需要，避免运行时循环依赖
    from membench.data.schema import Question


@dataclass
class JudgeResult:
    """单题判分结果。

    Attributes:
        correct: 是否判对。
        abstained_pred: 预测是否拒答。
        stale: 是否命中过期事实（gold_stale 出现在 prediction 中）。
        score: [0,1] 分数；启发式 = f1 阈值化，LLM = 0/1。
        judge: 判分来源标识，"heuristic" | "llm" | "llm-fallback"。
        reason: 判分理由（可选）。
    """

    correct: bool
    abstained_pred: bool
    stale: bool
    score: float
    judge: str
    reason: str = ""


class Judge(ABC):
    """判分器抽象基类。"""

    name: str = "base"

    @abstractmethod
    def grade(self, question: "Question", gold: str, prediction: str) -> JudgeResult:
        """给单条预测打分。

        Args:
            question: Question 数据类对象（含 category / gold_stale 等字段）。
            gold: 标准答案字符串（question.gold_answer）。
            prediction: 被测系统的回答文本。

        Returns:
            JudgeResult: 判分结果。
        """
        raise NotImplementedError
