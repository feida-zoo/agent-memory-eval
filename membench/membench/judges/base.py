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
        stale: 是否把过期旧值当成当前事实（口径见
            :func:`membench.judges.heuristic.is_stale`）。
        score: [0,1] 分数；启发式 = f1 阈值化，LLM = 0/1。
        judge: 判分来源标识，"heuristic" | "llm" | "llm-fallback"。
        reason: 判分理由（可选）。
        stale_checkable: 本题的 stale 是否**可机检**——即本次判分确实拿到了
            可用的旧值（启发式：有裸 ``gold_stale``；LLM：有 ``gold_stale``
            或 ``gold_stale_note``）。``stale_rate`` 的分母只含这类题；
            不可机检的题不参与均值，避免把"没有判定依据"记成"不陈旧"。
        stale_annotated: 本题是否有 stale 注解（``gold_stale`` 或
            ``gold_stale_note`` 非空）。与 ``stale_checkable`` 不同，它只描述
            数据标注、不依赖判分器：报告用 ``stale_annotated`` 的题数作覆盖率
            分母，回答"多少道 stale 题可机检"。
    """

    correct: bool
    abstained_pred: bool
    stale: bool
    score: float
    judge: str
    reason: str = ""
    stale_checkable: bool = False
    stale_annotated: bool = False


class Judge(ABC):
    """判分器抽象基类。"""

    name: str = "base"

    @abstractmethod
    def grade(self, question: "Question", gold: str, prediction: str) -> JudgeResult:
        """给单条预测打分。

        Args:
            question: Question 数据类对象（含 category / gold_stale /
                gold_stale_note 等字段）。
            gold: 标准答案字符串（question.gold_answer）。
            prediction: 被测系统的回答文本。

        Returns:
            JudgeResult: 判分结果。
        """
        raise NotImplementedError
