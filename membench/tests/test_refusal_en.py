"""拒答信号英文补充 + 大小写不敏感 的回归测试（修复 2）。"""

from __future__ import annotations

import pytest

from membench.judges.heuristic import HeuristicJudge, contains_refusal
from membench.data.schema import ABSTAIN_SENTINEL, Question


class TestEnglishRefusalSignals:
    """英文拒答句应被识别（大小写不敏感），正常英文回答不误伤。"""

    @pytest.mark.parametrize(
        "text",
        [
            "I don't have any information about that.",
            "I'm not sure.",
            "No record of this.",
            "I DON'T KNOW",
            "There is no information available.",
            "This was not mentioned in our conversation.",
            "I cannot determine the answer.",
            "Unable to answer based on my memory.",
            "I don't recall anything about it.",
            "No memory of that event.",
            "No relevant memories found.",
        ],
    )
    def test_english_refusal_detected(self, text: str) -> None:
        assert contains_refusal(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "The meeting is on Tuesday.",
            # 陷阱：含 "no" + "mentioned" 但不相邻，不应误判
            "There is no parking fee mentioned because it is free",
            "There is no charge for parking.",
            "I know the answer: it is 42.",
        ],
    )
    def test_normal_answer_not_flagged(self, text: str) -> None:
        assert contains_refusal(text) is False

    @pytest.mark.parametrize("text", ["我不知道。", "相关信息不在我的记忆中。", "无法确定"])
    def test_chinese_refusal_still_detected(self, text: str) -> None:
        assert contains_refusal(text) is True


class TestEnglishAbstentionGrading:
    """abstention 题（gold=<ABSTAIN>）英文拒答应判 correct。"""

    def test_english_abstention_correct(self) -> None:
        judge = HeuristicJudge()
        q = Question(
            qid="t_abs",
            question="What did the user say about their salary?",
            gold_answer=ABSTAIN_SENTINEL,
            category="abstention",
        )
        result = judge.grade(q, ABSTAIN_SENTINEL, "I don't have any information about that.")
        assert result.correct is True
        assert result.abstained_pred is True
