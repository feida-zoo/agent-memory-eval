"""HeuristicJudge / LLMJudge 单元测试。"""

from __future__ import annotations

from membench.data.schema import ABSTAIN_SENTINEL, Question
from membench.judges import HeuristicJudge, LLMJudge, get_judge


def _q(
    category: str = "single_hop",
    gold_stale: str | None = None,
    gold_answer: str = "杭州",
) -> Question:
    """构造一个测试用 Question。"""
    return Question(
        qid="t_q1",
        question="用户现在住在哪个城市？",
        gold_answer=gold_answer,
        category=category,
        evidence=[],
        gold_stale=gold_stale,
    )


class TestHeuristicJudge:
    """启发式判分：f1 阈值 / gold 覆盖 / stale / 拒答。"""

    def setup_method(self) -> None:
        self.judge = HeuristicJudge()

    def test_correct_via_high_f1(self) -> None:
        # prediction 与 gold 完全一致 → token_f1 = 1.0 ≥ 0.5。
        result = self.judge.grade(_q(), "杭州", "杭州")
        assert result.correct is True
        assert result.score == 1.0
        assert result.judge == "heuristic"
        assert result.stale is False
        assert result.abstained_pred is False

    def test_correct_via_gold_coverage(self) -> None:
        # 模板化长回答 f1 低，但完整覆盖 gold → 仍判对。
        result = self.judge.grade(_q(), "杭州", "根据我的记忆：现在住在杭州市余杭区。")
        assert result.correct is True

    def test_correct_via_multi_part_coverage(self) -> None:
        # 多片段 gold（逗号分隔）要求每个片段都被覆盖。
        result = self.judge.grade(
            _q(category="multi_hop"),
            "滨江公园，跑鞋",
            "根据我的记忆：每天早上去滨江公园跑步；我还买了专业的跑鞋。",
        )
        assert result.correct is True
        result2 = self.judge.grade(
            _q(category="multi_hop"),
            "滨江公园，跑鞋",
            "根据我的记忆：每天早上去滨江公园跑步。",
        )
        assert result2.correct is False

    def test_incorrect_when_no_overlap(self) -> None:
        result = self.judge.grade(_q(), "杭州", "我不知道。")
        # “我不知道。”触发拒答信号，但本题非 abstention → 仍判错。
        assert result.correct is False
        assert result.abstained_pred is True

    def test_stale_detection(self) -> None:
        result = self.judge.grade(
            _q(category="knowledge_update", gold_stale="北京"),
            "杭州",
            "我的狗叫旺财，我之前住在北京，现在搬到杭州了。",
        )
        assert result.stale is True
        # gold_stale 未命中则不为 stale。
        result2 = self.judge.grade(
            _q(category="knowledge_update", gold_stale="北京"),
            "杭州",
            "根据我的记忆：现在住在杭州市余杭区。",
        )
        assert result2.stale is False

    def test_stale_ignored_without_gold_stale(self) -> None:
        result = self.judge.grade(_q(), "杭州", "北京杭州都住过")
        assert result.stale is False

    def test_abstention_correct_refusal(self) -> None:
        result = self.judge.grade(
            _q(category="abstention", gold_answer=ABSTAIN_SENTINEL),
            ABSTAIN_SENTINEL,
            "我不知道，相关信息不在我的记忆中。",
        )
        assert result.abstained_pred is True
        assert result.correct is True
        assert result.score == 1.0

    def test_abstention_wrong_answer(self) -> None:
        result = self.judge.grade(
            _q(category="abstention", gold_answer=ABSTAIN_SENTINEL),
            ABSTAIN_SENTINEL,
            "根据我的记忆：用户住在杭州。",
        )
        assert result.abstained_pred is False
        assert result.correct is False
        assert result.score == 0.0

    def test_refusal_signals(self) -> None:
        for text in ("我不记得了", "无法确定这件事", "对话中没有相关信息"):
            assert self.judge.grade(
                _q(category="abstention", gold_answer=ABSTAIN_SENTINEL),
                ABSTAIN_SENTINEL,
                text,
            ).abstained_pred is True

    def test_question_accepts_plain_string(self) -> None:
        # question 传普通字符串时按无元信息处理，不应抛异常。
        result = self.judge.grade("任意问题", "杭州", "杭州")
        assert result.correct is True


class TestGetJudge:
    """判分器工厂。"""

    def test_get_heuristic(self) -> None:
        judge = get_judge("heuristic")
        assert isinstance(judge, HeuristicJudge)

    def test_get_llm_with_kwargs(self) -> None:
        judge = get_judge(
            "llm", base_url="http://127.0.0.1:9", api_key="k", model="m"
        )
        assert isinstance(judge, LLMJudge)
        assert judge.model == "m"

    def test_unknown_judge(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="未知判分器"):
            get_judge("nope")


class TestLLMJudgeFallback:
    """LLMJudge：任何异常都必须回退 HeuristicJudge 并标记 llm-fallback。"""

    def test_fallback_on_connection_error(self) -> None:
        # 指向一个不可达地址（端口 9 通常无服务），触发连接异常。
        judge = LLMJudge(base_url="http://127.0.0.1:9", api_key="k", model="m", timeout=1)
        result = judge.grade(_q(), "杭州", "杭州")
        assert result.judge == "llm-fallback"
        # 回退结果应与 HeuristicJudge 一致（gold 完全命中 → 判对）。
        assert result.correct is True
        assert "回退启发式" in result.reason

    def test_fallback_without_base_url(self) -> None:
        import os

        os.environ.pop("MEMBENCH_JUDGE_BASE_URL", None)
        os.environ.pop("OPENAI_BASE_URL", None)
        judge = LLMJudge(model="m")
        result = judge.grade(_q(), "杭州", "杭州")
        assert result.judge == "llm-fallback"
        assert result.correct is True

    def test_fallback_preserves_stale_and_abstain(self) -> None:
        judge = LLMJudge(base_url="http://127.0.0.1:9", api_key="k", model="m", timeout=1)
        result = judge.grade(
            _q(category="knowledge_update", gold_stale="北京"),
            "杭州",
            "我之前住在北京。",
        )
        assert result.judge == "llm-fallback"
        assert result.stale is True
