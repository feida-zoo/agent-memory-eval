"""HeuristicJudge / LLMJudge 单元测试。"""

from __future__ import annotations

from membench.data.schema import ABSTAIN_SENTINEL, Question
from membench.judges import HeuristicJudge, LLMJudge, get_judge


def _q(
    category: str = "single_hop",
    gold_stale: str | None = None,
    gold_answer: str = "杭州",
    gold_stale_note: str | None = None,
) -> Question:
    """构造一个测试用 Question。"""
    return Question(
        qid="t_q1",
        question="用户现在住在哪个城市？",
        gold_answer=gold_answer,
        category=category,
        evidence=[],
        gold_stale=gold_stale,
        gold_stale_note=gold_stale_note,
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
        # 旧值单独出现（未包含新值杭州）→ stale。
        result = self.judge.grade(
            _q(category="knowledge_update", gold_stale="北京"),
            "杭州",
            "我的狗叫旺财，我之前住在北京。",
        )
        assert result.stale is True
        # gold_stale 未命中则不为 stale。
        result2 = self.judge.grade(
            _q(category="knowledge_update", gold_stale="北京"),
            "杭州",
            "根据我的记忆：现在住在杭州市余杭区。",
        )
        assert result2.stale is False

    def test_stale_exempt_when_new_value_present(self) -> None:
        # 旧值仅作为变更上下文被引用、且新值（gold）同时命中 → 不判 stale。
        result = self.judge.grade(
            _q(category="knowledge_update", gold_stale="北京"),
            "杭州",
            "我的狗叫旺财，我之前住在北京，现在搬到杭州了。",
        )
        assert result.stale is False
        assert result.correct is True
        # 多片段 gold 全部被覆盖同样豁免。
        result2 = self.judge.grade(
            _q(category="knowledge_update", gold_stale="北京"),
            "杭州，余杭区",
            "之前在北京，现在住在杭州余杭区。",
        )
        assert result2.stale is False
        # 新值未完整覆盖（缺"余杭区"）→ 仍判 stale。
        result3 = self.judge.grade(
            _q(category="knowledge_update", gold_stale="北京"),
            "杭州，余杭区",
            "之前在北京，现在搬到杭州了。",
        )
        assert result3.stale is True
        assert "未先给出更新值" in result3.reason

    def test_stale_position_order_user_repro(self) -> None:
        """用户报告的两条 BEAM 样例：主答案旧值、括号新值 → stale；
        新值在前 → 非 stale。"""
        q = _q(
            category="knowledge_update",
            gold_stale="4 study sessions",
            gold_answer="5 study sessions",
        )
        # 例 1：主答案给 4（旧值），括号里出现 5 → 仍判陈旧。
        r = self.judge.grade(
            q, "5 study sessions", "You did 4 study sessions this month (not 5)."
        )
        assert r.stale is True
        # 新值在前、旧值只是对照 → 非 stale。
        r2 = self.judge.grade(
            q, "5 study sessions", "5 study sessions, not 4."
        )
        assert r2.stale is False
        # 旧值未出现 → 非 stale。
        r3 = self.judge.grade(q, "5 study sessions", "You did 5 study sessions.")
        assert r3.stale is False
        # gold_stale 为空 → 非 stale（也不可机检）。
        r4 = self.judge.grade(
            _q(category="knowledge_update", gold_answer="5 study sessions"),
            "5 study sessions",
            "You did 4 study sessions.",
        )
        assert r4.stale is False
        assert r4.stale_checkable is False

    def test_stale_numeric_boundary_and_prefix(self) -> None:
        """裸旧值的数字边界与"旧值是新值前缀"两类边界。"""
        # 14 不应被当成旧值 4。
        r = self.judge.grade(
            _q(
                category="knowledge_update",
                gold_stale="4 study sessions",
                gold_answer="5 study sessions",
            ),
            "5 study sessions",
            "You did 14 study sessions.",
        )
        assert r.stale is False
        # 旧值 "4 sessions" 整段落在新值 "4 sessions per week" 内 → 非 stale。
        r2 = self.judge.grade(
            _q(
                category="knowledge_update",
                gold_stale="4 sessions",
                gold_answer="4 sessions per week",
            ),
            "4 sessions per week",
            "The plan is 4 sessions per week.",
        )
        assert r2.stale is False

    def test_stale_checkable_and_annotated(self) -> None:
        """stale_checkable 只认裸 gold_stale；note-only 题可标注但不可机检。"""
        bare = self.judge.grade(
            _q(category="knowledge_update", gold_stale="北京"),
            "杭州",
            "之前在北京，现在在杭州。",
        )
        assert bare.stale_checkable is True
        assert bare.stale_annotated is True

        note_only = self.judge.grade(
            _q(
                category="knowledge_update",
                gold_stale_note="LLM might recall the original Beijing.",
            ),
            "杭州",
            "北京。",
        )
        assert note_only.stale is False
        assert note_only.stale_checkable is False
        assert note_only.stale_annotated is True

        none = self.judge.grade(_q(), "杭州", "杭州")
        assert none.stale_checkable is False
        assert none.stale_annotated is False

    def test_stale_exempt_when_new_value_paraphrased(self) -> None:
        # office 集 knowledge_update 题：回答正确给出新值、但
        # 改写/paraphrase 了表述，同时把旧值作为变更上下文提及，不应判 stale。
        # d10_q1：gold="下周一下午"，回答改写为"周一下午/9-14"。
        r1 = self.judge.grade(
            _q(category="knowledge_update", gold_stale="周四下午"),
            "下周一下午",
            "何斌的面试最终改到了 2026-09-14（周一下午），原定于 2026-09-07 那周的周四下午，9 月 9 日改期至下周一即 9 月 14 日下午。",
        )
        assert r1.stale is False
        # d11_q1：gold 含"流量高峰打满"，回答改写为"高峰打满"。
        r2 = self.judge.grade(
            _q(category="knowledge_update", gold_stale="Redis集群主从切换抖动"),
            "连接池配置把最大连接数从500改成了50，流量高峰打满",
            "最终根因是连接池配置把最大连接数从 500 改成了 50，导致高峰打满；最初怀疑的 Redis 集群主从切换抖动并非根因。",
        )
        assert r2.stale is False
        # 仍判 stale 的对照：只提旧值、新值完全没出现。
        r3 = self.judge.grade(
            _q(category="knowledge_update", gold_stale="周四下午"),
            "下周一下午",
            "何斌的面试在周四下午。",
        )
        assert r3.stale is True

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


class _FakeLLMResponse:
    """urlopen 的最小替身：返回一个 chat completions 响应。"""

    def __init__(self, content: str) -> None:
        import json

        self._body = json.dumps(
            {"choices": [{"message": {"content": content}}]}, ensure_ascii=False
        ).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeLLMResponse":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def _stub_llm(monkeypatch, content: str) -> LLMJudge:
    """让 LLMJudge 的 HTTP 调用固定返回 content。"""
    import urllib.request

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda req, timeout: _FakeLLMResponse(content)
    )
    return LLMJudge(base_url="http://stub", api_key="k", model="m")


class TestLLMJudgeStale:
    """LLM 的 stale 字段采用 / 回退口径。"""

    def test_model_stale_adopted(self, monkeypatch) -> None:
        # 模型判 False，即使启发式会判 True（旧值出现）也以模型为准。
        judge = _stub_llm(
            monkeypatch, '{"correct": true, "stale": false, "reason": "更新上下文"}'
        )
        result = judge.grade(
            _q(
                category="knowledge_update",
                gold_stale="4 study sessions",
                gold_answer="5 study sessions",
            ),
            "5 study sessions",
            "4 study sessions, updated to 5 study sessions.",
        )
        assert result.judge == "llm"
        assert result.stale is False
        assert result.stale_checkable is True
        assert result.stale_annotated is True

    def test_model_stale_from_note(self, monkeypatch) -> None:
        # 只有 gold_stale_note 时，模型仍可判 stale，且计入可机检分母。
        judge = _stub_llm(
            monkeypatch, '{"correct": false, "stale": true, "reason": "答了旧值"}'
        )
        result = judge.grade(
            _q(
                category="knowledge_update",
                gold_stale_note="previously 4 study sessions, but now 5 study sessions",
                gold_answer="5 study sessions",
            ),
            "5 study sessions",
            "You did 4 study sessions.",
        )
        assert result.stale is True
        assert result.stale_checkable is True
        assert result.stale_annotated is True

    def test_missing_stale_falls_back(self, monkeypatch) -> None:
        judge = _stub_llm(monkeypatch, '{"correct": true, "reason": "ok"}')
        q = _q(
            category="knowledge_update",
            gold_stale="4 study sessions",
            gold_answer="5 study sessions",
        )
        result = judge.grade(q, "5 study sessions", "You did 4 study sessions.")
        assert result.judge == "llm"
        assert result.stale is True  # 启发式按裸旧值判定
        assert result.stale_checkable is True

        note_only = judge.grade(
            _q(
                category="knowledge_update",
                gold_stale_note="previously 4, now 5",
                gold_answer="5 study sessions",
            ),
            "5 study sessions",
            "You did 4 study sessions.",
        )
        assert note_only.stale is False
        assert note_only.stale_checkable is False
        assert note_only.stale_annotated is True

    def test_bad_json_falls_back(self, monkeypatch) -> None:
        judge = _stub_llm(monkeypatch, "没有任何 JSON 的回答")
        result = judge.grade(
            _q(category="knowledge_update", gold_stale="北京"),
            "杭州",
            "之前在北京。",
        )
        assert result.judge == "llm-fallback"
        assert result.stale is True  # 回退启发式：只给旧值、新值未出现
        assert result.stale_checkable is True

        note_only = judge.grade(
            _q(category="knowledge_update", gold_stale_note="原来是北京"),
            "杭州",
            "北京。",
        )
        assert note_only.judge == "llm-fallback"
        assert note_only.stale_checkable is False
        assert note_only.stale_annotated is True
