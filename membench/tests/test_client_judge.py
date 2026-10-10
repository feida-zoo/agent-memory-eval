"""ClientLLMJudge 测试：复用 agent 内置 LLM 客户端的判分器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from membench.judges.client import ClientLLMJudge


@dataclass
class _Q:
    """模拟 Question 数据类（judge 只读 category/gold_stale/question）。"""

    category: str = "single_hop"
    gold_stale: Optional[str] = None
    question: str = "用户的狗叫什么名字？"


class _EchoClient:
    """模拟 agent 内置 LLM 客户端：确定性回显，回答含 gold 则判对。"""

    def chat(self, prompt: str) -> str:
        # prompt 中【标准答案】【系统回答】在 system 段也出现过，须从尾部取（rsplit）
        gold = prompt.rsplit("【标准答案】", 1)[1].split("\n")[0].strip()
        pred = prompt.rsplit("【系统回答】", 1)[1].split("\n")[0].strip()
        ok = gold in pred
        return '{"correct": %s, "reason": "echo"}' % ("true" if ok else "false")


class _BrokenClient:
    def chat(self, prompt: str) -> str:
        raise ConnectionError("模型服务不可用")


def _fn_client(prompt: str) -> str:
    return '{"correct": true, "reason": "fn"}'


class TestClientLLMJudge:
    def test_object_with_chat_method(self) -> None:
        judge = ClientLLMJudge(_EchoClient())
        r = judge.grade(_Q(), "旺财", "我的狗叫旺财")
        assert r.correct is True and r.judge == "client-llm"
        r2 = judge.grade(_Q(), "旺财", "我的狗叫来福")
        assert r2.correct is False

    def test_callable_client(self) -> None:
        judge = ClientLLMJudge(_fn_client)
        r = judge.grade(_Q(), "任意", "任意回答")
        assert r.correct is True

    def test_class_path_import(self) -> None:
        # 用模块内类路径（本文件自身）验证延迟导入
        judge = ClientLLMJudge("tests.test_client_judge:_EchoClient")
        r = judge.grade(_Q(), "9月28日", "截止日是9月28日")
        assert r.correct is True

    def test_class_path_without_colon_raises(self) -> None:
        with pytest.raises(ValueError, match="module:ClassName"):
            ClientLLMJudge("no_colon_path")

    def test_missing_client_raises(self) -> None:
        with pytest.raises(ValueError):
            ClientLLMJudge(None)

    def test_fallback_on_client_failure(self) -> None:
        judge = ClientLLMJudge(_BrokenClient())
        r = judge.grade(_Q(), "旺财", "我的狗叫旺财")
        # 回退启发式：gold 覆盖判定为正确，但 judge 字段带 -fallback 标记
        assert r.judge == "client-llm-fallback"
        assert "ConnectionError" in r.reason
        assert r.correct is True  # 启发式：gold 是 prediction 子串

    def test_stale_detection_rules_based(self) -> None:
        judge = ClientLLMJudge(_EchoClient())
        q = _Q(category="knowledge_update", gold_stale="9月20日")
        r = judge.grade(q, "9月28日", "截止日是9月28日（之前是9月20日）")
        assert r.stale is True  # 命中过期答案，无论 LLM 判分结果

    def test_abstention_rules_based(self) -> None:
        judge = ClientLLMJudge(_EchoClient())
        q = _Q(category="abstention")
        r = judge.grade(q, "<ABSTAIN>", "我不知道，对话中没有提到。")
        assert r.correct is True and r.abstained_pred is True
        r2 = judge.grade(q, "<ABSTAIN>", "狗狗今年3岁。")
        assert r2.correct is False and r2.abstained_pred is False

    def test_malformed_llm_output_falls_back(self) -> None:
        class _GarbageClient:
            def chat(self, prompt: str) -> str:
                return "这不是JSON"

        judge = ClientLLMJudge(_GarbageClient())
        r = judge.grade(_Q(), "旺财", "我的狗叫旺财")
        assert r.judge == "client-llm-fallback"
