"""LLM 判分器模板：复用 Agent 自己的 LLM 客户端做判分。

与 `llm.py`（OpenAI 兼容协议）不同，本模块面向"agent 自带 LLM SDK"的场景——
被测 Agent 项目里往往已经封装好了模型客户端（含密钥/代理/重试），判分器应直接
复用它，而不是再配一套 OpenAI 兼容端点。

用法（三种）::

    # 1) 对象包装：任何带 .chat(str)->str / .complete(str)->str 方法的对象
    from my_agent.llm import LLMClient
    judge = ClientLLMJudge(LLMClient())

    # 2) 可调用：judge = ClientLLMJudge(my_llm_fn)

    # 3) 类路径延迟导入（不需要在判分器文件里 import 被测项目）：
    judge = ClientLLMJudge("my_agent.llm:LLMClient")

CLI::

    python -m membench.cli run ... --judge client --client-llm "my_agent.llm:LLMClient"

注意（方法论）：judge 与被测 agent 同源时存在自评偏差，报告应注明；
stale/abstention 两个红线指标在本实现中始终由规则判定，不受 LLM 影响。
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Union

from membench.llm_ref import import_from_path

from .base import Judge, JudgeResult
from .heuristic import HeuristicJudge, contains_refusal

#: 判分提示词（与 llm.py 的 _SYSTEM_PROMPT + _USER_TEMPLATE 同口径）。
_SYSTEM_PROMPT = (
    "你是一个严格的记忆系统评测裁判。给定【问题】【标准答案】【系统回答】，"
    "判断系统回答是否正确。\n"
    "判对标准：系统回答包含了标准答案的关键信息，且不与标准答案矛盾；"
    "允许措辞不同，不苛求完全一致。\n"
    "若标准答案为 <ABSTAIN>，表示对话中没有答案，此时系统回答明确拒绝作答"
    "（如表示不知道/没有相关信息）才算正确。\n"
    "只输出 JSON，不要输出任何其他内容："
    '{"correct": true 或 false, "reason": "一句话理由"}'
)

_USER_TEMPLATE = (
    "【问题】{question}\n"
    "【标准答案】{gold}\n"
    "【系统回答】{prediction}\n"
    "请输出 JSON 判分结果。"
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class ClientLLMJudge(Judge):
    """用任意本地 LLM 客户端/函数判分，LLM 不可用时回退启发式。

    Args:
        client: 三选一——
            - 对象：需带 ``chat(prompt)`` 或 ``complete(prompt)`` 方法；
            - 可调用对象：签名 ``fn(prompt: str) -> str``；
            - 字符串 ``"module:ClassName"`` / ``"module:function"``，延迟导入
              （类会被无参实例化）。
    """

    name = "client-llm"

    def __init__(self, client: Union[Any, Callable[[str], str], str, None] = None) -> None:
        if client is None:
            raise ValueError(
                "ClientLLMJudge 需要一个 client：LLM 客户端对象、可调用函数，"
                "或 'module:ClassName' 类路径字符串。"
            )
        if isinstance(client, str):
            client = self._import_from_path(client)
        self._client = client
        self._fallback = HeuristicJudge()

    # ------------------------------------------------------------------
    @staticmethod
    def _import_from_path(spec: str) -> Any:
        """委托 :func:`membench.llm_ref.import_from_path`（保留历史调用点）。"""
        return import_from_path(spec)

    def _call(self, prompt: str) -> str:
        c = self._client
        if hasattr(c, "chat"):
            return str(c.chat(prompt))
        if hasattr(c, "complete"):
            return str(c.complete(prompt))
        if callable(c):
            return str(c(prompt))
        raise TypeError(
            "client 需带 .chat(str)/.complete(str) 方法，或直接是可调用对象"
        )

    # ------------------------------------------------------------------
    def grade(self, question: Any, gold: str, prediction: str) -> JudgeResult:
        category = str(getattr(question, "category", "") or "")
        gold_stale = str(getattr(question, "gold_stale", "") or "")

        # 红线指标一律规则判定，不交给（可能与被测系统同源的）LLM
        if category == "abstention":
            abstained = contains_refusal(prediction or "")
            return JudgeResult(
                correct=abstained,
                abstained_pred=abstained,
                stale=False,
                score=1.0 if abstained else 0.0,
                judge=self.name,
                reason="abstention 题按规则判定（拒答信号词匹配）",
            )

        stale = bool(gold_stale) and gold_stale in (prediction or "")

        try:
            raw = self._call(
                _SYSTEM_PROMPT
                + "\n\n"
                + _USER_TEMPLATE.format(
                    question=getattr(question, "question", ""),
                    gold=gold,
                    prediction=prediction,
                )
            )
            parsed = self._parse(raw)
            return JudgeResult(
                correct=bool(parsed["correct"]),
                abstained_pred=False,
                stale=stale,
                score=1.0 if parsed["correct"] else 0.0,
                judge=self.name,
                reason=str(parsed.get("reason", ""))[:500],
            )
        except Exception as exc:  # noqa: BLE001 — 判分失败绝不能中断评测
            fb = self._fallback.grade(question, gold, prediction)
            fb.judge = f"{self.name}-fallback"
            fb.reason = f"client LLM 调用失败({type(exc).__name__})，回退启发式；" + fb.reason
            return fb

    @staticmethod
    def _parse(raw: str) -> dict:
        """从模型输出中提取 {"correct": ..., "reason": ...}，容忍代码围栏与多余文本。"""
        m = _JSON_RE.search(raw.strip())
        if not m:
            raise ValueError(f"判分输出中未找到 JSON: {raw[:120]!r}")
        data = json.loads(m.group(0))
        if "correct" not in data:
            raise ValueError("判分输出缺少 'correct' 字段")
        return data
