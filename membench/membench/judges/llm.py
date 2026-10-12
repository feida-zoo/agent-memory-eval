"""LLMJudge：基于 OpenAI 兼容 chat completions 接口的 LLM-as-Judge。

调用 ``POST {base_url}/v1/chat/completions``（urllib 实现，超时 30s），
要求模型只输出 JSON：``{"correct": true/false, "reason": "..."}``。

任何异常（网络、超时、解析失败等）都会回退到 HeuristicJudge，
并把结果标记为 ``judge="llm-fallback"``，保证评测流程不中断。

配置来源（构造函数参数优先，其次环境变量）：
- base_url: ``MEMBENCH_JUDGE_BASE_URL`` 或 ``OPENAI_BASE_URL``
- api_key:  ``MEMBENCH_JUDGE_API_KEY`` 或 ``OPENAI_API_KEY``
- model:    ``MEMBENCH_JUDGE_MODEL``（默认 ``gpt-4o-mini``）
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from membench.data.schema import ABSTAIN_SENTINEL

from .base import Judge, JudgeResult
from .heuristic import HeuristicJudge, contains_refusal, is_stale

#: 从模型输出中提取 JSON 对象（允许被包裹在 ```json 代码块或多余文本中）。
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_output(content: str) -> dict:
    """三级解析模型输出为 JSON dict。

    1. 先尝试 ``json.loads`` 整个输出（模型严格只输出 JSON 时直接成功，
       避免贪婪正则把前后多余花括号文本一并吞入）；
    2. 失败再用 ``_JSON_RE`` 抠第一个 JSON 对象尝试解析；
    3. 再失败抛异常，交由调用方走 llm-fallback 回退路径。
    """
    text = (content or "").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, TypeError):
        pass
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError(f"模型输出不含 JSON：{content!r}")
    return json.loads(match.group(0))

#: 判分 prompt：要求模型只输出 JSON。
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


class LLMJudge(Judge):
    """LLM-as-Judge：OpenAI 兼容接口判分，失败回退启发式。"""

    name = "llm"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 90.0,
    ) -> None:
        """初始化 LLM judge。

        Args:
            base_url: OpenAI 兼容服务根地址（不含 ``/v1/chat/completions``）。
            api_key: API Key。
            model: 判分模型名。
            timeout: 请求超时时间（秒，默认 90，适配推理模型）。
        """
        self.base_url = (
            base_url
            or os.environ.get("MEMBENCH_JUDGE_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or ""
        ).rstrip("/")
        self.api_key = (
            api_key
            or os.environ.get("MEMBENCH_JUDGE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        self.model = model or os.environ.get("MEMBENCH_JUDGE_MODEL") or "gpt-4o-mini"
        self.timeout = timeout
        self._fallback = HeuristicJudge()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _chat(self, question_text: str, gold: str, prediction: str) -> tuple[bool, str]:
        """调用 chat completions 并解析 ``{"correct": ..., "reason": ...}``。

        Raises:
            ValueError: 未配置 base_url。
            urllib.error.URLError / TimeoutError / json.JSONDecodeError 等:
                网络或解析失败，交由调用方回退。
        """
        if not self.base_url:
            raise ValueError("未配置 LLM judge 的 base_url")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _USER_TEMPLATE.format(
                        question=question_text, gold=gold, prediction=prediction
                    ),
                },
            ],
            "temperature": 0,
        }
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        parsed = _parse_judge_output(content)
        return bool(parsed["correct"]), str(parsed.get("reason", ""))

    # ------------------------------------------------------------------
    # Judge 接口
    # ------------------------------------------------------------------

    def grade(self, question: "Question", gold: str, prediction: str) -> JudgeResult:  # noqa: F821
        """LLM 判分；任何异常回退 HeuristicJudge 并标记 ``llm-fallback``。"""
        question_text = getattr(question, "question", question)
        gold_stale = getattr(question, "gold_stale", None)

        # 拒答 / 陈旧检测保持确定性（与启发式一致），不依赖模型发挥。
        abstained = contains_refusal(prediction or "")
        stale = is_stale(gold_stale, gold, prediction or "")

        try:
            correct, reason = self._chat(str(question_text), gold, prediction or "")
            return JudgeResult(
                correct=correct,
                abstained_pred=abstained,
                stale=stale,
                score=1.0 if correct else 0.0,
                judge=self.name,
                reason=reason,
            )
        except Exception as exc:  # noqa: BLE001 - 任何异常都必须回退
            fb = self._fallback.grade(question, gold, prediction)
            return JudgeResult(
                correct=fb.correct,
                abstained_pred=fb.abstained_pred,
                stale=fb.stale,
                score=fb.score,
                judge="llm-fallback",
                reason=f"LLM 调用失败（{type(exc).__name__}: {exc}），回退启发式：{fb.reason}",
            )


#: 哨兵值 re-export（便于调用方从 judges 包取用）。
_ = ABSTAIN_SENTINEL
