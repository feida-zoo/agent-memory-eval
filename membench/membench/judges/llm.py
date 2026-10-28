"""LLMJudge：基于 OpenAI 兼容 chat completions 接口的 LLM-as-Judge。

调用 ``POST {base_url}/v1/chat/completions``（urllib 实现，超时默认 90s），
要求模型在同一调用里同时判正确性与陈旧性，只输出 JSON：
``{"correct": true/false, "stale": true/false, "reason": "..."}``。
stale 的语义是"回答把**旧值**当成当前事实"：题目给出 ``gold_stale``（裸旧值）
或 ``gold_stale_note``（原始描述）时，prompt 一并传给模型；模型没给
``stale`` 或输出解析失败时回退启发式 :func:`is_stale`。回退路径的
``gold_stale`` 先经 :func:`membench.data.stale.resolve_stale` 解析——旧缓存
里仍是整句描述时会先抽裸旧值，抽不出则按不可机检处理（不进 stale_rate 分母）。

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
from membench.data.stale import resolve_stale

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


def _parse_stale_field(parsed: dict) -> "bool | None":
    """取模型给出的 ``stale``；缺字段或不是 bool 时返回 ``None``（交给回退）。"""
    value = parsed.get("stale")
    # bool 是 int 的子类：这里只接受真正的 JSON true/false。
    return value if isinstance(value, bool) else None


#: 判分 prompt：要求模型只输出 JSON（correct + stale 同一次判完）。
_SYSTEM_PROMPT = (
    "你是一个严格的记忆系统评测裁判。给定【问题】【标准答案】【系统回答】，"
    "判断系统回答是否正确。\n"
    "判对标准：系统回答包含了标准答案的关键信息，且不与标准答案矛盾；"
    "允许措辞不同，不苛求完全一致。\n"
    "若标准答案为 <ABSTAIN>，表示对话中没有答案，此时系统回答明确拒绝作答"
    "（如表示不知道/没有相关信息）才算正确。\n"
    "【过期旧值】/【过期描述】给出这道题的旧事实（若有）。此时还要判断 "
    "stale：stale = 回答把旧值当成**当前事实**（给出了旧值、却没有把标准答案"
    "作为更新后的事实明确给出）；若旧值只是作为变更上下文被提及、且新值已"
    "正确给出，stale = false。没有给出旧值时 stale = false。\n"
    "只输出 JSON，不要输出任何其他内容："
    '{"correct": true 或 false, "stale": true 或 false, "reason": "一句话理由"}'
)


def _stale_block(gold_stale: "str | None", gold_stale_note: "str | None") -> str:
    """prompt 里的过期事实段：优先给裸旧值，描述句作为补充；都没有时明示。"""
    lines: list[str] = []
    if gold_stale:
        lines.append(f"【过期旧值】{gold_stale}")
    if gold_stale_note:
        lines.append(f"【过期描述】{gold_stale_note}")
    if not lines:
        lines.append("【过期旧值】（无，stale 输出 false）")
    return "\n".join(lines) + "\n"


def _user_prompt(
    question_text: str,
    gold: str,
    prediction: str,
    gold_stale: "str | None" = None,
    gold_stale_note: "str | None" = None,
) -> str:
    """拼出 user 段（system prompt 另发）。"""
    return (
        f"【问题】{question_text}\n"
        f"【标准答案】{gold}\n"
        f"【系统回答】{prediction}\n"
        f"{_stale_block(gold_stale, gold_stale_note)}"
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

    def _chat(
        self,
        question_text: str,
        gold: str,
        prediction: str,
        gold_stale: "str | None" = None,
        gold_stale_note: "str | None" = None,
    ) -> "tuple[bool, bool | None, str]":
        """调用 chat completions，解析 correct / stale / reason。

        Returns:
            ``(correct, stale, reason)``；模型没给合法 ``stale`` 时中间项为
            ``None``（由 :meth:`grade` 回退启发式）。

        Raises:
            ValueError: 未配置 base_url，或输出缺少 ``correct``。
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
                    "content": _user_prompt(
                        question_text, gold, prediction, gold_stale, gold_stale_note
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
        correct = parsed["correct"]
        if not isinstance(correct, bool):
            raise ValueError(f"模型输出的 correct 不是 bool：{correct!r}")
        return correct, _parse_stale_field(parsed), str(parsed.get("reason", ""))

    # ------------------------------------------------------------------
    # Judge 接口
    # ------------------------------------------------------------------

    def grade(self, question: "Question", gold: str, prediction: str) -> JudgeResult:  # noqa: F821
        """LLM 判分；任何异常回退 HeuristicJudge 并标记 ``llm-fallback``。"""
        question_text = getattr(question, "question", question)
        gold_stale = getattr(question, "gold_stale", None)
        gold_stale_note = getattr(question, "gold_stale_note", None)

        # 拒答检测保持确定性（与启发式一致）；stale 由模型同一次调用判出，
        # 没给或解析失败时回退启发式。
        abstained = contains_refusal(prediction or "")
        has_stale_annotation = bool(
            (gold_stale or "").strip() or (gold_stale_note or "").strip()
        )

        try:
            correct, model_stale, reason = self._chat(
                str(question_text), gold, prediction or "", gold_stale, gold_stale_note
            )
            if model_stale is not None and has_stale_annotation:
                stale = model_stale
                stale_checkable = True
            else:
                stale = is_stale(gold_stale, gold, prediction or "")
                # 回退规则时按 resolve_stale 的口径：旧缓存的整句描述会先
                # 抽裸旧值，抽不出即不可机检（不进入 stale_rate 分母）。
                _, stale_checkable = resolve_stale(gold_stale, gold)
            return JudgeResult(
                correct=correct,
                abstained_pred=abstained,
                stale=stale,
                score=1.0 if correct else 0.0,
                judge=self.name,
                reason=reason,
                stale_checkable=stale_checkable,
                stale_annotated=has_stale_annotation,
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
                stale_checkable=fb.stale_checkable,
                stale_annotated=fb.stale_annotated,
            )


#: 哨兵值 re-export（便于调用方从 judges 包取用）。
_ = ABSTAIN_SENTINEL
