"""检索证据命中判定：可插拔的匹配口径（substring / overlap / llm）。

``evidence_recall``（任一条证据命中）与 ``recall@5``（前 5 条覆盖率）共用
本模块的判定结果，口径由 :func:`build_evidence_matcher` 构造：

- ``substring``（默认，历史口径）：证据字符串原样是某条 retrieved 的子串。
  最严格、零歧义；要求适配器返回接近原文的检索片段。
- ``overlap``：证据与某条 retrieved 的内容 token（中文 bigram + 英文/数字
  词）包含度 >= 阈值（默认 0.6）即命中。对"原文被标点、包裹、轻微改写"
  的存储容错；对深度改写的摘要仍判 miss——它只放宽词面，不做语义。
- ``llm``：复用 OpenAI 兼容接口逐题做语义判定（一次调用覆盖该题的全部
  证据 × 检索条目），覆盖摘要/改写式记忆存储；调用或解析失败回退
  substring 并把次数记入 ``fallbacks``（runner 落账到 metrics.json）。

**session 级长证据的 containment 兜底**：substring / overlap 共用
:func:`containment_hit`——证据长度超过 :data:`SESSION_EVIDENCE_MIN_CHARS`
（视为整段 session 级证据）且子串 / token 判定都失败后，只要检索里存在一条
去空白后长度 >= :data:`CONTAINMENT_MIN_ITEM_CHARS` 的条目，且该条目是证据的
子串（或证据是该条目的子串），就算命中。这是**代理口径**：chunk 级记忆返回的
是会话里的一段，天然不可能是整段证据的子串，严格子串判定会让 recall@5 与
证据召回系统性趋零（LongMemEval S/M 的 session 证据中位数约 14k 字符）。
短证据（office / LoCoMo 的裸值）行为不变，仍要求原文子串命中。

三种口径的分数不可直接对比（llm >= overlap >= substring）；compare 的
可比性校验对 ``retrieval_match`` 不一致给出 WARN。
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.request
from typing import Any, Sequence

__all__ = [
    "SubstringMatcher",
    "OverlapMatcher",
    "LLMEvidenceMatcher",
    "build_evidence_matcher",
    "content_tokens",
    "containment_hit",
]

_ZH_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_RE = re.compile(r"[a-z0-9]+")
_WS_RE = re.compile(r"\s+")

#: 视为 session 级证据的最小字符数：超过它在子串判定失败后追加 containment 兜底。
SESSION_EVIDENCE_MIN_CHARS = 500

#: containment 兜底时检索条目的最小长度（去空白后）：过短的条目（裸值、单词）
#: 不作容器，避免 "9%" 之类短串把任意长证据判成命中。
CONTAINMENT_MIN_ITEM_CHARS = 20


def _squash(text: str) -> str:
    """去掉全部空白字符（换行、空格、缩进），供 containment 比较。"""
    return _WS_RE.sub("", text or "")


def containment_hit(evidence: str, retrieved: Sequence[str]) -> bool:
    """长证据的 containment 兜底：检索条目是否为证据的一部分（或反之）。

    只应由 :class:`SubstringMatcher` / :class:`OverlapMatcher` 在证据长度超过
    :data:`SESSION_EVIDENCE_MIN_CHARS` 时调用。比较前去空白，短于
    :data:`CONTAINMENT_MIN_ITEM_CHARS` 的条目不作容器。

    Args:
        evidence: 单条证据（session 级长文本）。
        retrieved: 检索条目列表。

    Returns:
        是否存在一条足够长、且与证据互为子串的检索条目。
    """
    ev = _squash(evidence)
    if not ev:
        return False
    for item in retrieved:
        squashed = _squash(item)
        if len(squashed) < CONTAINMENT_MIN_ITEM_CHARS:
            continue
        if squashed in ev or ev in squashed:
            return True
    return False


def content_tokens(text: str) -> set[str]:
    """内容 token 集合：中文按字符 bigram、英文/数字按小写词。"""
    tokens: set[str] = set()
    for run in _ZH_RUN_RE.findall(text or ""):
        if len(run) == 1:
            tokens.add(run)
        else:
            tokens.update(run[i : i + 2] for i in range(len(run) - 1))
    tokens.update(m.group(0) for m in _ASCII_RE.finditer((text or "").lower()))
    return tokens


class SubstringMatcher:
    """逐字子串命中（默认口径；长证据追加 containment 兜底）。"""

    name = "substring"

    def __init__(self) -> None:
        self.fallbacks = 0

    @staticmethod
    def _matches(evidence: str, retrieved: Sequence[str]) -> bool:
        """单条证据是否命中：原文子串；超长证据再看 containment。"""
        if any(evidence in item for item in retrieved):
            return True
        if len(evidence) > SESSION_EVIDENCE_MIN_CHARS:
            return containment_hit(evidence, retrieved)
        return False

    def evidence_hits(
        self, evidence: list[str], retrieved: list[str], k: int = 5
    ) -> tuple[bool, float]:
        """返回 ``(任一证据命中, 前 k 条覆盖率)``；空 evidence 返回 (False, 0.0)。"""
        if not evidence:
            return False, 0.0
        top = retrieved[:k]
        hit = any(self._matches(ev, retrieved) for ev in evidence)
        recall = sum(1 for ev in evidence if self._matches(ev, top)) / len(evidence)
        return hit, float(recall)


class OverlapMatcher:
    """token 包含度命中：对原文被标点/包裹/轻微改写的存储容错。"""

    name = "overlap"

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = float(threshold)
        self.fallbacks = 0

    def _evidence_hit(self, evidence: str, retrieved: Sequence[str]) -> bool:
        ev_tokens = content_tokens(evidence)
        if not ev_tokens:
            # 纯符号等无 token 的证据：退化为子串判定
            return any(evidence in item for item in retrieved)
        for item in retrieved:
            item_tokens = content_tokens(item)
            if not item_tokens:
                continue
            if len(ev_tokens & item_tokens) / len(ev_tokens) >= self.threshold:
                return True
        if len(evidence) > SESSION_EVIDENCE_MIN_CHARS:
            return containment_hit(evidence, retrieved)
        return False

    def evidence_hits(
        self, evidence: list[str], retrieved: list[str], k: int = 5
    ) -> tuple[bool, float]:
        """返回 ``(任一证据命中, 前 k 条覆盖率)``；空 evidence 返回 (False, 0.0)。"""
        if not evidence:
            return False, 0.0
        top = retrieved[:k]
        hit = any(self._evidence_hit(ev, retrieved) for ev in evidence)
        recall = sum(1 for ev in evidence if self._evidence_hit(ev, top)) / len(evidence)
        return hit, recall


class LLMEvidenceMatcher:
    """语义证据核对（OpenAI 兼容接口）；任何失败回退 substring。

    ``_chat`` 可被测试替换；回退次数累计在 ``fallbacks``。
    """

    name = "llm"

    _SYSTEM_PROMPT = (
        "你是检索证据核对器。给定【证据】与【检索条目】两个编号列表，判断每条"
        "证据是否被某条（或多条）检索条目支持：允许措辞不同、改写、摘要，只要"
        "语义上包含或支撑该证据即算支持。\n"
        "只输出 JSON，不要输出任何其他内容："
        '{"support": [[支持第 1 条证据的检索条目编号…], [支持第 2 条证据的…], …]}'
        "（编号从 1 开始；没有支持则为空列表 []；support 长度必须等于证据条数）"
    )

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 90.0,
        fallback: Any = None,
    ) -> None:
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
        self.timeout = float(timeout)
        self._fallback = fallback or SubstringMatcher()
        # 回退计数需线程安全：--jobs>1 时多题并发调用本匹配器
        self._fallbacks_lock = threading.Lock()
        self.fallbacks = 0

    def _chat(self, evidence: list[str], retrieved: list[str]) -> list[list[int]]:
        """调用 chat completions，返回每条证据的支撑条目编号（1 起始）。"""
        if not self.base_url:
            raise ValueError("未配置 evidence-match=llm 的 base_url")
        user = (
            "【证据】\n"
            + "\n".join(f"{i}. {ev}" for i, ev in enumerate(evidence, start=1))
            + "\n【检索条目】\n"
            + "\n".join(f"{j}. {doc}" for j, doc in enumerate(retrieved, start=1))
            + "\n请输出 JSON 核对结果。"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": user},
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
        # 复用 judge 的三级 JSON 解析（延迟导入，避免 metrics 与 judges 包循环）
        from membench.judges.llm import _parse_judge_output

        parsed = _parse_judge_output(content)
        support_raw = parsed.get("support") or []
        support: list[list[int]] = []
        for idx in range(len(evidence)):
            entries = support_raw[idx] if idx < len(support_raw) else []
            if not isinstance(entries, list):
                entries = []
            support.append([int(x) for x in entries if isinstance(x, (int, float))])
        return support

    def evidence_hits(
        self, evidence: list[str], retrieved: list[str], k: int = 5
    ) -> tuple[bool, float]:
        """返回 ``(任一证据命中, 前 k 条覆盖率)``；失败回退 substring 并计数。"""
        if not evidence:
            return False, 0.0
        try:
            support = self._chat(evidence, retrieved)
        except Exception:  # noqa: BLE001 - 任何失败都必须回退，不中断评测
            with self._fallbacks_lock:
                self.fallbacks += 1
            return self._fallback.evidence_hits(evidence, retrieved, k)
        top = len(retrieved[:k])
        hit = any(entries for entries in support)
        recall = sum(
            1 for entries in support if any(1 <= idx <= top for idx in entries)
        ) / len(evidence)
        return hit, recall


def build_evidence_matcher(
    name: str,
    *,
    threshold: float = 0.6,
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    llm_model: str | None = None,
    llm_timeout: float = 90.0,
) -> Any:
    """按名字构造匹配器；``llm`` 需要可用的 OpenAI 兼容配置。"""
    if name == "substring":
        return SubstringMatcher()
    if name == "overlap":
        return OverlapMatcher(threshold=threshold)
    if name == "llm":
        matcher = LLMEvidenceMatcher(
            base_url=llm_base_url,
            api_key=llm_api_key,
            model=llm_model,
            timeout=llm_timeout,
        )
        if not matcher.base_url:
            raise ValueError(
                "--evidence-match llm 需要 --llm-base-url 或 "
                "MEMBENCH_JUDGE_BASE_URL / OPENAI_BASE_URL"
            )
        return matcher
    raise ValueError(
        f"未知 evidence-match 口径: {name!r}（可选 substring / overlap / llm）"
    )
