"""内置 Mock 适配器：MockWeakAdapter / MockStrongAdapter（SPEC 第 4 节）。

两个适配器都不调用外部 API、输出确定，用于演示指标差异与回归测试：

- MockWeakAdapter：朴素基线。全部消息原样存入列表；查询时做大小写不敏感的
  词重叠检索取 top3，answer 直接拼接检索到的原文前 200 字符。
  预期缺陷：knowledge_update 答旧值、abstention 乱答、multi_hop 差。
- MockStrongAdapter：模拟优化后。ingest 按句切分并抽取 (subject, value) 事实，
  同主题新事实覆盖旧事实并记录时间戳；查询检索 top5，最高重叠得分 < 0.15
  时拒答；temporal 类问题优先取 timestamp 最新的事实。

分词策略（jieba 不可用时的轻量替代）：中文按字符 bigram、英文/数字按单词切分。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from .base import BaseMemoryAdapter, IngestStats, QueryResult, estimate_tokens

# ---------------------------------------------------------------------------
# 分词与重叠度评分
# ---------------------------------------------------------------------------

_EN_WORD_RE = re.compile(r"[a-z0-9]+")
_ZH_RUN_RE = re.compile(r"[一-鿿]+")

#: 疑问词等高频低信息量 bigram，打分前剔除，避免“用户/什么”之类虚词干扰检索。
_STOP_TOKENS = frozenset(
    {
        "用户",
        "什么",
        "怎么",
        "怎样",
        "多少",
        "哪个",
        "哪些",
        "哪里",
        "哪一",
        "是不是",
        "是否",
        "请问",
        "一下",
        "告诉",
        "吗",
        "呢",
        "the",
        "is",
        "a",
        "an",
        "of",
    }
)


def tokenize(text: str) -> list[str]:
    """轻量分词：英文/数字按单词（小写），中文按字符 bigram。

    单个汉字构成的片段保留为单字 token。结果不去重，调用方按需转 set。
    """
    tokens: list[str] = []
    for word in _EN_WORD_RE.findall(text.lower()):
        tokens.append(word)
    for run in _ZH_RUN_RE.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def content_tokens(text: str) -> set[str]:
    """分词后去停用词并去重，用于重叠度打分。"""
    return {t for t in tokenize(text) if t not in _STOP_TOKENS}


def overlap_score(question_tokens: set[str], doc_tokens: set[str]) -> float:
    """问题对文档的 token 包含度：|交集| / |问题 token 集|，取值 [0, 1]。"""
    if not question_tokens:
        return 0.0
    return len(question_tokens & doc_tokens) / len(question_tokens)


# ---------------------------------------------------------------------------
# MockWeakAdapter：朴素基线
# ---------------------------------------------------------------------------


class MockWeakAdapter(BaseMemoryAdapter):
    """朴素记忆基线：全量消息列表 + 词重叠 top3 检索 + 原文拼接作答。

    不做任何事实更新与拒答判断，因此 knowledge_update 题容易命中旧值、
    abstention 题会强行作答（SPEC 第 4 节的预期缺陷）。
    """

    name = "mock-weak"

    def __init__(self, top_k: int = 3, max_answer_chars: int = 200) -> None:
        self.top_k = top_k
        self.max_answer_chars = max_answer_chars
        self._messages: list[dict] = []

    def reset(self) -> None:
        """清空消息列表。"""
        self._messages = []

    def ingest(self, messages: list[dict]) -> IngestStats:
        """把全部消息原样追加进列表（不做任何加工）。"""
        t0 = time.perf_counter()
        self._messages.extend(messages)
        text = "".join(m.get("content", "") for m in messages)
        return IngestStats(
            input_tokens=estimate_tokens(text),
            llm_calls=0,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def query(self, question: str) -> QueryResult:
        """词重叠检索 top3，answer 拼接检索原文前 ``max_answer_chars`` 字符。

        得分相同时偏好**较早**的消息（朴素实现不懂时序，因此更容易命中旧事实）。
        """
        t0 = time.perf_counter()
        q_tokens = content_tokens(question)
        scored = [
            (overlap_score(q_tokens, content_tokens(m.get("content", ""))), idx, m)
            for idx, m in enumerate(self._messages)
        ]
        # 得分降序；同分按写入顺序升序（不区分新旧，旧消息优先）。
        scored.sort(key=lambda item: (-item[0], item[1]))
        hits = scored[: self.top_k]
        retrieved = [m.get("content", "") for _, _, m in hits]
        answer = "".join(retrieved)[: self.max_answer_chars]
        return QueryResult(
            answer=answer,
            retrieved=retrieved,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            input_tokens=estimate_tokens("".join(retrieved)),
            output_tokens=estimate_tokens(answer),
            llm_calls=0,
        )


# ---------------------------------------------------------------------------
# MockStrongAdapter：模拟优化后
# ---------------------------------------------------------------------------

#: 事实主题槽位：按顺序匹配，句子命中某槽位的任一关键词即归入该槽位。
#: 同槽位的新事实覆盖旧事实（模拟 UPDATE 操作）。
_SLOT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pet_name", ("狗叫", "狗狗叫", "改名", "改成", "的名字是")),
    ("pet_breed", ("金毛", "柯基", "泰迪", "品种")),
    ("city", ("住在", "搬到", "搬家", "定居")),
    ("employer", ("公司", "入职")),
    ("diet_spicy", ("吃辣", "戒辣", "爱吃辣")),
    ("exercise", ("健身房", "锻炼", "游泳", "瑜伽")),
    ("music_class", ("吉他", "报了个", "培训班", "改学")),
)

#: temporal 意图词：命中后同分时优先取时间戳最新的事实。
_TEMPORAL_MARKERS: tuple[str, ...] = ("现在", "最新", "目前", "最近", "如今", "当前")

#: 句子切分符（保留逗号级别的从句完整，避免切碎语义）。
_SENT_SPLIT_RE = re.compile(r"[。！？!?；;]+")

#: 拒答话术（与启发式 judge 的拒答信号词对齐）。
ABSTAIN_ANSWER = "我不知道，相关信息不在我的记忆中。"


@dataclass
class _MemoryItem:
    """一条记忆单元：原文 + 写入序号（充当 timestamp，单调递增）。"""

    text: str
    seq: int
    slot: str | None = None


class MockStrongAdapter(BaseMemoryAdapter):
    """模拟优化后的记忆系统：事实抽取 + 同主题覆盖 + 低置信拒答。

    - ingest：按句切分，正则/关键词识别“叫/是/在/搬/换”等句式的事实主题，
      同主题新事实**覆盖**旧事实并记录 timestamp（写入序号）。
    - query：词重叠检索 top5；若最高重叠得分 < ``threshold``（默认 0.15），
      回答“我不知道，相关信息不在我的记忆中。”；temporal 类问题在同分时
      优先取 timestamp 最新的事实。
    """

    name = "mock-strong"

    def __init__(
        self,
        top_k: int = 5,
        threshold: float = 0.15,
        max_answer_items: int = 3,
    ) -> None:
        self.top_k = top_k
        self.threshold = threshold
        self.max_answer_items = max_answer_items
        self._facts: dict[str, _MemoryItem] = {}
        self._plain: list[_MemoryItem] = []
        self._seq = 0

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """清空事实表与普通记忆。"""
        self._facts = {}
        self._plain = []
        self._seq = 0

    @staticmethod
    def _detect_slot(sentence: str) -> str | None:
        """按关键词识别句子所属的事实主题槽位（未命中返回 None）。"""
        for slot, keywords in _SLOT_KEYWORDS:
            if any(kw in sentence for kw in keywords):
                return slot
        return None

    def ingest(self, messages: list[dict]) -> IngestStats:
        """按句切分并抽取事实；同槽位新事实覆盖旧事实并更新 timestamp。"""
        t0 = time.perf_counter()
        for msg in messages:
            content = msg.get("content", "")
            for sentence in _SENT_SPLIT_RE.split(content):
                sentence = sentence.strip(" ，,、\t")
                if not sentence:
                    continue
                item = _MemoryItem(text=sentence, seq=self._seq)
                self._seq += 1
                slot = self._detect_slot(sentence)
                if slot is None:
                    self._plain.append(item)
                else:
                    # 同主题覆盖：新事实替换旧事实（模拟 UPDATE）。
                    item.slot = slot
                    self._facts[slot] = item
        text = "".join(m.get("content", "") for m in messages)
        return IngestStats(
            input_tokens=estimate_tokens(text),
            llm_calls=0,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    @staticmethod
    def _has_temporal_intent(question: str) -> bool:
        """问题是否包含“现在/最新/最近”等时间性意图词。"""
        return any(marker in question for marker in _TEMPORAL_MARKERS)

    def _rank(self, question: str) -> list[tuple[float, _MemoryItem]]:
        """对全部记忆单元打分并排序。

        排序键：得分降序，同分时按 timestamp（写入序号）降序——
        temporal 类问题因此优先拿到最新事实；非 temporal 问题同分取最新
        也无害（覆盖机制已保证同槽位只剩最新事实）。
        """
        q_tokens = content_tokens(question)
        items = list(self._facts.values()) + self._plain
        scored = [
            (overlap_score(q_tokens, content_tokens(item.text)), item)
            for item in items
        ]
        scored.sort(key=lambda pair: (-pair[0], -pair[1].seq))
        return scored

    def query(self, question: str) -> QueryResult:
        """检索 top5 并作答；最高得分低于阈值则拒答。"""
        t0 = time.perf_counter()
        ranked = self._rank(question)
        top = ranked[: self.top_k]
        retrieved = [item.text for _, item in top]
        best = top[0][0] if top else 0.0

        if best < self.threshold:
            # 低置信：模拟拒答（abstention）。
            answer = ABSTAIN_ANSWER
        else:
            # 只用真正有重叠（得分 > 0）的条目拼答案，避免噪声污染。
            picked = [item.text for score, item in top if score > 0.0]
            picked = picked[: self.max_answer_items]
            answer = "根据我的记忆：" + "；".join(picked) + "。"

        return QueryResult(
            answer=answer,
            retrieved=retrieved,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            input_tokens=estimate_tokens("".join(retrieved)),
            output_tokens=estimate_tokens(answer),
            llm_calls=0,
        )
