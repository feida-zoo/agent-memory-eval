"""记忆系统适配器基类与数据传输对象。

定义被测记忆系统接入 membench 的统一契约（SPEC 3.1）：
任何记忆系统只需实现 ``reset`` / ``ingest`` / ``query`` 三个方法即可接入评测。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class IngestStats:
    """一次 ingest 的资源消耗统计。

    Attributes:
        input_tokens: 写入内容的估算 token 数。
        llm_calls: ingest 过程中触发的 LLM 调用次数（无则填 0）。
        latency_ms: ingest 耗时（毫秒）。
    """

    input_tokens: int = 0
    llm_calls: int = 0
    latency_ms: float = 0.0


@dataclass
class QueryResult:
    """一次 query 的结果与资源消耗。

    Attributes:
        answer: 系统给出的回答文本。
        retrieved: 实际拼进回答 prompt 的记忆条目原文（用于 Recall@k / 证据
            命中判定）。口径 = **注入即口径**：报"注入的那一份全量"，按注入
            优先级顺序排列（``recall@5`` 只看前 5 条 = 最高优先级的 5 条）；
            多次检索先合并去重再返回（框架不合并、不去重、不截断），重复条目
            会白占前 5 窗口；不要报未经注入裁剪的内部原始 top-N，也不要把
            摘要/ID/答案当条目。
        latency_ms: 查询耗时（毫秒）。
        input_tokens: 注入 prompt 的记忆 token 数（估算即可）。
        output_tokens: 生成回答的 token 数（估算即可）。
        llm_calls: 查询过程中触发的 LLM 调用次数（无则填 0）。
    """

    answer: str
    retrieved: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0


class BaseMemoryAdapter(ABC):
    """记忆系统适配器抽象基类（SPEC 3.1）。

    适配器实现者负责把自家系统的 token / 调用数填入统计字段；
    填不了的字段填 0（报告中标注 n/a）。
    """

    name: str = "base"

    @abstractmethod
    def reset(self) -> None:
        """清空全部记忆状态。每个 dialogue 开始前调用。

        框架可能在调用前通过环境变量提供 reset 上下文（**可选提示**，可忽略）：
        ``MEMBENCH_RUN_DIR`` / ``MEMBENCH_DIALOGUE_ID`` / ``MEMBENCH_DIALOGUE_INDEX``。
        需要"清空前快照"这类回溯能力时（如调试 stale/误改记忆）自行把快照落盘；
        框架不读也不写适配器的记忆存储。
        """
        raise NotImplementedError

    @abstractmethod
    def ingest(self, messages: list[dict]) -> IngestStats:
        """写入一段对话。

        Args:
            messages: 消息列表，形如
                ``[{"role": "user"|"assistant", "content": str, "timestamp": str|None}]``。

        Returns:
            IngestStats: 写入过程的资源消耗。
        """
        raise NotImplementedError

    @abstractmethod
    def query(self, question: str) -> QueryResult:
        """基于已写入的记忆回答问题。

        Args:
            question: 自然语言问题。

        Returns:
            QueryResult: 回答文本、命中的记忆条目与资源消耗。
        """
        raise NotImplementedError


def estimate_tokens(text: str) -> int:
    """按字符数 / 2 粗估 token 数（中英文混合文本的经验值）。"""
    if not text:
        return 0
    return max(1, len(text) // 2)
