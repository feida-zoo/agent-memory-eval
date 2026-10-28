"""记忆系统适配器基类与数据传输对象。

定义被测记忆系统接入 membench 的统一契约（SPEC 3.1）：

- **必需**：``name`` 属性 + ``reset()`` / ``ingest()`` / ``query()``——实现这三个
  就完成了接入；
- **可选·快照组**：``snapshot_capabilities()`` / ``snapshot()`` / ``restore()`` /
  ``rebuild()``——用于 ``--replay`` 免重跑 ingest，**声明了就必须实现**；
- **可选·自描述**：``describe()``——返回配置 / 版本等自描述信息，写进
  ``metrics["adapter_describe"]``，不参与评分与可比性判定。
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

    必需接口是三个抽象方法（``reset`` / ``ingest`` / ``query``）；可选接口分
    快照组（``snapshot_capabilities`` / ``snapshot`` / ``restore`` / ``rebuild``，
    声明了就必须实现）与自描述（``describe``，不参与评分）。

    适配器实现者负责把自家系统的 token / 调用数填入统计字段；
    填不了的字段填 0（报告中标注 n/a）。
    """

    name: str = "base"

    @abstractmethod
    def reset(self) -> None:
        """清空全部记忆状态。每个 dialogue 开始前调用。

        框架可能在调用前通过环境变量提供 reset 上下文（**可选提示**，可忽略）：
        ``MEMBENCH_RUN_DIR`` / ``MEMBENCH_DIALOGUE_ID`` / ``MEMBENCH_DIALOGUE_INDEX``，
        供日志、追踪或自定义回溯使用；框架不读也不写适配器的记忆存储，
        也不内置 reset 前备份（要回看状态请用快照导出）。
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

    # ------------------------------------------------------------------
    # 可选能力：自描述（默认空；不参与评分与可比性）
    # ------------------------------------------------------------------

    def describe(self) -> dict:
        """自描述配置 / 版本（**可选**，默认空 dict）。

        返回值原样写进 ``metrics["adapter_describe"]``，并在报告「适配器自描述」
        小节展示：用途是让人事后看懂这次跑分背后被测系统长什么样，**不参与评分，
        也不是 compare 的可比性维度**。返回 ``{}`` 等价于"没什么可说的"，报告里
        该小节整节不出现；返回非 dict 或含不可 JSON 序列化的值会被框架忽略
        （记 ``None``）并警告。

        建议字段（按系统实际情况取舍，键名自由）：
        ``embedding_model``（嵌入模型）、``retrieval``（检索方式）、
        ``top_k``（注入条数）、``index_type``（索引类型）、``features``
        （开启的能力开关列表）。

        抛异常不会中断评测：框架打印一次警告，本次记 ``None``。
        """
        return {}

    # ------------------------------------------------------------------
    # 可选能力：状态快照与回放（默认不支持）
    # 见 membench/snapshot.py 与 references/adapter_guide.md「状态快照与回放」。
    # ------------------------------------------------------------------

    def snapshot_capabilities(self) -> dict:
        """声明快照能力（**可选**，默认不支持）。

        Returns:
            ``{"supported": False}``，或 ``{"supported": True, "restore": bool,
            "rebuild": bool, "format_version": Any, "compatible_from": [...]}``
            —— 声明 supported/restore/rebuild 就必须实现对应方法（缺了直接报错，
            不静默降级）。
        """
        return {"supported": False}

    def snapshot(self, target_dir: str, tag: str) -> dict:
        """把当前记忆状态导出到 ``target_dir``（tag 通常为对话 id）。

        由被测系统自己决定导出什么（原文 / 结构化记录 / 含派生结构的状态包）
        与格式版本；框架只给目录与 tag、算哈希、登记清单。

        Returns:
            可选的清单补充信息（如条目数、层级），会写进快照清单。
        """
        raise NotImplementedError

    def restore(self, source_dir: str, tag: str) -> None:
        """从快照恢复状态（同 ``format_version`` 前提下）。

        只在快照格式与当前代码兼容时使用：把快照里的状态直接放回去，不重建
        派生结构。
        """
        raise NotImplementedError

    def rebuild(self, source_dir: str, tag: str) -> None:
        """从快照重建（跨格式版本 / 重建派生结构），走系统自己的构建代码。

        与 :meth:`restore` 的分工：``restore`` 要求快照格式与当前代码兼容；
        ``rebuild`` 拿快照里的**源数据**，用当前代码的构建路径重建索引 / 向量 /
        摘要 / 缓存等派生结构——存储结构或 schema 变了也能回放，且不必重跑
        LLM 抽取。
        """
        raise NotImplementedError


def estimate_tokens(text: str) -> int:
    """按字符数 / 2 粗估 token 数（中英文混合文本的经验值）。"""
    if not text:
        return 0
    return max(1, len(text) // 2)
