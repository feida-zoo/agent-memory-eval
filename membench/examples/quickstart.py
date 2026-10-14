"""最小接入示例：继承 BaseMemoryAdapter 包装一个假想记忆系统，并跑一次评测。

运行方式（在 `membench/` 工程根目录下）：
    cd membench && python examples/quickstart.py

把 ``FakeMemorySystem`` 换成你自己的记忆系统 SDK，只改 MyAdapter 里的三行调用即可。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # 便于直接运行

from membench.adapters.base import BaseMemoryAdapter, IngestStats, QueryResult
from membench.runner import Runner


class FakeMemorySystem:
    """假想记忆系统：朴素地把消息存进列表，查询时做子串匹配。"""

    def __init__(self) -> None:
        self.docs: list[str] = []

    def clear(self) -> None:
        self.docs.clear()

    def add(self, messages: list[dict]) -> None:
        self.docs.extend(m["content"] for m in messages)

    def ask(self, question: str) -> tuple[str, list[str]]:
        hits = [d for d in self.docs if any(w and w in d for w in question.split())][:3]
        return ("根据我的记忆：" + "；".join(hits) if hits else "我不知道", hits)


class MyAdapter(BaseMemoryAdapter):
    """把 FakeMemorySystem 包装成 membench 适配器 —— 核心就这三方法。"""

    name = "my-system"

    def __init__(self) -> None:
        self.sys = FakeMemorySystem()

    def reset(self) -> None:
        self.sys.clear()

    def ingest(self, messages: list[dict]) -> IngestStats:
        self.sys.add(messages)
        return IngestStats(input_tokens=0, llm_calls=0, latency_ms=0.0)

    def query(self, question: str) -> QueryResult:
        answer, docs = self.sys.ask(question)
        return QueryResult(
            answer=answer, retrieved=docs,
            latency_ms=0.0, input_tokens=0, output_tokens=0, llm_calls=0,
        )


if __name__ == "__main__":
    dataset = os.path.join(
        os.path.dirname(__file__), "..", "membench", "data", "sample_dataset.jsonl"
    )
    run_dir = Runner(MyAdapter(), dataset, judge="heuristic", out_dir="runs").run()
    print(f"评测完成，报告见: {run_dir}/report.md")
