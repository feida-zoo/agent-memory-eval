"""membench：Agent 记忆系统评测框架。

快速开始::

    from membench import BaseMemoryAdapter, load_dataset
    from membench.adapters import ADAPTERS

    dialogues = load_dataset("membench/data/sample_dataset.jsonl")
    adapter = ADAPTERS["mock-strong"]()
"""

from __future__ import annotations

from .adapters.base import BaseMemoryAdapter, IngestStats, QueryResult
from .agent_loop import AgentLoopAdapter, run_tool_loop
from .data.schema import Dataset, Dialogue, Question, load_dataset
from .runner import Runner

__version__ = "0.1.0"

__all__ = [
    "BaseMemoryAdapter",
    "IngestStats",
    "QueryResult",
    "AgentLoopAdapter",
    "run_tool_loop",
    "Dataset",
    "Dialogue",
    "Question",
    "Runner",
    "load_dataset",
    "__version__",
]
