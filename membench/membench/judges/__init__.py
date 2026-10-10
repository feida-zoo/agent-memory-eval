"""Judge 工厂与判分器导出（SPEC 第 2 节：``get_judge(name) -> Judge``）。"""

from __future__ import annotations

from .base import Judge, JudgeResult
from .heuristic import HeuristicJudge
from .llm import LLMJudge

_JUDGES = {
    "heuristic": HeuristicJudge,
    "llm": LLMJudge,
}


def get_judge(name: str, **kwargs: object) -> Judge:
    """按名称实例化判分器。

    Args:
        name: "heuristic" 或 "llm"。
        **kwargs: 透传给判分器构造函数（LLMJudge 支持
            ``base_url=`` / ``api_key=`` / ``model=``）。

    Raises:
        ValueError: 未知判分器名。
    """
    if name not in _JUDGES:
        raise ValueError(f"未知判分器 '{name}'，可用：{sorted(_JUDGES)}")
    return _JUDGES[name](**kwargs)  # type: ignore[arg-type]


__all__ = ["Judge", "JudgeResult", "HeuristicJudge", "LLMJudge", "get_judge"]
