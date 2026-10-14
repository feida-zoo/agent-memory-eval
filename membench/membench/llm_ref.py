"""``module:Attr`` 类路径延迟导入（判分器与 agent 循环共用同一约定）。

接入方代码与 membench 框架解耦：被测 Agent 的 LLM 客户端、工具 handler 等
都可以只给出 ``"my_agent.llm:LLMClient"`` 形式的类路径，由 membench 在运行时
导入，不需要 membench 反向依赖被测项目。
"""

from __future__ import annotations

import importlib
from typing import Any


def import_from_path(spec: str) -> Any:
    """按 ``"module:Attr"`` 延迟导入对象；``Attr`` 是类时无参实例化。

    Args:
        spec: 形如 ``"my_agent.llm:LLMClient"`` 的类路径。

    Returns:
        导入得到的对象（类会被无参实例化后返回）。

    Raises:
        ValueError: spec 不含 ``":"`` 分隔符。
        ImportError / AttributeError: 模块或属性不存在（原样抛出，便于定位）。
    """
    if ":" not in spec:
        raise ValueError(f"类路径格式应为 'module:ClassName'，收到: {spec!r}")
    module_name, attr = spec.split(":", 1)
    module = importlib.import_module(module_name)
    obj = getattr(module, attr)
    return obj() if isinstance(obj, type) else obj
