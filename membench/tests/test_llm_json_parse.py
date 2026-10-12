"""LLM judge JSON 输出三级解析测试（_parse_judge_output）。"""

from __future__ import annotations

import pytest

from membench.judges.llm import _parse_judge_output


def test_parse_pure_json() -> None:
    """纯 JSON 输出：整体 json.loads 直接成功。"""
    parsed = _parse_judge_output('{"correct": true, "reason": "命中关键信息"}')
    assert parsed == {"correct": True, "reason": "命中关键信息"}


def test_parse_json_with_thinking_text() -> None:
    """前后带思考文字的 JSON：回退到正则抠第一个 JSON 对象。"""
    content = (
        "让我想想……系统回答包含了标准答案。\n"
        '{"correct": true, "reason": "覆盖关键信息"}\n'
        "以上是我的判分。"
    )
    parsed = _parse_judge_output(content)
    assert parsed["correct"] is True
    assert parsed["reason"] == "覆盖关键信息"


def test_parse_json_in_code_fence() -> None:
    content = '```json\n{"correct": false, "reason": "答非所问"}\n```'
    parsed = _parse_judge_output(content)
    assert parsed["correct"] is False


def test_parse_no_json_raises() -> None:
    """完全无 JSON 的输出抛 ValueError（由调用方走 llm-fallback 回退）。"""
    with pytest.raises(ValueError, match="不含 JSON"):
        _parse_judge_output("系统回答完全错误，没有任何结构。")


def test_parse_empty_raises() -> None:
    with pytest.raises(ValueError):
        _parse_judge_output("")


def test_parse_invalid_json_object_raises() -> None:
    """有花括号但不是合法 JSON：两级均失败，抛异常走回退。"""
    with pytest.raises(ValueError):
        _parse_judge_output("{correct: 这是非法 JSON}")
