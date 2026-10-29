"""TranscriptAdapter：离线回放按问题精确匹配，匹配前剥掉提问日期后缀。"""

from __future__ import annotations

import json
from pathlib import Path

from membench.adapters.transcript import TranscriptAdapter, strip_question_date

#: LongMemEval 转换器追加在问题文本行尾的提问日期后缀（见 convert_longmemeval）。
_SUFFIX = "\n（提问日期：2023-04-10T23:07）"


def _adapter(tmp_path: Path, records: list[dict]) -> TranscriptAdapter:
    path = tmp_path / "transcript.jsonl"
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return TranscriptAdapter(path)


def test_query_matches_after_stripping_question_date(tmp_path: Path) -> None:
    """带日期后缀的查询能匹配到按原始问题录制的回放记录。"""
    adapter = _adapter(
        tmp_path,
        [{"question": "What is my dog's name?", "answer": "Rex", "retrieved": ["Rex"]}],
    )
    result = adapter.query(f"What is my dog's name?{_SUFFIX}")
    assert result.answer == "Rex"
    assert result.retrieved == ["Rex"]


def test_query_matches_record_keyed_with_suffix(tmp_path: Path) -> None:
    """转录文件按带后缀的原文录制时，带后缀的查询仍能匹配（回退原问题查找）。"""
    adapter = _adapter(
        tmp_path,
        [
            {
                "question": f"What is my dog's name?{_SUFFIX}",
                "answer": "Rex",
                "retrieved": ["Rex"],
            }
        ],
    )
    assert adapter.query(f"What is my dog's name?{_SUFFIX}").answer == "Rex"


def test_query_without_suffix_unchanged(tmp_path: Path) -> None:
    """没有后缀的问题照旧精确匹配。"""
    adapter = _adapter(
        tmp_path, [{"question": "What is my dog's name?", "answer": "Rex"}]
    )
    assert adapter.query("What is my dog's name?").answer == "Rex"


def test_unmatched_query_returns_empty(tmp_path: Path) -> None:
    """未命中的问题返回空答案与空检索列表（行为不变）。"""
    adapter = _adapter(tmp_path, [{"question": "What is my dog's name?", "answer": "Rex"}])
    result = adapter.query("Some other question")
    assert result.answer == ""
    assert result.retrieved == []


def test_suffix_stripped_only_at_line_end() -> None:
    """只剥行尾的整段后缀；正文中出现的同形文字不受影响。"""
    assert strip_question_date(f"What is my dog's name?{_SUFFIX}") == (
        "What is my dog's name?"
    )
    inline = "He asked （提问日期：2023-04-10T23:07） at the meeting."
    assert strip_question_date(inline) == inline
    # 后缀后面还有内容 → 不是行尾后缀，不剥
    assert strip_question_date(f"Q{_SUFFIX} extra") == f"Q{_SUFFIX} extra"
