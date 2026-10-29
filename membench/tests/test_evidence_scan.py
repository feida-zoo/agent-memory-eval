"""``evidence_in_state`` 的状态扫描：长证据多 needle、目录缓存与读取上限。

覆盖：
- 长证据按句拆 needle：事实在中段（前 200 字符没有）也能命中；
- 多 needle 任一命中即算写入；纯标点 / 过短句跳过，无可用句时退回前 200 字符；
- 同一状态目录只读一次（进程内 LRU），目录被改写（大小 / mtime 变化）即失效；
- 超过单文件 / 单目录读取上限记 ``unknown``（None）且只告警一次。
"""

from __future__ import annotations

from pathlib import Path

import pytest

import membench.runner as runner_mod
from membench.runner import (
    EVIDENCE_NEEDLE_CHARS,
    evidence_needles,
    scan_state_for_evidence,
)

#: 只在长证据中段出现的事实（≥20 字符，够格当单句 needle）。
FACT = "项目预算在第二季度调整为四十二万元整这是最终结论"


def _write_state(tmp_path: Path, text: str, name: str = "state") -> Path:
    """建一个只有一个文本文件的状态目录。"""
    state = tmp_path / name
    state.mkdir()
    (state / "memory.txt").write_text(text, encoding="utf-8")
    return state


def _normalized(text: str) -> str:
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# needle 生成与匹配
# ---------------------------------------------------------------------------


def test_long_evidence_fact_in_middle_is_found(tmp_path: Path) -> None:
    """长证据的事实只在中段：旧口径（前 200 字符）判 False，多 needle 判 True。"""
    evidence = "开场白。" * 60 + FACT + "。" + "结尾补充。" * 40
    assert len(_normalized(evidence)) > EVIDENCE_NEEDLE_CHARS
    state = _write_state(
        tmp_path, "无关内容。\n" + FACT + "。\n更多无关内容。"
    )

    old_needle = _normalized(evidence)[:EVIDENCE_NEEDLE_CHARS]
    assert FACT not in old_needle
    assert old_needle not in (state / "memory.txt").read_text(encoding="utf-8")

    needles = evidence_needles(evidence)
    assert FACT in needles
    assert old_needle in needles  # 前 200 字符仍是兜底 needle
    assert scan_state_for_evidence(str(state), [evidence]) is True


def test_any_needle_hit_counts(tmp_path: Path) -> None:
    """多句证据：状态里只有其中一句也算已写入；都不在则 False。"""
    sentences = [
        "第一段说明" + "甲" * 40,
        "第二段说明" + "乙" * 50,
        "第三段说明" + "丙" * 35,
        "第四段说明" + "丁" * 45,
        "第五段说明" + "戊" * 38,
    ]
    evidence = "。".join(sentences) + "。"
    assert len(_normalized(evidence)) > EVIDENCE_NEEDLE_CHARS

    only_fourth = _write_state(tmp_path, sentences[3] + "。", name="only-fourth")
    assert scan_state_for_evidence(str(only_fourth), [evidence]) is True

    absent = _write_state(tmp_path, "完全无关的记忆内容。", name="absent")
    assert scan_state_for_evidence(str(absent), [evidence]) is False


def test_needles_skip_short_and_punctuation_sentences() -> None:
    """短句与纯标点句不当 needle；长证据仍保留前 200 字符兜底。"""
    evidence = (
        ("好的。" + "！！！？？。" + "嗯。") * 30
        + "这一段是有意义的完整事实句子需要被保留下来作为候选片段"
    )
    needles = evidence_needles(evidence)
    assert "好的" not in needles
    assert "嗯" not in needles
    assert all(any(char.isalnum() for char in needle) for needle in needles)
    assert _normalized(evidence)[:EVIDENCE_NEEDLE_CHARS] in needles


def test_needles_fallback_when_no_long_sentence() -> None:
    """长文本全是短句（每句 <20 字符）：退回前 200 字符这一条。"""
    evidence = "短句。" * 120
    needles = evidence_needles(evidence)
    assert needles == [_normalized(evidence)[:EVIDENCE_NEEDLE_CHARS]]


def test_short_evidence_uses_whole_text() -> None:
    """≤200 字符的证据整段作为 needle（行为不变）。"""
    assert evidence_needles("alpha fact") == ["alpha fact"]
    assert evidence_needles("") == []


# ---------------------------------------------------------------------------
# 状态目录缓存
# ---------------------------------------------------------------------------


def test_state_dir_scanned_once_per_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同一状态目录被多题引用时只读一次；目录被改写后缓存失效、重新扫。"""
    state = _write_state(tmp_path, "m1 alpha fact")
    calls: list[str] = []
    real_reader = runner_mod._read_state_texts

    def counted(state_dir: str):
        calls.append(state_dir)
        return real_reader(state_dir)

    monkeypatch.setattr(runner_mod, "_read_state_texts", counted)

    assert scan_state_for_evidence(str(state), ["alpha fact"]) is True
    assert scan_state_for_evidence(str(state), ["alpha fact"]) is True
    assert scan_state_for_evidence(str(state), ["missing"]) is False
    assert len(calls) == 1

    # 目录被改写（大小变化 → 指纹变化）：重新扫描并看到新内容
    (state / "memory.txt").write_text(
        "m1 alpha fact\nm2 beta fact（追加内容让文件大小变化）", encoding="utf-8"
    )
    assert scan_state_for_evidence(str(state), ["beta fact"]) is True
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# 读取上限
# ---------------------------------------------------------------------------


def test_oversize_total_is_unknown_and_warns_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """单目录合计超限：两次扫描都记 unknown，stderr 只告警一次。"""
    monkeypatch.setattr(runner_mod, "STATE_SCAN_MAX_TOTAL_BYTES", 10)
    monkeypatch.setattr(runner_mod, "_STATE_SCAN_OVERSIZE_WARNED", False)
    state = _write_state(tmp_path, "x" * 100)

    assert scan_state_for_evidence(str(state), ["x"]) is None
    assert scan_state_for_evidence(str(state), ["x"]) is None
    assert capsys.readouterr().err.count("超过扫描上限") == 1


def test_oversize_single_file_is_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """单文件超限同样记 unknown（不能只读一部分就声称"没写入"）。"""
    monkeypatch.setattr(runner_mod, "STATE_SCAN_MAX_FILE_BYTES", 10)
    monkeypatch.setattr(runner_mod, "STATE_SCAN_MAX_TOTAL_BYTES", 1 << 20)
    monkeypatch.setattr(runner_mod, "_STATE_SCAN_OVERSIZE_WARNED", False)
    state = _write_state(tmp_path, "x" * 100, name="single-file")

    assert scan_state_for_evidence(str(state), ["x"]) is None
