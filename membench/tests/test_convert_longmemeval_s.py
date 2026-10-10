"""LongMemEval S 变体转换测试：session 级 evidence 解析与 oracle 向后兼容。"""

from __future__ import annotations

import json
from pathlib import Path

from membench.data.convert_longmemeval import EVIDENCE_MAX_CHARS, convert_longmemeval
from membench.data.schema import load_dataset

# ---------------------------------------------------------------------------
# 微型 S 变体 fixture：2 题，每题 3 个 session，证据指向 s1
# ---------------------------------------------------------------------------

_LME_S_SAMPLE = [
    {
        "question_id": "lme_s_q1",
        "question": "What is my dog's name?",
        "answer": "Rex",
        "question_type": "single-session-user",
        "question_date": "2024-03-01",
        "haystack_sessions": [
            [{"role": "user", "content": "I like hiking."},
             {"role": "assistant", "content": "Great hobby."}],
            [{"role": "user", "content": "My dog Rex turned two."},
             {"role": "assistant", "content": "Happy birthday Rex!"}],
            [{"role": "user", "content": "I work in Berlin."},
             {"role": "assistant", "content": "Nice city."}],
        ],
        "haystack_dates": ["2024-01-01", "2024-01-15", "2024-02-01"],
        "haystack_session_ids": ["s0", "s1", "s2"],
        "answer_session_ids": ["s1"],
    },
    {
        "question_id": "lme_s_q2",
        "question": "What genre do I prefer?",
        "answer": "jazz",
        "question_type": "single-session-preference",
        "question_date": "2024-03-02",
        "haystack_sessions": [
            [{"role": "user", "content": "I cook on weekends."}],
            [{"role": "user", "content": "I love jazz music."},
             {"role": "assistant", "content": "Smooth choice."}],
            [{"role": "user", "content": "I run every morning."}],
        ],
        "haystack_dates": ["2024-01-02", "2024-01-16", "2024-02-02"],
        "haystack_session_ids": ["s0", "s1", "s2"],
        "answer_session_ids": ["s1"],
    },
]


def _write(tmp_path: Path, name: str, obj: object) -> str:
    p = tmp_path / name
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return str(p)


def test_s_variant_fills_evidence(tmp_path: Path) -> None:
    """S 变体：evidence 非空且含 answer_session_ids 指向的 session 内容。"""
    src = _write(tmp_path, "lme_s.json", _LME_S_SAMPLE)
    out = str(tmp_path / "lme_s.jsonl")
    stats = convert_longmemeval(src, out)

    assert stats["questions"] == 2
    assert stats["evidence_filled"] == 2
    assert stats["evidence_missing"] == 0

    dialogues = load_dataset(out)  # 必须通过 schema 校验
    assert len(dialogues) == 2
    q1 = dialogues[0].questions[0]
    assert len(q1.evidence) == 1
    assert "My dog Rex turned two." in q1.evidence[0]
    assert "Happy birthday Rex!" in q1.evidence[0]
    # 不混入其他 session 内容
    assert "I like hiking." not in q1.evidence[0]

    q2 = dialogues[1].questions[0]
    assert "I love jazz music." in q2.evidence[0]


def test_s_variant_unmatched_session_id(tmp_path: Path) -> None:
    """answer_session_ids 指向不存在的 id：跳过并计入 evidence_missing。"""
    bad = dict(_LME_S_SAMPLE[0])
    bad["answer_session_ids"] = ["no_such_id"]
    src = _write(tmp_path, "lme_bad_sid.json", [bad])
    out = str(tmp_path / "out.jsonl")
    stats = convert_longmemeval(src, out)

    assert stats["evidence_filled"] == 0
    assert stats["evidence_missing"] == 1
    dialogues = load_dataset(out)
    assert dialogues[0].questions[0].evidence == []


def test_oracle_variant_backward_compatible(tmp_path: Path) -> None:
    """oracle 风格实例（无 haystack_session_ids）：evidence 为空、evidence_filled==0。"""
    oracle = {
        "question_id": "lme_o_q1",
        "question": "What city did I move to?",
        "answer": "Berlin",
        "question_type": "knowledge-update",
        "haystack_sessions": [
            [{"role": "user", "content": "I moved to Berlin last week."}]
        ],
        "haystack_dates": ["2024-01-01"],
        "answer_session_ids": ["s1"],
    }
    src = _write(tmp_path, "lme_oracle.json", [oracle])
    out = str(tmp_path / "lme_oracle.jsonl")
    stats = convert_longmemeval(src, out)

    assert stats["evidence_filled"] == 0
    assert stats["evidence_missing"] == 0
    dialogues = load_dataset(out)
    assert dialogues[0].questions[0].evidence == []


def test_s_variant_evidence_truncation(tmp_path: Path) -> None:
    """超长证据 session 截断到 EVIDENCE_MAX_CHARS。"""
    inst = dict(_LME_S_SAMPLE[0])
    inst["haystack_sessions"] = [
        [{"role": "user", "content": "x" * (EVIDENCE_MAX_CHARS + 500)}]
    ]
    inst["haystack_session_ids"] = ["s0"]
    inst["answer_session_ids"] = ["s0"]
    src = _write(tmp_path, "lme_long.json", [inst])
    out = str(tmp_path / "out.jsonl")
    stats = convert_longmemeval(src, out)

    assert stats["evidence_filled"] == 1
    dialogues = load_dataset(out)
    assert len(dialogues[0].questions[0].evidence[0]) == EVIDENCE_MAX_CHARS
