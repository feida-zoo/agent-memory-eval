"""公开基准转换器测试：LoCoMo 与 LongMemEval。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from membench.data.convert_locomo import convert_locomo
from membench.data.convert_longmemeval import convert_longmemeval
from membench.data.schema import load_dataset

# ---------------------------------------------------------------------------
# 微型 fixture
# ---------------------------------------------------------------------------

_LOCOMO_SAMPLE = [
    {
        "sample_id": "conv-test",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1_date_time": "1:56 pm on 8 May, 2023",  # LoCoMo 原始形态（12 小时制）
            "session_1": [
                {"speaker": "Alice", "dia_id": "D1:1", "text": "I adopted a dog named Rex."},
                {"speaker": "Bob", "dia_id": "D1:2", "text": "Nice! How old is he?"},
            ],
            "session_2_date_time": "10:00 am on 1 June, 2023",
            "session_2": [
                {"speaker": "Alice", "dia_id": "D2:1", "text": "Rex turned three yesterday."},
            ],
        },
        "qa": [
            {"question": "What is the dog's name?", "answer": "Rex", "evidence": ["D1:1"], "category": 4},
            {"question": "When did Rex turn three?", "answer": "2023", "evidence": ["D2:1"], "category": 2},
            {"question": "What is Alice's favorite color?", "evidence": [], "category": 5,
             "adversarial_answer": "blue"},
        ],
    }
]

_LME_SAMPLE = [
    {
        "question_id": "lme_q1",
        "question": "What city did I move to?",
        "answer": "Berlin",
        "question_type": "knowledge-update",
        "haystack_sessions": [
            [{"role": "user", "content": "I moved to Berlin last week."},
             {"role": "assistant", "content": "Congrats!"}],
            [{"role": "user", "content": "I work remotely now."},
             {"role": "assistant", "content": "Nice."}],
        ],
        "haystack_dates": ["2023/04/10 (Mon) 17:50", "2024/02/01 (Thu) 09:05"],
        "answer_session_ids": ["s1"],
    },
    {
        "question_id": "lme_q2",
        "question": "What is my favorite genre?",
        "answer": "jazz",
        "question_type": "single-session-preference",
        "haystack_sessions": [
            [{"role": "user", "content": "I love jazz."},
             {"role": "assistant", "content": "Noted."}]
        ],
        "haystack_dates": ["2024/01/03 (Wed) 08:30"],
        "answer_session_ids": ["s1"],
    },
]


def _write(tmp_path: Path, name: str, obj: object) -> str:
    p = tmp_path / name
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# LoCoMo
# ---------------------------------------------------------------------------

def test_locomo_convert_and_load(tmp_path: Path) -> None:
    src = _write(tmp_path, "locomo.json", _LOCOMO_SAMPLE)
    out = str(tmp_path / "locomo.jsonl")
    stats = convert_locomo(src, out)
    assert stats["dialogues"] == 1 and stats["questions"] == 3

    dialogues = load_dataset(out)  # 必须通过 schema 校验
    d = dialogues[0]
    assert d.dialogue_id == "conv-test"
    assert len(d.sessions) == 2
    # 角色映射：speaker_a -> user
    assert d.sessions[0][0]["role"] == "user"
    assert d.sessions[0][1]["role"] == "assistant"
    assert d.sessions[0][0]["timestamp"] == "2023-05-08T13:56"  # 归一为 ISO，精度仍是到分

    by_cat = {q.category: q for q in d.questions}
    assert by_cat["single_hop"].gold_answer == "Rex"
    assert by_cat["single_hop"].evidence == ["I adopted a dog named Rex."]
    assert by_cat["temporal"].gold_answer == "2023"
    # 对抗题 -> 拒答哨兵
    assert by_cat["abstention"].gold_answer == "<ABSTAIN>"


def test_locomo_max_dialogues(tmp_path: Path) -> None:
    src = _write(tmp_path, "locomo.json", _LOCOMO_SAMPLE * 3)
    out = str(tmp_path / "locomo.jsonl")
    stats = convert_locomo(src, out, max_dialogues=2)
    assert stats["dialogues"] == 2


# ---------------------------------------------------------------------------
# LongMemEval
# ---------------------------------------------------------------------------

def test_longmemeval_convert_and_load(tmp_path: Path) -> None:
    src = _write(tmp_path, "lme.json", _LME_SAMPLE)
    out = str(tmp_path / "lme.jsonl")
    stats = convert_longmemeval(src, out)
    assert stats["dialogues"] == 2 and stats["questions"] == 2
    assert stats["categories"] == {"knowledge_update": 1, "preference": 1}

    dialogues = load_dataset(out)
    d0 = dialogues[0]
    assert len(d0.sessions) == 2
    assert d0.sessions[0][1]["role"] == "assistant"
    assert d0.sessions[1][0]["timestamp"] == "2024-02-01T09:05"
    q = d0.questions[0]
    assert q.qid == "lme_q1" and q.gold_answer == "Berlin"
    assert q.category == "knowledge_update"
    assert q.evidence == []  # LongMemEval 无 per-turn 证据


def test_longmemeval_unknown_type_raises(tmp_path: Path) -> None:
    bad = dict(_LME_SAMPLE[0])
    bad["question_type"] = "not-a-type"
    src = _write(tmp_path, "lme_bad.json", [bad])
    with pytest.raises(ValueError, match="not-a-type"):
        convert_longmemeval(src, str(tmp_path / "out.jsonl"))


def test_longmemeval_limit(tmp_path: Path) -> None:
    src = _write(tmp_path, "lme.json", _LME_SAMPLE)
    out = str(tmp_path / "lme.jsonl")
    stats = convert_longmemeval(src, out, limit=1)
    assert stats["questions"] == 1
