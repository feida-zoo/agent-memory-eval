"""LongMemEval 转换测试：session 级 evidence、日期排序、question_date、证据不截断。"""

from __future__ import annotations

import json
from pathlib import Path

from membench.data.convert_longmemeval import convert_longmemeval
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


def _first_contents(out: str) -> list[str]:
    """第一条 dialogue 各 session 的首条消息内容（用于验证排序）。"""
    dialogues = load_dataset(out)
    return [session[0]["content"] for session in dialogues[0].sessions]


def test_s_variant_fills_evidence(tmp_path: Path) -> None:
    """S 变体：evidence 非空且含 answer_session_ids 指向的 session 内容。"""
    src = _write(tmp_path, "lme_s.json", _LME_S_SAMPLE)
    out = str(tmp_path / "lme_s.jsonl")
    stats = convert_longmemeval(src, out)

    assert stats["questions"] == 2
    assert stats["evidence_filled"] == 2
    assert stats["evidence_missing"] == 0
    assert stats["n_unsorted"] == 0

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
    # 没有 question_date 时不追加日期行
    assert "提问日期" not in dialogues[0].questions[0].question


def test_sessions_sorted_by_date_and_ids_aligned(tmp_path: Path) -> None:
    """乱序 haystack 按日期升序输出；session_id 跟着 session 走，evidence 仍对。"""
    inst = dict(_LME_S_SAMPLE[0])
    inst["haystack_sessions"] = [
        [{"role": "user", "content": "C-third"}],
        [{"role": "user", "content": "A-first"}],
        [{"role": "user", "content": "B-second"}],
    ]
    inst["haystack_dates"] = [
        "2024/02/01 (Thu) 09:00",
        "2024/01/01 (Mon) 09:00",
        "2024/01/15 (Mon) 09:00",
    ]
    inst["haystack_session_ids"] = ["s_late", "s_early", "s_mid"]
    inst["answer_session_ids"] = ["s_mid"]

    src = _write(tmp_path, "lme_unsorted.json", [inst])
    out = str(tmp_path / "out.jsonl")
    stats = convert_longmemeval(src, out)

    assert stats["n_unsorted"] == 1
    dialogues = load_dataset(out)
    assert _first_contents(out) == ["A-first", "B-second", "C-third"]
    # timestamp 跟着原 session 的日期走，而不是输入位置
    assert [s[0]["timestamp"] for s in dialogues[0].sessions] == [
        "2024-01-01T09:00",
        "2024-01-15T09:00",
        "2024-02-01T09:00",
    ]
    # evidence 按 answer_session_id 解析，排序后仍指向原 session 内容
    assert dialogues[0].questions[0].evidence == ["B-second"]


def test_same_date_keeps_input_order(tmp_path: Path) -> None:
    """同日期稳定排序：保持输入原序，且不计入 n_unsorted。"""
    inst = dict(_LME_S_SAMPLE[0])
    inst["haystack_sessions"] = [
        [{"role": "user", "content": "first"}],
        [{"role": "user", "content": "second"}],
    ]
    inst["haystack_dates"] = ["2024-01-01", "2024-01-01"]
    inst["haystack_session_ids"] = ["s0", "s1"]
    inst["answer_session_ids"] = ["s1"]

    src = _write(tmp_path, "lme_same_day.json", [inst])
    out = str(tmp_path / "out.jsonl")
    stats = convert_longmemeval(src, out)

    assert stats["n_unsorted"] == 0
    assert _first_contents(out) == ["first", "second"]


def test_question_date_saved_and_appended(tmp_path: Path) -> None:
    """question_date 归一为 ISO、落库，并追加到问题文本末尾。"""
    inst = dict(_LME_S_SAMPLE[0])
    inst["question_date"] = "2023/04/10 (Mon) 23:07"
    src = _write(tmp_path, "lme_qdate.json", [inst])
    out = str(tmp_path / "out.jsonl")
    stats = convert_longmemeval(src, out)

    assert stats["question_date_filled"] == 1
    q = load_dataset(out)[0].questions[0]
    assert q.question_date == "2023-04-10T23:07"
    assert q.question == "What is my dog's name?\n（提问日期：2023-04-10T23:07）"


def test_question_date_unparseable_kept(tmp_path: Path) -> None:
    """无法解析的 question_date 保留原串（辅助信息，不阻断转换）。"""
    inst = dict(_LME_S_SAMPLE[0])
    inst["question_date"] = "last spring"
    src = _write(tmp_path, "lme_qdate_bad.json", [inst])
    out = str(tmp_path / "out.jsonl")
    convert_longmemeval(src, out)

    q = load_dataset(out)[0].questions[0]
    assert q.question_date == "last spring"
    assert q.question.endswith("（提问日期：last spring）")


def test_evidence_not_truncated(tmp_path: Path) -> None:
    """超长 session 证据原样保留（不再截断到 2000 字符）。"""
    long_text = "x" * 5000
    inst = dict(_LME_S_SAMPLE[0])
    inst["haystack_sessions"] = [[{"role": "user", "content": long_text}]]
    inst["haystack_dates"] = ["2024-01-01"]
    inst["haystack_session_ids"] = ["s0"]
    inst["answer_session_ids"] = ["s0"]
    src = _write(tmp_path, "lme_long.json", [inst])
    out = str(tmp_path / "out.jsonl")
    stats = convert_longmemeval(src, out)

    assert stats["evidence_filled"] == 1
    evidence = load_dataset(out)[0].questions[0].evidence
    assert evidence == [long_text]
    assert len(evidence[0]) == 5000
