"""BEAM 转换器测试：题型映射、abstention 哨兵、gold_stale、schema 兼容。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from membench.data.convert_beam import (
    ABSTAIN_SENTINEL,
    convert_beam,
    convert_beam_split,
    extract_stale_value,
)
from membench.data.schema import load_dataset

# ---------------------------------------------------------------------------
# 微型 fixture
# ---------------------------------------------------------------------------

_CHAT = [
    {
        "batch_number": 1,
        "time_anchor": None,
        "turns": [
            [
                {"role": "user", "id": 0, "time_anchor": "March-01-2024", "index": "1,1",
                 "question_type": "main_question", "content": "帮我搭建 React 项目。"},
                {"role": "assistant", "id": 1, "content": "好的，建议如下……"},
            ],
            [
                {"role": "user", "id": 2, "index": "1,2",
                 "question_type": "followup_question", "content": "后端用哪个数据库？"},
                {"role": "assistant", "id": 3, "content": "PostgreSQL 14。"},
            ],
        ],
    },
    {
        "batch_number": 2,
        "time_anchor": None,
        "turns": [
            [
                {"role": "user", "id": 4, "time_anchor": "March-12-2024", "index": "2,1",
                 "question_type": "main_question", "content": "截止日期改到 3 月 18 日。"},
                {"role": "assistant", "id": 5, "content": "已更新。"},
            ],
        ],
    },
]

_PQ = {
    "information_extraction": [
        {"question": "我用的是哪个数据库？", "answer": "PostgreSQL 14",
         "difficulty": "easy", "conversation_references": ["Batch 1"]},
    ],
    "multi_session_reasoning": [
        {"question": "前后端版本分别是多少？", "answer": "React 18.2 + Node.js 18"},
    ],
    "event_ordering": [
        {"question": "讨论按什么顺序推进？", "answer": "先搭框架后换库"},
    ],
    "temporal_reasoning": [
        {"question": "两次讨论相隔几天？", "answer": "11 天"},
    ],
    "knowledge_update": [
        {"question": "模块截止日期是哪天？", "answer": "March 18, 2024",
         "potential_confusion": "LLM might incorrectly recall the original March 15, 2024 deadline"},
        {"question": "缓存 TTL 是多少？", "answer": "20 分钟", "potential_confusion": ""},
    ],
    "contradiction_resolution": [
        {"question": "我到底做没做过这个服务？", "ideal_answer": "你前后说法矛盾，请澄清。"},
    ],
    "instruction_following": [
        {"question": "API 失败怎么办？", "expected_compliance": "回答需包含兜底策略。"},
    ],
    "preference_following": [
        {"question": "推荐个翻译 API？", "expected_compliance": "应推荐低延迟方案。"},
    ],
    "summarization": [
        {"question": "总结一下整个项目。", "ideal_summary": "项目涵盖前端与后端……"},
    ],
    "abstention": [
        {"question": "我在哪里读的心理学？", "ideal_response": "对话中未提及。"},
    ],
    "unknown_future_type": [
        {"question": "未知题型应被跳过", "answer": "x"},
    ],
}


def _write_chat(root: Path, chat_id: str, pq_name: str = "probing_questions.json") -> Path:
    """在临时目录写一个迷你 BEAM chat。"""
    chat_dir = root / chat_id
    chat_dir.mkdir(parents=True)
    (chat_dir / "chat.json").write_text(json.dumps(_CHAT, ensure_ascii=False), encoding="utf-8")
    (chat_dir / pq_name).write_text(json.dumps(_PQ, ensure_ascii=False), encoding="utf-8")
    return chat_dir


# ---------------------------------------------------------------------------
# convert_beam：单个 chat
# ---------------------------------------------------------------------------


def test_convert_beam_dialogue_structure(tmp_path: Path) -> None:
    chat_dir = _write_chat(tmp_path, "1")
    dialogue = convert_beam(chat_dir, scale="1m", chat_id="1")

    assert dialogue["dialogue_id"] == "beam_1m_1"
    # 每个 batch 一个 session，嵌套 turns 被拍平
    assert len(dialogue["sessions"]) == 2
    assert len(dialogue["sessions"][0]) == 4
    assert len(dialogue["sessions"][1]) == 2
    # timestamp：消息级 time_anchor 优先；缺失时为 None（batch 级也是 null）
    # 归一为 ISO 日期（原始只到日，不补 00:00）
    assert dialogue["sessions"][0][0]["timestamp"] == "2024-03-01"
    assert dialogue["sessions"][0][1]["timestamp"] is None
    assert all(m["role"] in ("user", "assistant") for s in dialogue["sessions"] for m in s)


def test_convert_beam_category_mapping(tmp_path: Path) -> None:
    """BEAM 10 类按 1:1 直传，question.category 保留 BEAM 原始 key
    （信息抽取/多 session 推理/事件排序/时序推理/知识更新/矛盾消解/指令遵循/
    偏好遵循/摘要/拒答）；不再合并到 membench 7 类通用约定。
    """
    chat_dir = _write_chat(tmp_path, "1")
    dialogue = convert_beam(chat_dir, scale="1m", chat_id="1")
    by_qid = {q["qid"]: q for q in dialogue["questions"]}

    expected = {
        "beam_1m_1_information_extraction_1": "information_extraction",
        "beam_1m_1_multi_session_reasoning_1": "multi_session_reasoning",
        "beam_1m_1_event_ordering_1": "event_ordering",
        "beam_1m_1_temporal_reasoning_1": "temporal_reasoning",
        "beam_1m_1_knowledge_update_1": "knowledge_update",
        "beam_1m_1_knowledge_update_2": "knowledge_update",
        "beam_1m_1_contradiction_resolution_1": "contradiction_resolution",
        "beam_1m_1_instruction_following_1": "instruction_following",
        "beam_1m_1_preference_following_1": "preference_following",
        "beam_1m_1_summarization_1": "summarization",
        "beam_1m_1_abstention_1": "abstention",
    }
    assert {qid: q["category"] for qid, q in by_qid.items()} == expected
    # 未知题型被跳过
    assert not any("unknown_future_type" in qid for qid in by_qid)
    # evidence 一律为空
    assert all(q["evidence"] == [] for q in by_qid.values())


def test_convert_beam_answer_fields(tmp_path: Path) -> None:
    chat_dir = _write_chat(tmp_path, "1")
    dialogue = convert_beam(chat_dir, scale="1m", chat_id="1")
    by_qid = {q["qid"]: q for q in dialogue["questions"]}

    # 不同题型的异名答案字段均被正确拾取
    assert by_qid["beam_1m_1_information_extraction_1"]["gold_answer"] == "PostgreSQL 14"
    assert by_qid["beam_1m_1_contradiction_resolution_1"]["gold_answer"].startswith("你前后说法矛盾")
    assert by_qid["beam_1m_1_instruction_following_1"]["gold_answer"] == "回答需包含兜底策略。"
    assert by_qid["beam_1m_1_summarization_1"]["gold_answer"].startswith("项目涵盖")


def test_convert_beam_abstention_and_gold_stale(tmp_path: Path) -> None:
    chat_dir = _write_chat(tmp_path, "1")
    dialogue = convert_beam(chat_dir, scale="1m", chat_id="1")
    by_qid = {q["qid"]: q for q in dialogue["questions"]}

    # abstention：gold_answer 固定哨兵，忽略原始 ideal_response
    abst = by_qid["beam_1m_1_abstention_1"]
    assert abst["gold_answer"] == ABSTAIN_SENTINEL == "<ABSTAIN>"

    # knowledge_update：描述句没有 "instead of / now" 对照，抽不出裸旧值 →
    # gold_stale 留空，原始描述保留在 gold_stale_note（交给 LLM judge）。
    ku1 = by_qid["beam_1m_1_knowledge_update_1"]
    assert ku1.get("gold_stale") is None
    assert ku1["gold_stale_note"] == (
        "LLM might incorrectly recall the original March 15, 2024 deadline"
    )
    # potential_confusion 为空 → 两个字段都不填
    ku2 = by_qid["beam_1m_1_knowledge_update_2"]
    assert ku2.get("gold_stale") is None
    assert ku2.get("gold_stale_note") is None


@pytest.mark.parametrize(
    ("confusion", "gold_answer", "expected"),
    [
        # 三种句式：original X instead of Y / previously X ... now Y /
        # originally X ... now Y
        (
            "original 4 study sessions instead of 5 study sessions",
            "5 study sessions",
            "4 study sessions",
        ),
        (
            "previously 4 study sessions, but now 5 study sessions",
            "5 study sessions",
            "4 study sessions",
        ),
        (
            "originally 4 study sessions; now 5 study sessions",
            "5 study sessions",
            "4 study sessions",
        ),
        # 限定词 / 标点 / 引号 / 单复数 / 主语前缀
        (
            "The original '4 study sessions' instead of updated '5 study sessions'.",
            "5 study sessions",
            "4 study sessions",
        ),
        (
            "Originally, 1 study session, now 2 study sessions.",
            "2 study sessions",
            "1 study session",
        ),
        (
            "The user originally completed 4 study sessions instead of 5 study sessions.",
            "5 study sessions",
            "4 study sessions",
        ),
    ],
)
def test_extract_stale_value_patterns(
    confusion: str, gold_answer: str, expected: str
) -> None:
    """三种句式及常见变体都抽出裸旧值（不含主语 / 系动词前缀）。"""
    assert extract_stale_value(confusion, gold_answer) == expected


@pytest.mark.parametrize(
    ("confusion", "gold_answer"),
    [
        # 没有新旧对照，无法判断哪段是旧值 -> 交给 note + LLM
        ("LLM might incorrectly recall the original March 15, 2024 deadline", "March 18, 2024"),
        ("The user changed the deadline.", "March 18, 2024"),
        # 抽取结果与 gold_answer 相同：不是"旧值"
        ("original 5 study sessions instead of 5 study sessions", "5 study sessions"),
        # 过长：多半是整句描述而不是值
        ("original " + "x" * 81 + " instead of 5", "5"),
        ("", "5"),
    ],
)
def test_extract_stale_value_rejects(confusion: str, gold_answer: str) -> None:
    assert extract_stale_value(confusion, gold_answer) is None


def test_convert_beam_split_stale_value_and_note_roundtrip(tmp_path: Path) -> None:
    """转换产物经 load_dataset 加载后：gold_stale 是裸旧值，note 是原句。"""
    chat_dir = _write_chat(tmp_path, "1")
    pq = {
        "knowledge_update": [
            {
                "question": "完成了多少次学习？",
                "answer": "5 study sessions",
                "potential_confusion": (
                    "previously 4 study sessions, but now 5 study sessions"
                ),
            },
            {
                "question": "截止日期是哪天？",
                "answer": "March 18, 2024",
                "potential_confusion": (
                    "LLM might incorrectly recall the original March 15, 2024 deadline"
                ),
            },
        ]
    }
    (chat_dir / "probing_questions.json").write_text(
        json.dumps(pq, ensure_ascii=False), encoding="utf-8"
    )
    out = tmp_path / "out.jsonl"
    convert_beam_split(tmp_path, "1m", out)
    questions = {q.qid: q for q in load_dataset(out)[0].questions}

    extracted = questions["beam_1m_1_knowledge_update_1"]
    assert extracted.gold_stale == "4 study sessions"
    assert extracted.gold_stale_note == (
        "previously 4 study sessions, but now 5 study sessions"
    )

    failed = questions["beam_1m_1_knowledge_update_2"]
    assert failed.gold_stale is None
    assert failed.gold_stale_note == (
        "LLM might incorrectly recall the original March 15, 2024 deadline"
    )


def test_convert_beam_pq_fallback_name(tmp_path: Path) -> None:
    """官方发布中探针题文件也可能名为 pq.json。"""
    chat_dir = _write_chat(tmp_path, "7", pq_name="pq.json")
    dialogue = convert_beam(chat_dir, scale="10m", chat_id="7")
    assert dialogue["dialogue_id"] == "beam_10m_7"
    assert len(dialogue["questions"]) == 11


def test_convert_beam_invalid_role_normalized(tmp_path: Path) -> None:
    chat_dir = _write_chat(tmp_path, "1")
    chat = json.loads((chat_dir / "chat.json").read_text(encoding="utf-8"))
    chat[0]["turns"][0][1]["role"] = "AI Assistant"  # 非法 role
    chat[0]["turns"][0][0]["role"] = "human"
    (chat_dir / "chat.json").write_text(json.dumps(chat), encoding="utf-8")

    dialogue = convert_beam(chat_dir, "1m", "1")
    assert dialogue["sessions"][0][0]["role"] == "user"
    assert dialogue["sessions"][0][1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# convert_beam_split：批量 + 损坏跳过 + schema 兼容
# ---------------------------------------------------------------------------


def test_convert_beam_split(tmp_path: Path) -> None:
    root = tmp_path / "beam_1m"
    _write_chat(root, "1")
    _write_chat(root, "2")
    broken = root / "3"
    broken.mkdir()
    (broken / "chat.json").write_text("{损坏的 json", encoding="utf-8")

    out = tmp_path / "out.jsonl"
    stats = convert_beam_split(root, "1m", out)

    assert stats["dialogues"] == 2
    assert stats["skipped"] == 1
    assert stats["questions"] == 22
    assert stats["categories"]["abstention"] == 2
    # knowledge_update / contradiction_resolution 分开统计：
    # knowledge_update 2 题/chat × 2 chat = 4；contradiction_resolution 1 题/chat × 2 chat = 2
    assert stats["categories"]["knowledge_update"] == 4
    assert stats["categories"]["contradiction_resolution"] == 2

    # 输出可被 schema.load_dataset 加载
    dialogues = load_dataset(out)
    assert len(dialogues) == 2
    assert dialogues[0].dialogue_id == "beam_1m_1"


def test_convert_beam_split_limit(tmp_path: Path) -> None:
    root = tmp_path / "beam_1m"
    _write_chat(root, "1")
    _write_chat(root, "2")
    out = tmp_path / "out.jsonl"
    stats = convert_beam_split(root, "1m", out, limit=1)
    assert stats["dialogues"] == 1
    assert load_dataset(out)[0].dialogue_id == "beam_1m_1"
