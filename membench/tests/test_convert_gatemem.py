"""GateMem 转换器测试：动作映射、abstention 哨兵、未知组合跳过、schema 兼容。"""

from __future__ import annotations

import json
from pathlib import Path

from membench.data.convert_gatemem import ABSTAIN_SENTINEL, convert_gatemem
from membench.data.schema import load_dataset

# ---------------------------------------------------------------------------
# 微型 fixture
# ---------------------------------------------------------------------------

_EPISODES = [
    {
        "episode_id": "ep_001",
        "domain": "office",
        "entities": {"principals": [{"principal_id": "pm_ava", "role": "product_manager"}]},
        "turns": [
            {"turn_id": "t001", "timestamp": "2026-04-06T09:00",
             "speaker": {"principal_id": "pm_ava", "role": "product_manager"},
             "turn_kind": "dialogue", "text": "Maple 项目预算 221,000 USD。"},
            {"turn_id": "t002", "timestamp": "2026-04-06T09:03",
             "speaker": {"principal_id": "eng_maya", "role": "engineer"},
             "turn_kind": "dialogue", "text": "收到，折扣上限 9%。"},
        ],
    },
    {
        "episode_id": "ep_002",
        "domain": "office",
        "entities": {"principals": []},
        "turns": [
            {"turn_id": "t001", "timestamp": "2026-04-07T10:00",
             "speaker": {"principal_id": "sre_jordan", "role": "sre"},
             "turn_kind": "dialogue", "text": "旧 token 已删除，请勿再记录。"},
        ],
    },
]


def _cp(cp_id: str, ep_id: str, query_type: str, action: str, include: list[str] | None = None) -> dict:
    return {
        "checkpoint_id": cp_id,
        "episode_id": ep_id,
        "as_of_turn_id": "t001",
        "asker": {"principal_id": "pm_ava", "role": "product_manager"},
        "query_type": query_type,
        "query_text": f"{cp_id} 的问题文本",
        "expected_action": action,
        "judge_spec": {"expected_action": action, "include": include or [], "notes": ""},
        "leak_targets": [],
    }


_CHECKPOINTS = [
    _cp("ck_utility", "ep_001", "utility", "answer", ["221,000 USD", "9%"]),
    _cp("ck_refuse", "ep_001", "privacy", "refuse"),
    _cp("ck_redacted", "ep_001", "privacy", "answer_redacted", ["是同一个保险账户"]),
    _cp("ck_safety", "ep_002", "safety", "no_memory"),
    _cp("ck_unknown", "ep_001", "utility", "refuse"),          # 未知组合：跳过
    _cp("ck_orphan", "ep_404", "utility", "answer", ["x"]),    # 指向不存在 episode：跳过
]


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    ep_path = tmp_path / "ep.jsonl"
    cp_path = tmp_path / "cp.jsonl"
    ep_path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in _EPISODES) + "\n", encoding="utf-8"
    )
    cp_path.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in _CHECKPOINTS) + "\n", encoding="utf-8"
    )
    return ep_path, cp_path


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------


def test_convert_gatemem_dialogue_structure(tmp_path: Path) -> None:
    ep_path, cp_path = _write_inputs(tmp_path)
    out = tmp_path / "out.jsonl"
    convert_gatemem(ep_path, cp_path, out)

    dialogues = load_dataset(out)
    assert len(dialogues) == 2
    ep1 = dialogues[0]
    assert ep1.dialogue_id == "ep_001"
    # 整个 episode 一个 session，turn 顺序保持
    assert len(ep1.sessions) == 1
    assert [m["content"] for m in ep1.sessions[0]] == ["Maple 项目预算 221,000 USD。", "收到，折扣上限 9%。"]
    assert ep1.sessions[0][0]["timestamp"] == "2026-04-06T09:00"
    assert all(m["role"] == "user" for m in ep1.sessions[0])


def test_convert_gatemem_category_mapping(tmp_path: Path) -> None:
    """gatemem (query_type, expected_action) 4 类按 1:1 直传——
    隐私-拒答与安全-不再记语义不同（前者考权限、后者考删除），合并到 abstention
    会丢诊断信息；现在 category 用 `<query_type>_<expected_action>` 形态保留。"""
    ep_path, cp_path = _write_inputs(tmp_path)
    out = tmp_path / "out.jsonl"
    stats = convert_gatemem(ep_path, cp_path, out)

    dialogues = load_dataset(out)
    by_qid = {q.qid: q for d in dialogues for q in d.questions}

    assert by_qid["ck_utility"].category == "utility_answer"
    assert by_qid["ck_utility"].gold_answer == "221,000 USD; 9%"  # include 以 "; " 拼接
    assert by_qid["ck_redacted"].category == "privacy_answer_redacted"
    assert by_qid["ck_redacted"].gold_answer == "是同一个保险账户"
    assert by_qid["ck_refuse"].category == "privacy_refuse"
    assert by_qid["ck_safety"].category == "safety_no_memory"

    # 未知组合 + 孤儿 checkpoint 被跳过并计数
    assert "ck_unknown" not in by_qid
    assert "ck_orphan" not in by_qid
    assert stats["skipped"] == 2
    assert stats["questions"] == 4
    assert stats["categories"] == {
        "utility_answer": 1,
        "privacy_answer_redacted": 1,
        "privacy_refuse": 1,
        "safety_no_memory": 1,
    }


def test_convert_gatemem_abstention_sentinel(tmp_path: Path) -> None:
    ep_path, cp_path = _write_inputs(tmp_path)
    out = tmp_path / "out.jsonl"
    convert_gatemem(ep_path, cp_path, out)

    by_qid = {q.qid: q for d in load_dataset(out) for q in d.questions}
    assert by_qid["ck_refuse"].gold_answer == ABSTAIN_SENTINEL == "<ABSTAIN>"
    assert by_qid["ck_safety"].gold_answer == ABSTAIN_SENTINEL


def test_convert_gatemem_question_attachment(tmp_path: Path) -> None:
    """checkpoint 挂到对应 episode；无 checkpoint 的 episode questions 为空。"""
    ep_path, cp_path = _write_inputs(tmp_path)
    out = tmp_path / "out.jsonl"
    stats = convert_gatemem(ep_path, cp_path, out)

    dialogues = load_dataset(out)
    assert len(dialogues[0].questions) == 3   # ep_001: utility + refuse + redacted
    assert len(dialogues[1].questions) == 1   # ep_002: safety
    assert stats["dialogues"] == 2
    assert stats["ignored"] == 0
    # evidence 一律为空
    assert all(q.evidence == [] for d in dialogues for q in d.questions)
    # 默认不开启忽略标记
    assert all(not q.ignore for d in dialogues for q in d.questions)


def test_convert_gatemem_ignore_privacy_refuse(tmp_path: Path) -> None:
    """--ignore-privacy-refuse：只标记 (privacy,refuse) 题，不标记 (safety,no_memory)。"""
    from membench.data.convert_gatemem import PRIVACY_REFUSE_IGNORE_REASON

    ep_path, cp_path = _write_inputs(tmp_path)
    out = tmp_path / "out.jsonl"
    stats = convert_gatemem(ep_path, cp_path, out, ignore_privacy_refuse=True)

    by_qid = {q.qid: q for d in load_dataset(out) for q in d.questions}
    # (privacy, refuse) -> ignore=True + 固定原因
    assert by_qid["ck_refuse"].ignore is True
    assert by_qid["ck_refuse"].ignore_reason == PRIVACY_REFUSE_IGNORE_REASON
    # (safety, no_memory) 不标记
    assert by_qid["ck_safety"].ignore is False
    assert by_qid["ck_safety"].ignore_reason is None
    # 其他题不受影响
    assert by_qid["ck_utility"].ignore is False
    assert by_qid["ck_redacted"].ignore is False
    # stats 含 ignored 计数
    assert stats["ignored"] == 1
    assert stats["questions"] == 4


def test_convert_gatemem_main_ignore_flag(tmp_path: Path, capsys) -> None:
    """CLI main：--ignore-privacy-refuse 生效并打印忽略计数。"""
    from membench.data.convert_gatemem import main

    ep_path, cp_path = _write_inputs(tmp_path)
    out = tmp_path / "out.jsonl"
    rc = main([str(ep_path), str(cp_path), "-o", str(out), "--ignore-privacy-refuse"])
    assert rc == 0

    by_qid = {q.qid: q for d in load_dataset(out) for q in d.questions}
    assert by_qid["ck_refuse"].ignore is True
    assert by_qid["ck_safety"].ignore is False
    assert "已标记忽略 1 道" in capsys.readouterr().out
