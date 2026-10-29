"""GateMem 转换器测试：动作映射、abstention 哨兵、未知组合跳过、schema 兼容。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def _cp(
    cp_id: str,
    ep_id: str,
    query_type: str,
    action: str,
    include: list[str] | None = None,
    not_include: list[str] | None = None,
    leak_targets: list[str] | None = None,
) -> dict:
    return {
        "checkpoint_id": cp_id,
        "episode_id": ep_id,
        "as_of_turn_id": "t001",
        "asker": {"principal_id": "pm_ava", "role": "product_manager"},
        "query_type": query_type,
        "query_text": f"{cp_id} 的问题文本",
        "expected_action": action,
        "judge_spec": {
            "expected_action": action,
            "include": include or [],
            "not_include": not_include or [],
            "notes": "",
        },
        "leak_targets": leak_targets or [],
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
    # 每个 episode 一段对话、整段标 interleaved（按 checkpoint 的 as_of 边 ingest 边问）
    assert all(d.ingest_strategy == "interleaved" for d in dialogues)


def test_convert_gatemem_as_of_from_turn_position(tmp_path: Path) -> None:
    """``as_of`` = ``as_of_turn_id`` 在 turns 里的下标 + 1（已 ingest 的消息条数）。"""
    ep_path, cp_path = _write_inputs(tmp_path)
    out = tmp_path / "out.jsonl"
    convert_gatemem(ep_path, cp_path, out)

    by_qid = {q.qid: q for d in load_dataset(out) for q in d.questions}
    # 夹具里所有 checkpoint 都挂 t001（每个 episode 的第一条消息）
    assert by_qid["ck_utility"].as_of == 1
    assert by_qid["ck_safety"].as_of == 1


def test_convert_gatemem_as_of_uses_later_turn(tmp_path: Path) -> None:
    """挂在第 2 个 turn 上的 checkpoint：as_of=2；找不到 turn 的直接跳过并计数。"""
    checkpoints = [
        {
            **_cp("ck_t2", "ep_001", "utility", "answer", ["9%"]),
            "as_of_turn_id": "t002",
        },
        {
            **_cp("ck_missing_turn", "ep_001", "utility", "answer", ["x"]),
            "as_of_turn_id": "t999",
        },
    ]
    ep_path, cp_path = _write_custom_checkpoints(tmp_path, checkpoints)
    out = tmp_path / "out.jsonl"
    stats = convert_gatemem(ep_path, cp_path, out)

    by_qid = {q.qid: q for d in load_dataset(out) for q in d.questions}
    assert by_qid["ck_t2"].as_of == 2
    assert "ck_missing_turn" not in by_qid
    assert stats["as_of_missing"] == 1
    assert stats["skipped"] == 1
    assert (stats["as_of_min"], stats["as_of_max"]) == (2, 2)


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


def _write_custom_checkpoints(tmp_path: Path, checkpoints: list[dict]) -> tuple[Path, Path]:
    """复用默认 episodes，写一组自定义 checkpoints。"""
    ep_path = tmp_path / "ep.jsonl"
    ep_path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in _EPISODES) + "\n",
        encoding="utf-8",
    )
    cp_path = tmp_path / "cp.jsonl"
    cp_path.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in checkpoints) + "\n",
        encoding="utf-8",
    )
    return ep_path, cp_path


def test_judge_spec_written_to_question(tmp_path: Path) -> None:
    """include / not_include / leak_targets 原样落进 Question.judge_spec。"""
    checkpoints = [
        _cp(
            "ck_regex",
            "ep_001",
            "utility",
            "answer",
            include=["no[- ]auto[- ]renewal", "9%"],
        ),
        _cp(
            "ck_leak",
            "ep_001",
            "privacy",
            "refuse",
            not_include=["Granite Mutual", "active"],
            leak_targets=["Granite Mutual"],
        ),
        _cp(
            "ck_redacted",
            "ep_001",
            "privacy",
            "answer_redacted",
            include=["是同一个保险账户"],
            leak_targets=["mp_stg_4R2N"],
        ),
    ]
    ep_path, cp_path = _write_custom_checkpoints(tmp_path, checkpoints)
    out = tmp_path / "out.jsonl"
    convert_gatemem(ep_path, cp_path, out)

    by_qid = {q.qid: q for d in load_dataset(out) for q in d.questions}
    assert by_qid["ck_regex"].judge_spec == {
        "include": ["no[- ]auto[- ]renewal", "9%"],
        "not_include": [],
        "leak_targets": [],
    }
    assert by_qid["ck_leak"].judge_spec == {
        "include": [],
        "not_include": ["Granite Mutual", "active"],
        "leak_targets": ["Granite Mutual"],
    }
    assert by_qid["ck_redacted"].judge_spec["leak_targets"] == ["mp_stg_4R2N"]
    # gold_answer 仍保留 include 的可读拼接（报告展示用），拒答题为哨兵
    assert by_qid["ck_regex"].gold_answer == "no[- ]auto[- ]renewal; 9%"
    assert by_qid["ck_leak"].gold_answer == ABSTAIN_SENTINEL


def test_judge_spec_omitted_when_all_sections_empty(tmp_path: Path) -> None:
    """三段全空时不写 judge_spec（空 rubric 会把任意回答判对）。"""
    checkpoints = [_cp("ck_empty", "ep_001", "utility", "answer", include=[])]
    ep_path, cp_path = _write_custom_checkpoints(tmp_path, checkpoints)
    out = tmp_path / "out.jsonl"
    convert_gatemem(ep_path, cp_path, out)

    by_qid = {q.qid: q for d in load_dataset(out) for q in d.questions}
    assert by_qid["ck_empty"].judge_spec is None


# ---------------------------------------------------------------------------
# 真实 GateMem 数据（按需下载；仓库不内置，缺失则跳过）
# ---------------------------------------------------------------------------

_GATEMEM_RAW = Path.home() / ".membench/datasets/.gatemem_office_raw"


@pytest.mark.skipif(
    not (_GATEMEM_RAW / "episodes.jsonl").is_file()
    or not (_GATEMEM_RAW / "checkpoints.jsonl").is_file(),
    reason="GateMem 原始数据未下载（dataset fetch gatemem_office）",
)
def test_convert_real_gatemem_as_of_and_counts(tmp_path: Path) -> None:
    """真实数据：17 个 episode、547 个 checkpoint（忽略 140 privacy-refuse 后 407 有效），
    as_of 全部落在各 episode 的 1..消息总数，ingest_strategy 统一 interleaved。"""
    episodes = [
        json.loads(line)
        for line in (_GATEMEM_RAW / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    checkpoints = [
        json.loads(line)
        for line in (_GATEMEM_RAW / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    turns_by_episode = {
        str(e["episode_id"]): [str(t.get("turn_id")) for t in e.get("turns") or []]
        for e in episodes
    }
    expected_as_of = {
        str(cp["checkpoint_id"]): turns_by_episode[str(cp["episode_id"])].index(
            str(cp["as_of_turn_id"])
        )
        + 1
        for cp in checkpoints
        if str(cp.get("episode_id")) in turns_by_episode
        and str(cp.get("as_of_turn_id")) in turns_by_episode[str(cp["episode_id"])]
    }

    out = tmp_path / "gatemem.jsonl"
    stats = convert_gatemem(
        _GATEMEM_RAW / "episodes.jsonl",
        _GATEMEM_RAW / "checkpoints.jsonl",
        out,
        ignore_privacy_refuse=True,
    )
    dialogues = load_dataset(out)
    questions = [q for d in dialogues for q in d.questions]

    assert stats["dialogues"] == len(dialogues) == 17
    assert stats["questions"] == len(questions) == 547
    assert stats["ignored"] == 140
    assert sum(1 for q in questions if not q.ignore) == 407
    assert all(d.ingest_strategy == "interleaved" for d in dialogues)
    assert all(len(d.sessions) == 1 for d in dialogues)
    assert all(q.as_of is not None for q in questions)
    for dialogue in dialogues:
        assert dialogue.n_messages in (240, 260)
        for question in dialogue.questions:
            assert 1 <= question.as_of <= dialogue.n_messages
    # as_of 与原始 as_of_turn_id 的下标 +1 逐题一致
    assert {q.qid: q.as_of for q in questions} == expected_as_of
    # judge_spec 仍按 checkpoint 原样落盘（include 正则 / not_include / leak_targets）
    by_qid = {q.qid: q for q in questions}
    utility = next(q for q in questions if q.category == "utility_answer")
    assert utility.judge_spec is not None
    assert utility.judge_spec["include"] == [
        str(item)
        for item in next(
            cp for cp in checkpoints if str(cp["checkpoint_id"]) == utility.qid
        )["judge_spec"]["include"]
    ]
    refuse = next(q for q in questions if q.category == "privacy_refuse")
    assert refuse.gold_answer == ABSTAIN_SENTINEL
    assert by_qid[refuse.qid].ignore is True
