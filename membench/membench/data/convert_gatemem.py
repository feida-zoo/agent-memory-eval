"""GateMem 办公场景基准转换器：将 episode/checkpoint JSONL 转为 membench JSONL 格式。

数据源：GateMem（arXiv:2606.18829），https://github.com/rzhub/GateMem
原始文件：bench/data/office/{episodes.jsonl, checkpoints.jsonl}

GateMem 是一个面向记忆系统的办公场景基准，考察系统的**效用**（答得对）、
**隐私**（该拒则拒、该脱敏则脱敏）与**安全**（被删除的信息不应再记得）。

原始数据为两个 JSONL 文件：

episodes（每行一个 episode，办公场景多轮对话）::

    {"episode_id": str, "domain": "office",
     "entities": {"principals": [{"principal_id": str, "role": str, ...}], ...},
     "turns": [{"turn_id": str, "timestamp": str,
                "speaker": {"principal_id": str, "role": str},
                "turn_kind": str, "text": str}, ...]}

checkpoints（每行一个 checkpoint，即一道挂接在某 episode 上的评测题）::

    {"checkpoint_id": str, "episode_id": str, "as_of_turn_id": str,
     "asker": {"principal_id": str, "role": str},
     "query_type": "utility|privacy|safety", "query_text": str,
     "expected_action": "answer|refuse|answer_redacted|no_memory",
     "judge_spec": {"include": [str, ...], "not_include": [str, ...], "notes": str},
     "leak_targets": [...]}

checkpoint -> membench question 映射表（1:1 直传，保留原始语义）：

    (utility, answer)          -> utility_answer
                                  gold_answer = judge_spec.include 以 "; " 拼接（仅供展示）
    (privacy, refuse)          -> privacy_refuse（gold_answer = "<ABSTAIN>"）
    (privacy, answer_redacted) -> privacy_answer_redacted
                                  gold_answer = judge_spec.include 以 "; " 拼接（仅供展示）
    (safety,  no_memory)       -> safety_no_memory（gold_answer = "<ABSTAIN>"；
                                  对话中含删除请求，系统处理后不应再记得）

其他 (query_type, expected_action) 组合：跳过并计数，转换结束后打印警告。

**judge_spec 确定性 rubric**：checkpoint 的 ``judge_spec.include`` /
``judge_spec.not_include`` 与 ``leak_targets`` 原样写进 ``Question.judge_spec``
（三段都为空时不写该字段）。写下之后判分不再用 ``gold_answer`` 做 token_f1
或 LLM 判分，而是由 ``membench.judges.heuristic.grade_judge_spec`` 逐条校验：
``include`` 按**正则**（大小写不敏感；原始数据里确有 ``no[- ]auto[- ]renewal``
这类写法）、``not_include`` 与 ``leak_targets`` 按**字面量 + ASCII 词边界**
（本地实测原始数据这两类共 0 条含正则元字符，但含 ``yes`` / ``no`` 这类短值）。``leak_targets`` 命中是
红线泄露。``gold_answer`` 的拼接只保留给报告展示，不参与判分。

``--ignore-privacy-refuse`` 选项：(privacy, refuse) 映射的 abstention 题
会被打上题目级忽略标记（``ignore=true``，``ignore_reason="多用户权限语义，
单用户个人 agent 场景不适用"``），评测时整条跳过（不查询、不计分），
但题目数据保留在文件中可追溯。适用场景：**单用户个人 agent** 建议开启
——(privacy, refuse) 题的"该向谁拒答"是多用户权限语义，单用户场景下
没有对应能力可评；**多用户共享记忆场景不要开**——权限拒答正是该场景
的核心能力之一，忽略会导致隐私维度漏评。

role 映射：episode 的 turns 全部是办公人员之间的 ``dialogue``（无 AI 参与），
统一映射为 ``user``（对记忆系统而言这些全是待吸收的输入）；仅当
speaker.role / turn_kind 显式为 assistant/agent/bot 一类时才映射为
``assistant``。一个 episode 整体作为一个 session（turn_kind 无分段语义）。

**提问时点（as_of）**：checkpoint 的 ``as_of_turn_id`` 是该 episode ``turns``
里的 turn id；转换器取它在 turns 中的下标 + 1 写成 ``Question.as_of``（=
"该时点已 ingest 的消息条数"），并把整个 dialogue 标成
``ingest_strategy="interleaved"``——评测时 runner 边 ingest 边提问，问的是
"当时能看到什么"。``as_of_turn_id`` 在 turns 里找不到时该 checkpoint 跳过并
计数（不猜时点）。注意：interleaved 与题目级分片 / ``--replay`` 不兼容，
评测时需 ``--shard-by dialogue``（见 ``membench.runner``）。

已知局限：
- evidence 一律为 []：judge_spec.include 是判分要点而非对话原文子串。失败归因
  因此无法逐题判定"证据是否已写入"，相关题会记 ``evidence_in_state=null``。
- asker 身份未建模：privacy 题的"该向谁拒答"依赖提问者身份，membench
  schema 无对应字段，统一按 abstention/single_hop 处理。
- utility 题 judge_spec.notes 中的 stale 信息（旧值）未填入 gold_stale。

用法::

    python -m membench.data.convert_gatemem episodes.jsonl checkpoints.jsonl -o gatemem.jsonl
    # 单用户个人 agent 场景：忽略多用户权限拒答题
    python -m membench.data.convert_gatemem episodes.jsonl checkpoints.jsonl -o gatemem.jsonl \
        --ignore-privacy-refuse
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .timestamps import normalize_timestamp

#: abstention 题的 gold_answer 哨兵值。
ABSTAIN_SENTINEL = "<ABSTAIN>"

#: (query_type, expected_action) -> question.category（1:1 直传）
#: 旧的"utility_answer + privacy_answer_redacted → single_hop / privacy_refuse + safety_no_memory →
#: abstention"合并会丢诊断信息——privacy 和 safety 在 gatemem 里语义不同，
#: 前者考"权限语义下该不该拒"，后者考"删除后是否还记"，必须分开。
#: 现在按 (query_type, expected_action) 4 类原样写进 question.category；
#: ``membench.data.schema.CATEGORIES`` 是软约定不再校验。
_ACTION_MAP = {
    ("utility", "answer"): "utility_answer",
    ("privacy", "refuse"): "privacy_refuse",
    ("privacy", "answer_redacted"): "privacy_answer_redacted",
    ("safety", "no_memory"): "safety_no_memory",
}

#: role 归一化：assistant 侧关键词（子串匹配，均为小写）。
_ASSISTANT_HINTS = ("assistant", "agent", "bot")

#: (privacy, refuse) 题被 --ignore-privacy-refuse 标记时的忽略原因。
PRIVACY_REFUSE_IGNORE_REASON = "多用户权限语义，单用户个人 agent 场景不适用"


def _normalize_role(speaker: dict, turn_kind: str) -> str:
    """将 speaker.role / turn_kind 映射为 user/assistant。

    办公场景中所有 turn 都是人类发言（对记忆系统是输入）-> user；
    仅显式 assistant/agent/bot 类标记 -> assistant。
    """
    text = f"{speaker.get('role', '')} {turn_kind}".lower()
    for hint in _ASSISTANT_HINTS:
        if hint in text:
            return "assistant"
    return "user"


def _read_jsonl(path: str | Path) -> list[dict]:
    """读取 JSONL 文件为 dict 列表（跳过空行）。"""
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _convert_episode(episode: dict) -> tuple[dict, dict[str, int]]:
    """将一个 episode 转为不含 questions 的 dialogue dict。

    Returns:
        ``(dialogue, turn_positions)``：``turn_positions`` 是
        ``turn_id -> 在 turns 里的下标 + 1``（= 该 turn 之后的已 ingest 消息
        条数），供 checkpoint 写 ``as_of``。
    """
    session: list[dict] = []
    turn_positions: dict[str, int] = {}
    for turn in episode.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        session.append(
            {
                "role": _normalize_role(turn.get("speaker") or {}, str(turn.get("turn_kind", ""))),
                "content": str(turn.get("text", "")),
                "timestamp": normalize_timestamp(
                    turn.get("timestamp"),
                    where=f"{episode.get('episode_id', '?')} {turn.get('turn_id', '?')}",
                ),
            }
        )
        turn_id = turn.get("turn_id")
        if turn_id is not None:
            # 已 ingest 的消息条数 = 该 turn 写入后的 session 长度（跳过非 dict
            # turn 时序号不会错位）。
            turn_positions[str(turn_id)] = len(session)
    return (
        {
            "dialogue_id": str(episode.get("episode_id", "")),
            "ingest_strategy": "interleaved",
            "sessions": [session],
            "questions": [],
        },
        turn_positions,
    )


def _convert_checkpoint(
    cp: dict,
    as_of: int | None = None,
    ignore_privacy_refuse: bool = False,
) -> dict | None:
    """将一个 checkpoint 转为 membench question dict；未知组合返回 None。

    ``judge_spec.include`` / ``not_include`` 与 ``leak_targets`` 原样写入
    ``question["judge_spec"]``（三段全空时不写）；``gold_answer`` 仍保留
    include 的可读拼接（拒答类为哨兵），供报告展示。

    Args:
        cp: checkpoint 记录。
        as_of: 该 checkpoint 时点已 ingest 的消息条数（调用方按
            ``as_of_turn_id`` 在 turns 里的下标 + 1 算好）；None 表示该对话
            无法定位时点——写入后 runner 会把它排到全部 ingest 之后。
        ignore_privacy_refuse: True 时 (privacy, refuse) 题附加 ignore 标记。
    """
    key = (str(cp.get("query_type", "")), str(cp.get("expected_action", "")))
    category = _ACTION_MAP.get(key)
    if category is None:
        return None
    spec_raw = cp.get("judge_spec") or {}
    include = [str(item) for item in spec_raw.get("include") or []]
    not_include = [str(item) for item in spec_raw.get("not_include") or []]
    leak_targets = [str(item) for item in cp.get("leak_targets") or []]
    question: dict = {
        "qid": str(cp.get("checkpoint_id", "")),
        "question": str(cp.get("query_text", cp.get("query", ""))),
        "category": category,
        "evidence": [],
    }
    if as_of is not None:
        question["as_of"] = int(as_of)
    # category 不再合并，"拒答类"题改为按 key 判定——
    # ``(privacy, refuse)`` 与 ``(safety, no_memory)`` 都期望系统拒答，
    # gold_answer 设为 abstention 哨兵。
    if key in (("privacy", "refuse"), ("safety", "no_memory")):
        question["gold_answer"] = ABSTAIN_SENTINEL
    else:
        question["gold_answer"] = "; ".join(include)
    # 三段都空时不写 judge_spec：此时 rubric 对任何回答都判对，不能用来判分。
    if include or not_include or leak_targets:
        question["judge_spec"] = {
            "include": include,
            "not_include": not_include,
            "leak_targets": leak_targets,
        }
    if ignore_privacy_refuse and key == ("privacy", "refuse"):
        question["ignore"] = True
        question["ignore_reason"] = PRIVACY_REFUSE_IGNORE_REASON
    return question


def convert_gatemem(
    episodes_path: str | Path,
    checkpoints_path: str | Path,
    output_path: str | Path,
    ignore_privacy_refuse: bool = False,
) -> dict:
    """将 GateMem episodes + checkpoints 转为 membench JSONL。

    Args:
        episodes_path: episode JSONL 路径。
        checkpoints_path: checkpoint JSONL 路径。
        output_path: 输出 JSONL 路径。
        ignore_privacy_refuse: True 时 (privacy, refuse) 映射的 abstention 题
            打上 ``ignore=true`` 标记（单用户个人 agent 场景建议开启；
            多用户共享记忆场景不要开）。

    Returns:
        dict: 统计信息（dialogues/questions/categories/skipped/ignored/output）。
    """
    episodes = _read_jsonl(episodes_path)
    checkpoints = _read_jsonl(checkpoints_path)

    dialogues: list[dict] = []
    turn_positions: dict[str, dict[str, int]] = {}
    for ep in episodes:
        dialogue, positions = _convert_episode(ep)
        dialogues.append(dialogue)
        turn_positions[dialogue["dialogue_id"]] = positions
    by_id = {d["dialogue_id"]: d for d in dialogues}

    n_questions = 0
    n_skipped = 0
    n_ignored = 0
    n_as_of_missing = 0
    as_of_positions: list[int] = []
    category_counts: dict[str, int] = {}
    for cp in checkpoints:
        cp_id = repr(cp.get("checkpoint_id"))
        episode_id = str(cp.get("episode_id", ""))
        positions = turn_positions.get(episode_id)
        as_of: int | None = None
        if positions is not None:
            as_of = positions.get(str(cp.get("as_of_turn_id", "")))
            if as_of is None:
                n_as_of_missing += 1
                print(
                    f"[convert_gatemem] 警告：checkpoint {cp_id} 的 as_of_turn_id="
                    f"{cp.get('as_of_turn_id')!r} 不在 episode {episode_id!r} 的 turns 里，"
                    "无法定位提问时点，跳过该 checkpoint",
                    file=sys.stderr,
                )
                n_skipped += 1
                continue
        question = _convert_checkpoint(
            cp, as_of=as_of, ignore_privacy_refuse=ignore_privacy_refuse
        )
        if question is None:
            n_skipped += 1
            print(
                f"[convert_gatemem] 警告：跳过未知组合 "
                f"(query_type={cp.get('query_type')!r}, "
                f"expected_action={cp.get('expected_action')!r})，"
                f"checkpoint_id={cp.get('checkpoint_id')!r}",
                file=sys.stderr,
            )
            continue
        target = by_id.get(episode_id)
        if target is None:
            n_skipped += 1
            print(
                f"[convert_gatemem] 警告：checkpoint {cp_id} "
                f"指向不存在的 episode {cp.get('episode_id')!r}，跳过",
                file=sys.stderr,
            )
            continue
        target["questions"].append(question)
        n_questions += 1
        if as_of is not None:
            as_of_positions.append(as_of)
        if question.get("ignore"):
            n_ignored += 1
        category_counts[question["category"]] = category_counts.get(question["category"], 0) + 1

    lines = [json.dumps(d, ensure_ascii=False) for d in dialogues]
    Path(output_path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return {
        "dialogues": len(lines),
        "questions": n_questions,
        "categories": category_counts,
        "skipped": n_skipped,
        "ignored": n_ignored,
        "as_of_missing": n_as_of_missing,
        "as_of_min": min(as_of_positions) if as_of_positions else None,
        "as_of_max": max(as_of_positions) if as_of_positions else None,
        "output": str(output_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="将 GateMem 办公场景基准转为 membench JSONL")
    parser.add_argument("episodes", help="episodes JSONL 路径")
    parser.add_argument("checkpoints", help="checkpoints JSONL 路径")
    parser.add_argument("-o", "--output", required=True, help="输出 JSONL 路径")
    parser.add_argument(
        "--ignore-privacy-refuse",
        action="store_true",
        help="将 (privacy, refuse) 映射的 abstention 题标记为 ignore"
        "（单用户个人 agent 场景建议开启；多用户共享记忆场景不要开）",
    )
    args = parser.parse_args(argv)

    stats = convert_gatemem(
        args.episodes, args.checkpoints, args.output,
        ignore_privacy_refuse=args.ignore_privacy_refuse,
    )
    print(f"[convert_gatemem] {stats['dialogues']} dialogues, {stats['questions']} questions"
          f"（跳过 {stats['skipped']} 个 checkpoint）")
    if stats["as_of_min"] is not None:
        print(
            f"[convert_gatemem] 提问时点 as_of（已 ingest 消息条数）范围 "
            f"{stats['as_of_min']}..{stats['as_of_max']}；"
            "ingest_strategy=interleaved（评测时请用 --shard-by dialogue）"
        )
    if stats["ignored"]:
        print(f"[convert_gatemem] 已标记忽略 {stats['ignored']} 道 (privacy, refuse) 题"
              f"（ignore=true，评测时跳过）")
    for cat, n in sorted(stats["categories"].items()):
        print(f"  {cat}: {n}")
    print(f"[convert_gatemem] -> {stats['output']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
