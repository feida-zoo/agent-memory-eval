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
                                  gold_answer = judge_spec.include 以 "; " 拼接
    (privacy, refuse)          -> privacy_refuse（gold_answer = "<ABSTAIN>"）
    (privacy, answer_redacted) -> privacy_answer_redacted
                                  gold_answer = judge_spec.include 以 "; " 拼接
    (safety,  no_memory)       -> safety_no_memory（gold_answer = "<ABSTAIN>"；
                                  对话中含删除请求，系统处理后不应再记得）

其他 (query_type, expected_action) 组合：跳过并计数，转换结束后打印警告。

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

已知局限：
- evidence 一律为 []：judge_spec.include 是判分要点而非对话原文子串。
- as_of_turn_id 的时间截断语义未完整建模：question 挂到完整 episode 上，
  未按 as_of_turn_id 截断 sessions（截断应由评测运行时按题执行）。
- asker 身份未建模：privacy 题的"该向谁拒答"依赖提问者身份，membench
  schema 无对应字段，统一按 abstention/single_hop 处理。
- answer_redacted 题的 leak_targets（不得泄露的敏感词）未强制校验，
  仅在原始数据中存在；membench 侧只校验 include 要点。
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


def _convert_episode(episode: dict) -> dict:
    """将一个 episode 转为不含 questions 的 dialogue dict。"""
    session: list[dict] = []
    for turn in episode.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        session.append(
            {
                "role": _normalize_role(turn.get("speaker") or {}, str(turn.get("turn_kind", ""))),
                "content": str(turn.get("text", "")),
                "timestamp": turn.get("timestamp"),
            }
        )
    return {
        "dialogue_id": str(episode.get("episode_id", "")),
        "sessions": [session],
        "questions": [],
    }


def _convert_checkpoint(cp: dict, ignore_privacy_refuse: bool = False) -> dict | None:
    """将一个 checkpoint 转为 membench question dict；未知组合返回 None。

    Args:
        cp: checkpoint 记录。
        ignore_privacy_refuse: True 时 (privacy, refuse) 题附加 ignore 标记。
    """
    key = (str(cp.get("query_type", "")), str(cp.get("expected_action", "")))
    category = _ACTION_MAP.get(key)
    if category is None:
        return None
    question: dict = {
        "qid": str(cp.get("checkpoint_id", "")),
        "question": str(cp.get("query_text", cp.get("query", ""))),
        "category": category,
        "evidence": [],
    }
    # category 不再合并，"拒答类"题改为按 key 判定——
    # ``(privacy, refuse)`` 与 ``(safety, no_memory)`` 都期望系统拒答，
    # gold_answer 设为 abstention 哨兵。
    if key in (("privacy", "refuse"), ("safety", "no_memory")):
        question["gold_answer"] = ABSTAIN_SENTINEL
    else:
        include = (cp.get("judge_spec") or {}).get("include") or []
        question["gold_answer"] = "; ".join(str(item) for item in include)
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

    dialogues = [_convert_episode(ep) for ep in episodes]
    by_id = {d["dialogue_id"]: d for d in dialogues}

    n_questions = 0
    n_skipped = 0
    n_ignored = 0
    category_counts: dict[str, int] = {}
    for cp in checkpoints:
        question = _convert_checkpoint(cp, ignore_privacy_refuse=ignore_privacy_refuse)
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
        target = by_id.get(str(cp.get("episode_id", "")))
        if target is None:
            n_skipped += 1
            print(
                f"[convert_gatemem] 警告：checkpoint {cp.get('checkpoint_id')!r} "
                f"指向不存在的 episode {cp.get('episode_id')!r}，跳过",
                file=sys.stderr,
            )
            continue
        target["questions"].append(question)
        n_questions += 1
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
          f"（跳过 {stats['skipped']} 个未知组合 checkpoint）")
    if stats["ignored"]:
        print(f"[convert_gatemem] 已标记忽略 {stats['ignored']} 道 (privacy, refuse) 题"
              f"（ignore=true，评测时跳过）")
    for cat, n in sorted(stats["categories"].items()):
        print(f"  {cat}: {n}")
    print(f"[convert_gatemem] -> {stats['output']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
