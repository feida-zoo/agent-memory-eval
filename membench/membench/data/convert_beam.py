"""BEAM 数据集转换器：将官方原始数据转为 membench JSONL 格式。

数据源：
- 论文：arXiv:2510.27246
- 代码与数据：https://github.com/mohammadtavakoli78/BEAM

原始数据按规模分目录（如 ``beam_1m/``、``beam_10m/``），每个 chat 一个
数字目录（``1/``、``2/`` …），目录内含两个文件::

    chat.json               对话本体（数 MB）
    probing_questions.json  探针题（部分发布版文件名为 pq.json，两者均识别）

chat.json 为 batch 列表，每个 batch 形如::

    {"batch_number": int,
     "time_anchor": str | null,          # batch 级时间锚点（可能为 null）
     "turns": [[{"role": "user"|"assistant", "id": int, "content": str,
                 "time_anchor": str,      # 仅 batch 首条 user 消息携带
                 "index": str, "question_type": str}, ...], ...]}

注意 ``turns`` 是嵌套列表（每个元素是一组消息），转换时拍平；每个 batch
对应输出中的一个 session。

probing_questions.json 为 dict，key 为 10 种题型，value 为题目列表。
题型 -> membench category 映射表：

    information_extraction   -> single_hop
    multi_session_reasoning  -> multi_hop
    event_ordering           -> temporal
    temporal_reasoning       -> temporal
    knowledge_update         -> knowledge_update（potential_confusion 非空
                                字符串时填入 gold_stale）
    contradiction_resolution -> knowledge_update
    instruction_following    -> preference
    preference_following     -> preference
    summarization            -> open_ended
    abstention               -> abstention（gold_answer 固定 "<ABSTAIN>"，
                                忽略原始 answer/ideal_response 字段）

不同题型的答案字段名不同，按优先级取第一个非空字段：
``answer`` -> ``ideal_answer`` -> ``ideal_response`` -> ``ideal_summary``
-> ``expected_compliance``。

已知局限：
- evidence 一律为 []：conversation_references 是会话索引/摘要而非原文
  子串，无法作为检索证据。
- rubric 字段（判分细则列表）未利用，judge 只能基于 gold_answer。
- gold_stale 取自 potential_confusion 的自然语言描述（如 "LLM might
  incorrectly recall the original March 15 ..."），是描述而非纯粹的过期
  答案，stale 判定需做包含式匹配。
- 消息 timestamp 稀疏：仅每个 batch 首条 user 消息携带 time_anchor，
  其余消息回退到 batch 级 time_anchor（也可能为 null）。

用法::

    python -m membench.data.convert_beam /path/to/beam_1m --scale 1m -o beam_1m.jsonl [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: BEAM 题型 -> membench category
QUESTION_TYPE_MAP = {
    "information_extraction": "single_hop",
    "multi_session_reasoning": "multi_hop",
    "event_ordering": "temporal",
    "temporal_reasoning": "temporal",
    "knowledge_update": "knowledge_update",
    "contradiction_resolution": "knowledge_update",
    "instruction_following": "preference",
    "preference_following": "preference",
    "summarization": "open_ended",
    "abstention": "abstention",
}

#: abstention 题的 gold_answer 哨兵值。
ABSTAIN_SENTINEL = "<ABSTAIN>"

#: 不同题型的答案字段名不一致，按此优先级取第一个非空字符串字段。
_ANSWER_FIELDS = ("answer", "ideal_answer", "ideal_response", "ideal_summary", "expected_compliance")

#: 探针题文件的可能文件名（官方发布中两种命名均出现过）。
_PQ_FILENAMES = ("probing_questions.json", "pq.json")

#: role 归一化：assistant 侧关键词（子串匹配，均为小写）。
_ASSISTANT_HINTS = ("assistant", "agent", "bot", "system", "ai")


def _normalize_role(raw: object) -> str:
    """将原始 role 映射为 schema 允许的 user/assistant 中最接近者。"""
    role = str(raw or "").strip().lower()
    if role in ("user", "assistant"):
        return role
    for hint in _ASSISTANT_HINTS:
        if hint in role:
            return "assistant"
    return "user"


def _flatten_turns(turns: object) -> list[dict]:
    """拍平 batch 的嵌套 turns 结构为一维消息列表。"""
    messages: list[dict] = []
    if not isinstance(turns, list):
        return messages
    for group in turns:
        items = group if isinstance(group, list) else [group]
        for msg in items:
            if isinstance(msg, dict):
                messages.append(msg)
    return messages


def _convert_sessions(chat: list[dict]) -> list[list[dict]]:
    """将 chat.json 的 batch 列表转为 sessions：一个 batch 一个 session。"""
    sessions: list[list[dict]] = []
    for batch in chat:
        if not isinstance(batch, dict):
            continue
        batch_anchor = batch.get("time_anchor")
        session: list[dict] = []
        for msg in _flatten_turns(batch.get("turns")):
            session.append(
                {
                    "role": _normalize_role(msg.get("role")),
                    "content": str(msg.get("content", "")),
                    "timestamp": msg.get("time_anchor") or batch_anchor,
                }
            )
        sessions.append(session)
    return sessions


def _pick_answer(item: dict) -> str:
    """按优先级从多个候选字段中取答案文本。"""
    for field_name in _ANSWER_FIELDS:
        value = item.get(field_name)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _convert_questions(pq: dict, id_prefix: str) -> list[dict]:
    """将 probing_questions dict 转为 membench question 列表。"""
    questions: list[dict] = []
    for qtype, items in pq.items():
        category = QUESTION_TYPE_MAP.get(qtype)
        if category is None:
            print(f"[convert_beam] 警告：未知题型 {qtype!r}，跳过", file=sys.stderr)
            continue
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            question: dict = {
                "qid": f"{id_prefix}_{qtype}_{idx}",
                "question": str(item.get("question", "")),
                "category": category,
                "evidence": [],  # conversation_references 是索引非原文，不可用
            }
            if category == "abstention":
                question["gold_answer"] = ABSTAIN_SENTINEL
            else:
                question["gold_answer"] = _pick_answer(item)
            if qtype == "knowledge_update":
                confusion = item.get("potential_confusion")
                if isinstance(confusion, str) and confusion.strip():
                    question["gold_stale"] = confusion
            questions.append(question)
    return questions


def _find_pq_path(chat_dir: Path) -> Path | None:
    """在 chat 目录中定位探针题文件（兼容两种官方命名）。"""
    for name in _PQ_FILENAMES:
        candidate = chat_dir / name
        if candidate.is_file():
            return candidate
    return None


def convert_beam(chat_dir: str | Path, scale: str, chat_id: str) -> dict:
    """将单个 BEAM chat 目录转为一个 dialogue dict。

    Args:
        chat_dir: 含 chat.json 与 probing_questions.json（或 pq.json）的目录。
        scale: 数据规模标识（如 "1m"/"10m"），用于拼 dialogue_id/qid。
        chat_id: chat 标识（通常为目录名，如 "1"）。

    Returns:
        dict: 符合 membench schema 的 dialogue（dict 形式）。

    Raises:
        FileNotFoundError: chat.json 或探针题文件缺失。
    """
    chat_dir = Path(chat_dir)
    chat = json.loads((chat_dir / "chat.json").read_text(encoding="utf-8"))
    pq_path = _find_pq_path(chat_dir)
    if pq_path is None:
        raise FileNotFoundError(
            f"{chat_dir}: 未找到探针题文件（尝试过 {list(_PQ_FILENAMES)}）"
        )
    pq = json.loads(pq_path.read_text(encoding="utf-8"))

    id_prefix = f"beam_{scale}_{chat_id}"
    return {
        "dialogue_id": id_prefix,
        "sessions": _convert_sessions(chat if isinstance(chat, list) else []),
        "questions": _convert_questions(pq if isinstance(pq, dict) else {}, id_prefix),
    }


def convert_beam_split(
    input_root: str | Path,
    scale: str,
    output_path: str | Path,
    limit: int | None = None,
) -> dict:
    """遍历 input_root 下所有数字目录，批量转换为 JSONL。

    损坏（JSON 解析失败 / 文件缺失）的 chat 会打印警告并跳过。

    Args:
        input_root: BEAM 某规模的根目录（其下为数字编号的 chat 目录）。
        scale: 数据规模标识（如 "1m"）。
        output_path: 输出 JSONL 路径。
        limit: 只转换前 N 个 chat（冒烟测试用）。

    Returns:
        dict: 统计信息（dialogues/questions/categories/skipped/output）。
    """
    root = Path(input_root)
    chat_dirs = sorted(
        (d for d in root.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )
    if limit is not None:
        chat_dirs = chat_dirs[:limit]

    n_questions = 0
    n_skipped = 0
    category_counts: dict[str, int] = {}
    lines: list[str] = []

    for chat_dir in chat_dirs:
        try:
            dialogue = convert_beam(chat_dir, scale, chat_dir.name)
        except (OSError, json.JSONDecodeError) as exc:
            n_skipped += 1
            print(f"[convert_beam] 警告：跳过损坏的 chat {chat_dir}: {exc}", file=sys.stderr)
            continue
        for q in dialogue["questions"]:
            n_questions += 1
            category_counts[q["category"]] = category_counts.get(q["category"], 0) + 1
        lines.append(json.dumps(dialogue, ensure_ascii=False))

    Path(output_path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return {
        "dialogues": len(lines),
        "questions": n_questions,
        "categories": category_counts,
        "skipped": n_skipped,
        "output": str(output_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="将 BEAM 原始数据转为 membench JSONL")
    parser.add_argument("input_root", help="BEAM 某规模根目录（其下为数字 chat 目录）")
    parser.add_argument("--scale", required=True, help="数据规模标识，如 1m / 10m")
    parser.add_argument("-o", "--output", required=True, help="输出 JSONL 路径")
    parser.add_argument("--limit", type=int, default=None, help="只转换前 N 个 chat（冒烟测试用）")
    args = parser.parse_args(argv)

    stats = convert_beam_split(args.input_root, args.scale, args.output, args.limit)
    print(f"[convert_beam] {stats['dialogues']} dialogues, {stats['questions']} questions"
          f"（跳过 {stats['skipped']} 个损坏 chat）")
    for cat, n in sorted(stats["categories"].items()):
        print(f"  {cat}: {n}")
    print(f"[convert_beam] -> {stats['output']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
