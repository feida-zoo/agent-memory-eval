"""LongMemEval 数据集转换器：将官方 JSON 转为 membench JSONL 格式。

数据源：https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
三个变体::

    longmemeval_oracle.json    ~5 MB   只含证据会话（推理上限/冒烟测试）
    longmemeval_s_cleaned.json ~250 MB 每题约 115k token、~40 个历史会话（公开榜单标准）
    longmemeval_m_cleaned.json ~2 GB   每题约 500 个会话

原始结构（每个实例一题，共 500 题）::

    {"question_id": str, "question": str, "answer": str,
     "question_type": str,            # 见 QUESTION_TYPE_MAP
     "haystack_sessions": [[{"role": "user"|"assistant", "content": str}, ...], ...],
     "haystack_dates": [str, ...],
     "haystack_session_ids": [str, ...],   # 仅 S/M 变体；oracle 变体没有
     "answer_session_ids": [str, ...]}

变体自动检测与 evidence 填充：
- **S/M 变体**（实例含 ``haystack_session_ids`` 且 ``answer_session_ids`` 非空）：
  对每个 ``answer_session_id``，在 ``haystack_session_ids`` 中定位索引，把对应
  session 的全部 turn content 用 ``"\\n"`` 拍平为一条字符串（超过
  ``EVIDENCE_MAX_CHARS`` 截断防爆）写入 ``evidence`` 列表。找不到的 id 跳过并
  计数警告（``evidence_missing``）。
- **oracle 变体**（无 ``haystack_session_ids`` 或 ``answer_session_ids`` 为空）：
  LongMemEval 没有 per-turn 证据标注，``evidence`` 保持为空列表（向后兼容）。

S 变体的用途：oracle 变体只含证据会话，是推理上限；S 变体混入 ~50 个历史
会话模拟真实检索场景。同一系统在 S 与 oracle 上的分差即检索召回 / 精确率
损失，配合填充后的 ``evidence`` 可直接用 membench 的 recall@5 量化检索质量。

注意事项：
- knowledge-update 题不提供"过期答案"，gold_stale 无法填充，stale_rate 在该
  基准上恒为 0 —— 测陈旧召回请使用 membench 自建黄金集或 LoCoMo。
- cleaned 版不含 abstention 题型。

用法::

    python -m membench.data.convert_longmemeval longmemeval_s_cleaned.json -o lme_s.jsonl [--limit 50]
    python -m membench.data.convert_longmemeval longmemeval_oracle.json -o lme_oracle.jsonl [--limit 50]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

#: 单条 evidence 的最大字符数（超出截断，防止超长会话撑爆证据字段）。
EVIDENCE_MAX_CHARS = 2000

#: LongMemEval question_type -> membench category
QUESTION_TYPE_MAP = {
    "single-session-user": "single_hop",
    "single-session-assistant": "single_hop",
    "multi-session": "multi_hop",
    "temporal-reasoning": "temporal",
    "knowledge-update": "knowledge_update",
    "single-session-preference": "preference",
}


def _extract_evidence(inst: dict) -> tuple[list[str], int]:
    """从 S/M 变体实例解析 session 级证据。

    对每个 ``answer_session_id`` 在 ``haystack_session_ids`` 中定位索引，
    将对应 session 的 turn content 用换行拍平成一条字符串（超过
    ``EVIDENCE_MAX_CHARS`` 截断）。找不到的 id 跳过并计入返回值。

    Args:
        inst: 原始 LongMemEval 实例。

    Returns:
        tuple[list[str], int]: (evidence 列表, 未匹配的 answer_session_id 数)。
    """
    sids: list[str] = inst.get("haystack_session_ids") or []
    answer_sids: list[str] = inst.get("answer_session_ids") or []
    if not sids or not answer_sids:
        return [], 0

    sessions: list[list[dict]] = inst.get("haystack_sessions") or []
    evidence: list[str] = []
    n_missing = 0
    for asid in answer_sids:
        try:
            idx = sids.index(asid)
        except ValueError:
            n_missing += 1
            continue
        if idx >= len(sessions):
            n_missing += 1
            continue
        flat = "\n".join(str(turn.get("content", "")) for turn in sessions[idx])
        evidence.append(flat[:EVIDENCE_MAX_CHARS])
    return evidence, n_missing


def convert_longmemeval(input_path: str, output_path: str, limit: int | None = None) -> dict:
    """执行转换，返回统计信息。

    自动检测变体：S/M 变体解析 session 级 evidence，oracle 变体保持
    evidence 为空列表。统计含 ``evidence_filled``（成功填充 evidence 的
    题目数）与 ``evidence_missing``（answer_session_ids 非空但存在未匹配
    id 的题目数）。
    """
    instances = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if limit is not None:
        instances = instances[:limit]

    n_questions = 0
    n_evidence_filled = 0
    n_evidence_missing = 0
    category_counts: dict[str, int] = {}
    lines: list[str] = []

    for inst in instances:
        qtype = inst.get("question_type")
        category = QUESTION_TYPE_MAP.get(qtype)
        if category is None:
            raise ValueError(f"{inst.get('question_id')}: 未知 question_type={qtype!r}")

        dates = inst.get("haystack_dates") or []
        sessions: list[list[dict]] = []
        for idx, session in enumerate(inst.get("haystack_sessions") or []):
            ts = dates[idx] if idx < len(dates) else None
            sessions.append(
                [
                    {
                        "role": turn.get("role", "user"),
                        "content": str(turn.get("content", "")),
                        "timestamp": ts,
                    }
                    for turn in session
                ]
            )

        evidence, n_unmatched = _extract_evidence(inst)
        if evidence:
            n_evidence_filled += 1
        if n_unmatched:
            n_evidence_missing += 1
            logger.warning(
                "%s: %d 个 answer_session_id 在 haystack_session_ids 中未找到，跳过",
                inst.get("question_id"),
                n_unmatched,
            )

        qid = str(inst.get("question_id", f"lme_{len(lines)}"))
        questions = [
            {
                "qid": qid,
                "question": str(inst["question"]),
                "gold_answer": str(inst["answer"]),
                "category": category,
                "evidence": evidence,  # S/M 变体为 session 级证据；oracle 变体为空
            }
        ]
        n_questions += 1
        category_counts[category] = category_counts.get(category, 0) + 1

        lines.append(
            json.dumps(
                {"dialogue_id": qid, "sessions": sessions, "questions": questions},
                ensure_ascii=False,
            )
        )

    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "dialogues": len(lines),
        "questions": n_questions,
        "categories": category_counts,
        "evidence_filled": n_evidence_filled,
        "evidence_missing": n_evidence_missing,
        "output": output_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="将 LongMemEval 官方 JSON 转为 membench JSONL")
    parser.add_argument("input", help="longmemeval_*.json 路径")
    parser.add_argument("-o", "--output", required=True, help="输出 JSONL 路径")
    parser.add_argument("--limit", type=int, default=None, help="只转换前 N 题（冒烟测试用）")
    args = parser.parse_args(argv)

    stats = convert_longmemeval(args.input, args.output, args.limit)
    print(f"[convert_longmemeval] {stats['dialogues']} dialogues, {stats['questions']} questions")
    for cat, n in sorted(stats["categories"].items()):
        print(f"  {cat}: {n}")
    print(
        f"[convert_longmemeval] evidence_filled={stats['evidence_filled']}, "
        f"evidence_missing={stats['evidence_missing']}"
    )
    print(f"[convert_longmemeval] -> {stats['output']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
