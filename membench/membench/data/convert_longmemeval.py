"""LongMemEval 数据集转换器：将官方 JSON 转为 membench JSONL 格式。

数据源：https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
三个变体::

    longmemeval_oracle.json    ~5 MB   只含证据会话（推理上限/冒烟测试）
    longmemeval_s_cleaned.json ~250 MB 每题约 115k token、~40 个历史会话（公开榜单标准）
    longmemeval_m_cleaned.json ~2 GB   每题约 500 个会话

原始结构（每个实例一题，共 500 题）::

    {"question_id": str, "question": str, "answer": str,
     "question_type": str,            # 见 QUESTION_TYPE_MAP
     "question_date": str,            # 如 "2023/04/10 (Mon) 23:07"，500/500 题都有
     "haystack_sessions": [[{"role": "user"|"assistant", "content": str}, ...], ...],
     "haystack_dates": [str, ...],
     "haystack_session_ids": [str, ...],
     "answer_session_ids": [str, ...]}

**会话按时间排序**：``haystack_sessions`` 与 ``haystack_session_ids`` 按
``haystack_dates`` 升序对齐输出（稳定排序，同日期保持输入原序，缺日期的会话
排在最后）。原始数据里存在无规律的日期倒序（本地实测 oracle 500 题中 34 题
非升序），照原样输出会让"先发生的在后"进入 ingest，时序题与知识更新的
新/旧判断都会失真。统计里的 ``n_unsorted`` 记录输入中非升序的实例数，便于
审计排序行为。

**提问日期**：``question_date`` 归一为 ISO 8601（解析不了时保留原串并告警），
写入 ``Question.question_date``，并在问题文本末尾追加一行
``（提问日期：<ISO>）``——这是不改适配器契约的前提下让作答模型知道"现在是什么
时候"的唯一入口（时序题否则无从判断"上个月""昨天"）。

变体自动检测与 evidence 填充：

- **S/M 变体**（实例含 ``haystack_session_ids`` 且 ``answer_session_ids`` 非空）：
  对每个 ``answer_session_id``，在排序后的 ``haystack_session_ids`` 中定位索引，
  把对应 session 的全部 turn content 用 ``"\\n"`` 拍平为一条字符串写入
  ``evidence`` 列表。**不截断**：session 级证据的字符数中位数约 14k，早先按
  2000 字符截断会让证据只剩前 14%，chunk 级记忆根本不可能命中被砍掉的尾部。
  找不到的 id 跳过并计数警告（``evidence_missing``）。
- **无 session 级 id 的实例**（缺 ``haystack_session_ids`` 或
  ``answer_session_ids`` 为空）：``evidence`` 保持为空列表（向后兼容）。
  LongMemEval 没有 per-turn 证据标注；注意 oracle 变体同样带 session 级 id，
  实际会填出完整 session 文本（本地实测 500/500 题非空）——"oracle 无证据"
  只是说没有 per-turn 标注。

S 变体的用途：oracle 变体只含证据会话，是推理上限；S 变体混入 ~50 个历史
会话模拟真实检索场景。同一系统在 S 与 oracle 上的分差即检索召回 / 精确率
损失，配合填充后的 ``evidence`` 可直接用 membench 的 recall@5 量化检索质量。
session 级长证据的命中判定见 ``membench/metrics/evidence.py`` 的 containment
兜底（长证据在子串判定失败后再看"检索条目是否为证据的子串"）。

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

from .timestamps import normalize_timestamp

logger = logging.getLogger(__name__)

#: LongMemEval question_type -> membench category
QUESTION_TYPE_MAP = {
    "single-session-user": "single_hop",
    "single-session-assistant": "single_hop",
    "multi-session": "multi_hop",
    "temporal-reasoning": "temporal",
    "knowledge-update": "knowledge_update",
    "single-session-preference": "preference",
}


def _order_sessions(inst: dict, qid: str) -> tuple[list[list[dict]], list[str | None], int]:
    """按 ``haystack_dates`` 升序对齐 session 与 session_id。

    Args:
        inst: 原始 LongMemEval 实例。
        qid: 题目 id（用于时间戳报错的来源标注）。

    Returns:
        ``(sessions, session_ids, n_unsorted)``；``session_ids`` 与 ``sessions``
        等长（原始 id 缺失的会话为 ``None``），``n_unsorted`` 为输入顺序与
        日期升序不一致的实例数（0 或 1）。日期缺失的会话排在最后。
    """
    raw_sessions = inst.get("haystack_sessions") or []
    dates = inst.get("haystack_dates") or []
    raw_sids = inst.get("haystack_session_ids") or []

    entries: list[tuple[str | None, int, str | None, list[dict]]] = []
    for idx, session in enumerate(raw_sessions):
        ts = normalize_timestamp(
            dates[idx] if idx < len(dates) else None,
            where=f"{qid} haystack_dates[{idx}]",
        )
        sid = str(raw_sids[idx]) if idx < len(raw_sids) else None
        entries.append((ts, idx, sid, session))

    n_unsorted = int(
        [(ts is None, ts or "") for ts, _, _, _ in entries]
        != sorted((ts is None, ts or "") for ts, _, _, _ in entries)
    )
    # 稳定排序：同日期保持输入原序；缺日期（None）排最后。
    ordered = sorted(entries, key=lambda e: (e[0] is None, e[0] or ""))

    sessions = [
        [
            {
                "role": turn.get("role", "user"),
                "content": str(turn.get("content", "")),
                "timestamp": ts,
            }
            for turn in session
        ]
        for ts, _, _, session in ordered
    ]
    sids = [sid for _, _, sid, _ in ordered]
    return sessions, sids, n_unsorted


def _extract_evidence(
    sessions: list[list[dict]], sids: list[str | None], inst: dict
) -> tuple[list[str], int]:
    """从 S/M 变体实例解析 session 级证据（完整 session 文本，不截断）。

    对每个 ``answer_session_id`` 在排好序的 ``haystack_session_ids`` 中定位
    索引，将对应 session 的 turn content 用换行拍平成一条字符串。

    Args:
        sessions: 已按日期排序的 session 列表。
        sids: 与 ``sessions`` 等长的 session id 列表。
        inst: 原始 LongMemEval 实例。

    Returns:
        tuple[list[str], int]: (evidence 列表, 未匹配的 answer_session_id 数)。
    """
    answer_sids: list[str] = inst.get("answer_session_ids") or []
    if not answer_sids or not any(sid is not None for sid in sids):
        return [], 0

    evidence: list[str] = []
    n_missing = 0
    for asid in answer_sids:
        try:
            idx = sids.index(str(asid))
        except ValueError:
            n_missing += 1
            continue
        evidence.append(
            "\n".join(str(turn.get("content", "")) for turn in sessions[idx])
        )
    return evidence, n_missing


def _normalize_question_date(raw: object, qid: str) -> str | None:
    """归一 ``question_date``；形态不认识时保留原串（告警，不中断）。

    提问日期是辅助信息，不该像会话时间戳那样让 ``dataset fetch`` 失败。
    """
    text = str(raw).strip() if raw is not None else ""
    if not text:
        return None
    try:
        return normalize_timestamp(text, where=f"{qid} question_date")
    except ValueError:
        logger.warning("%s: question_date=%r 形态不认识，保留原串", qid, text)
        return text


def convert_longmemeval(input_path: str, output_path: str, limit: int | None = None) -> dict:
    """执行转换，返回统计信息。

    含 session 级 id 的实例解析 evidence；缺 id 或 answer_session_ids 为空的
    实例保持空列表。统计含 ``evidence_filled``（成功填充 evidence 的
    题目数）、``evidence_missing``（answer_session_ids 非空但存在未匹配
    id 的题目数）、``n_unsorted``（输入顺序与日期升序不一致的实例数）与
    ``question_date_filled``（带提问日期的题目数）。
    """
    instances = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if limit is not None:
        instances = instances[:limit]

    n_questions = 0
    n_evidence_filled = 0
    n_evidence_missing = 0
    n_unsorted = 0
    n_question_date = 0
    category_counts: dict[str, int] = {}
    lines: list[str] = []

    for inst in instances:
        qtype = inst.get("question_type")
        category = QUESTION_TYPE_MAP.get(qtype)
        if category is None:
            raise ValueError(f"{inst.get('question_id')}: 未知 question_type={qtype!r}")

        qid = str(inst.get("question_id", f"lme_{len(lines)}"))
        sessions, sids, unsorted = _order_sessions(inst, qid)
        n_unsorted += unsorted

        evidence, n_unmatched = _extract_evidence(sessions, sids, inst)
        if evidence:
            n_evidence_filled += 1
        if n_unmatched:
            n_evidence_missing += 1
            logger.warning(
                "%s: %d 个 answer_session_id 在 haystack_session_ids 中未找到，跳过",
                qid,
                n_unmatched,
            )

        question_text = str(inst["question"])
        question_date = _normalize_question_date(inst.get("question_date"), qid)
        if question_date is not None:
            n_question_date += 1
            question_text = f"{question_text}\n（提问日期：{question_date}）"

        questions = [
            {
                "qid": qid,
                "question": question_text,
                "gold_answer": str(inst["answer"]),
                "category": category,
                "evidence": evidence,  # S/M 变体为完整 session 级证据；oracle 变体为空
                "question_date": question_date,
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
        "n_unsorted": n_unsorted,
        "question_date_filled": n_question_date,
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
    print(
        f"[convert_longmemeval] n_unsorted={stats['n_unsorted']}"
        f"（输入会话日期非升序的题，已按日期升序输出）, "
        f"question_date_filled={stats['question_date_filled']}"
    )
    print(f"[convert_longmemeval] -> {stats['output']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
