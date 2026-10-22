"""LoCoMo 数据集转换器：将官方 locomo10.json 转为 membench JSONL 格式。

数据源：https://github.com/snap-research/locomo （data/locomo10.json, 10 个长对话样本）

LoCoMo 原始结构::

    [{"sample_id": str,
      "conversation": {"speaker_a": str, "speaker_b": str,
                       "session_1_date_time": str, "session_1": [turn, ...], ...},
      "qa": [{"question": str, "answer": str | int | None,
              "evidence": ["D1:3", ...], "category": int}, ...]}]

turn = {"speaker": str, "dia_id": "D1:3", "text": str}

题型映射（LoCoMo 论文章节 3.2）::

    1 -> multi_hop        多跳推理
    2 -> temporal         时序推理
    3 -> open_ended       单跳+常识/世界知识
    4 -> single_hop       单跳事实
    5 -> abstention       对抗题（对话中无答案，answer 为 None,
                          正确行为是拒答；adversarial_answer 是"看似合理"的错误答案）

用法::

    python -m membench.data.convert_locomo locomo10.json -o locomo.jsonl [--max-dialogues 10]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .timestamps import normalize_timestamp

#: LoCoMo category int -> membench category str
CATEGORY_MAP = {
    1: "multi_hop",
    2: "temporal",
    3: "open_ended",
    4: "single_hop",
    5: "abstention",
}

#: membench 拒答哨兵（与 schema.py 约定一致）
ABSTAIN_SENTINEL = "<ABSTAIN>"


def _resolve_evidence(conversation: dict, dia_ids: list[str]) -> list[str]:
    """把 LoCoMo 的 evidence dia_id（如 "D2:3"）解析为对应发言原文。

    dia_id 格式为 ``D{session_idx}:{turn_idx}``，turn_idx 从 1 开始。
    解析失败的 id 静默跳过（数据中存在少量越界引用）。
    """
    texts: list[str] = []
    for dia_id in dia_ids or []:
        try:
            head, tail = dia_id.split(":")
            session_idx = int(head.lstrip("D"))
            turn_idx = int(tail)
            session = conversation[f"session_{session_idx}"]
            texts.append(str(session[turn_idx - 1]["text"]))
        except (KeyError, IndexError, ValueError, TypeError):
            continue
    return texts


def convert_locomo(input_path: str, output_path: str, max_dialogues: int | None = None) -> dict:
    """执行转换，返回统计信息。"""
    samples = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if max_dialogues is not None:
        samples = samples[:max_dialogues]

    n_questions = 0
    category_counts: dict[str, int] = {}
    lines: list[str] = []

    for sample in samples:
        conv = sample["conversation"]
        speaker_a = conv.get("speaker_a", "speaker_a")

        # 会话：session_N / session_N_date_time 成对出现，按 N 排序
        session_ids = sorted(
            int(k[len("session_"):])
            for k in conv
            if k.startswith("session_") and not k.endswith("_date_time")
        )
        sessions: list[list[dict]] = []
        for sid in session_ids:
            ts = normalize_timestamp(
                conv.get(f"session_{sid}_date_time"),
                where=f"{sample.get('sample_id', '?')} session_{sid}",
            )
            turns = [
                {
                    "role": "user" if t.get("speaker") == speaker_a else "assistant",
                    "content": str(t.get("text", "")),
                    "timestamp": ts,
                }
                for t in conv[f"session_{sid}"]
            ]
            sessions.append(turns)

        questions: list[dict] = []
        for i, qa in enumerate(sample.get("qa", [])):
            cat_int = qa.get("category")
            category = CATEGORY_MAP.get(cat_int)
            if category is None:
                raise ValueError(f"{sample.get('sample_id')} qa[{i}]: 未知 category={cat_int}")

            answer = qa.get("answer")
            if category == "abstention" or answer is None:
                gold = ABSTAIN_SENTINEL
            else:
                gold = str(answer)

            questions.append(
                {
                    "qid": f"{sample.get('sample_id', 'locomo')}_q{i + 1}",
                    "question": str(qa["question"]),
                    "gold_answer": gold,
                    "category": category,
                    "evidence": _resolve_evidence(conv, qa.get("evidence") or []),
                }
            )
            n_questions += 1
            category_counts[category] = category_counts.get(category, 0) + 1

        lines.append(
            json.dumps(
                {
                    "dialogue_id": str(sample.get("sample_id", f"locomo_{len(lines)}")),
                    "sessions": sessions,
                    "questions": questions,
                },
                ensure_ascii=False,
            )
        )

    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "dialogues": len(lines),
        "questions": n_questions,
        "categories": category_counts,
        "output": output_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="将 LoCoMo locomo10.json 转为 membench JSONL")
    parser.add_argument("input", help="locomo10.json 路径")
    parser.add_argument("-o", "--output", required=True, help="输出 JSONL 路径")
    parser.add_argument("--max-dialogues", type=int, default=None, help="只转换前 N 个对话（调试用）")
    args = parser.parse_args(argv)

    stats = convert_locomo(args.input, args.output, args.max_dialogues)
    print(f"[convert_locomo] {stats['dialogues']} dialogues, {stats['questions']} questions")
    for cat, n in sorted(stats["categories"].items()):
        print(f"  {cat}: {n}")
    print(f"[convert_locomo] -> {stats['output']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
