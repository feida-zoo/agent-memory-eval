"""数据集 Schema 与 JSONL 加载器（SPEC 3.2）。

数据集为 JSONL 文件，每行一个 dialogue::

    {
      "dialogue_id": "d001",
      "sessions": [[{"role": "user", "content": "...", "timestamp": "..."}], ...],
      "questions": [
        {"qid": "d001_q1", "question": "...", "gold_answer": "...",
         "category": "single_hop | ...", "evidence": ["..."], "gold_stale": "...",
         "ignore": false, "ignore_reason": "..."}
      ]
    }

其中 ``ignore`` / ``ignore_reason`` 为可选的题目级忽略标记：``ignore=true``
的题在评测时被整条跳过（不查询、不计分），但数据保留在文件中可追溯。

``timestamp`` 约定为 **ISO 8601、精度可变**（``YYYY-MM-DD`` / ``YYYY-MM-DDTHH:MM`` /
``YYYY-MM-DDTHH:MM:SS``，可省缺）：加载器只保证它是字符串，不理解时间语义，
原样透传给 ``ingest``；公开数据集的转换器统一按此约定输出（见
``membench/data/timestamps.py``），用户自建集请直接写 ISO。

加载器做字段校验，缺字段 / 非法取值时报错并给出**行号**。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

#: 题目类别"软约定"全集（SPEC 3.2，7 种）。
#: 不同数据集设计意图不同，category 集合本来就该不重合——locomo 5 /
#: longmemeval 6 / gatemem 4 / BEAM 10 各自保留自己的 raw category；
#: loader 不再硬校验 ∈ CATEGORIES。CATEGORIES 仍保留为"通用约定"用于
#: adapter 文档、check_comparability 等需要"列出 membench 通用类型"的
#: 公共逻辑；convert 阶段不再做多对一合并。
CATEGORIES: tuple[str, ...] = (
    "single_hop",
    "multi_hop",
    "temporal",
    "knowledge_update",
    "abstention",
    "preference",
    "open_ended",
)

#: abstention 题的 gold_answer 哨兵值：表示期望系统拒答。
ABSTAIN_SENTINEL = "<ABSTAIN>"


def is_abstention(category: str, gold_answer: str) -> bool:
    """是否「期望拒答」题：category 为 abstention 或 gold 为哨兵值。

    数据集保留原始题型后（如 gatemem 的 privacy_refuse / safety_no_memory），
    拒答判定不能再只认 ``category == "abstention"``——一律以 gold 哨兵值为准，
    category 仅作兼容兜底。
    """
    return category == "abstention" or gold_answer == ABSTAIN_SENTINEL


def question_capabilities(questions: Iterable[Question]) -> dict[str, bool]:
    """扫描题目集合，得出能力标注 has_evidence / has_stale / has_abstention。

    ignore=true 的题不参与评分，能力标注同样排除它们；runner 与
    :func:`membench.evaluate.scan_capabilities` 共享同一口径，避免两处
    扫描语义漂移。
    """
    has_evidence = False
    has_stale = False
    has_abstention = False
    for q in questions:
        if q.ignore:
            continue
        if q.evidence:
            has_evidence = True
        if q.gold_stale:
            has_stale = True
        if is_abstention(q.category, q.gold_answer):
            has_abstention = True
    return {
        "has_evidence": has_evidence,
        "has_stale": has_stale,
        "has_abstention": has_abstention,
    }


#: 消息允许的 role 取值。
_VALID_ROLES = ("user", "assistant")


@dataclass
class Question:
    """一道评测题。

    Attributes:
        qid: 题目唯一标识。
        question: 问题文本。
        gold_answer: 标准答案；abstention 题固定为 ``"<ABSTAIN>"``。
        category: 题目类别，必须是 ``CATEGORIES`` 之一。
        evidence: 支持证据（对话原文子串），用于检索召回判定。
        gold_stale: 仅 knowledge_update 题使用：过期答案，命中即陈旧。
        ignore: 题目级忽略标记；True 时评测引擎整条跳过该题
            （不查询、不计分），数据仍保留在文件中可追溯。
        ignore_reason: 忽略原因（可选说明，供追溯）。
    """

    qid: str
    question: str
    gold_answer: str
    category: str
    evidence: list[str] = field(default_factory=list)
    gold_stale: str | None = None
    ignore: bool = False
    ignore_reason: str | None = None


@dataclass
class Dialogue:
    """一段多 session 对话及其挂接的题目。

    Attributes:
        dialogue_id: 对话唯一标识。
        sessions: session 列表，每个 session 是消息列表；
            消息形如 ``{"role": str, "content": str, "timestamp": str|None}``。
        questions: 挂接在该对话上的评测题。
    """

    dialogue_id: str
    sessions: list[list[dict]]
    questions: list[Question] = field(default_factory=list)


@dataclass
class Dataset:
    """数据集容器：名称 + dialogue 列表。"""

    name: str
    dialogues: list[Dialogue] = field(default_factory=list)

    @property
    def n_questions(self) -> int:
        """题目总数。"""
        return sum(len(d.questions) for d in self.dialogues)


# ---------------------------------------------------------------------------
# 加载与校验
# ---------------------------------------------------------------------------


def _err(path: Path, lineno: int, msg: str) -> ValueError:
    """构造带文件路径与行号的 ValueError。"""
    return ValueError(f"{path}:{lineno}: {msg}")


def _parse_question(raw: object, path: Path, lineno: int) -> Question:
    """解析并校验单条 question 记录。"""
    if not isinstance(raw, dict):
        raise _err(path, lineno, "question 必须是 JSON 对象")
    for key in ("qid", "question", "gold_answer", "category", "evidence"):
        if key not in raw:
            raise _err(path, lineno, f"question 缺少必填字段 '{key}'")
    category = raw["category"]
    if not isinstance(category, str) or not category.strip():
        raise _err(path, lineno, "category 必须是非空字符串")
    # 注：不再硬校验 category ∈ CATEGORIES。
    # 各数据集（BEAM 10 类 / gatemem 4 类 / locomo 5 类 / longmemeval 6 类）
    # 保留自己的 raw category；CATEGORIES 仅作"通用约定"用于公共逻辑。
    evidence = raw["evidence"]
    if not isinstance(evidence, list) or not all(isinstance(e, str) for e in evidence):
        raise _err(path, lineno, "evidence 必须是字符串列表")
    gold_stale = raw.get("gold_stale")
    if gold_stale is not None and not isinstance(gold_stale, str):
        raise _err(path, lineno, "gold_stale 必须是字符串")
    # 可选的题目级忽略标记：缺省视为不忽略（向后兼容旧数据集）。
    ignore = raw.get("ignore", False)
    if not isinstance(ignore, bool):
        raise _err(path, lineno, "ignore 必须是 bool")
    ignore_reason = raw.get("ignore_reason")
    if ignore_reason is not None and not isinstance(ignore_reason, str):
        raise _err(path, lineno, "ignore_reason 必须是字符串或 null")
    return Question(
        qid=str(raw["qid"]),
        question=str(raw["question"]),
        gold_answer=str(raw["gold_answer"]),
        category=category,
        evidence=list(evidence),
        gold_stale=gold_stale,
        ignore=ignore,
        ignore_reason=ignore_reason,
    )


def _parse_dialogue(raw: object, path: Path, lineno: int) -> Dialogue:
    """解析并校验单行 dialogue 记录。"""
    if not isinstance(raw, dict):
        raise _err(path, lineno, "每行必须是 JSON 对象")
    for key in ("dialogue_id", "sessions", "questions"):
        if key not in raw:
            raise _err(path, lineno, f"dialogue 缺少必填字段 '{key}'")

    sessions = raw["sessions"]
    if not isinstance(sessions, list):
        raise _err(path, lineno, "sessions 必须是列表")
    for si, session in enumerate(sessions):
        if not isinstance(session, list):
            raise _err(path, lineno, f"sessions[{si}] 必须是消息列表")
        for mi, msg in enumerate(session):
            if not isinstance(msg, dict):
                raise _err(path, lineno, f"sessions[{si}][{mi}] 必须是对象")
            for key in ("role", "content"):
                if key not in msg:
                    raise _err(
                        path,
                        lineno,
                        f"sessions[{si}][{mi}] 缺少必填字段 '{key}'",
                    )
            if msg["role"] not in _VALID_ROLES:
                raise _err(
                    path,
                    lineno,
                    f"sessions[{si}][{mi}] 的 role 非法：'{msg['role']}'"
                    f"（合法取值 {list(_VALID_ROLES)}）",
                )
            msg.setdefault("timestamp", None)

    questions_raw = raw["questions"]
    if not isinstance(questions_raw, list):
        raise _err(path, lineno, "questions 必须是列表")
    questions = [_parse_question(q, path, lineno) for q in questions_raw]

    return Dialogue(
        dialogue_id=str(raw["dialogue_id"]),
        sessions=sessions,
        questions=questions,
    )


def load_dataset(path: str | Path) -> list[Dialogue]:
    """加载 JSONL 数据集。

    Args:
        path: 数据集文件路径（每行一个 dialogue 的 JSONL）。

    Returns:
        list[Dialogue]: 解析并校验通过的 dialogue 列表。

    Raises:
        ValueError: 字段缺失 / 类型非法 / category 非法时抛出，错误信息带行号。
        FileNotFoundError: 文件不存在。
    """
    p = Path(path)
    dialogues: list[Dialogue] = []
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise _err(p, lineno, f"非法 JSON：{exc}") from exc
            dialogues.append(_parse_dialogue(raw, p, lineno))
    return dialogues


def dataset_fingerprint(path: str | Path) -> dict:
    """数据集指纹：内容 sha256（前 16 位十六进制）+ 字节数 + 文件 basename。

    用于 A/B 可比性前置校验：数据集变了但没重跑基线时，
    ``compare`` 能凭指纹机检发现，而不是静默给 PASS。

    Args:
        path: 数据集 JSONL 路径。

    Returns:
        ``{"sha256": str|None, "bytes": int|None, "name": str}``；文件不存在时
        ``sha256`` / ``bytes`` 为 None（不抛异常，交由调用方决定严重性）。
    """
    p = Path(path)
    if not p.is_file():
        return {"sha256": None, "bytes": None, "name": p.name}
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return {
        "sha256": digest.hexdigest()[:16],
        "bytes": os.path.getsize(p),
        "name": p.name,
    }
