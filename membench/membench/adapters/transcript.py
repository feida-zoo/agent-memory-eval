"""TranscriptAdapter：离线日志回放适配器。

用于对已经跑过一遍的系统做**离线回放评测**：事先把该系统在数据集上的
``问题 -> 回答`` 记录成 JSON/JSONL 文件，评测时不再调用真实系统，
而是按问题精确匹配回放答案，保证结果可复现、零成本重跑。

转录文件格式（JSONL，每行一条）::

    {"question": "用户的狗叫什么名字？", "answer": "豆豆", "retrieved": ["..."]}

其中 ``retrieved`` 可省略（缺省为空列表）。
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .base import BaseMemoryAdapter, IngestStats, QueryResult, estimate_tokens

#: LongMemEval 转换器追加在问题文本**行尾**的提问日期后缀（见
#: ``membench.data.convert_longmemeval``）：它是展示用的时间提示，不是问题
#: 本身。转录文件很可能是在后缀引入前录制的（键里没有该行），匹配前先剥掉。
_QUESTION_DATE_SUFFIX_RE = re.compile(r"\n（提问日期：[^）]*）$")


def strip_question_date(question: str) -> str:
    """剥掉问题文本行尾的 ``\\n（提问日期：…）`` 后缀；没有后缀原样返回。"""
    return _QUESTION_DATE_SUFFIX_RE.sub("", question or "")


class TranscriptAdapter(BaseMemoryAdapter):
    """离线日志回放适配器：ingest 仅记录消息，query 按问题精确匹配回放。

    匹配不到的问题返回空答案并视为未命中（retrieved 为空列表）。
    """

    name = "transcript"

    def __init__(self, transcript_path: str | Path) -> None:
        """从 JSONL 转录文件加载 ``question -> {answer, retrieved}`` 映射。

        Args:
            transcript_path: 转录文件路径（JSONL）。

        Raises:
            ValueError: 文件格式非法（缺 question/answer 字段）时抛出，带行号。
        """
        self._records: dict[str, dict] = {}
        path = Path(transcript_path)
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: 非法 JSON: {exc}") from exc
                for key in ("question", "answer"):
                    if key not in obj:
                        raise ValueError(f"{path}:{lineno}: 缺少必填字段 '{key}'")
                self._records[obj["question"]] = {
                    "answer": obj["answer"],
                    "retrieved": list(obj.get("retrieved", [])),
                }
        self._ingested_chars = 0

    def reset(self) -> None:
        """回放适配器无内部记忆状态，仅重置统计计数。"""
        self._ingested_chars = 0

    def ingest(self, messages: list[dict]) -> IngestStats:
        """记录写入量（回放模式下不产生任何副作用）。"""
        t0 = time.perf_counter()
        self._ingested_chars += sum(len(m.get("content", "")) for m in messages)
        return IngestStats(
            input_tokens=estimate_tokens("".join(m.get("content", "") for m in messages)),
            llm_calls=0,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def query(self, question: str) -> QueryResult:
        """按问题精确匹配回放记录；未命中返回空答案。

        匹配前先剥掉行尾的提问日期后缀（LongMemEval 转换器追加的展示用时间
        提示）；转录文件本身若按带后缀的原文录制，再回退到原问题查找——两种
        录制时点都能回放。
        """
        t0 = time.perf_counter()
        stripped = strip_question_date(question)
        rec = self._records.get(stripped)
        if rec is None and stripped != question:
            rec = self._records.get(question)
        if rec is None:
            answer, retrieved = "", []
        else:
            answer, retrieved = rec["answer"], list(rec["retrieved"])
        return QueryResult(
            answer=answer,
            retrieved=retrieved,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            input_tokens=estimate_tokens("".join(retrieved)),
            output_tokens=estimate_tokens(answer),
            llm_calls=0,
        )
