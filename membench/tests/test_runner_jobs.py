"""--jobs 题目级并发 的回归测试（修复 3）。

要点：
- ingest 阶段始终串行（记录调用顺序验证）；
- jobs=4 与 jobs=1 的 results.jsonl 记录顺序一致、metrics 一致；
- jobs<1 报错。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from membench.adapters.base import BaseMemoryAdapter, IngestStats, QueryResult
from membench.runner import Runner


def _write_mini_dataset(path: Path) -> None:
    """写一个 2 dialogue × 3 题的迷你数据集。"""
    lines = []
    for di in range(1, 3):
        record = {
            "dialogue_id": f"d{di:03d}",
            "sessions": [
                [
                    {"role": "user", "content": f"我住在城市{di}号。", "timestamp": "2025-01-05T10:00:00"},
                    {"role": "assistant", "content": "好的。", "timestamp": "2025-01-05T10:00:10"},
                ],
                [
                    {"role": "user", "content": f"我的狗叫旺财{di}。", "timestamp": "2025-01-06T10:00:00"},
                ],
            ],
            "questions": [
                {
                    "qid": f"q{di}_{qi}",
                    "question": f"问题 {di}-{qi}？",
                    "gold_answer": f"答案{di}{qi}",
                    "category": "single_hop",
                    "evidence": [],
                }
                for qi in range(1, 4)
            ],
        }
        lines.append(json.dumps(record, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class RecordingAdapter(BaseMemoryAdapter):
    """记录 ingest 调用顺序（含线程名）的测试适配器；query 线程安全。"""

    name = "recording"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.ingest_calls: list[tuple[str, str]] = []  # (thread_name, first_content)
        self._messages: list[dict] = []

    def reset(self) -> None:
        self._messages = []

    def ingest(self, messages: list[dict]) -> IngestStats:
        with self._lock:
            self.ingest_calls.append(
                (threading.current_thread().name, messages[0].get("content", ""))
            )
        self._messages.extend(messages)
        return IngestStats()

    def query(self, question: str) -> QueryResult:
        t0 = time.perf_counter()
        # 确定性回答：从题号推导 gold（"问题 1-2？" -> "答案12"）
        digits = "".join(ch for ch in question if ch.isdigit())
        answer = "答案" + digits if digits else "不知道"
        # 加一点耗时让并发真正交错
        time.sleep(0.005)
        return QueryResult(
            answer=answer,
            retrieved=[],
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )


def _run_once(out_dir: Path, dataset: Path, jobs: int) -> tuple[Path, RecordingAdapter]:
    adapter = RecordingAdapter()
    runner = Runner(adapter, str(dataset), judge="heuristic", out_dir=str(out_dir), jobs=jobs)
    run_dir = Path(runner.run())
    return run_dir, adapter


def test_jobs4_matches_jobs1(tmp_path: Path) -> None:
    """jobs=4 与 jobs=1：results.jsonl 记录顺序一致、metrics 一致。"""
    dataset = tmp_path / "mini.jsonl"
    _write_mini_dataset(dataset)

    run_dir_1, _ = _run_once(tmp_path / "out1", dataset, jobs=1)
    run_dir_4, _ = _run_once(tmp_path / "out4", dataset, jobs=4)

    records_1 = [
        json.loads(line)
        for line in (run_dir_1 / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    records_4 = [
        json.loads(line)
        for line in (run_dir_4 / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [r["qid"] for r in records_1] == [r["qid"] for r in records_4]
    # latency_ms 是单题墙钟时间，并发调度下允许有微小差异，逐字段对比时剔除
    strip = lambda rs: [{k: v for k, v in r.items() if k != "latency_ms"} for r in rs]
    assert strip(records_1) == strip(records_4)

    metrics_1 = json.loads((run_dir_1 / "metrics.json").read_text(encoding="utf-8"))
    metrics_4 = json.loads((run_dir_4 / "metrics.json").read_text(encoding="utf-8"))
    # 去掉 run_id / 路径 / 时间戳等运行相关字段后应完全一致
    for m in (metrics_1, metrics_4):
        for key in ("run_id", "dataset_path", "timestamp"):
            m.pop(key, None)
        # 延迟分位数随并发调度波动，不属于一致性口径
        for key in ("latency_p50_ms", "latency_p95_ms"):
            m.get("overall", {}).pop(key, None)
    assert metrics_1 == metrics_4


def test_ingest_is_serial(tmp_path: Path) -> None:
    """jobs=4 时 ingest 仍在主线程按 session 顺序串行执行。"""
    dataset = tmp_path / "mini.jsonl"
    _write_mini_dataset(dataset)
    _, adapter = _run_once(tmp_path / "out", dataset, jobs=4)

    assert len(adapter.ingest_calls) == 4  # 2 dialogues × 2 sessions
    assert all(thread == "MainThread" for thread, _ in adapter.ingest_calls)
    contents = [content for _, content in adapter.ingest_calls]
    assert contents == [
        "我住在城市1号。", "我的狗叫旺财1。", "我住在城市2号。", "我的狗叫旺财2。",
    ]


def test_jobs_zero_rejected(tmp_path: Path) -> None:
    """jobs=0 报错。"""
    dataset = tmp_path / "mini.jsonl"
    _write_mini_dataset(dataset)
    with pytest.raises(ValueError, match="jobs"):
        Runner(RecordingAdapter(), str(dataset), judge="heuristic",
               out_dir=str(tmp_path / "out"), jobs=0)
