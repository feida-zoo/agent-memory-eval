"""--jobs 题目级并发 的回归测试（修复 3）。

要点：
- ingest 阶段始终串行（记录调用顺序验证）；
- jobs=4 与 jobs=1 的数据集级与对话级产物记录顺序一致、metrics 一致；
- jobs<1 报错。

产物落位见 ``membench/layout.py``：``Runner(out_dir=<实验目录>)``，
``run()`` 返回 ``<实验目录>/runs/<dataset>``，报告在其 ``report/`` 下。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from membench import layout
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


def _run_once(experiment_dir: Path, dataset: Path, jobs: int) -> tuple[Path, RecordingAdapter]:
    """跑一场评测，返回 ``(数据集目录, 适配器)``。"""
    adapter = RecordingAdapter()
    runner = Runner(
        adapter, str(dataset), judge="heuristic", out_dir=str(experiment_dir), jobs=jobs
    )
    return Path(runner.run()), adapter


def _read_records(path: Path) -> list[dict]:
    """读逐题明细（results.jsonl）。"""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _strip_latency(records: list[dict]) -> list[dict]:
    """剔除单题墙钟延迟：并发调度下允许微差，不属于一致性口径。"""
    return [{k: v for k, v in r.items() if k != "latency_ms"} for r in records]


def _normalize_metrics(metrics: dict) -> dict:
    """剔除身份 / 路径 / 时间戳 / 延迟分位后的可比口径。"""
    normalized = json.loads(json.dumps(metrics))
    for key in ("run_id", "dataset_path", "timestamp", "experiment_dir", "dataset_dir"):
        normalized.pop(key, None)
    for key in ("latency_p50_ms", "latency_p95_ms"):
        normalized.get("overall", {}).pop(key, None)
    return normalized


def test_jobs4_matches_jobs1(tmp_path: Path) -> None:
    """jobs=4 与 jobs=1：数据集级与对话级产物的记录顺序、指标一致。"""
    dataset = tmp_path / "mini.jsonl"
    _write_mini_dataset(dataset)

    exp_1, exp_4 = tmp_path / "eval_jobs1", tmp_path / "eval_jobs4"
    dataset_dir_1, _ = _run_once(exp_1, dataset, jobs=1)
    dataset_dir_4, _ = _run_once(exp_4, dataset, jobs=4)
    assert dataset_dir_1 == exp_1 / layout.DIR_RUNS / "mini"
    assert dataset_dir_4 == exp_4 / layout.DIR_RUNS / "mini"

    records_1 = _read_records(dataset_dir_1 / layout.DIR_REPORT / layout.FILE_RESULTS)
    records_4 = _read_records(dataset_dir_4 / layout.DIR_REPORT / layout.FILE_RESULTS)
    assert [r["qid"] for r in records_1] == [r["qid"] for r in records_4]
    assert _strip_latency(records_1) == _strip_latency(records_4)

    metrics_1 = json.loads(
        (dataset_dir_1 / layout.DIR_REPORT / layout.FILE_METRICS).read_text(encoding="utf-8")
    )
    metrics_4 = json.loads(
        (dataset_dir_4 / layout.DIR_REPORT / layout.FILE_METRICS).read_text(encoding="utf-8")
    )
    assert _normalize_metrics(metrics_1) == _normalize_metrics(metrics_4)

    # 对话级产物同样逐对话一致：先落盘的子集口径不受并发影响
    for dialogue_id in ("d001", "d002"):
        dialogue_1 = dataset_dir_1 / dialogue_id
        dialogue_4 = dataset_dir_4 / dialogue_id
        assert (dialogue_1 / layout.DIR_RUNTIME).is_dir()
        assert (dialogue_4 / layout.DIR_RUNTIME).is_dir()
        dlg_records_1 = _read_records(dialogue_1 / layout.DIR_REPORT / layout.FILE_RESULTS)
        dlg_records_4 = _read_records(dialogue_4 / layout.DIR_REPORT / layout.FILE_RESULTS)
        assert [r["qid"] for r in dlg_records_1] == [r["qid"] for r in dlg_records_4]
        assert _strip_latency(dlg_records_1) == _strip_latency(dlg_records_4)
        dlg_metrics_1 = json.loads(
            (dialogue_1 / layout.DIR_REPORT / layout.FILE_METRICS).read_text(encoding="utf-8")
        )
        dlg_metrics_4 = json.loads(
            (dialogue_4 / layout.DIR_REPORT / layout.FILE_METRICS).read_text(encoding="utf-8")
        )
        assert dlg_metrics_1["dialogue_id"] == dlg_metrics_4["dialogue_id"] == dialogue_id
        assert _normalize_metrics(dlg_metrics_1) == _normalize_metrics(dlg_metrics_4)


def test_ingest_is_serial(tmp_path: Path) -> None:
    """jobs=4 时 ingest 仍在主线程按 session 顺序串行执行。"""
    dataset = tmp_path / "mini.jsonl"
    _write_mini_dataset(dataset)
    _, adapter = _run_once(tmp_path / "eval_serial", dataset, jobs=4)

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
