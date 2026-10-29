"""interleaved ingest 策略与失败归因（--trace-state）的回归测试。

覆盖：
- schema 的 ``ingest_strategy`` / ``as_of`` 默认值与非法值（另见 test_schema.py）；
- interleaved 的调用顺序：按 as_of 升序切片 ingest、组内 query、全部组完成后补齐
  剩余消息、中间不 reset、结果按原始题序落盘；
- GateMem 语义夹具：t60 问旧预算、t150 更新后仍答旧值（interleaved 对 / sequence 错）；
- 约束：interleaved + 题目级分片 / --replay 直接 SystemExit；+ --snapshot 允许；
- 失败归因四桶与 ``evidence_in_state``（无快照能力 → 全 unknown）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from membench import layout
from membench.adapters.base import BaseMemoryAdapter, IngestStats, QueryResult
from membench.judges.base import JudgeResult
from membench.runner import Runner


# ---------------------------------------------------------------------------
# 测试适配器
# ---------------------------------------------------------------------------


class _StateFileAdapter(BaseMemoryAdapter):
    """把已 ingest 的消息写进 ``state.txt`` 的假适配器（可声明快照能力）。

    ``retrieved_by_qid`` 让用例按题控制"检索到了什么"（问题文本以 ``qid:`` 开头）。
    """

    name = "state-file"

    def __init__(
        self,
        retrieved_by_qid: dict[str, list[str]] | None = None,
        *,
        snapshot_support: bool = True,
    ) -> None:
        self._messages: list[dict] = []
        self.retrieved_by_qid = dict(retrieved_by_qid or {})
        self.snapshot_support = snapshot_support

    # ---- 三动词 ----
    def reset(self) -> None:
        self._messages = []

    def ingest(self, messages: list[dict]) -> IngestStats:
        self._messages.extend(messages)
        return IngestStats()

    def query(self, question: str) -> QueryResult:
        qid = question.split(":", 1)[0].strip()
        return QueryResult(
            answer="错误回答",
            retrieved=list(self.retrieved_by_qid.get(qid, [])),
        )

    # ---- 可选快照 ----
    def snapshot_capabilities(self) -> dict:
        if not self.snapshot_support:
            return {"supported": False}
        return {
            "supported": True,
            "restore": False,
            "rebuild": False,
            "format_version": 1,
        }

    def snapshot(self, target_dir: str, tag: str) -> dict:
        path = os.path.join(target_dir, "state.txt")
        with open(path, "w", encoding="utf-8") as handle:
            for message in self._messages:
                handle.write(str(message.get("content", "")) + "\n")
        return {"n_messages": len(self._messages)}


class _RecordingAdapter(_StateFileAdapter):
    """在 :class:`_StateFileAdapter` 之上记录调用序列（reset / ingest / query）。"""

    name = "recording-interleaved"

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple] = []

    def reset(self) -> None:
        self.events.append(("reset", len(self._messages)))
        super().reset()

    def ingest(self, messages: list[dict]) -> IngestStats:
        self.events.append(("ingest", [str(m.get("content", "")) for m in messages]))
        return super().ingest(messages)

    def query(self, question: str) -> QueryResult:
        qid = question.split(":", 1)[0].strip()
        self.events.append(("query", qid, len(self._messages)))
        return super().query(question)


class _BudgetAdapter(BaseMemoryAdapter):
    """只答"最新一条提到预算的消息"：模拟"看到多少就答多少"的记忆系统。"""

    name = "budget"

    def __init__(self) -> None:
        self._messages: list[dict] = []

    def reset(self) -> None:
        self._messages = []

    def ingest(self, messages: list[dict]) -> IngestStats:
        self._messages.extend(messages)
        return IngestStats()

    def query(self, question: str) -> QueryResult:
        hits = [m["content"] for m in self._messages if "预算" in m["content"]]
        return QueryResult(answer=hits[-1] if hits else "不知道", retrieved=hits)


class _AlwaysWrongJudge:
    """恒定判错：归因测试只关心"判错题怎么分桶"，不关心判分质量。"""

    name = "always-wrong"

    def grade(self, question, gold: str, prediction: str) -> JudgeResult:
        return JudgeResult(
            correct=False,
            abstained_pred=False,
            stale=False,
            score=0.0,
            judge=self.name,
        )


# ---------------------------------------------------------------------------
# 数据集辅助
# ---------------------------------------------------------------------------


def _write_dataset(path: Path, dialogues: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in dialogues) + "\n",
        encoding="utf-8",
    )
    return path


def _question(qid: str, *, as_of: int | None, evidence: list[str] | None = None) -> dict:
    return {
        "qid": qid,
        "question": f"{qid}: 问题？",
        "gold_answer": "标准答案",
        "category": "single_hop",
        "evidence": list(evidence or []),
        **({"as_of": as_of} if as_of is not None else {}),
    }


def _order_dataset(path: Path) -> Path:
    """2 个 session × 2 条消息；题目原始顺序与 as_of 顺序刻意不同。"""
    return _write_dataset(
        path,
        [
            {
                "dialogue_id": "d1",
                "ingest_strategy": "interleaved",
                "sessions": [
                    [
                        {"role": "user", "content": "m1", "timestamp": None},
                        {"role": "user", "content": "m2", "timestamp": None},
                    ],
                    [
                        {"role": "user", "content": "m3", "timestamp": None},
                        {"role": "user", "content": "m4", "timestamp": None},
                    ],
                ],
                "questions": [
                    _question("q3", as_of=3),
                    _question("q1", as_of=1),
                    _question("q_none", as_of=None),
                    _question("q2", as_of=3),
                ],
            }
        ],
    )


def _records(dataset_dir: Path) -> list[dict]:
    path = Path(dataset_dir) / layout.DIR_REPORT / layout.FILE_RESULTS
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _metrics(dataset_dir: Path) -> dict:
    path = Path(dataset_dir) / layout.DIR_REPORT / layout.FILE_METRICS
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# interleaved 执行顺序
# ---------------------------------------------------------------------------


def test_interleaved_ingest_order_and_record_order(tmp_path: Path) -> None:
    """按 as_of 升序分组 ingest/query，剩余消息最后补齐，结果按原始题序落盘。"""
    dataset = _order_dataset(tmp_path / "order.jsonl")
    adapter = _RecordingAdapter()
    dataset_dir = Runner(
        adapter,
        str(dataset),
        judge="heuristic",
        out_dir=str(tmp_path / "exp"),
        trace_state=True,
    ).run()

    assert adapter.events == [
        ("reset", 0),
        ("ingest", ["m1"]),
        ("query", "q1", 1),
        ("ingest", ["m2", "m3"]),
        ("query", "q3", 3),
        ("query", "q2", 3),
        ("ingest", ["m4"]),
        ("query", "q_none", 4),
    ]
    # 中间没有 reset（整段对话只 reset 一次）
    assert [event for event in adapter.events if event[0] == "reset"] == [("reset", 0)]

    # 落盘按原始题序，而不是按提问顺序
    assert [r["qid"] for r in _records(Path(dataset_dir))] == [
        "q3",
        "q1",
        "q_none",
        "q2",
    ]
    # 每题记录带上自己的 as_of，None 原样保留
    as_of_by_qid = {r["qid"]: r["as_of"] for r in _records(Path(dataset_dir))}
    assert as_of_by_qid == {"q3": 3, "q1": 1, "q_none": None, "q2": 3}

    metrics = _metrics(Path(dataset_dir))
    assert metrics["ingest_strategy"] == "interleaved"
    assert metrics["as_of_questions"] == 3
    assert metrics["as_of_enforced"] is True
    assert metrics["trace_state"]["role"] == "exported"
    # 有提问的 as_of 各一份（q_none 不产生 t4）
    assert metrics["trace_state"]["n_states"] == 2


def test_interleaved_trace_state_dirs_and_final_snapshot(tmp_path: Path) -> None:
    """trace_state/t<as_of>/ 与末尾 memory_snapshot/：目录、清单与内容时点正确。"""
    dataset = _order_dataset(tmp_path / "order.jsonl")
    dataset_dir = Path(
        Runner(
            _StateFileAdapter(),
            str(dataset),
            judge="heuristic",
            out_dir=str(tmp_path / "exp"),
            trace_state=True,
        ).run()
    )
    dialogue_dir = dataset_dir / "d1"

    def _content(path: Path) -> list[str]:
        return (path / "state.txt").read_text(encoding="utf-8").splitlines()

    t1 = dialogue_dir / layout.DIR_TRACE_STATE / "t1"
    t3 = dialogue_dir / layout.DIR_TRACE_STATE / "t3"
    assert _content(t1) == ["m1"]
    assert _content(t3) == ["m1", "m2", "m3"]
    assert (t1 / "manifest.json").is_file()
    assert (t3 / "manifest.json").is_file()
    # 没有 as_of=4 的题：不产生 t4
    assert not (dialogue_dir / layout.DIR_TRACE_STATE / "t4").exists()
    # 末尾快照 = 全部 ingest 完之后的状态（含最后一条消息）
    snapshot_dir = dialogue_dir / layout.DIR_SNAPSHOT
    assert _content(snapshot_dir) == ["m1", "m2", "m3", "m4"]
    assert (snapshot_dir / "manifest.json").is_file()


# ---------------------------------------------------------------------------
# 硬约束
# ---------------------------------------------------------------------------


def test_interleaved_rejects_question_sharding(tmp_path: Path) -> None:
    """interleaved + 题目级分片：开跑前 SystemExit，且不留产物目录。"""
    dataset = _order_dataset(tmp_path / "order.jsonl")
    out_dir = tmp_path / "exp"
    with pytest.raises(SystemExit, match="shard-by dialogue"):
        Runner(
            _StateFileAdapter(),
            str(dataset),
            judge="heuristic",
            out_dir=str(out_dir),
            shards=2,
            shard_index=0,
            shard_by="question",
        ).run()
    assert not out_dir.exists()


def test_interleaved_rejects_replay(tmp_path: Path) -> None:
    """interleaved + --replay：开跑前 SystemExit（回放只有末尾快照）。"""
    dataset = _order_dataset(tmp_path / "order.jsonl")
    out_dir = tmp_path / "exp"
    with pytest.raises(SystemExit, match="不支持 --replay"):
        Runner(
            _StateFileAdapter(),
            str(dataset),
            judge="heuristic",
            out_dir=str(out_dir),
            replay=str(tmp_path / "some-source"),
        ).run()
    assert not out_dir.exists()


def test_trace_state_rejects_replay(tmp_path: Path) -> None:
    """--trace-state + --replay：回放不 ingest，没有 as_of 可追踪，直接拒绝。"""
    dataset = _write_dataset(
        tmp_path / "seq.jsonl",
        [
            {
                "dialogue_id": "d1",
                "sessions": [[{"role": "user", "content": "m1", "timestamp": None}]],
                "questions": [_question("q1", as_of=None)],
            }
        ],
    )
    with pytest.raises(SystemExit, match="trace-state"):
        Runner(
            _StateFileAdapter(),
            str(dataset),
            judge="heuristic",
            out_dir=str(tmp_path / "exp"),
            replay=str(tmp_path / "some-source"),
            trace_state=True,
        ).run()


def test_interleaved_allows_snapshot_at_end(tmp_path: Path) -> None:
    """interleaved + --snapshot（默认开）：快照在全部 ingest 完成后导出（末尾态）。"""
    dataset = _order_dataset(tmp_path / "order.jsonl")
    dataset_dir = Path(
        Runner(
            _StateFileAdapter(),
            str(dataset),
            judge="heuristic",
            out_dir=str(tmp_path / "exp"),
            snapshot=True,
        ).run()
    )
    metrics = _metrics(dataset_dir)
    assert metrics["state_snapshot"]["role"] == "exported"
    content = (
        dataset_dir / "d1" / layout.DIR_SNAPSHOT / "state.txt"
    ).read_text(encoding="utf-8").splitlines()
    assert content == ["m1", "m2", "m3", "m4"]


# ---------------------------------------------------------------------------
# GateMem 语义夹具：as_of 决定答案对错
# ---------------------------------------------------------------------------


def _budget_dataset(path: Path, strategy: str) -> Path:
    """t1 记旧预算 100 万、t2 更新为 200 万；题目问 t1 时点的预算。"""
    return _write_dataset(
        path,
        [
            {
                "dialogue_id": "d1",
                "ingest_strategy": strategy,
                "sessions": [
                    [
                        {"role": "user", "content": "项目预算 100 万。", "timestamp": None},
                        {"role": "user", "content": "预算更新：200 万。", "timestamp": None},
                    ]
                ],
                "questions": [
                    {
                        "qid": "q_budget",
                        "question": "项目预算是多少？",
                        "gold_answer": "100 万",
                        "category": "knowledge_update",
                        "evidence": ["项目预算 100 万"],
                        **({"as_of": 1} if strategy == "interleaved" else {}),
                    }
                ],
            }
        ],
    )


def test_gatemem_update_semantics_interleaved_vs_sequence(tmp_path: Path) -> None:
    """同一份数据：interleaved 问 t1 时点答旧值（正确），sequence 看到新值（错误）。"""
    interleaved = Path(
        Runner(
            _BudgetAdapter(),
            str(_budget_dataset(tmp_path / "inter.jsonl", "interleaved")),
            judge="heuristic",
            out_dir=str(tmp_path / "exp-inter"),
        ).run()
    )
    sequence = Path(
        Runner(
            _BudgetAdapter(),
            str(_budget_dataset(tmp_path / "seq.jsonl", "sequence")),
            judge="heuristic",
            out_dir=str(tmp_path / "exp-seq"),
        ).run()
    )

    inter_rec = _records(interleaved)[0]
    seq_rec = _records(sequence)[0]
    assert "100 万" in inter_rec["prediction"] and "200 万" not in inter_rec["prediction"]
    assert inter_rec["judge"]["correct"] is True
    assert "200 万" in seq_rec["prediction"]
    assert seq_rec["judge"]["correct"] is False

    # sequence 下 as_of 不强制：metrics 明确标注，报告给"时点错位"提示
    seq_metrics = _metrics(sequence)
    assert seq_metrics["ingest_strategy"] == "sequence"
    assert seq_metrics["as_of_questions"] == 0  # 该夹具 sequence 版本没有写 as_of
    report = (sequence / layout.DIR_REPORT / layout.FILE_REPORT_MD).read_text(encoding="utf-8")
    assert "未强制" not in report


def test_sequence_with_as_of_annotation_warns_in_report(tmp_path: Path) -> None:
    """数据带 as_of 但按 sequence 跑：报告必须给出"未强制 / 时点错位"提示。"""
    dataset = _write_dataset(
        tmp_path / "seq.jsonl",
        [
            {
                "dialogue_id": "d1",
                "ingest_strategy": "sequence",
                "sessions": [
                    [{"role": "user", "content": "项目预算 100 万。", "timestamp": None}],
                    [{"role": "user", "content": "预算更新：200 万。", "timestamp": None}],
                ],
                "questions": [
                    {
                        "qid": "q_budget",
                        "question": "项目预算是多少？",
                        "gold_answer": "100 万",
                        "category": "knowledge_update",
                        "evidence": ["项目预算 100 万"],
                        "as_of": 1,
                    }
                ],
            }
        ],
    )
    dataset_dir = Path(
        Runner(
            _BudgetAdapter(),
            str(dataset),
            judge="heuristic",
            out_dir=str(tmp_path / "exp"),
        ).run()
    )
    metrics = _metrics(dataset_dir)
    assert metrics["as_of_enforced"] is False
    report = (dataset_dir / layout.DIR_REPORT / layout.FILE_REPORT_MD).read_text(
        encoding="utf-8"
    )
    assert "未强制" in report
    assert "时点错位" in report


# ---------------------------------------------------------------------------
# 失败归因
# ---------------------------------------------------------------------------


def _attribution_dataset(path: Path) -> Path:
    """两条消息 + 4 道判错题：未写入 / 未检索到 / 检索到但答错 / 无 evidence。"""
    questions = [
        {
            "qid": "q_not_written",
            "question": "q_not_written: 问题？",
            "gold_answer": "标准答案",
            "category": "single_hop",
            "evidence": ["gamma-missing"],
            "as_of": 1,
        },
        {
            "qid": "q_not_retrieved",
            "question": "q_not_retrieved: 问题？",
            "gold_answer": "标准答案",
            "category": "multi_hop",
            "evidence": ["alpha fact"],
            "as_of": 1,
        },
        {
            "qid": "q_wrong_answer",
            "question": "q_wrong_answer: 问题？",
            "gold_answer": "标准答案",
            "category": "multi_hop",
            "evidence": ["alpha fact"],
            "as_of": 1,
        },
        {
            "qid": "q_no_evidence",
            "question": "q_no_evidence: 问题？",
            "gold_answer": "标准答案",
            "category": "temporal",
            "evidence": [],
            "as_of": 1,
        },
    ]
    return _write_dataset(
        path,
        [
            {
                "dialogue_id": "d1",
                "ingest_strategy": "interleaved",
                "sessions": [
                    [
                        {"role": "user", "content": "m1 alpha fact", "timestamp": None},
                        {"role": "user", "content": "m2 beta fact", "timestamp": None},
                    ]
                ],
                "questions": questions,
            }
        ],
    )


def _attribution_run(tmp_path: Path, *, snapshot_support: bool) -> dict:
    dataset = _attribution_dataset(tmp_path / "attr.jsonl")
    adapter = _StateFileAdapter(
        # 只有 q_wrong_answer 把证据检索进了回答上下文
        {"q_wrong_answer": ["m1 alpha fact"]},
        snapshot_support=snapshot_support,
    )
    dataset_dir = Path(
        Runner(
            adapter,
            str(dataset),
            judge=_AlwaysWrongJudge(),
            out_dir=str(tmp_path / f"exp-{snapshot_support}"),
            trace_state=True,
        ).run()
    )
    return {
        "metrics": _metrics(dataset_dir),
        "records": _records(dataset_dir),
        "dataset_dir": dataset_dir,
    }


def test_attribution_four_buckets(tmp_path: Path) -> None:
    """三类可判定样例各进一桶；无 evidence 的判错题单独计数、不塞进四桶。"""
    result = _attribution_run(tmp_path, snapshot_support=True)
    records = {r["qid"]: r for r in result["records"]}
    assert records["q_not_written"]["evidence_in_state"] is False
    assert records["q_not_retrieved"]["evidence_in_state"] is True
    assert records["q_wrong_answer"]["evidence_in_state"] is True
    assert records["q_no_evidence"]["evidence_in_state"] is None

    attribution = result["metrics"]["attribution"]
    assert attribution["n_wrong"] == 4
    assert attribution["n_wrong_with_evidence"] == 3
    assert attribution["n_wrong_without_evidence"] == 1
    assert attribution["n_state_readable"] == 3
    assert attribution["buckets"] == {
        "not_written": 1,
        "not_retrieved": 1,
        "wrong_answer": 1,
        "unknown": 0,
    }
    assert {s["bucket"] for s in attribution["samples"]} == {
        "not_written",
        "not_retrieved",
        "wrong_answer",
    }
    # 报告小节出现，且带四桶与样例
    report = (
        result["dataset_dir"] / layout.DIR_REPORT / layout.FILE_REPORT_MD
    ).read_text(encoding="utf-8")
    assert "## 失败归因" in report
    assert "未写入" in report and "未检索到" in report and "检索到但答错" in report
    assert "### 归因样例" in report


def test_attribution_all_unknown_without_snapshot_support(tmp_path: Path) -> None:
    """适配器不支持快照：状态不可读，带 evidence 的判错题全部记 unknown。"""
    result = _attribution_run(tmp_path, snapshot_support=False)
    records = {r["qid"]: r for r in result["records"]}
    assert all(r["evidence_in_state"] is None for r in records.values())
    attribution = result["metrics"]["attribution"]
    assert attribution["n_wrong_with_evidence"] == 3
    assert attribution["n_state_readable"] == 0
    assert attribution["buckets"] == {
        "not_written": 0,
        "not_retrieved": 0,
        "wrong_answer": 0,
        "unknown": 3,
    }


def test_attribution_absent_without_evidence_keys(tmp_path: Path) -> None:
    """没有 evidence 标注的数据集：四桶为 0，报告明确说"无法归因"而不是"没有失败"。"""
    dataset = _write_dataset(
        tmp_path / "noev.jsonl",
        [
            {
                "dialogue_id": "d1",
                "ingest_strategy": "interleaved",
                "sessions": [[{"role": "user", "content": "m1", "timestamp": None}]],
                "questions": [_question("q1", as_of=1)],
            }
        ],
    )
    dataset_dir = Path(
        Runner(
            _StateFileAdapter(),
            str(dataset),
            judge=_AlwaysWrongJudge(),
            out_dir=str(tmp_path / "exp"),
        ).run()
    )
    attribution = _metrics(dataset_dir)["attribution"]
    assert attribution["n_wrong"] == 1
    assert attribution["n_wrong_with_evidence"] == 0
    assert sum(attribution["buckets"].values()) == 0
    report = (dataset_dir / layout.DIR_REPORT / layout.FILE_REPORT_MD).read_text(
        encoding="utf-8"
    )
    assert "无法归因" in report


# ---------------------------------------------------------------------------
# 状态扫描的边界与磁盘告警
# ---------------------------------------------------------------------------


def test_scan_state_for_evidence_truncates_long_needle(tmp_path: Path) -> None:
    """长证据只取前 200 字符；二进制文件跳过；没有可读文本 / 无证据返回 None。"""
    from membench.runner import EVIDENCE_NEEDLE_CHARS, scan_state_for_evidence

    state = tmp_path / "state"
    state.mkdir()
    (state / "text.txt").write_text("a" * 200 + "TAIL", encoding="utf-8")
    (state / "binary.bin").write_bytes(b"\x00\x01\x02TAIL")
    (state / "manifest.json").write_text("TAIL", encoding="utf-8")

    assert EVIDENCE_NEEDLE_CHARS == 200
    assert scan_state_for_evidence(str(state), ["a" * 200]) is True
    # 超过 200 字符的部分不参与匹配（只取前 200）
    assert scan_state_for_evidence(str(state), ["a" * 200 + "TAIL"]) is True
    assert scan_state_for_evidence(str(state), ["missing"]) is False
    # 只有二进制 + manifest：没有可读文本，判"不可读"而不是"没写入"
    binary_only = tmp_path / "binary-only"
    binary_only.mkdir()
    (binary_only / "index.bin").write_bytes(b"\x00\x01")
    (binary_only / "manifest.json").write_text("x", encoding="utf-8")
    assert scan_state_for_evidence(str(binary_only), ["x"]) is None
    # 没有 evidence / 没有状态目录
    assert scan_state_for_evidence(str(state), []) is None
    assert scan_state_for_evidence(None, ["a"]) is None


def test_trace_state_disk_warnings_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """单份超限 / 总份数超限各提示一次（阈值收紧到可触发）。"""
    import membench.runner as runner_mod

    monkeypatch.setattr(runner_mod, "TRACE_STATE_WARN_BYTES", 1)
    monkeypatch.setattr(runner_mod, "TRACE_STATE_WARN_COUNT", 1)
    dataset = _order_dataset(tmp_path / "order.jsonl")
    Runner(
        _StateFileAdapter(),
        str(dataset),
        judge="heuristic",
        out_dir=str(tmp_path / "exp"),
        trace_state=True,
    ).run()
    out = capsys.readouterr().out
    assert out.count("单份追踪状态") == 1
    assert out.count("追踪状态已导出") == 1


def test_interleaved_jobs4_matches_jobs1(tmp_path: Path) -> None:
    """组内并发（jobs=4）与串行：记录内容与原始题序一致，ingest 顺序不变。"""
    dataset = _order_dataset(tmp_path / "order.jsonl")

    def _run(name: str, jobs: int) -> tuple[list[dict], list[tuple]]:
        adapter = _RecordingAdapter()
        dataset_dir = Runner(
            adapter,
            str(dataset),
            judge="heuristic",
            out_dir=str(tmp_path / name),
            jobs=jobs,
        ).run()
        return _records(Path(dataset_dir)), adapter.events

    records_1, events_1 = _run("jobs1", 1)
    records_4, events_4 = _run("jobs4", 4)

    strip = lambda records: [
        {k: v for k, v in record.items() if k != "latency_ms"} for record in records
    ]
    assert [r["qid"] for r in records_1] == [r["qid"] for r in records_4] == [
        "q3",
        "q1",
        "q_none",
        "q2",
    ]
    assert strip(records_1) == strip(records_4)
    # ingest 阶段始终串行：切片顺序与内容不受 jobs 影响
    assert [e for e in events_1 if e[0] != "query"] == [
        e for e in events_4 if e[0] != "query"
    ]


# ---------------------------------------------------------------------------
# 摄入块（ingest_blocks）与 interleaved 并发告警
# ---------------------------------------------------------------------------


def test_ingest_blocks_match_ingest_calls(tmp_path: Path) -> None:
    """块数 = ingest 调用次数（含补齐块），min / median / max 按消息条数算。"""
    dataset = _order_dataset(tmp_path / "order.jsonl")
    adapter = _RecordingAdapter()
    dataset_dir = Path(
        Runner(
            adapter,
            str(dataset),
            judge="heuristic",
            out_dir=str(tmp_path / "exp"),
        ).run()
    )
    ingest_blocks = [len(event[1]) for event in adapter.events if event[0] == "ingest"]
    assert ingest_blocks == [1, 2, 1]

    metrics = _metrics(dataset_dir)
    assert metrics["ingest_blocks"] == {"n": 3, "min": 1, "median": 1, "max": 2}
    # 对话级 metrics 同样给该对话的摄入块
    dialogue_metrics = _metrics(
        dataset_dir / "d1"
    )
    assert dialogue_metrics["ingest_blocks"] == metrics["ingest_blocks"]
    # 报告「评测配置」表出现「摄入块」一行
    report = (dataset_dir / layout.DIR_REPORT / layout.FILE_REPORT_MD).read_text(
        encoding="utf-8"
    )
    assert "| 摄入块 | 3 块（1~2 条/块，中位 1） |" in report


def test_ingest_blocks_absent_for_sequence(tmp_path: Path) -> None:
    """sequence 对话不产出 ingest_blocks：对话级与数据集级都没有该键。"""
    dataset = _write_dataset(
        tmp_path / "seq.jsonl",
        [
            {
                "dialogue_id": "d1",
                "ingest_strategy": "sequence",
                "sessions": [
                    [{"role": "user", "content": "m1", "timestamp": None}],
                    [{"role": "user", "content": "m2", "timestamp": None}],
                ],
                "questions": [_question("q1", as_of=None)],
            }
        ],
    )
    dataset_dir = Path(
        Runner(
            _StateFileAdapter(),
            str(dataset),
            judge="heuristic",
            out_dir=str(tmp_path / "exp"),
        ).run()
    )
    assert "ingest_blocks" not in _metrics(dataset_dir)
    assert "ingest_blocks" not in _metrics(dataset_dir / "d1")
    report = (dataset_dir / layout.DIR_REPORT / layout.FILE_REPORT_MD).read_text(
        encoding="utf-8"
    )
    assert "摄入块" not in report


def test_interleaved_jobs_warning_once(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """interleaved + jobs>1 警告一次；jobs=1 不警告（也不强制降 jobs）。"""
    dataset = _order_dataset(tmp_path / "order.jsonl")
    Runner(
        _StateFileAdapter(),
        str(dataset),
        judge="heuristic",
        out_dir=str(tmp_path / "jobs2"),
        jobs=2,
    ).run()
    out = capsys.readouterr().out
    assert out.count("interleaved 下同组 query 并发执行") == 1
    assert "--jobs 1" in out

    Runner(
        _StateFileAdapter(),
        str(dataset),
        judge="heuristic",
        out_dir=str(tmp_path / "jobs1"),
        jobs=1,
    ).run()
    assert "interleaved 下同组 query 并发执行" not in capsys.readouterr().out


def test_attribution_excludes_abstention_questions() -> None:
    """拒答题不进四桶：它们由 state_leak / 判分 reason 解释，另记 n_wrong_abstention。"""
    from membench.metrics.summary import attribution_summary

    records = [
        {
            "qid": "q_safety",
            "dialogue_id": "d",
            "category": "safety_no_memory",
            "gold_answer": "<ABSTAIN>",
            "evidence_in_state": True,
            "evidence_hit": None,
            "judge": {"correct": False},
        },
        {
            "qid": "q_util",
            "dialogue_id": "d",
            "category": "multi_hop",
            "gold_answer": "answer",
            "evidence_in_state": True,
            "evidence_hit": False,
            "judge": {"correct": False},
        },
    ]
    out = attribution_summary(records, {("d", "q_safety"), ("d", "q_util")})
    assert out["n_wrong"] == 2
    assert out["n_wrong_abstention"] == 1
    assert out["n_wrong_with_evidence"] == 1
    assert out["n_wrong_without_evidence"] == 0
    assert out["buckets"]["not_retrieved"] == 1
    assert out["buckets"]["not_written"] == 0
    assert out["by_category"]["safety_no_memory"]["n_wrong_abstention"] == 1
