"""token 记账测试：run 落账 token_usage、分片求和、run / evaluate 报告展示。

产物结构（见 :mod:`membench.layout`）：``Runner(out_dir=实验目录)`` 落
``<实验目录>/runs/<dataset>/report/``（数据集级）与
``<实验目录>/runs/<dataset>/<对话>/report/``（对话级），任务级报告在
``<实验目录>/reports/``。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from membench import layout
from membench.adapters.base import BaseMemoryAdapter, IngestStats, QueryResult
from membench.evaluate import DatasetSpec, evaluate
from membench.fuse import fuse_gate, fuse_runs
from membench.runner import Runner


class _TokenAdapter(BaseMemoryAdapter):
    """固定上报 token 的适配器：每 session 写入 10 token / 1 次调用，每题 100 进 / 20 出。"""

    name = "mock-tokens"

    def reset(self) -> None:
        pass

    def ingest(self, messages: list[dict]) -> IngestStats:
        return IngestStats(input_tokens=10, llm_calls=1, latency_ms=1.0)

    def query(self, question: str) -> QueryResult:
        return QueryResult(answer="周五", input_tokens=100, output_tokens=20, llm_calls=1)


def _dataset(path: Path, dialogue_id: str, qids: tuple[str, ...], sessions: int = 2) -> Path:
    """单对话数据集：``sessions`` 段会话、``qids`` 道题（single_hop，gold=周五）。"""
    row = {
        "dialogue_id": dialogue_id,
        "sessions": [
            [{"role": "user", "content": f"第{i}次记录：项目截止日是周五。", "timestamp": None}]
            for i in range(sessions)
        ],
        "questions": [
            {
                "qid": qid,
                "question": "项目截止日是哪天？",
                "gold_answer": "周五",
                "category": "single_hop",
                "evidence": ["项目截止日是周五"],
            }
            for qid in qids
        ],
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _report_metrics(report_owner: str) -> dict:
    """某层级的 ``report/metrics.json``（数据集目录或对话目录均可）。"""
    return json.loads(
        (Path(report_owner) / "report" / "metrics.json").read_text(encoding="utf-8")
    )


def test_run_accounts_ingest_and_query_tokens(tmp_path: Path) -> None:
    """ingest 按对话真实计一次、query 逐题累计；报告与控制台同步可见。"""
    ds = _dataset(tmp_path / "d.jsonl", "d1", ("q1", "q2"))
    out_dir = tmp_path / "exp"
    dataset_dir = Runner(
        _TokenAdapter(), str(ds), judge="heuristic", out_dir=str(out_dir)
    ).run()
    # run() 返回数据集目录（<实验目录>/runs/<dataset>），数据集级报告在其 report/ 下
    assert dataset_dir == str(out_dir / "runs" / "d")
    usage = _report_metrics(dataset_dir)["token_usage"]
    assert usage == {
        "ingest_input_tokens": 20,  # 2 session × 10
        "ingest_llm_calls": 2,
        "query_input_tokens": 200,  # 2 题 × 100
        "query_output_tokens": 40,
        "query_llm_calls": 2,
        "n_dialogues": 1,
        "n_questions": 2,
    }
    records = [
        json.loads(line)
        for line in (Path(dataset_dir) / "report" / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [r["ingest_input_tokens"] for r in records] == [20, 20]

    md = (Path(dataset_dir) / "report" / "report.md").read_text(encoding="utf-8")
    assert "## Token 消耗（全流程）" in md
    assert "260" in md  # ingest 20 + query 输入 200 + 输出 40

    # 对话级与数据集级同一套记账口径（单对话时两者逐字段相等）
    dialogue_metrics = _report_metrics(str(Path(dataset_dir) / "d1"))
    assert dialogue_metrics["dialogue_id"] == "d1"
    assert dialogue_metrics["token_usage"] == usage

    # 单数据集 run 的任务级报告 = 数据集级报告内容（write_task_reports 默认 True）
    task_dir = out_dir / "reports"
    assert (task_dir / "report.md").read_text(encoding="utf-8") == md
    assert (task_dir / "metrics.json").is_file()


class _SilentAdapter(BaseMemoryAdapter):
    """什么都不上报的适配器：IngestStats / QueryResult 全用默认值（token 字段为 0）。"""

    name = "mock-silent"

    def reset(self) -> None:
        pass

    def ingest(self, messages: list[dict]) -> IngestStats:
        return IngestStats()

    def query(self, question: str) -> QueryResult:
        return QueryResult(answer="周五", retrieved=["项目截止日是周五"])


def test_unreported_tokens_marked_as_such(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """适配器未上报（字段为 0）时控制台标注"未上报"，而非读成零消耗。"""
    ds = _dataset(tmp_path / "d.jsonl", "d1", ("q1",))
    Runner(_SilentAdapter(), str(ds), judge="heuristic", out_dir=str(tmp_path / "runs")).run()
    out = capsys.readouterr().out
    assert "适配器未上报该字段" in out


def test_fuse_sums_token_usage_across_shards(tmp_path: Path) -> None:
    """分片各自计自己的 ingest（题目分片下同一对话被多片写入是真实成本，求和不去重）。"""
    ds = _dataset(tmp_path / "d.jsonl", "d1", ("q1", "q2"))
    shard_dirs = [
        Runner(
            _TokenAdapter(),
            str(ds),
            judge="heuristic",
            out_dir=str(tmp_path / f"s{i}"),
            shards=2,
            shard_index=i,
        ).run()
        for i in (0, 1)
    ]
    fused = fuse_runs(shard_dirs, str(tmp_path / "fused"))
    # fuse 的 out_dir 是完整实验目录：数据集产物在其 runs/<dataset>/，任务级报告在其 reports/
    assert fused["dataset_dir"] == str(tmp_path / "fused" / "runs" / "d")
    assert (tmp_path / "fused" / "reports" / "report.md").is_file()
    usage = _report_metrics(fused["dataset_dir"])["token_usage"]
    assert usage["ingest_input_tokens"] == 40  # 两片各 ingest 一次该对话
    assert usage["query_input_tokens"] == 200  # 题目不重复，2 题 × 100
    assert usage["query_output_tokens"] == 40
    assert usage["n_questions"] == 2
    # 融合产物的逐题明细与指标同在数据集 report/ 目录
    assert (tmp_path / "fused" / "runs" / "d" / "report" / "results.jsonl").is_file()
    assert (tmp_path / "fused" / "runs" / "d" / "d1" / "report" / "metrics.json").is_file()


def test_fuse_omits_token_usage_when_a_shard_lacks_it(tmp_path: Path) -> None:
    """任一分片未落账则不落账整体——避免"部分求和"被当成全量消耗。"""
    ds = _dataset(tmp_path / "d.jsonl", "d1", ("q1", "q2"))
    shard_dirs = [
        Runner(
            _TokenAdapter(),
            str(ds),
            judge="heuristic",
            out_dir=str(tmp_path / f"s{i}"),
            shards=2,
            shard_index=i,
        ).run()
        for i in (0, 1)
    ]
    old = Path(shard_dirs[1]) / "report" / "metrics.json"
    metrics = json.loads(old.read_text(encoding="utf-8"))
    del metrics["token_usage"]
    old.write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")

    fused = fuse_runs(shard_dirs, str(tmp_path / "fused"))
    assert "token_usage" not in _report_metrics(fused["dataset_dir"])


def test_fuse_gate_skips_when_stale_rate_none() -> None:
    """stale_rate 未计算（None）时门禁 SKIP——此前 float(None) 直接抛 TypeError。"""
    records = [{"qid": "q1", "category": "knowledge_update", "judge": {"stale": True}}]
    gate = fuse_gate(records, {"overall": {"stale_rate": None}})
    assert gate["status"] == "SKIP"
    assert "未计算" in gate["detail"]


def test_evaluate_report_shows_token_totals(tmp_path: Path) -> None:
    """evaluate 报告给出分数据集 token 消耗与合计，summary.json 同步落账。"""
    a = _dataset(tmp_path / "a.jsonl", "d_a", ("q_a1",))
    b = _dataset(tmp_path / "b.jsonl", "d_b", ("q_b1",))
    rc = evaluate(
        _TokenAdapter(),
        "heuristic",
        report_out=str(tmp_path / "out"),
        specs=[
            DatasetSpec(key="a", name="a", path=str(a), source="user-specified"),
            DatasetSpec(key="b", name="b", path=str(b), source="user-specified"),
        ],
    )
    assert rc == 0
    eval_dir = next((tmp_path / "out").glob("eval_*"))
    md = (eval_dir / "reports" / "report.md").read_text(encoding="utf-8")
    assert "## Token 消耗（全流程）" in md
    assert "合计（2/2 个数据集上报）" in md
    summary = json.loads(
        (eval_dir / "reports" / "evaluate_summary.json").read_text(encoding="utf-8")
    )
    assert summary["token_usage"]["datasets_reported"] == 2
    total = summary["token_usage"]["total"]
    assert total["ingest_input_tokens"] == 40  # 两个数据集各 1 对话 × 2 session × 10
    assert total["query_input_tokens"] == 200


def test_evaluate_report_marks_missing_token_usage(tmp_path: Path) -> None:
    """旧 run（无 token_usage）标「—」并说明合计只覆盖已上报部分。"""
    a = _dataset(tmp_path / "a.jsonl", "d_a", ("q_a1",))
    b = _dataset(tmp_path / "b.jsonl", "d_b", ("q_b1",))
    rc = evaluate(
        _TokenAdapter(),
        "heuristic",
        report_out=str(tmp_path / "out"),
        specs=[
            DatasetSpec(key="a", name="a", path=str(a), source="user-specified"),
            DatasetSpec(key="b", name="b", path=str(b), source="user-specified"),
        ],
    )
    assert rc == 0
    eval_dir = next((tmp_path / "out").glob("eval_*"))
    summary = json.loads(
        (eval_dir / "reports" / "evaluate_summary.json").read_text(encoding="utf-8")
    )
    for rel in summary["runs"].values():
        metrics_path = eval_dir / rel / "report" / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        del metrics["token_usage"]
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")
    # 复用旧 run 重出报告（纯汇总模式）
    from membench.evaluate import evaluate as _evaluate

    from_runs = {k: str(eval_dir / rel) for k, rel in summary["runs"].items()}
    rc = _evaluate(
        None,
        None,
        report_out=str(tmp_path / "out2"),
        from_runs=from_runs,
    )
    assert rc == 0
    eval_dir2 = next((tmp_path / "out2").glob("eval_*"))
    md = (eval_dir2 / "reports" / "report.md").read_text(encoding="utf-8")
    assert "均未落账 `token_usage`" in md


def test_runner_exports_reset_context(tmp_path: Path) -> None:
    """reset 前把 run 目录 / 对话 id 与序号作为可选提示导出给适配器，用完即撤。"""
    import os

    seen: list[dict[str, str]] = []

    class _RecordingAdapter(BaseMemoryAdapter):
        name = "mock-recording"

        def reset(self) -> None:
            seen.append(
                {
                    "run_dir": os.environ.get("MEMBENCH_RUN_DIR", ""),
                    "dialogue_id": os.environ.get("MEMBENCH_DIALOGUE_ID", ""),
                    "dialogue_index": os.environ.get("MEMBENCH_DIALOGUE_INDEX", ""),
                }
            )

        def ingest(self, messages: list[dict]) -> IngestStats:
            return IngestStats()

        def query(self, question: str) -> QueryResult:
            return QueryResult(answer="周五", retrieved=["项目截止日是周五"])

    ds = _dataset(tmp_path / "d.jsonl", "d1", ("q1",))
    out_dir = tmp_path / "exp"
    dataset_dir = Runner(
        _RecordingAdapter(), str(ds), judge="heuristic", out_dir=str(out_dir)
    ).run()
    assert dataset_dir == str(out_dir / "runs" / "d")
    # MEMBENCH_RUN_DIR 指本对话目录（适配器把运行时数据就近落它的 runtime/ 下）
    assert seen == [
        {
            "run_dir": layout.dialogue_dir(str(out_dir), "d", "d1"),
            "dialogue_id": "d1",
            "dialogue_index": "1",
        }
    ]
    assert (out_dir / "runs" / "d" / "d1" / "runtime").is_dir()
    assert "MEMBENCH_RUN_DIR" not in os.environ
    assert "MEMBENCH_DIALOGUE_ID" not in os.environ


def test_http_adapter_forwards_reset_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """HttpAdapter 把 reset 上下文放进 /reset 请求体；无上下文时退化为空体。"""
    import sys
    import types

    from membench.adapters.http import HttpAdapter

    calls: list[dict] = []

    class _Resp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {}

    fake_requests = types.SimpleNamespace(
        post=lambda url, json=None, headers=None, timeout=None: (
            calls.append({"url": url, "json": json}) or _Resp()
        )
    )
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    adapter = HttpAdapter("http://example.invalid:9")
    monkeypatch.delenv("MEMBENCH_RUN_DIR", raising=False)
    monkeypatch.delenv("MEMBENCH_DIALOGUE_ID", raising=False)
    monkeypatch.delenv("MEMBENCH_DIALOGUE_INDEX", raising=False)
    adapter.reset()
    assert calls[-1] == {"url": "http://example.invalid:9/reset", "json": {}}

    monkeypatch.setenv("MEMBENCH_RUN_DIR", "/tmp/run1")
    monkeypatch.setenv("MEMBENCH_DIALOGUE_ID", "d7")
    monkeypatch.setenv("MEMBENCH_DIALOGUE_INDEX", "7")
    adapter.reset()
    assert calls[-1]["json"] == {
        "run_dir": "/tmp/run1",
        "dialogue_id": "d7",
        "dialogue_index": "7",
    }
