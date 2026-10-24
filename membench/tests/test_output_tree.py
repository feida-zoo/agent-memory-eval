"""输出树结构测试：逐层形态、数据集与逐对话聚合一致、对话级不出图、目录冲突、重复运行拒绝、旧结构拒绝。

契约见 :mod:`membench.layout`——一次实验一个自包含子目录::

    eval_<name>_<platform>_<ts>/
    ├── runs/<dataset>/
    │   ├── report/                    # 数据集级：md + metrics + 逐题明细 + 两图
    │   └── <对话>/{runtime/,report/}   # 对话级报告不出图
    └── reports/                       # 任务级（单数据集 run = 数据集级报告内容）

用固定上报 token 的脚本化适配器跑一个两对话数据集，使逐层数值都能手算核对。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from membench import cli, layout
from membench.adapters.base import BaseMemoryAdapter, IngestStats, QueryResult
from membench.evaluate import evaluate
from membench.evaluate_compare import evaluate_compare
from membench.fuse import fuse_runs
from membench.runner import Runner

#: 数据集文件名干（= 产物里的数据集目录名）。
DATASET_NAME = "tree"


class _TokenAdapter(BaseMemoryAdapter):
    """固定上报 token 的适配器：每 session 写入 10 token，每题 100 进 / 20 出。"""

    name = "mock-tokens"

    def reset(self) -> None:
        pass

    def ingest(self, messages: list[dict]) -> IngestStats:
        return IngestStats(input_tokens=10, llm_calls=1, latency_ms=1.0)

    def query(self, question: str) -> QueryResult:
        return QueryResult(
            answer="周五",
            retrieved=["项目截止日是周五"],
            input_tokens=100,
            output_tokens=20,
            llm_calls=1,
        )


def _write_dataset(path: Path) -> Path:
    """两对话数据集：d1 两段会话 / 2 题，d2 一段会话 / 1 题。"""
    rows = [
        {
            "dialogue_id": dialogue_id,
            "sessions": [
                [{"role": "user", "content": f"第{i}次记录：项目截止日是周五。", "timestamp": None}]
                for i in range(sessions)
            ],
            "questions": [
                {
                    "qid": f"{dialogue_id}_q{i}",
                    "question": "项目截止日是哪天？",
                    "gold_answer": "周五",
                    "category": "single_hop",
                    "evidence": ["项目截止日是周五"],
                }
                for i in range(questions)
            ],
        }
        for dialogue_id, sessions, questions in (("d1", 2, 2), ("d2", 1, 1))
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _run_experiment(tmp_path: Path) -> tuple[Path, Path]:
    """跑一次单数据集 run，返回 (实验目录, 数据集目录)。"""
    out_dir = tmp_path / "out"
    dataset_path = _write_dataset(tmp_path / f"{DATASET_NAME}.jsonl")
    dataset_dir = Runner(
        _TokenAdapter(), str(dataset_path), judge="heuristic", out_dir=str(out_dir)
    ).run()
    return out_dir, Path(dataset_dir)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_experiment_tree_layers(tmp_path: Path) -> None:
    """实验目录逐层形态：runs/<dataset>/{report,<对话>/{runtime,report}} + reports/。"""
    out_dir, dataset_dir = _run_experiment(tmp_path)

    # run() 返回数据集目录，且只落实验目录之内
    assert dataset_dir == out_dir / "runs" / DATASET_NAME
    assert sorted(p.name for p in out_dir.iterdir()) == ["reports", "runs"]
    assert sorted(p.name for p in (out_dir / "runs").iterdir()) == [DATASET_NAME]

    # 数据集级报告层：md + metrics + 逐题明细 + 两张图
    assert sorted(p.name for p in (dataset_dir / "report").iterdir()) == [
        "fig_category_accuracy.png",
        "fig_latency_hist.png",
        "metrics.json",
        "report.md",
        "results.jsonl",
    ]
    # 数据集目录里只有报告层与对话目录：分片账本只在分片评测时出现
    assert sorted(p.name for p in dataset_dir.iterdir()) == ["d1", "d2", "report"]

    # 对话层：report/（纯文字）与 runtime/ 始终建出；快照默认导出但该适配器不声明
    # 快照能力，跳过导出后自然没有 memory_snapshot/
    for dialogue_id in ("d1", "d2"):
        dialogue_dir = dataset_dir / dialogue_id
        assert sorted(p.name for p in dialogue_dir.iterdir()) == ["report", "runtime"]
        assert sorted(p.name for p in (dialogue_dir / "report").iterdir()) == [
            "metrics.json",
            "report.md",
            "results.jsonl",
        ]
        assert list((dialogue_dir / "runtime").iterdir()) == []

    # 任务级报告层：单数据集 run = 数据集级报告内容（md + metrics + 它引用的两张图）
    assert sorted(p.name for p in (out_dir / "reports").iterdir()) == [
        "fig_category_accuracy.png",
        "fig_latency_hist.png",
        "metrics.json",
        "report.md",
    ]
    assert (out_dir / "reports" / "report.md").read_text(encoding="utf-8") == (
        dataset_dir / "report" / "report.md"
    ).read_text(encoding="utf-8")

    # run_id 只作为身份字符串写进 metrics，不再占一个目录层级
    metrics = _read_json(dataset_dir / "report" / "metrics.json")
    assert metrics["run_id"]
    assert metrics["run_id"] not in [p.name for p in (out_dir / "runs").iterdir()]
    assert not (out_dir / "runs" / metrics["run_id"]).exists()
    # 产物归属元数据：读取方据此定位，不必猜层级
    assert metrics["dataset"] == DATASET_NAME
    assert Path(metrics["experiment_dir"]).resolve() == out_dir.resolve()
    assert Path(metrics["dataset_dir"]).resolve() == dataset_dir.resolve()
    assert metrics["platform"] == "default"


def test_dataset_totals_match_dialogue_sums(tmp_path: Path) -> None:
    """数据集级 metrics / results 与逐对话之和一致（题目、token、分题型题数）。"""
    _, dataset_dir = _run_experiment(tmp_path)
    dataset_report = dataset_dir / "report"
    ds_metrics = _read_json(dataset_report / "metrics.json")
    ds_records = _read_records(dataset_report / "results.jsonl")

    dialogue_metrics: list[dict] = []
    dialogue_records: list[dict] = []
    for dialogue_id in ("d1", "d2"):
        metrics = _read_json(dataset_dir / dialogue_id / "report" / "metrics.json")
        records = _read_records(dataset_dir / dialogue_id / "report" / "results.jsonl")
        assert metrics["dialogue_id"] == dialogue_id
        assert {r["dialogue_id"] for r in records} == {dialogue_id}
        assert metrics["overall"]["n_questions"] == len(records)
        dialogue_metrics.append(metrics)
        dialogue_records.extend(records)

    # 逐题明细：数据集级 = 逐对话明细按对话顺序拼接（记录顺序与 jobs=1 一致）
    assert ds_records == dialogue_records
    assert ds_metrics["overall"]["n_questions"] == sum(
        m["overall"]["n_questions"] for m in dialogue_metrics
    )
    assert [r["qid"] for r in ds_records] == ["d1_q0", "d1_q1", "d2_q0"]

    # token 账目：数据集级逐字段 = 逐对话之和
    usage_fields = (
        "ingest_input_tokens",
        "ingest_llm_calls",
        "query_input_tokens",
        "query_output_tokens",
        "query_llm_calls",
        "n_dialogues",
        "n_questions",
    )
    for field in usage_fields:
        assert ds_metrics["token_usage"][field] == sum(
            m["token_usage"][field] for m in dialogue_metrics
        ), field
    assert ds_metrics["token_usage"] == {
        "ingest_input_tokens": 30,  # d1 两段会话 + d2 一段，各 10
        "ingest_llm_calls": 3,
        "query_input_tokens": 300,  # 3 题 × 100
        "query_output_tokens": 60,
        "query_llm_calls": 3,
        "n_dialogues": 2,
        "n_questions": 3,
    }

    # 分题型题数：数据集级 = 逐对话之和
    categories = set(ds_metrics["by_category"]) | {
        cat for m in dialogue_metrics for cat in m["by_category"]
    }
    assert categories
    for cat in categories:
        assert ds_metrics["by_category"][cat]["n"] == sum(
            m["by_category"].get(cat, {"n": 0})["n"] for m in dialogue_metrics
        ), cat


def test_dialogue_report_has_no_figures(tmp_path: Path) -> None:
    """对话级报告目录不出图，md 也不引用任何图片（不留打不开的相对路径）。"""
    _, dataset_dir = _run_experiment(tmp_path)

    for dialogue_id in ("d1", "d2"):
        report_dir = dataset_dir / dialogue_id / "report"
        assert list(report_dir.glob("*.png")) == []
        md = (report_dir / "report.md").read_text(encoding="utf-8")
        assert "![" not in md
        assert ".png" not in md

    # 对照：数据集级报告出图并同目录引用（否则上面的断言可能因"报告根本不带图"而假通过）
    dataset_md = (dataset_dir / "report" / "report.md").read_text(encoding="utf-8")
    assert "![分题型准确率](fig_category_accuracy.png)" in dataset_md
    assert "![查询延迟分布](fig_latency_hist.png)" in dataset_md
    for name in ("fig_category_accuracy.png", "fig_latency_hist.png"):
        assert (dataset_dir / "report" / name).is_file()


def test_experiment_dir_conflict_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """实验目录非空时拒绝再次使用：同一秒内的第二次运行不得静默覆盖上次产物。"""
    out_root = tmp_path / "out"
    monkeypatch.setenv("MEMBENCH_OUTPUT_DIR", str(out_root))
    # 固定时间戳 = 同一秒内第二次运行（真实撞车场景），实验目录名只精确到秒
    monkeypatch.setattr(layout, "ts_now", lambda now=None: "20260101-000000")
    dataset_path = _write_dataset(tmp_path / f"{DATASET_NAME}.jsonl")
    argv = [
        "run",
        "--adapter", "mock-strong",
        "--dataset", str(dataset_path),
        "--judge", "heuristic",
    ]

    assert cli.main(argv) == 0
    experiment_dir = out_root / "eval_mock-strong_default_20260101-000000"
    assert (experiment_dir / "runs" / DATASET_NAME / "report" / "metrics.json").is_file()

    with pytest.raises(SystemExit, match="已存在且非空"):
        cli.main(argv)
    # 拒绝发生时原产物保持不变（仍是第一次运行留下的 run_id）
    assert (experiment_dir / "runs" / DATASET_NAME / "report" / "metrics.json").is_file()

    # 仅当目标为空目录时才允许复用（校验阶段失败留下的空目录）
    empty_name = "eval_mock-strong_default_20260101-000001"
    empty_dir = layout.experiment_path(str(out_root), empty_name)
    assert layout.experiment_path(str(out_root), empty_name) == empty_dir
    assert Path(empty_dir).is_dir() and not list(Path(empty_dir).iterdir())


def test_legacy_and_incomplete_products_rejected(tmp_path: Path) -> None:
    """旧结构（平铺 run 目录 / 旧 ab_ 实验目录）与半成品产物统一被拒。"""
    legacy_run = tmp_path / "legacy_run"
    legacy_run.mkdir()
    (legacy_run / "metrics.json").write_text("{}", encoding="utf-8")
    (legacy_run / "results.jsonl").write_text("", encoding="utf-8")

    legacy_ab = tmp_path / "ab_t1_20260101-000000"
    (legacy_ab / "A").mkdir(parents=True)
    (legacy_ab / "B").mkdir()
    (legacy_ab / "compare_report.md").write_text("", encoding="utf-8")

    for target in (legacy_run, legacy_ab):
        with pytest.raises(SystemExit, match=re.escape(layout.LEGACY_LAYOUT_MSG)):
            layout.resolve_dataset_dir(str(target))

    # 读取侧入口同样拒绝（不是只拦 layout 一处），且都在建目录之前
    with pytest.raises(SystemExit, match=re.escape(layout.LEGACY_LAYOUT_MSG)):
        evaluate(
            None,
            None,
            report_out=str(tmp_path / "out"),
            from_runs={"a": str(legacy_run)},
        )
    assert not (tmp_path / "out").exists()
    with pytest.raises(SystemExit, match=re.escape(layout.LEGACY_LAYOUT_MSG)):
        evaluate_compare(
            None,
            None,
            None,
            report_out=str(tmp_path / "cmp"),
            baseline_runs={"a": str(legacy_run)},
            candidate_runs={"a": str(legacy_run)},
        )
    with pytest.raises(SystemExit, match=re.escape(layout.LEGACY_LAYOUT_MSG)):
        fuse_runs([str(legacy_run), str(legacy_run)], str(tmp_path / "fused"))

    # 半成品（有 report/ 但缺 metrics.json）另有明确报错，不误判成旧结构
    partial = tmp_path / "partial_exp" / "runs" / DATASET_NAME
    (partial / "report").mkdir(parents=True)
    with pytest.raises(SystemExit, match="产物不完整"):
        layout.resolve_dataset_dir(str(partial))


def test_rerun_same_dataset_in_same_experiment_dir_rejected(tmp_path: Path) -> None:
    """同一实验目录下二次跑同一数据集：写盘前拒绝、原产物不被覆盖；换目录 / 换数据集可跑。"""
    out_dir, dataset_dir = _run_experiment(tmp_path)
    dataset_path = tmp_path / f"{DATASET_NAME}.jsonl"
    metrics_path = dataset_dir / "report" / "metrics.json"
    before_content = metrics_path.read_text(encoding="utf-8")
    before_mtime = metrics_path.stat().st_mtime_ns

    with pytest.raises(SystemExit, match="已在本实验目录下完成评测"):
        Runner(
            _TokenAdapter(), str(dataset_path), judge="heuristic", out_dir=str(out_dir)
        ).run()

    # 拒绝发生在任何写盘之前：不再有"同一路径被重写、timestamp/mtime 变化"的静默覆盖
    assert metrics_path.read_text(encoding="utf-8") == before_content
    assert metrics_path.stat().st_mtime_ns == before_mtime

    # 换 --report-out（新的实验目录）后可跑，旧实验目录的产物依旧不变
    rerun_dir = Runner(
        _TokenAdapter(), str(dataset_path), judge="heuristic", out_dir=str(tmp_path / "out2")
    ).run()
    assert Path(rerun_dir) == tmp_path / "out2" / "runs" / DATASET_NAME
    assert metrics_path.read_text(encoding="utf-8") == before_content

    # 保护只针对"同一数据集"：同一实验目录下跑另一个数据集不受影响（分片同理）
    other_dataset = _write_dataset(tmp_path / "other.jsonl")
    other_dir = Runner(
        _TokenAdapter(), str(other_dataset), judge="heuristic", out_dir=str(out_dir)
    ).run()
    assert Path(other_dir) == out_dir / "runs" / "other"
    assert metrics_path.read_text(encoding="utf-8") == before_content
