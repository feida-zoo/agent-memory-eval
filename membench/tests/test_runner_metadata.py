"""Runner platform/commit 元数据、report.md 评测配置表与 --report-out 复制行为测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from membench.adapters import MockStrongAdapter
from membench.data import SAMPLE_DATASET_PATH
from membench.report import copy_report
from membench.runner import Runner


def _run(tmp_path: Path, **kwargs) -> tuple[Path, dict]:
    runner = Runner(
        MockStrongAdapter(),
        str(SAMPLE_DATASET_PATH),
        judge="heuristic",
        out_dir=str(tmp_path / "runs"),
        **kwargs,
    )
    run_dir = Path(runner.run())
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return run_dir, metrics


def test_metrics_contain_platform_commit_source(tmp_path: Path) -> None:
    run_dir, metrics = _run(tmp_path, platform="wx", commit="a1b2c3")
    assert metrics["platform"] == "wx"
    assert metrics["commit"] == "a1b2c3"
    assert metrics["dataset_source"] == "user-specified"
    # run_id 形如 {adapter}_{platform}_{commit前8位}_{dataset}_{timestamp}
    assert metrics["run_id"].startswith("mock-strong_wx_a1b2c3_sample_dataset_")


def test_metrics_defaults_without_platform(tmp_path: Path) -> None:
    _, metrics = _run(tmp_path)
    assert metrics["platform"] == "default"
    assert metrics["commit"] is None
    assert metrics["dataset_source"] == "user-specified"
    assert metrics["run_id"].startswith("mock-strong_default_sample_dataset_")


def test_metrics_dataset_source_passthrough(tmp_path: Path) -> None:
    _, metrics = _run(tmp_path, dataset_source="builtin-office")
    assert metrics["dataset_source"] == "builtin-office"


def test_report_has_config_table_at_top(tmp_path: Path) -> None:
    run_dir, _ = _run(tmp_path, platform="wx", commit="a1b2c3")
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    head = report.split("## 摘要", 1)[0]  # 标题之后、正文第一段之前
    assert "## 评测配置" in head
    assert "测试集路径" in head
    assert "测试集来源" in head
    assert "平台" in head
    assert "commit" in head
    assert "判分器" in head
    assert "驱动 LLM" in head
    assert "判分模型" in head
    assert "adapter 版本" in head
    assert "数据集指纹" in head
    assert "wx" in head and "a1b2c3" in head
    assert "user-specified" in head


def test_copy_report_with_figures(tmp_path: Path) -> None:
    """copy_report：md 与引用的 png 一并复制，png 加时间戳前缀并修正引用。"""
    run_dir, _ = _run(tmp_path)
    dest_dir = tmp_path / "exported"
    ts = "260918_120000"
    dest_md = copy_report(
        str(run_dir / "report.md"),
        str(dest_dir),
        f"eval_report_mock-strong_wx_{ts}.md",
        ts,
    )
    assert Path(dest_md).is_file()
    # 引用的两张图已复制为带前缀文件名
    assert (dest_dir / f"{ts}_fig_category_accuracy.png").is_file()
    assert (dest_dir / f"{ts}_fig_latency_hist.png").is_file()
    content = Path(dest_md).read_text(encoding="utf-8")
    assert f"{ts}_fig_category_accuracy.png" in content
    assert f"{ts}_fig_latency_hist.png" in content
    # 原引用已被替换
    assert "](fig_category_accuracy.png)" not in content
    assert "](fig_latency_hist.png)" not in content


def test_metrics_contain_new_metadata(tmp_path: Path) -> None:
    """metrics.json 顶层新增 schema_version / 可比性元数据 / 数据集指纹。"""
    _, metrics = _run(tmp_path)
    assert metrics["schema_version"] == "0.2"
    assert metrics["driver_llm"] is None
    assert metrics["adapter_version"] is None
    # heuristic judge 无 .model 属性 -> judge_model 保持 None
    assert metrics["judge_model"] is None

    fingerprint = metrics["dataset_fingerprint"]
    assert fingerprint["name"] == Path(SAMPLE_DATASET_PATH).name
    assert fingerprint["bytes"] == Path(SAMPLE_DATASET_PATH).stat().st_size
    assert isinstance(fingerprint["sha256"], str)
    assert len(fingerprint["sha256"]) == 16


def test_metadata_passthrough_and_judge_model_autofill(tmp_path: Path) -> None:
    """显式元数据原样记录；judge_model 缺省时从 judge 对象的 .model 自动补全。"""

    class _ModelJudge:
        """带 .model 的极简 judge，用于验证 judge_model 自动补全。"""

        name = "stub-judge"
        model = "auto-model"

        def grade(self, question: object, gold_answer: str, answer: str) -> dict:
            return {"correct": True}

    runner = Runner(
        MockStrongAdapter(),
        str(SAMPLE_DATASET_PATH),
        judge=_ModelJudge(),
        out_dir=str(tmp_path / "runs"),
        driver_llm="gpt-4o",
        adapter_version="v9",
    )
    run_dir = Path(runner.run())
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["judge"] == "stub-judge"
    assert metrics["judge_model"] == "auto-model"
    assert metrics["driver_llm"] == "gpt-4o"
    assert metrics["adapter_version"] == "v9"


def test_report_out_creates_experiment_dir(tmp_path: Path) -> None:
    """report_out 非空 -> 生成 eval_* 实验目录（含 run 副本），metrics.json 记 experiment_dir。"""
    out = tmp_path / "out"
    runner = Runner(
        MockStrongAdapter(),
        str(SAMPLE_DATASET_PATH),
        judge="heuristic",
        out_dir=str(tmp_path / "runs"),
        platform="wx",
        report_out=str(out),
    )
    run_dir = Path(runner.run())

    subs = [p for p in out.iterdir() if p.is_dir()]
    assert len(subs) == 1
    exp_dir = subs[0]
    assert exp_dir.name.startswith("eval_mock-strong_wx_")
    for fname in (
        "report.md",
        "metrics.json",
        "results.jsonl",
        "fig_category_accuracy.png",
        "fig_latency_hist.png",
    ):
        assert (exp_dir / fname).is_file(), f"缺少 {fname}"

    run_copies = [p for p in (exp_dir / "runs").iterdir() if p.is_dir()]
    assert len(run_copies) == 1
    assert (run_copies[0] / "metrics.json").is_file()

    # experiment_dir 写回 run 与实验目录各一份 metrics.json
    run_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert run_metrics["experiment_dir"] == str(exp_dir)
    exp_metrics = json.loads((exp_dir / "metrics.json").read_text(encoding="utf-8"))
    assert exp_metrics["experiment_dir"] == str(exp_dir)


def test_default_out_dir_uses_paths_runs_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """out_dir 缺省（None）时用 paths.runs_dir()，且该默认路径能真正跑通。"""
    monkeypatch.setenv("MEMBENCH_OUTPUT_DIR", str(tmp_path / "output"))
    runner = Runner(MockStrongAdapter(), str(SAMPLE_DATASET_PATH), judge="heuristic")
    assert runner.out_dir == str(tmp_path / "output" / "runs")
    run_dir = Path(runner.run())
    assert run_dir.parent == tmp_path / "output" / "runs"
    assert (run_dir / "report.md").is_file()
