"""Runner platform/commit 元数据、report.md 评测配置表与实验目录归属测试。

产物结构见 :mod:`membench.layout`：``out_dir`` 即**实验目录**，数据集级
metrics / 逐题明细 / 报告落 ``<实验目录>/runs/<dataset>/report/``，任务级报告落
``<实验目录>/reports/``；本文件的断言默认读数据集级那份。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from membench import layout
from membench.adapters import MockStrongAdapter
from membench.data import SAMPLE_DATASET_PATH
from membench.report import copy_report
from membench.runner import Runner


def _run(exp_dir: Path, **kwargs) -> tuple[Path, dict]:
    """把 ``exp_dir`` 当实验目录跑一次，返回（数据集产物目录, 数据集级 metrics）。"""
    runner = Runner(
        MockStrongAdapter(),
        str(SAMPLE_DATASET_PATH),
        judge="heuristic",
        out_dir=str(exp_dir),
        **kwargs,
    )
    dataset_dir = runner.run()
    assert dataset_dir is not None
    metrics = json.loads(
        (Path(dataset_dir) / "report" / "metrics.json").read_text(encoding="utf-8")
    )
    return Path(dataset_dir), metrics


def test_metrics_contain_platform_commit_source(tmp_path: Path) -> None:
    dataset_dir, metrics = _run(tmp_path / "exp", platform="wx", commit="a1b2c3")
    assert metrics["platform"] == "wx"
    assert metrics["commit"] == "a1b2c3"
    assert metrics["dataset_source"] == "user-specified"
    # run_id 形如 {adapter}_{platform}_{commit前8位}_{dataset}_{timestamp}
    assert metrics["run_id"].startswith("mock-strong_wx_a1b2c3_sample_dataset_")
    # run_id 不再是目录名：数据集产物固定落 runs/<dataset>/
    assert dataset_dir == tmp_path / "exp" / "runs" / "sample_dataset"


def test_metrics_defaults_without_platform(tmp_path: Path) -> None:
    _, metrics = _run(tmp_path / "exp")
    assert metrics["platform"] == "default"
    assert metrics["commit"] is None
    assert metrics["dataset_source"] == "user-specified"
    assert metrics["run_id"].startswith("mock-strong_default_sample_dataset_")


def test_metrics_dataset_source_passthrough(tmp_path: Path) -> None:
    _, metrics = _run(tmp_path / "exp", dataset_source="builtin-office")
    assert metrics["dataset_source"] == "builtin-office"


def test_report_has_config_table_at_top(tmp_path: Path) -> None:
    dataset_dir, _ = _run(tmp_path / "exp", platform="wx", commit="a1b2c3")
    report = (dataset_dir / "report" / "report.md").read_text(encoding="utf-8")
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
    """copy_report：md 与引用的 png 一并复制（保持同名同目录引用），extra_files 一并带出。"""
    dataset_dir, _ = _run(tmp_path / "exp")
    report_dir = dataset_dir / "report"
    dest_dir = tmp_path / "exported"
    dest_md = copy_report(
        str(report_dir / "report.md"),
        str(dest_dir),
        layout.FILE_REPORT_MD,
        extra_files=[str(report_dir / "metrics.json")],
    )
    assert Path(dest_md) == dest_dir / layout.FILE_REPORT_MD
    content = Path(dest_md).read_text(encoding="utf-8")
    # 引用的两张图按原文件名复制到同一目录，引用文本仍然可解析
    for fig in (layout.FIG_CATEGORY, layout.FIG_LATENCY):
        assert (dest_dir / fig).is_file(), f"缺少 {fig}"
        assert f"]({fig})" in content
    assert (dest_dir / "metrics.json").is_file()
    # 不再有 ts_prefix 改名：目标目录里只有原文件名
    assert not [p.name for p in dest_dir.iterdir() if re.match(r"^\d{6}_\d{6}_", p.name)]
    # 引用一律同目录相对文件名（无绝对路径 / 越界路径）
    refs = [r for r in re.findall(r"\]\(([^)]+)\)", content) if "://" not in r]
    assert refs
    for ref in refs:
        assert not ref.startswith("/") and ".." not in ref, f"非相对同目录引用: {ref}"

    # 非本地引用（http/https）原样保留，不当作文件复制
    remote_src = tmp_path / "remote" / "report.md"
    remote_src.parent.mkdir()
    remote_src.write_text(
        "# r\n\n![远端](https://example.invalid/a.png)\n", encoding="utf-8"
    )
    remote_dest = tmp_path / "remote_dest"
    copy_report(str(remote_src), str(remote_dest), layout.FILE_REPORT_MD)
    assert "](https://example.invalid/a.png)" in (
        remote_dest / layout.FILE_REPORT_MD
    ).read_text(encoding="utf-8")
    assert not (remote_dest / "a.png").exists()


def test_metrics_contain_new_metadata(tmp_path: Path) -> None:
    """metrics.json 顶层新增 schema_version / 可比性元数据 / 数据集指纹。"""
    _, metrics = _run(tmp_path / "exp")
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
        out_dir=str(tmp_path / "exp"),
        driver_llm="gpt-4o",
        adapter_version="v9",
    )
    dataset_dir = Path(runner.run())
    metrics = json.loads(
        (dataset_dir / "report" / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["judge"] == "stub-judge"
    assert metrics["judge_model"] == "auto-model"
    assert metrics["driver_llm"] == "gpt-4o"
    assert metrics["adapter_version"] == "v9"


def test_runner_out_dir_is_experiment_dir(tmp_path: Path) -> None:
    """out_dir 即实验目录：产物归属落账 metrics，任务级 reports/ 由数据集级报告派生。"""
    exp_dir = tmp_path / "exp"
    dataset_dir, metrics = _run(exp_dir, platform="wx")

    assert dataset_dir == exp_dir / "runs" / "sample_dataset"
    assert metrics["experiment_dir"] == str(exp_dir)
    assert metrics["dataset_dir"] == str(dataset_dir)
    assert metrics["dataset"] == "sample_dataset"
    assert metrics["run_mode"] == "full"
    assert metrics["shards"] == 1 and metrics["shard_by"] == "question"

    # 任务级 reports/：md + 两图 + metrics（内容 = 数据集级），不重复落逐题明细
    task_dir = exp_dir / "reports"
    for fname in (
        layout.FILE_REPORT_MD,
        layout.FILE_METRICS,
        layout.FIG_CATEGORY,
        layout.FIG_LATENCY,
    ):
        assert (task_dir / fname).is_file(), f"缺少 reports/{fname}"
    assert not (task_dir / layout.FILE_RESULTS).exists()
    assert json.loads((task_dir / layout.FILE_METRICS).read_text(encoding="utf-8")) == metrics

    # 对话级 metrics 记录同一实验 / 数据集归属与自己的对话 id
    dlg_metrics = json.loads(
        (dataset_dir / "d001" / "report" / "metrics.json").read_text(encoding="utf-8")
    )
    assert dlg_metrics["experiment_dir"] == str(exp_dir)
    assert dlg_metrics["dataset_dir"] == str(dataset_dir)
    assert dlg_metrics["dialogue_id"] == "d001"
    assert dlg_metrics["token_usage"]["n_dialogues"] == 1


def test_experiment_path_uses_output_root_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """experiment_path 缺省 base 用 paths.output_root()，且该默认路径能真正跑通。"""
    root = tmp_path / "output"
    monkeypatch.setenv("MEMBENCH_OUTPUT_DIR", str(root))
    exp_dir = Path(
        layout.experiment_path(
            None, layout.eval_subdir("mock-strong", None, ts="20261024-120000")
        )
    )
    assert exp_dir == root / "eval_mock-strong_default_20261024-120000"

    dataset_dir, metrics = _run(exp_dir)
    assert (dataset_dir / "report" / "report.md").is_file()
    assert metrics["experiment_dir"] == str(exp_dir)


def test_retrieval_match_metadata_and_overlap(tmp_path: Path) -> None:
    """检索命中口径落账 + overlap 对近原文改写的命中（runner 集成）。"""
    from membench.adapters.base import BaseMemoryAdapter, IngestStats, QueryResult
    from membench.metrics.evidence import OverlapMatcher

    class _WrappedAdapter(BaseMemoryAdapter):
        name = "wrapped"

        def reset(self) -> None:
            pass

        def ingest(self, messages: list[dict]) -> IngestStats:
            return IngestStats()

        def query(self, question: str) -> QueryResult:
            # 检索条目插入了空白：子串 miss、token 包含度命中
            return QueryResult(answer="周五", retrieved=["项目截止日 是 周五"])

    dataset = tmp_path / "wrapped.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "dialogue_id": "d1",
                "sessions": [[{"role": "user", "content": "随便聊聊。", "timestamp": None}]],
                "questions": [
                    {
                        "qid": "q1",
                        "question": "项目截止日是哪天？",
                        "gold_answer": "周五",
                        "category": "single_hop",
                        "evidence": ["项目截止日是周五"],
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    # 默认 substring：命中失败，口径落账
    runner = Runner(
        _WrappedAdapter(), str(dataset), judge="heuristic", out_dir=str(tmp_path / "exp1")
    )
    dataset_dir = Path(runner.run())
    report_dir = dataset_dir / "report"
    metrics = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["retrieval_match"] == "substring"
    rec = json.loads(
        (report_dir / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
    )
    assert rec["evidence_hit"] is False and rec["recall@5"] == 0.0

    # overlap：命中，阈值随口径落账
    runner = Runner(
        _WrappedAdapter(),
        str(dataset),
        judge="heuristic",
        out_dir=str(tmp_path / "exp2"),
        evidence_matcher=OverlapMatcher(threshold=0.6),
    )
    dataset_dir = Path(runner.run())
    report_dir = dataset_dir / "report"
    metrics = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["retrieval_match"] == "overlap"
    assert metrics["retrieval_match_threshold"] == pytest.approx(0.6)
    rec = json.loads(
        (report_dir / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
    )
    assert rec["evidence_hit"] is True and rec["recall@5"] == 1.0
    # 对话级 metrics 与数据集级同一口径
    dlg_metrics = json.loads(
        (dataset_dir / "d1" / "report" / "metrics.json").read_text(encoding="utf-8")
    )
    assert dlg_metrics["retrieval_match"] == "overlap"
    assert dlg_metrics["retrieval_match_threshold"] == pytest.approx(0.6)


def test_retrieval_match_fallbacks_per_run_delta(tmp_path: Path) -> None:
    """matcher 实例跨 run 共享时，metrics.json 记本次 run 的回退增量。"""
    from membench.data.schema import load_dataset
    from membench.metrics.evidence import LLMEvidenceMatcher

    matcher = LLMEvidenceMatcher(base_url="http://x")

    def _boom(evidence: list[str], retrieved: list[str]) -> list[list[int]]:
        raise ConnectionError("model service down")

    matcher._chat = _boom  # type: ignore[method-assign]
    n_evidence = sum(
        1
        for d in load_dataset(SAMPLE_DATASET_PATH)
        for q in d.questions
        if q.evidence and not q.ignore
    )
    assert n_evidence > 0

    for name in ("r1", "r2"):
        runner = Runner(
            MockStrongAdapter(),
            str(SAMPLE_DATASET_PATH),
            judge="heuristic",
            out_dir=str(tmp_path / name),
            evidence_matcher=matcher,
        )
        dataset_dir = Path(runner.run())
        metrics = json.loads(
            (dataset_dir / "report" / "metrics.json").read_text(encoding="utf-8")
        )
        # 每次 run 只记自己的增量（而非累计值）
        assert metrics["retrieval_match_fallbacks"] == n_evidence
    assert matcher.fallbacks == 2 * n_evidence


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
