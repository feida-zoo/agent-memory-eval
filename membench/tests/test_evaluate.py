"""evaluate 全量评估测试：宏平均融合、stale 门禁、CLI 参数行为、报告内容。

用 tmp_path 造 3 个迷你合成数据集（每个 <=3 对话 <=6 题）：
- A：全标注（evidence / gold_stale / abstention 均有），充当办公域槽位；
- B：无 evidence、无 gold_stale、无 abstention；
- C：有 abstention、无 evidence。

用 ScriptedAdapter（按问题文本返回固定答案与检索条目）+ heuristic judge，
使各数据集的指标值完全确定、可手算核对。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from membench import cli
from membench.adapters.base import BaseMemoryAdapter, IngestStats, QueryResult
from membench.evaluate import (
    DatasetSpec,
    evaluate,
    fuse_metrics,
    plan_datasets,
    scan_capabilities,
)

#: 拒答话术（与 heuristic judge 的拒答信号词对齐）。
REFUSAL = "我不知道，相关信息不在我的记忆中。"


class ScriptedAdapter(BaseMemoryAdapter):
    """脚本化适配器：按问题文本查表返回固定 (answer, retrieved)，结果完全确定。"""

    name = "scripted"

    def __init__(self, script: dict[str, tuple[str, list[str]]]) -> None:
        self._script = script

    def reset(self) -> None:
        """无状态，无需清空。"""

    def ingest(self, messages: list[dict]) -> IngestStats:
        """不做任何存储。"""
        return IngestStats()

    def query(self, question: str) -> QueryResult:
        answer, retrieved = self._script[question]
        return QueryResult(answer=answer, retrieved=list(retrieved))


def _write_dataset(path: Path, dialogues: list[dict]) -> Path:
    """把 dialogue dict 列表写成 JSONL 数据集文件。"""
    with path.open("w", encoding="utf-8") as f:
        for d in dialogues:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return path


def _session() -> list[dict]:
    return [{"role": "user", "content": "随便聊聊。", "timestamp": None}]


@pytest.fixture()
def mini_datasets(tmp_path: Path) -> dict[str, Path]:
    """写出 A / B / C 三个迷你数据集并返回路径。"""
    a = _write_dataset(
        tmp_path / "a.jsonl",
        [
            {
                "dialogue_id": "a_d1",
                "sessions": [_session()],
                "questions": [
                    {
                        "qid": "a_q1",
                        "question": "项目截止日是哪天？",
                        "gold_answer": "周五",
                        "category": "single_hop",
                        "evidence": ["项目截止日是周五"],
                    },
                    {
                        "qid": "a_q2",
                        "question": "现在项目截止日是哪天？",
                        "gold_answer": "周五",
                        "category": "knowledge_update",
                        "evidence": ["截止日已改为周五"],
                        "gold_stale": "周三",
                    },
                    {
                        "qid": "a_q3",
                        "question": "老板的私人电话是多少？",
                        "gold_answer": "<ABSTAIN>",
                        "category": "abstention",
                        "evidence": [],
                    },
                ],
            }
        ],
    )
    b = _write_dataset(
        tmp_path / "b.jsonl",
        [
            {
                "dialogue_id": "b_d1",
                "sessions": [_session()],
                "questions": [
                    {
                        "qid": "b_q1",
                        "question": "封面是什么颜色？",
                        "gold_answer": "蓝色",
                        "category": "single_hop",
                        "evidence": [],
                    },
                    {
                        "qid": "b_q2",
                        "question": "logo 是什么颜色？",
                        "gold_answer": "绿色",
                        "category": "single_hop",
                        "evidence": [],
                    },
                ],
            }
        ],
    )
    c = _write_dataset(
        tmp_path / "c.jsonl",
        [
            {
                "dialogue_id": "c_d1",
                "sessions": [_session()],
                "questions": [
                    {
                        "qid": "c_q1",
                        "question": "服务器 root 密码是什么？",
                        "gold_answer": "<ABSTAIN>",
                        "category": "abstention",
                        "evidence": [],
                    },
                    {
                        "qid": "c_q2",
                        "question": "下午茶订了什么水果？",
                        "gold_answer": "苹果",
                        "category": "single_hop",
                        "evidence": [],
                    },
                ],
            }
        ],
    )
    return {"A": a, "B": b, "C": c}


@pytest.fixture()
def script() -> dict[str, tuple[str, list[str]]]:
    """问题 -> (answer, retrieved) 脚本；答案设计使指标值可手算。"""
    return {
        # A：q1 对、q2 答旧值（stale 且错）、q3 正确拒答
        "项目截止日是哪天？": ("周五", ["项目截止日是周五"]),
        "现在项目截止日是哪天？": ("截止日是周三", ["截止日已改为周五"]),
        "老板的私人电话是多少？": (REFUSAL, []),
        # B：一对一错，均无 evidence
        "封面是什么颜色？": ("蓝色", ["今天的会议纪要"]),
        "logo 是什么颜色？": ("红色", ["今天的会议纪要"]),
        # C：拒答对 + 常规题对
        "服务器 root 密码是什么？": (REFUSAL, []),
        "下午茶订了什么水果？": ("苹果", ["购物清单"]),
    }


def _specs(mini: dict[str, Path], office_key: str = "A") -> list[DatasetSpec]:
    """构造 A/B/C 的 DatasetSpec 清单，office_key 指定办公域槽位。"""
    return [
        DatasetSpec(
            key=k.lower(),
            name=k,
            path=str(mini[k]),
            source="user-specified",
            is_office_slot=(k == office_key),
        )
        for k in ("A", "B", "C")
    ]


def _run_evaluate(
    tmp_path: Path,
    mini: dict[str, Path],
    script: dict[str, tuple[str, list[str]]],
    office_key: str = "A",
) -> dict:
    """执行全量评估并返回解析后的 evaluate_summary.json（附带 _rc 与 _eval_dir）。"""
    rc = evaluate(
        ScriptedAdapter(script),
        "heuristic",
        out_dir=str(tmp_path / "runs"),
        specs=_specs(mini, office_key=office_key),
    )
    eval_dirs = sorted((tmp_path / "runs").glob("evaluate_*"))
    assert eval_dirs, "缺少 evaluate_<时间戳> 目录"
    eval_dir = eval_dirs[-1]
    summary_path = eval_dir / "evaluate_summary.json"
    assert summary_path.is_file(), "缺少 evaluate_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["_rc"] = rc
    summary["_eval_dir"] = str(eval_dir)
    return summary


def test_scan_capabilities(mini_datasets: dict[str, Path]) -> None:
    """能力标注扫描：A 全有、B 全无、C 仅 abstention。"""
    assert scan_capabilities(mini_datasets["A"]) == {
        "has_evidence": True,
        "has_stale": True,
        "has_abstention": True,
    }
    assert scan_capabilities(mini_datasets["B"]) == {
        "has_evidence": False,
        "has_stale": False,
        "has_abstention": False,
    }
    assert scan_capabilities(mini_datasets["C"]) == {
        "has_evidence": False,
        "has_stale": False,
        "has_abstention": True,
    }


def test_scan_capabilities_sentinel_reject_category(tmp_path: Path) -> None:
    """category 非 abstention、gold 为哨兵值也算拒答题（raw category 数据集）。"""
    path = _write_dataset(
        tmp_path / "sentinel.jsonl",
        [
            {
                "dialogue_id": "d1",
                "sessions": [_session()],
                "questions": [
                    {
                        "qid": "q1",
                        "question": "同事的家庭住址？",
                        "gold_answer": "<ABSTAIN>",
                        "category": "privacy_refuse",
                        "evidence": [],
                    }
                ],
            }
        ],
    )
    assert scan_capabilities(path) == {
        "has_evidence": False,
        "has_stale": False,
        "has_abstention": True,
    }


def test_fusion_macro_average(
    tmp_path: Path,
    mini_datasets: dict[str, Path],
    script: dict[str, tuple[str, list[str]]],
) -> None:
    """宏平均融合：参与规则与手算均值核对。

    手算（heuristic judge + 脚本答案）：
    - A: accuracy=2/3, evidence_recall=1.0, stale_rate=1.0, abstention_accuracy=1.0
    - B: accuracy=0.5（无 evidence/stale/abstention 标注）
    - C: accuracy=1.0, abstention_accuracy=1.0（无 evidence 标注）
    融合 accuracy = (2/3 + 0.5 + 1.0) / 3 = 13/18。
    """
    summary = _run_evaluate(tmp_path, mini_datasets, script)
    fused = summary["fused"]

    assert fused["evidence_recall"]["participants"] == ["A"]
    assert fused["evidence_recall"]["value"] == pytest.approx(1.0)
    assert fused["stale_rate"]["participants"] == ["A"]
    assert fused["stale_rate"]["value"] == pytest.approx(1.0)
    assert fused["abstention_accuracy"]["participants"] == ["A", "C"]
    assert fused["abstention_accuracy"]["value"] == pytest.approx(1.0)

    for metric in ("accuracy", "token_f1", "bleu1"):
        assert fused[metric]["participants"] == ["A", "B", "C"]
    assert fused["accuracy"]["value"] == pytest.approx(13 / 18)


def test_gate_fail_exit_code(
    tmp_path: Path,
    mini_datasets: dict[str, Path],
    script: dict[str, tuple[str, list[str]]],
) -> None:
    """办公域槽位 stale_rate=1.0 > 0.03 -> 门禁 FAIL、总评不达标、退出码 1。"""
    summary = _run_evaluate(tmp_path, mini_datasets, script, office_key="A")
    assert summary["gate"]["status"] == "FAIL"
    assert summary["_rc"] == 1

    report = (Path(summary["_eval_dir"]) / "evaluate_report.md").read_text(encoding="utf-8")
    assert "门禁结论： FAIL" in report
    assert "总评结论： 不达标" in report


def test_gate_skip_without_stale_annotation(
    tmp_path: Path,
    mini_datasets: dict[str, Path],
    script: dict[str, tuple[str, list[str]]],
) -> None:
    """办公域槽位无 gold_stale 标注（用 B 充当）-> 门禁 SKIP、退出码 0。"""
    summary = _run_evaluate(tmp_path, mini_datasets, script, office_key="B")
    assert summary["gate"]["status"] == "SKIP"
    assert "无 gold_stale 标注" in summary["gate"]["detail"]
    assert summary["_rc"] == 0


def test_report_contents(
    tmp_path: Path,
    mini_datasets: dict[str, Path],
    script: dict[str, tuple[str, list[str]]],
) -> None:
    """报告含综合评级、"评价（百分比）"样式与"参与数据集"列。"""
    summary = _run_evaluate(tmp_path, mini_datasets, script)
    report = (Path(summary["_eval_dir"]) / "evaluate_report.md").read_text(encoding="utf-8")

    assert "**综合评级：" in report
    assert "/4.0）**" in report
    # "评价（百分比）"样式，如 优秀（100.0%）
    for label in ("优秀（", "良好（", "中等（", "较差（"):
        if label in report:
            break
    else:  # pragma: no cover - 防御性断言
        raise AssertionError("报告中未出现 评价（百分比） 样式")
    assert "%）" in report
    assert "参与数据集" in report
    assert "## 分数据集明细" in report
    assert "## 诊断指标（不参与评级）" in report
    # 效率成本段 + 报告配图
    assert "## 效率成本（分数据集）" in report
    assert "avg_output_tokens" in report
    assert "avg_ingest_latency_ms" in report
    assert (Path(summary["_eval_dir"]) / "fig_dataset_overview.png").is_file()
    assert "](fig_dataset_overview.png)" in report
    # 综合评级保留两位小数（不强行截断到一位）
    import re

    m = re.search(r"综合评级：\S+（(\d+\.\d+)/4.0）", report)
    assert m is not None and len(m.group(1).split(".")[1]) == 2
    # 数据集范围行：先确认本次实际评测的数据集 key 与来源
    assert "数据集范围" in report
    assert "a（user-specified）" in report
    # 报告目录下复制了各数据集 run 副本，链接指向相对路径 runs/<run_id>/report.md
    report_dir = Path(summary["_eval_dir"])
    assert summary["experiment_dir"] == str(report_dir)
    for rel in summary["runs"].values():
        assert rel.startswith("runs/")
        assert (report_dir / rel / "report.md").is_file()
    assert "(runs/" in report
    # B 不参与 stale_rate，明细表中应显示 "未定义"（避免业务方把 "未定义"
    # 误读为 "0% 召回"，统一用 "未定义" 取代 "—" 占位）。
    detail_section = report.split("## 分数据集明细")[1]
    assert "未定义" in detail_section


def test_fuse_metrics_empty_participants() -> None:
    """所有集都无某能力标注时，该指标融合值为 None、参与集为空。"""
    from membench.evaluate import DatasetResult

    res = DatasetResult(
        spec=DatasetSpec(key="b", name="B", path="x", source="builtin"),
        run_dir="x",
        metrics={"overall": {"accuracy": 0.5, "evidence_recall": 0.0}},
        capabilities={"has_evidence": False, "has_stale": False, "has_abstention": False},
    )
    fused = fuse_metrics([res])
    assert fused["evidence_recall"]["value"] is None
    assert fused["evidence_recall"]["participants"] == []
    assert fused["accuracy"]["value"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# CLI 行为
# ---------------------------------------------------------------------------


def test_cli_quick_and_beam10m_mutually_exclusive() -> None:
    """--quick 与 --beam-10m 互斥，argparse 报错退出。"""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["evaluate", "--adapter", "mock-strong", "--quick", "--beam-10m"])
    assert exc_info.value.code == 2


def _stub_fetched(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """让公开集解析到虚构路径（离线环境无下载），使 plan_datasets 继续后续检查。"""
    monkeypatch.setattr(
        "membench.evaluate.resolve_available_path",
        lambda key: tmp_path / f"{key}.jsonl",
    )


def test_cli_beam_missing_systemexit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """无 --quick 且 BEAM 缺失 -> SystemExit 提示 dataset fetch beam_1m。"""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # BEAM 必然不存在
    monkeypatch.setenv("MEMBENCH_USER_DATA_DIR", str(tmp_path / "no_user_data"))
    _stub_fetched(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["evaluate", "--adapter", "mock-strong", "--judge", "heuristic"])
    assert "fetch beam_1m" in str(exc_info.value)


def test_cli_quick_skips_beam_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--quick 不检查 BEAM 文件存在，且清单中无 BEAM 槽位。"""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MEMBENCH_USER_DATA_DIR", str(tmp_path / "no_user_data"))
    _stub_fetched(monkeypatch, tmp_path)
    specs = plan_datasets(quick=True)
    assert [s.key for s in specs] == ["office", "locomo", "longmemeval_oracle", "gatemem_office"]
    # 公开集统一标注 fetch 来源（不再被当内置集）。
    assert all(s.source == "fetch" for s in specs[1:])

    # CLI 层面：--quick 时不触发 SystemExit（stub 掉真正评测避免跑内置大数据集）。
    captured: dict[str, object] = {}

    def _stub_evaluate(adapter, judge, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return 0

    import membench.evaluate as evaluate_mod

    monkeypatch.setattr(evaluate_mod, "evaluate", _stub_evaluate)
    rc = cli.main(["evaluate", "--adapter", "mock-strong", "--quick"])
    assert rc == 0
    assert captured["quick"] is True
    assert captured["beam_10m"] is False


def test_plan_datasets_missing_fetch_systemexit(monkeypatch: pytest.MonkeyPatch) -> None:
    """公开集未下载（resolve_available_path 返回 None）-> SystemExit 提示 dataset fetch。"""
    monkeypatch.setattr("membench.evaluate.resolve_available_path", lambda key: None)
    with pytest.raises(SystemExit) as exc_info:
        plan_datasets(quick=True)
    msg = str(exc_info.value)
    assert "dataset fetch" in msg
    assert "locomo" in msg


def test_plan_datasets_missing_lme_s_systemexit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--include-lme-s 且 LongMemEval-S 未下载 -> SystemExit 提示 fetch longmemeval_s。"""
    monkeypatch.setattr(
        "membench.evaluate.resolve_available_path",
        lambda key: None if key == "longmemeval_s" else tmp_path / f"{key}.jsonl",
    )
    with pytest.raises(SystemExit) as exc_info:
        plan_datasets(quick=True, include_lme_s=True)
    assert "dataset fetch longmemeval_s" in str(exc_info.value)


def test_run_datasets_forwards_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--jobs 必须透传到 Runner。

    历史 bug：run_datasets 接受并文档承诺 jobs 透传，但从未传给 Runner，
    导致 evaluate / evaluate-compare 的 --jobs 静默失效（含 evidence-match
    的 LLM 逐题调用也因此完全串行）。
    """
    import membench.evaluate as ev_mod

    dataset = _write_dataset(
        tmp_path / "d.jsonl",
        [
            {
                "dialogue_id": "d1",
                "sessions": [_session()],
                "questions": [
                    {
                        "qid": "q1",
                        "question": "q?",
                        "gold_answer": "a",
                        "category": "single_hop",
                        "evidence": [],
                    }
                ],
            }
        ],
    )
    captured: dict[str, object] = {}

    class _FakeRunner:
        def __init__(self, adapter, path, **kwargs):  # noqa: ANN001
            captured.update(kwargs)

        def run(self) -> str:
            run_dir = tmp_path / "fake_run"
            run_dir.mkdir(exist_ok=True)
            (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
            return str(run_dir)

    monkeypatch.setattr(ev_mod, "Runner", _FakeRunner)
    specs = [DatasetSpec(key="k", name="K", path=str(dataset), source="user-specified")]
    results = ev_mod.run_datasets(object(), "heuristic", specs, str(tmp_path / "runs"), jobs=7)
    assert captured["jobs"] == 7
    assert results and results[0].metrics == {}


def test_cli_run_forwards_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI `run --jobs N` 必须传给 Runner（历史 bug：从未传）。"""
    from membench.data import SAMPLE_DATASET_PATH

    captured: dict[str, object] = {}

    class _FakeRunner:
        def __init__(self, adapter, path, **kwargs):  # noqa: ANN001
            captured.update(kwargs)

        def run(self) -> str:
            return str(tmp_path)

    monkeypatch.setattr(cli, "Runner", _FakeRunner)
    rc = cli.main(
        [
            "run",
            "--adapter", "mock-weak",
            "--dataset", str(SAMPLE_DATASET_PATH),
            "--jobs", "5",
            "--out", str(tmp_path / "runs"),
        ]
    )
    assert rc == 0
    assert captured["jobs"] == 5
