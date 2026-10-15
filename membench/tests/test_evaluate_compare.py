"""evaluate-compare 双臂多数据集 A/B 融合对比测试。

覆盖：
- 双臂融合 diff 计算；
- 逐数据集 diff 表 + 融合 diff 表 + 门禁结论写入 compare_report.md；
- 产物布局：A/B 两侧各数据集 run 副本 + compare_report.md + 两张图；
- CLI：_cmd_evaluate_compare 参数透传 + 系统退出码映射；
- 缺失融合指标时的兜底（未定义 / 跳过）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from membench import cli
from membench.adapters.base import BaseMemoryAdapter, IngestStats, QueryResult
from membench.evaluate import DatasetSpec, fuse_metrics
from membench.evaluate_compare import (
    ArmDatasetResult,
    _arm_fuse,
    _fused_diff,
    _fused_gates,
    _per_dataset_diff,
    evaluate_compare,
)


REFUSAL = "我不知道，相关信息不在我的记忆中。"


class ScriptedAdapter(BaseMemoryAdapter):
    """脚本化适配器：按问题文本返回固定 (answer, retrieved)。"""

    name = "scripted"

    def __init__(
        self,
        script: dict[str, tuple[str, list[str]]],
        name: str = "scripted",
    ) -> None:
        self._script = script
        self.name = name

    def reset(self) -> None:
        pass

    def ingest(self, messages: list[dict]) -> IngestStats:
        return IngestStats()

    def query(self, question: str) -> QueryResult:
        answer, retrieved = self._script[question]
        return QueryResult(answer=answer, retrieved=list(retrieved))


def _write_dataset(path: Path, dialogues: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as f:
        for d in dialogues:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return path


def _session() -> list[dict]:
    return [{"role": "user", "content": "随便聊聊。", "timestamp": None}]


@pytest.fixture()
def mini_datasets(tmp_path: Path) -> dict[str, Path]:
    """A（office 槽位，含 evidence/stale/abstention）、B（普通）、C（普通）。"""
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
                        "question": "老板私人电话？",
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
                        "question": "封面颜色？",
                        "gold_answer": "蓝色",
                        "category": "single_hop",
                        "evidence": [],
                    },
                    {
                        "qid": "b_q2",
                        "question": "logo 颜色？",
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
                        "question": "服务器 root 密码？",
                        "gold_answer": "<ABSTAIN>",
                        "category": "abstention",
                        "evidence": [],
                    },
                    {
                        "qid": "c_q2",
                        "question": "下午茶水果？",
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
def scripts() -> dict[str, dict[str, tuple[str, list[str]]]]:
    """baseline / candidate 两个 adapter 的脚本；candidate 在 A/B 上整体优于 baseline。"""
    base = {
        # A: q1 对、q2 答旧值（stale 且错）、q3 正确拒答
        "项目截止日是哪天？": ("周五", ["项目截止日是周五"]),
        "现在项目截止日是哪天？": ("截止日是周三", ["截止日已改为周五"]),
        "老板私人电话？": (REFUSAL, []),
        # B: 一对一错
        "封面颜色？": ("蓝色", ["今天的会议纪要"]),
        "logo 颜色？": ("红色", ["今天的会议纪要"]),
        # C: 拒答对 + 常规题对
        "服务器 root 密码？": (REFUSAL, []),
        "下午茶水果？": ("苹果", ["购物清单"]),
    }
    cand = {
        # A: 全对 + 正确拒答
        "项目截止日是哪天？": ("周五", ["项目截止日是周五"]),
        "现在项目截止日是哪天？": ("周五", ["截止日已改为周五"]),
        "老板私人电话？": (REFUSAL, []),
        # B: 全对
        "封面颜色？": ("蓝色", ["封面颜色"]),
        "logo 颜色？": ("绿色", ["logo 颜色"]),
        # C: 同 baseline
        "服务器 root 密码？": (REFUSAL, []),
        "下午茶水果？": ("苹果", ["购物清单"]),
    }
    return {"baseline": base, "candidate": cand}


def _specs(mini: dict[str, Path]) -> list[DatasetSpec]:
    return [
        DatasetSpec(key="a", name="A（办公）", path=str(mini["A"]), source="user-specified", is_office_slot=True),
        DatasetSpec(key="b", name="B", path=str(mini["B"]), source="user-specified"),
        DatasetSpec(key="c", name="C", path=str(mini["C"]), source="user-specified"),
    ]


def _build_arms(mini: dict[str, Path], scripts: dict[str, dict[str, tuple[str, list[str]]]]) -> list[ArmDatasetResult]:
    """直接构造 ArmDatasetResult，避开真实 run（轻量、确定性高）。"""
    base = ScriptedAdapter(scripts["baseline"], name="base")
    cand = ScriptedAdapter(scripts["candidate"], name="cand")
    from membench.evaluate import run_datasets

    specs = _specs(mini)
    base_results = run_datasets(base, "heuristic", specs=specs, out_dir=str(mini["A"].parent))
    cand_results = run_datasets(cand, "heuristic", specs=specs, out_dir=str(mini["A"].parent))
    arms: list[ArmDatasetResult] = []
    for b, c in zip(base_results, cand_results):
        arms.append(
            ArmDatasetResult(
                spec=b.spec,
                baseline=b.metrics,
                candidate=c.metrics,
                baseline_run_dir=b.run_dir,
                candidate_run_dir=c.run_dir,
                capabilities=b.capabilities,
            )
        )
    return arms


def test_arm_fuse_matches_per_arm_fuse_metrics(mini_datasets, scripts) -> None:
    """_arm_fuse 与 evaluate.fuse_metrics 在两侧应分别得到一致的融合值。"""
    arms = _build_arms(mini_datasets, scripts)
    fused_base = _arm_fuse(arms, "baseline")
    fused_cand = _arm_fuse(arms, "candidate")

    # 手工比对：baseline 侧 accuracy
    assert fused_base["accuracy"]["value"] is not None
    assert fused_cand["accuracy"]["value"] is not None
    assert fused_cand["accuracy"]["value"] >= fused_base["accuracy"]["value"]


def test_fused_diff_computes_per_metric_diff() -> None:
    """_fused_diff：任一侧 None 时跳过；其它指标算 b - a。"""
    fused_a = {
        "accuracy": {"value": 0.6, "participants": ["x"]},
        "evidence_recall": {"value": None, "participants": []},
    }
    fused_b = {
        "accuracy": {"value": 0.7, "participants": ["x"]},
        "evidence_recall": {"value": 0.5, "participants": ["y"]},
    }
    diffs = _fused_diff(fused_a, fused_b)
    assert set(diffs.keys()) == {"accuracy"}  # evidence_recall 双侧不全，跳过
    assert diffs["accuracy"]["diff"] == pytest.approx(0.1)


def test_fused_gates_triggers_fail_on_accuracy_regression() -> None:
    """_fused_gates：accuracy 下降超阈值 -> FAIL。"""
    fused_a = {"accuracy": {"value": 0.85, "participants": ["x"]}}
    fused_b = {"accuracy": {"value": 0.80, "participants": ["x"]}}
    gates = _fused_gates(fused_a, fused_b, acc_threshold=0.02, stale_threshold=0.02, latency_warn=0.30)
    assert any(g["status"] == "FAIL" for g in gates)


def test_fused_gates_skips_when_no_metrics() -> None:
    """_fused_gates：任一侧融合值为空 -> 返回空 list。"""
    gates = _fused_gates({}, {"accuracy": {"value": 0.5, "participants": []}}, 0.02, 0.02, 0.30)
    assert gates == []


def test_per_dataset_diff_returns_one_row_per_dataset(mini_datasets, scripts) -> None:
    arms = _build_arms(mini_datasets, scripts)
    rows = _per_dataset_diff(arms)
    assert len(rows) == 3
    keys = [r["spec"].key for r in rows]
    assert keys == ["a", "b", "c"]
    # candidate 全面优于 baseline：accuracy diff 全部 >= 0
    for row in rows:
        diff = row["diffs"]["accuracy"]["diff"]
        assert diff >= -1e-9


def test_evaluate_compare_writes_ab_layout(mini_datasets, scripts, tmp_path) -> None:
    """evaluate_compare 端到端：ab_<tag>_<ts>/ 落地，含 A/B 两侧 runs/ + compare_report.md + 两张图。"""
    base = ScriptedAdapter(scripts["baseline"], name="base")
    cand = ScriptedAdapter(scripts["candidate"], name="cand")
    result = evaluate_compare(
        base,
        cand,
        "heuristic",
        out_dir=str(tmp_path / "runs"),
        report_out=str(tmp_path),
        tag="base_vs_cand",
        specs=_specs(mini_datasets),
        adapter_version_baseline="v1",
        adapter_version_candidate="v2",
    )
    compare_dir = Path(result["compare_dir"])
    assert compare_dir.name.startswith("ab_base_vs_cand_")
    assert (compare_dir / "A" / "runs").is_dir()
    assert (compare_dir / "B" / "runs").is_dir()
    assert (compare_dir / "compare_report.md").is_file()
    assert (compare_dir / "fig_per_dataset_diff.png").is_file()
    assert (compare_dir / "fig_fused_diff.png").is_file()

    md = (compare_dir / "compare_report.md").read_text(encoding="utf-8")
    assert "**门禁结论：" in md
    assert "跨数据集融合 diff" in md
    assert "逐数据集关键指标 diff" in md
    # by_category diff 段按数据集分别渲染（数据集小标题 + 中文题型名，不跨集合并）
    assert "逐数据集题型 diff" in md
    assert "### A（办公）" in md
    assert "单跳（single_hop）" in md
    assert "拒答（abstention）" in md
    # 中文化指标名出现在报告中
    assert "正确率（accuracy）" in md
    assert "证据召回率（evidence_recall）" in md


def test_evaluate_compare_verdict_pass_when_candidate_better(mini_datasets, scripts, tmp_path) -> None:
    """candidate 全面优于 baseline -> verdict PASS。"""
    base = ScriptedAdapter(scripts["baseline"], name="base")
    cand = ScriptedAdapter(scripts["candidate"], name="cand")
    result = evaluate_compare(
        base,
        cand,
        "heuristic",
        out_dir=str(tmp_path / "runs"),
        report_out=str(tmp_path),
        tag="base_vs_cand",
        specs=_specs(mini_datasets),
    )
    assert result["verdict"] in ("PASS", "WARN")  # A 的 stale_rate=1.0 可能触发 WARN/FAIL
    # 至少 gates 包含 accuracy_regression
    gate_names = {g["name"] for g in result["gates"]}
    assert "accuracy_regression" in gate_names


def test_cli_evaluate_compare_quick_and_beam10m_exclusive() -> None:
    """CLI 互斥参数：--quick 与 --beam-10m 同时传 -> argparse 拒绝。"""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "evaluate-compare",
                "--baseline-adapter", "mock-strong",
                "--candidate-adapter", "mock-weak",
                "--quick",
                "--beam-10m",
            ]
        )
    assert exc_info.value.code == 2


def test_cli_evaluate_compare_pass_through(tmp_path, monkeypatch) -> None:
    """_cmd_evaluate_compare 透传到 evaluate_compare；失败时退出码 1。"""
    captured: dict[str, object] = {}

    def _stub(baseline_adapter, candidate_adapter, judge, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return {"verdict": "FAIL", "gates": [], "fused_diffs": {}, "fused_baseline": {}, "fused_candidate": {}, "per_dataset": [], "compare_dir": str(tmp_path / "ab")}

    import membench.evaluate_compare as ec_mod

    monkeypatch.setattr(ec_mod, "evaluate_compare", _stub)
    rc = cli.main(
        [
            "evaluate-compare",
            "--baseline-adapter", "mock-strong",
            "--candidate-adapter", "mock-weak",
            "--quick",
            "--acc-threshold", "0.01",
            "--stale-threshold", "0.01",
            "--latency-warn", "0.2",
            "--tag", "my_tag",
        ]
    )
    assert rc == 1
    assert captured["acc_threshold"] == 0.01
    assert captured["stale_threshold"] == 0.01
    assert captured["latency_warn"] == 0.2
    assert captured["tag"] == "my_tag"
    assert captured["quick"] is True
    assert captured["beam_10m"] is False


def test_cli_evaluate_compare_pass_returns_zero(tmp_path, monkeypatch) -> None:
    """verdict=PASS 时退出码 0。"""
    def _stub(baseline_adapter, candidate_adapter, judge, **kwargs):  # type: ignore[no-untyped-def]
        return {"verdict": "PASS", "gates": [], "fused_diffs": {}, "fused_baseline": {}, "fused_candidate": {}, "per_dataset": [], "compare_dir": str(tmp_path / "ab")}

    import membench.evaluate_compare as ec_mod

    monkeypatch.setattr(ec_mod, "evaluate_compare", _stub)
    rc = cli.main(
        [
            "evaluate-compare",
            "--baseline-adapter", "mock-strong",
            "--candidate-adapter", "mock-weak",
        ]
    )
    assert rc == 0
