"""诊断项落盘与报告呈现：evidence 原文、要点命中率、检索条数 / 空转。

口径来源：GateMem 实跑发现「accuracy 全或无」会把"只差一条要点"记成完全答错，
且 `recall@5` 与适配器返回条数强耦合、空检索没有单独计数。这三项都是**诊断**
指标（不参与评级），但必须落进 ``results.jsonl`` / ``metrics.json`` / 报告。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from membench import layout
from membench.adapters.base import BaseMemoryAdapter, IngestStats, QueryResult
from membench.judges.base import JudgeResult
from membench.runner import EVIDENCE_RECORD_CHARS, Runner

EVIDENCE_LONG = "Alpha 项目的预算从 200,000 USD 调整到 221,000 USD。" + "补充说明。" * 80


class _RetrievalAdapter(BaseMemoryAdapter):
    """``q_empty`` 返回空检索（模拟 ingest 未落盘），其余题返回 8 条命中。"""

    name = "diag"

    def reset(self) -> None:
        return None

    def ingest(self, messages: list[dict]) -> IngestStats:
        return IngestStats()

    def query(self, question: str) -> QueryResult:
        if "q_empty" in question:
            return QueryResult(answer="记忆里没有。", retrieved=[])
        return QueryResult(
            answer="预算 221,000 USD，续约条款未变。",
            retrieved=[EVIDENCE_LONG] + [f"噪声条目 {index}" for index in range(7)],
        )


class _FixedJudge:
    """恒定判错：把三条题都记进判错集合，便于检查诊断项独立于判分。"""

    name = "fixed"

    def grade(self, question: Any, gold: str, prediction: str) -> JudgeResult:
        return JudgeResult(
            correct=False,
            abstained_pred=False,
            stale=False,
            score=0.0,
            judge=self.name,
        )


def _dataset(path: Path) -> Path:
    dialogue = {
        "dialogue_id": "d1",
        "sessions": [[{"role": "user", "content": "记一下预算。", "timestamp": None}]],
        "questions": [
            {
                "qid": "q_spec",
                "question": "当前预算和续约条款是什么？",
                "gold_answer": "221,000 USD",
                "category": "utility_answer",
                "evidence": [EVIDENCE_LONG],
                # 只命中 1 条要点（budget），另一条（renewal）没写进回答
                "judge_spec": {
                    "include": ["221,000 USD", "mutual written renewal"],
                    "not_include": [],
                    "leak_targets": [],
                },
            },
            {
                "qid": "q_empty",
                "question": "q_empty：预算多少？",
                "gold_answer": "221,000 USD",
                "category": "utility_answer",
                "evidence": ["Alpha 项目的预算调整为 221,000 USD。"],
            },
        ],
    }
    path.write_text(json.dumps(dialogue, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _run(tmp_path: Path) -> tuple[Path, dict, dict[str, dict], str]:
    dataset_dir = Path(
        Runner(
            _RetrievalAdapter(),
            str(_dataset(tmp_path / "diag.jsonl")),
            judge=_FixedJudge(),
            out_dir=str(tmp_path / "exp-diag"),
            trace_state=False,
        ).run()
    )
    report_dir = dataset_dir / layout.DIR_REPORT
    metrics = json.loads((report_dir / layout.FILE_METRICS).read_text(encoding="utf-8"))
    records = {
        record["qid"]: record
        for record in (
            json.loads(line)
            for line in (report_dir / layout.FILE_RESULTS).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    report = (report_dir / layout.FILE_REPORT_MD).read_text(encoding="utf-8")
    return dataset_dir, metrics, records, report


def test_evidence_text_recorded_and_truncated(tmp_path: Path) -> None:
    """``results.jsonl`` 落 evidence 原文（截断到 needle 口径），事后可复算命中。"""
    _dir, _metrics, records, _report = _run(tmp_path)
    evidence = records["q_spec"]["evidence"]
    assert len(evidence) == 1
    assert len(evidence[0]) <= EVIDENCE_RECORD_CHARS
    assert evidence[0].startswith("Alpha 项目的预算")
    # 非 judge_spec 题同样落原文
    assert records["q_empty"]["evidence"] == ["Alpha 项目的预算调整为 221,000 USD。"]


def test_include_hit_rate_and_rubric_item_recall(tmp_path: Path) -> None:
    """要点命中率：逐题 1/2，数据集均值同为 0.5；非 judge_spec 题为 None。"""
    _dir, metrics, records, report = _run(tmp_path)
    assert records["q_spec"]["include_hit_rate"] == pytest.approx(0.5)
    assert records["q_empty"]["include_hit_rate"] is None
    assert metrics["overall"]["rubric_item_recall"] == pytest.approx(0.5)
    assert metrics["by_category"]["utility_answer"]["rubric_item_recall"] == pytest.approx(
        0.5
    )
    assert "要点命中率" in report


def test_empty_and_median_retrieval_counted(tmp_path: Path) -> None:
    """空检索单独计数；检索条数中位来自非拒答题（含空列表，如实反映规模）。"""
    _dir, metrics, _records, report = _run(tmp_path)
    assert metrics["overall"]["n_empty_retrieved"] == 1
    # q_spec 返回 8 条、q_empty 返回 0 条 → 中位 4（线性插值）
    assert metrics["overall"]["retrieved_median"] == pytest.approx(4.0)
    assert "检索诊断" in report
    assert "检索结果为空" in report


def test_mixed_membench_versions_flagged_across_datasets(tmp_path: Path) -> None:
    """汇总模式跨数据集代码版本不一致：返回说明，报告头部据此提示"混合口径"。"""
    from membench.evaluate import DatasetResult, DatasetSpec, metadata_inconsistencies

    def result(key: str, hash_value: str | None) -> DatasetResult:
        metrics = {"overall": {"accuracy": 0.5}, "platform": "default"}
        if hash_value:
            metrics["membench_version"] = {
                "version": "0.1.0",
                "hash": hash_value,
                "hash_source": "build",
            }
        return DatasetResult(
            spec=DatasetSpec(key=key, name=key, path=f"/data/{key}.jsonl", source="fetch"),
            run_dir=str(tmp_path / key),
            metrics=metrics,
            capabilities={},
        )

    notes = metadata_inconsistencies(
        [result("gatemem_office", "new123"), result("locomo", "old456")]
    )
    assert any("membench_hash" in note for note in notes)
    assert "new123" in notes[0] and "old456" in notes[0]
    # 两侧同版本 / 都没有该字段：不产生提示
    assert metadata_inconsistencies(
        [result("a", "same1"), result("b", "same1")]
    ) == []
    assert metadata_inconsistencies([result("a", None), result("b", None)]) == []


def test_abstention_question_has_no_token_f1(tmp_path: Path) -> None:
    """拒答题的 gold 是哨兵：字面重合无意义，token_f1 / bleu1 记 None 而不是 0。

    记 0 会把"该拒答"题当成答错的字面重合，拉低数据集均值（locomo 446 题
    会让总体 token_f1 掉约 22%）。
    """
    question = {
        "qid": "q_abstain",
        "question": "What did we decide about the Tokyo trip?",
        "gold_answer": "<ABSTAIN>",
        "category": "abstention",
        "evidence": [],
    }
    dialogue = {
        "dialogue_id": "d1",
        "sessions": [[{"role": "user", "content": "记一条。", "timestamp": None}]],
        "questions": [question],
    }
    dataset = tmp_path / "abs.jsonl"
    dataset.write_text(json.dumps(dialogue, ensure_ascii=False) + "\n", encoding="utf-8")

    class _RefuseAdapter(BaseMemoryAdapter):
        name = "abs-tokens"

        def reset(self) -> None:
            return None

        def ingest(self, messages: list[dict]) -> IngestStats:
            return IngestStats()

        def query(self, question: str) -> QueryResult:
            return QueryResult(answer="That isn't something I can recall.", retrieved=[])

    dataset_dir = Path(
        Runner(
            _RefuseAdapter(),
            str(dataset),
            judge=_FixedJudge(),
            out_dir=str(tmp_path / "exp-abs"),
            trace_state=False,
        ).run()
    )
    report_dir = dataset_dir / layout.DIR_REPORT
    record = json.loads(
        (report_dir / layout.FILE_RESULTS).read_text(encoding="utf-8").strip()
    )
    assert record["token_f1"] is None and record["bleu1"] is None
    metrics = json.loads((report_dir / layout.FILE_METRICS).read_text(encoding="utf-8"))
    assert metrics["by_category"]["abstention"]["token_f1"] is None


def test_worst_categories_ranking_and_issue() -> None:
    """最需要优化榜：按准确率升序、题量 <10 不进榜、失败形态可机检。"""
    from membench.report import worst_categories

    def rec(cat, correct, *, abstained=False, ev=None, state=None, gold="x"):
        return {
            "category": cat,
            "gold_answer": gold,
            "judge": {"correct": correct, "abstained_pred": abstained},
            "evidence_hit": ev,
            "evidence_in_state": state,
        }

    records = [rec("utility_answer", False, ev=False) for _ in range(12)]
    records += [rec("safety_no_memory", False, gold="<ABSTAIN>") for _ in range(10)]
    records += [rec("privacy_answer_redacted", True, ev=True, abstained=True) for _ in range(11)]
    records += [rec("tiny", False, ev=False) for _ in range(3)]  # 题量不够，不进榜

    rows = worst_categories(records)
    cats = [row["category"] for row in rows]
    # 同为 0%：题量大的排在前面（utility 12 题 > safety 10 题）
    assert cats[:2] == ["utility_answer", "safety_no_memory"]
    assert "tiny" not in cats
    by_cat = {row["category"]: row for row in rows}
    assert by_cat["utility_answer"]["issue"] == "证据未命中 12"
    assert by_cat["safety_no_memory"]["issue"] == "未拒答 10/10"
    assert by_cat["privacy_answer_redacted"]["accuracy"] == pytest.approx(1.0)


def test_worst_categories_empty_when_all_small() -> None:
    from membench.report import worst_categories

    assert worst_categories([{"category": "c", "judge": {"correct": False}}]) == []


def test_report_renders_worst_categories_section(tmp_path: Path) -> None:
    """单版本评估报告要有「最需要优化的题型」小节（题量不足时给说明）。"""
    _dir, _metrics, _records, report = _run(tmp_path)
    assert "## 最需要优化的题型（TOP 5）" in report
