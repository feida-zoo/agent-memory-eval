"""judge_fallback 指标与报告警示测试（llm-fallback 占比进 metrics 与报告头部）。"""

from __future__ import annotations

import pytest

from membench.metrics import aggregate
from membench.report import write_run_report_md


def _record(judge_name: str, correct: bool = True) -> dict:
    return {
        "qid": "q1",
        "dialogue_id": "d0",
        "category": "single_hop",
        "question": "q",
        "gold_answer": "g",
        "prediction": "p",
        "retrieved": [],
        "evidence_hit": True,
        "token_f1": 0.5,
        "bleu1": 0.25,
        "recall@5": 1.0,
        "judge": {
            "correct": correct,
            "abstained_pred": False,
            "stale": False,
            "score": 1.0,
            "judge": judge_name,
            "reason": "",
        },
        "latency_ms": 10.0,
        "input_tokens": 10,
        "output_tokens": 5,
        "llm_calls": 0,
        "ingest_latency_ms": 1.0,
        "ingest_llm_calls": 0,
    }


def _metrics(judge_name: str, count: int) -> dict:
    metrics = aggregate([_record(judge_name) for _ in range(count)])
    metrics.update({"run_id": "t", "overall": metrics["overall"]})
    return metrics


class TestFallbackAggregate:
    def test_no_llm_judge_omits_key(self) -> None:
        """无 LLM judge（全部 heuristic）时省略 judge_fallback 键。"""
        metrics = aggregate([_record("heuristic") for _ in range(3)])
        assert "judge_fallback" not in metrics

    def test_llm_judge_without_fallback_is_zero(self) -> None:
        """有 LLM judge 但无回退时 count=0、ratio=0.0。"""
        metrics = aggregate([_record("llm") for _ in range(4)])
        assert metrics["judge_fallback"] == {"count": 0, "ratio": 0.0}

    def test_fallback_count_and_ratio(self) -> None:
        records = [_record("llm") for _ in range(3)] + [_record("llm-fallback")]
        metrics = aggregate(records)
        assert metrics["judge_fallback"]["count"] == 1
        assert metrics["judge_fallback"]["ratio"] == pytest.approx(0.25)


class TestFallbackReport:
    def _render(self, tmp_path, metrics) -> str:
        write_run_report_md(str(tmp_path), [], metrics)
        return (tmp_path / "report.md").read_text(encoding="utf-8")

    def test_report_shows_fallback_line(self, tmp_path) -> None:
        metrics = _metrics("llm-fallback", 2)
        report = self._render(tmp_path, metrics)
        assert "LLM judge 回退" in report
        assert "100.0%" in report

    def test_report_warns_above_5_percent(self, tmp_path) -> None:
        metrics = _metrics("llm-fallback", 2)
        report = self._render(tmp_path, metrics)
        assert "判分质量可能退化" in report

    def test_report_no_warning_below_5_percent(self, tmp_path) -> None:
        # 1/20 = 5%（不超过阈值）→ 展示回退行但不警示。
        records = [_record("llm") for _ in range(19)] + [_record("llm-fallback")]
        metrics = aggregate(records)
        report = self._render(tmp_path, metrics)
        assert "判分质量可能退化" not in report
        assert "LLM judge 回退" in report

    def test_report_without_fallback_key(self, tmp_path) -> None:
        metrics = aggregate([_record("heuristic")])
        report = self._render(tmp_path, metrics)
        assert "LLM judge 回退" not in report
