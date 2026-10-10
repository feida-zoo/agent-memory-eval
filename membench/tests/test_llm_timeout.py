"""LLMJudge 判分超时默认值 90s 与 --llm-timeout 的回归测试（修复 4）。"""

from __future__ import annotations

from membench.cli import _build_parser, _make_judge
from membench.judges.llm import LLMJudge


def test_default_timeout_is_90() -> None:
    assert LLMJudge().timeout == 90.0


def test_explicit_timeout_respected() -> None:
    assert LLMJudge(timeout=5).timeout == 5.0


def test_cli_run_llm_timeout_parsing() -> None:
    """run 子命令解析 --llm-timeout 并透传给 LLMJudge。"""
    parser = _build_parser()
    args = parser.parse_args(
        ["run", "--adapter", "mock-strong", "--judge", "llm", "--llm-timeout", "5"]
    )
    assert args.llm_timeout == 5.0
    judge = _make_judge(args)
    assert isinstance(judge, LLMJudge)
    assert judge.timeout == 5.0


def test_cli_run_llm_timeout_default_none() -> None:
    """不传 --llm-timeout 时用 judge 内建默认值 90s。"""
    parser = _build_parser()
    args = parser.parse_args(["run", "--adapter", "mock-strong", "--judge", "llm"])
    assert args.llm_timeout is None
    judge = _make_judge(args)
    assert isinstance(judge, LLMJudge)
    assert judge.timeout == 90.0


def test_cli_evaluate_llm_timeout_parsing() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        ["evaluate", "--adapter", "mock-strong", "--judge", "llm",
         "--llm-timeout", "12.5", "--quick"]
    )
    assert args.llm_timeout == 12.5
    judge = _make_judge(args)
    assert isinstance(judge, LLMJudge)
    assert judge.timeout == 12.5


def test_cli_jobs_parsing() -> None:
    """run / evaluate 的 --jobs 解析（修复 3 的 CLI 部分）。"""
    parser = _build_parser()
    args = parser.parse_args(["run", "--adapter", "mock-strong", "--jobs", "4"])
    assert args.jobs == 4
    args = parser.parse_args(["run", "--adapter", "mock-strong"])
    assert args.jobs == 1
    args = parser.parse_args(["evaluate", "--adapter", "mock-strong", "--jobs", "8"])
    assert args.jobs == 8
