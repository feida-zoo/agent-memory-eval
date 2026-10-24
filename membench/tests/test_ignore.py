"""题目级忽略标记（ignore=true）测试：runner 跳过、报告注明、能力扫描排除。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from membench import layout
from membench.adapters import MockStrongAdapter
from membench.evaluate import scan_capabilities
from membench.report import render_dialogue_report
from membench.runner import Runner


def _write_dataset(path: Path) -> Path:
    """写出含 2 道忽略题（共 4 题）的迷你数据集。"""
    dialogue = {
        "dialogue_id": "d1",
        "sessions": [
            [
                {"role": "user", "content": "项目截止日是周五。", "timestamp": None},
                {"role": "assistant", "content": "好的，已记住。", "timestamp": None},
            ]
        ],
        "questions": [
            {
                "qid": "d1_q1",
                "question": "项目截止日是哪天？",
                "gold_answer": "周五",
                "category": "single_hop",
                "evidence": ["项目截止日是周五"],
            },
            {
                "qid": "d1_q2",
                "question": "助手回复了什么？",
                "gold_answer": "好的，已记住",
                "category": "single_hop",
                "evidence": ["好的，已记住"],
            },
            {
                "qid": "d1_q3",
                "question": "同事的薪资是多少？",
                "gold_answer": "<ABSTAIN>",
                "category": "abstention",
                "evidence": [],
                "ignore": True,
                "ignore_reason": "多用户权限语义，单用户个人 agent 场景不适用",
            },
            {
                "qid": "d1_q4",
                "question": "老板的私人电话是多少？",
                "gold_answer": "<ABSTAIN>",
                "category": "abstention",
                "evidence": [],
                "ignore": True,
                "ignore_reason": "多用户权限语义，单用户个人 agent 场景不适用",
            },
        ],
    }
    path.write_text(json.dumps(dialogue, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def test_runner_skips_ignored_questions(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """忽略题不查询、不写 results.jsonl、不计入 n_questions，报告注明忽略数。"""
    ds_path = _write_dataset(tmp_path / "ds.jsonl")
    runner = Runner(
        MockStrongAdapter(),
        str(ds_path),
        judge="heuristic",
        out_dir=str(tmp_path / "eval_ignore"),
    )
    dataset_dir = Path(runner.run())
    report_dir = dataset_dir / layout.DIR_REPORT

    results = [
        json.loads(line)
        for line in (report_dir / layout.FILE_RESULTS).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    qids = {r["qid"] for r in results}
    assert qids == {"d1_q1", "d1_q2"}  # 被忽略题完全不产生记录

    metrics = json.loads((report_dir / layout.FILE_METRICS).read_text(encoding="utf-8"))
    assert metrics["overall"]["n_questions"] == 2  # 分母不含被忽略题
    assert metrics["n_ignored"] == 2
    assert "abstention" not in metrics["by_category"]  # 忽略题不进分类聚合

    report = (report_dir / layout.FILE_REPORT_MD).read_text(encoding="utf-8")
    assert "- 忽略题数: 2（ignore=true，未参与评分）" in report

    # 对话级产物：同一套忽略口径（分母与忽略数都按该对话统计）
    dialogue_report_dir = dataset_dir / "d1" / layout.DIR_REPORT
    assert (dataset_dir / "d1" / layout.DIR_RUNTIME).is_dir()
    dialogue_metrics = json.loads(
        (dialogue_report_dir / layout.FILE_METRICS).read_text(encoding="utf-8")
    )
    assert dialogue_metrics["dialogue_id"] == "d1"
    assert dialogue_metrics["overall"]["n_questions"] == 2
    assert dialogue_metrics["n_ignored"] == 2
    dialogue_qids = {
        json.loads(line)["qid"]
        for line in (dialogue_report_dir / layout.FILE_RESULTS)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    }
    assert dialogue_qids == {"d1_q1", "d1_q2"}
    assert "- 忽略题数: 2（ignore=true，未参与评分）" in (
        dialogue_report_dir / layout.FILE_REPORT_MD
    ).read_text(encoding="utf-8")

    out = capsys.readouterr().out
    assert "[membench] 已忽略 2 道标记题（ignore=true）" in out


def test_report_omits_ignore_line_when_zero(tmp_path: Path) -> None:
    """n_ignored 为 0 / 缺失时报告不输出忽略题数行；对话级报告不引用图片。"""
    out_path = render_dialogue_report(
        str(tmp_path / "d1" / layout.DIR_REPORT),
        records=[],
        metrics={
            "run_id": "t",
            "dialogue_id": "d1",
            "overall": {"n_questions": 0},
            "by_category": {},
        },
    )
    assert Path(out_path).name == layout.FILE_REPORT_MD
    report = Path(out_path).read_text(encoding="utf-8")
    assert "忽略题数" not in report
    assert "![" not in report  # 对话级不出图，也不留打不开的相对引用


def test_scan_capabilities_excludes_ignored(tmp_path: Path) -> None:
    """全部 abstention 题都被忽略的极端集：has_abstention 应为 False。"""
    ds_path = _write_dataset(tmp_path / "ds.jsonl")
    caps = scan_capabilities(ds_path)
    assert caps == {
        "has_evidence": True,
        "has_stale": False,
        "has_abstention": False,  # 两道 abstention 题均被忽略
    }
